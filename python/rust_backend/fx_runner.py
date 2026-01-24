from __future__ import annotations

from typing import Any, Callable, Dict
import logging
import operator

import numpy as np
import torch
import torch.nn.functional as F
from torch.fx import GraphModule

import rustorch

_LOG = logging.getLogger(__name__)

_SUPPORTED_FUNCTIONS = {
    torch.add,
    operator.add,
    torch.matmul,
    operator.matmul,
    torch.relu,
    torch.nn.functional.relu,
    torch.flatten,
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.t.default,
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.relu.default,
    torch.ops.aten.sum.dim_IntList,
    torch.ops.aten.threshold_backward.default,
    torch.ops.aten.sym_size.int,
    F.linear,
}

if hasattr(torch, "_C") and hasattr(torch._C, "_nn") and hasattr(torch._C._nn, "linear"):
    _SUPPORTED_FUNCTIONS.add(torch._C._nn.linear)

_SUPPORTED_METHODS = {
    "relu",
    "add",
    "__add__",
    "matmul",
    "__matmul__",
    "flatten",
    "view",
}


def _to_pytensor(value: Any) -> rustorch.PyTensor:
    if isinstance(value, rustorch.PyTensor):
        return value
    if isinstance(value, torch.Tensor):
        data = value.detach().cpu().numpy()
        return rustorch.PyTensor(data, requires_grad=False)
    if hasattr(value, "numpy") and not isinstance(value, torch.Tensor):
        data = value.numpy()
        return rustorch.PyTensor(data, requires_grad=False)
    if isinstance(value, (int, float)):
        return rustorch.PyTensor(np.array(value, dtype=np.float32), requires_grad=False)
    raise TypeError(f"Unsupported input type for PyTensor conversion: {type(value)}")


def _to_torch(value: Any) -> torch.Tensor:
    if value is None:
        return value
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value
    if isinstance(value, rustorch.PyTensor):
        return torch.from_numpy(value.numpy())
    if hasattr(value, "numpy"):
        return torch.from_numpy(value.numpy())
    if isinstance(value, (int, float)):
        return value
    raise TypeError(f"Unsupported output type for torch conversion: {type(value)}")


def _to_torch_tree(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        converted = [_to_torch_tree(item) for item in value]
        return type(value)(converted)
    return _to_torch(value)


def _normalize_dim(dim: int, rank: int) -> int:
    if dim < 0:
        dim += rank
    return dim


def _flatten_shape(shape: tuple[int, ...], start_dim: int, end_dim: int) -> list[int]:
    rank = len(shape)
    start_dim = _normalize_dim(start_dim, rank)
    end_dim = _normalize_dim(end_dim, rank)
    if start_dim > end_dim:
        raise ValueError(f"flatten start_dim ({start_dim}) > end_dim ({end_dim})")
    leading = list(shape[:start_dim])
    flattened = int(np.prod(shape[start_dim : end_dim + 1], dtype=np.int64))
    trailing = list(shape[end_dim + 1 :])
    return leading + [flattened] + trailing


def _flatten_tensor(value: rustorch.PyTensor, start_dim: int, end_dim: int) -> rustorch.PyTensor:
    new_shape = _flatten_shape(tuple(value.shape()), start_dim, end_dim)
    return value.reshape(new_shape)

def _as_int(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, torch.Tensor):
        return int(value.item())
    if isinstance(value, rustorch.PyTensor):
        return int(_to_torch(value).item())
    if hasattr(value, "numpy"):
        return int(_to_torch(value).item())
    return int(value)


def _as_float(value: Any) -> float:
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    if isinstance(value, torch.Tensor):
        return float(value.item())
    if isinstance(value, rustorch.PyTensor):
        return float(_to_torch(value).item())
    if hasattr(value, "numpy"):
        return float(_to_torch(value).item())
    return float(value)

def _shape_from_arg(arg: Any) -> list[int]:
    if isinstance(arg, (list, tuple, torch.Size)):
        return [_as_int(dim) for dim in arg]
    if isinstance(arg, torch.Tensor):
        if arg.numel() == 1:
            return [int(arg.item())]
        return [int(dim) for dim in arg.flatten().tolist()]
    if isinstance(arg, rustorch.PyTensor) or hasattr(arg, "numpy"):
        tensor = _to_torch(arg)
        if tensor.numel() == 1:
            return [int(tensor.item())]
        return [int(dim) for dim in tensor.flatten().tolist()]
    return [int(arg)]


def _shape_from_args(args: tuple[Any, ...]) -> list[int]:
    if len(args) == 2:
        return _shape_from_arg(args[1])
    dims: list[int] = []
    for arg in args[1:]:
        dims.extend(_shape_from_arg(arg))
    return dims


def _resolve_reshape_shape(shape: tuple[int, ...], numel: int) -> list[int]:
    resolved = list(shape)
    if resolved.count(-1) > 1:
        raise ValueError("reshape can only infer one dimension")
    if -1 in resolved:
        known = 1
        for dim in resolved:
            if dim != -1:
                known *= dim
        if known == 0:
            raise ValueError("cannot infer reshape dimension with zero known size")
        inferred = numel // known
        if inferred * known != numel:
            raise ValueError("inferred reshape size does not match numel")
        resolved[resolved.index(-1)] = inferred
    return resolved


def _reshape_tensor(value: rustorch.PyTensor, shape: tuple[int, ...]) -> rustorch.PyTensor:
    numel = int(np.prod(value.shape(), dtype=np.int64))
    resolved = _resolve_reshape_shape(shape, numel)
    return value.reshape(resolved)


def _transpose_tensor(value: rustorch.PyTensor) -> rustorch.PyTensor:
    arr = value.numpy().T
    return rustorch.PyTensor(arr, requires_grad=False)


def _scale_tensor(value: rustorch.PyTensor, scale: float) -> rustorch.PyTensor:
    if scale == 1.0:
        return value
    return value.mul(_to_pytensor(scale))


def _sum_tensor(value: rustorch.PyTensor, dims: list[int], keepdim: bool) -> rustorch.PyTensor:
    arr = value.numpy()
    for dim in sorted(dims):
        arr = arr.sum(axis=dim, keepdims=keepdim)
    return rustorch.PyTensor(arr, requires_grad=False)


def _threshold_backward(grad_out: rustorch.PyTensor, inp: rustorch.PyTensor, threshold: float) -> rustorch.PyTensor:
    grad = grad_out.numpy()
    mask = inp.numpy() > threshold
    return rustorch.PyTensor(grad * mask, requires_grad=False)


def _sym_size_int(value: Any, dim: int) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.shape[dim])
    if isinstance(value, rustorch.PyTensor) or hasattr(value, "shape"):
        return int(value.shape()[dim])
    raise TypeError(f"Unsupported type for sym_size.int: {type(value)}")


def _run_graph(gm: GraphModule, *inputs: torch.Tensor) -> torch.Tensor:
    env: Dict[str, Any] = {}
    input_iter = iter(inputs)

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            env[node.name] = _to_pytensor(next(input_iter))
        elif node.op == "call_function":
            if node.target not in _SUPPORTED_FUNCTIONS:
                raise RuntimeError(f"Unsupported call_function: {node.target}")
            args = torch.fx.node.map_arg(node.args, lambda n: env[n.name] if hasattr(n, "name") else n)
            if node.target in (torch.add, operator.add):
                left = _to_pytensor(args[0])
                right = _to_pytensor(args[1])
                env[node.name] = left.add(right)
            elif node.target in (torch.matmul, operator.matmul):
                left = _to_pytensor(args[0])
                right = _to_pytensor(args[1])
                env[node.name] = left.matmul(right)
            elif node.target in (torch.relu, torch.nn.functional.relu):
                value = _to_pytensor(args[0])
                env[node.name] = value.relu()
            elif node.target in (F.linear, getattr(torch._C._nn, "linear", None)):
                inp = _to_pytensor(args[0])
                weight = args[1]
                if isinstance(weight, torch.Tensor):
                    weight = weight.t()
                elif isinstance(weight, rustorch.PyTensor):
                    weight = _to_torch(weight).t()
                else:
                    raise TypeError(f"Unsupported weight type for linear: {type(weight)}")
                weight = _to_pytensor(weight)
                out = inp.matmul(weight)
                if len(args) > 2 and args[2] is not None:
                    bias = _to_pytensor(args[2])
                    out = out.add(bias)
                env[node.name] = out
            elif node.target is torch.flatten:
                value = _to_pytensor(args[0])
                start_dim = _as_int(args[1]) if len(args) > 1 else 0
                end_dim = _as_int(args[2]) if len(args) > 2 else -1
                env[node.name] = _flatten_tensor(value, start_dim, end_dim)
            elif node.target in (torch.ops.aten.view.default, torch.ops.aten.reshape.default):
                value = _to_pytensor(args[0])
                shape = tuple(_shape_from_args(args))
                env[node.name] = _reshape_tensor(value, shape)
            elif node.target is torch.ops.aten.detach.default:
                value = _to_pytensor(args[0])
                env[node.name] = value
            elif node.target is torch.ops.aten.t.default:
                value = _to_pytensor(args[0])
                env[node.name] = _transpose_tensor(value)
            elif node.target is torch.ops.aten.mm.default:
                left = _to_pytensor(args[0])
                right = _to_pytensor(args[1])
                env[node.name] = left.matmul(right)
            elif node.target is torch.ops.aten.addmm.default:
                inp = _to_pytensor(args[0])
                mat1 = _to_pytensor(args[1])
                mat2 = _to_pytensor(args[2])
                beta = _as_float(args[3]) if len(args) > 3 else 1.0
                alpha = _as_float(args[4]) if len(args) > 4 else 1.0
                out = mat1.matmul(mat2)
                out = _scale_tensor(out, alpha)
                inp = _scale_tensor(inp, beta)
                env[node.name] = out.add(inp)
            elif node.target is torch.ops.aten.relu.default:
                value = _to_pytensor(args[0])
                env[node.name] = value.relu()
            elif node.target is torch.ops.aten.sum.dim_IntList:
                value = _to_pytensor(args[0])
                dims = _shape_from_arg(args[1])
                keepdim = bool(args[2]) if len(args) > 2 else False
                env[node.name] = _sum_tensor(value, dims, keepdim)
            elif node.target is torch.ops.aten.threshold_backward.default:
                grad_out = _to_pytensor(args[0])
                inp = _to_pytensor(args[1])
                threshold = _as_float(args[2]) if len(args) > 2 else 0.0
                env[node.name] = _threshold_backward(grad_out, inp, threshold)
            elif node.target is torch.ops.aten.sym_size.int:
                value = args[0]
                dim = _as_int(args[1])
                env[node.name] = _sym_size_int(value, dim)
            else:
                raise RuntimeError(f"Unhandled call_function: {node.target}")
        elif node.op == "call_method":
            if node.target not in _SUPPORTED_METHODS:
                raise RuntimeError(f"Unsupported call_method: {node.target}")
            args = torch.fx.node.map_arg(node.args, lambda n: env[n.name] if hasattr(n, "name") else n)
            if node.target in ("relu",):
                env[node.name] = _to_pytensor(args[0]).relu()
            elif node.target in ("add", "__add__"):
                env[node.name] = _to_pytensor(args[0]).add(_to_pytensor(args[1]))
            elif node.target in ("matmul", "__matmul__"):
                env[node.name] = _to_pytensor(args[0]).matmul(_to_pytensor(args[1]))
            elif node.target in ("flatten",):
                value = _to_pytensor(args[0])
                start_dim = _as_int(args[1]) if len(args) > 1 else 0
                end_dim = _as_int(args[2]) if len(args) > 2 else -1
                env[node.name] = _flatten_tensor(value, start_dim, end_dim)
            elif node.target in ("view",):
                value = _to_pytensor(args[0])
                shape = tuple(_shape_from_args(args))
                env[node.name] = _reshape_tensor(value, shape)
            else:
                raise RuntimeError(f"Unhandled call_method: {node.target}")
        elif node.op == "call_module":
            module = gm.get_submodule(node.target)
            args = torch.fx.node.map_arg(node.args, lambda n: env[n.name] if hasattr(n, "name") else n)
            if isinstance(module, torch.nn.ReLU):
                env[node.name] = _to_pytensor(args[0]).relu()
            elif isinstance(module, torch.nn.Linear):
                inp = _to_pytensor(args[0])
                weight = _to_pytensor(module.weight.t())
                out = inp.matmul(weight)
                if module.bias is not None:
                    bias = _to_pytensor(module.bias)
                    out = out.add(bias)
                env[node.name] = out
            elif isinstance(module, torch.nn.Flatten):
                value = _to_pytensor(args[0])
                env[node.name] = _flatten_tensor(value, module.start_dim, module.end_dim)
            else:
                raise RuntimeError(f"Unsupported call_module: {type(module)}")
        elif node.op == "output":
            result = torch.fx.node.map_arg(node.args[0], lambda n: env[n.name] if hasattr(n, "name") else n)
            return _to_torch_tree(result)
        else:
            raise RuntimeError(f"Unsupported node op: {node.op}")

    raise RuntimeError("FX graph had no output node")


def run_fx(gm: GraphModule, example_inputs: list[torch.Tensor]) -> Callable:
    warned = False

    def compiled(*inputs: torch.Tensor) -> torch.Tensor:
        nonlocal warned
        try:
            return _run_graph(gm, *inputs)
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            if not warned:
                _LOG.warning("rustorch FX runner fallback to eager: %s", exc)
                warned = True
            return gm(*inputs)

    return compiled
