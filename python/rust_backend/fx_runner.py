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
    operator.getitem,
    torch.matmul,
    operator.matmul,
    torch.relu,
    torch.nn.functional.relu,
    torch.nn.functional.batch_norm,
    torch.nn.functional.conv2d,
    torch.nn.functional.dropout,
    torch.nn.functional.log_softmax,
    torch.nn.functional.max_pool2d,
    torch.flatten,
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.t.default,
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.relu.default,
    torch.ops.aten.batch_norm.default,
    torch.ops.aten.native_batch_norm.default,
    torch.ops.aten.native_batch_norm_backward.default,
    torch.ops.aten.convolution.default,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.dropout.default,
    torch.ops.aten.native_dropout.default,
    torch.ops.aten.native_dropout_backward.default,
    torch.ops.aten.log_softmax.default,
    torch.ops.aten._log_softmax.default,
    torch.ops.aten.sum.dim_IntList,
    torch.ops.aten.threshold_backward.default,
    torch.ops.aten.sym_size.int,
    torch.ops.aten.max_pool2d_with_indices.default,
    torch.ops.aten.max_pool2d_with_indices_backward.default,
    torch.ops.aten.nll_loss_forward.default,
    torch.ops.aten.nll_loss_backward.default,
    torch.ops.aten.convolution_backward.default,
    torch.ops.aten.log_softmax_backward_data.default,
    torch.ops.aten._log_softmax_backward_data.default,
    F.cross_entropy,
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


def _is_torch_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor)


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

def _to_torch_arg(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        converted = [_to_torch_arg(item) for item in value]
        return type(value)(converted)
    if isinstance(value, dict):
        return {key: _to_torch_arg(val) for key, val in value.items()}
    if _is_torch_tensor(value) or isinstance(value, rustorch.PyTensor):
        return _to_torch(value)
    return value


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


def _flatten_tensor(value: Any, start_dim: int, end_dim: int) -> Any:
    if _is_torch_tensor(value):
        return torch.flatten(value, start_dim=start_dim, end_dim=end_dim)
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


def _conv2d_from_args(args: tuple[Any, ...]) -> torch.Tensor:
    inp = _to_torch(args[0])
    weight = _to_torch(args[1])
    bias = _to_torch(args[2]) if len(args) > 2 and args[2] is not None else None
    stride = args[3] if len(args) > 3 else 1
    padding = args[4] if len(args) > 4 else 0
    dilation = args[5] if len(args) > 5 else 1
    if len(args) > 8:
        groups = _as_int(args[8])
    elif len(args) > 6:
        groups = _as_int(args[6])
    else:
        groups = 1
    return F.conv2d(inp, weight, bias, stride, padding, dilation, groups)


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


def _reshape_tensor(value: Any, shape: tuple[int, ...]) -> Any:
    if _is_torch_tensor(value):
        return value.reshape(shape)
    numel = int(np.prod(value.shape(), dtype=np.int64))
    resolved = _resolve_reshape_shape(shape, numel)
    return value.reshape(resolved)


def _transpose_tensor(value: Any) -> Any:
    if _is_torch_tensor(value):
        return value.t()
    arr = value.numpy().T
    return rustorch.PyTensor(arr, requires_grad=False)


def _scale_tensor(value: Any, scale: float) -> Any:
    if scale == 1.0:
        return value
    if _is_torch_tensor(value):
        return value * scale
    return value.mul(_to_pytensor(scale))


def _sum_tensor(value: Any, dims: list[int], keepdim: bool) -> Any:
    if _is_torch_tensor(value):
        return torch.sum(value, dim=tuple(dims), keepdim=keepdim)
    arr = value.numpy()
    for dim in sorted(dims):
        arr = arr.sum(axis=dim, keepdims=keepdim)
    return rustorch.PyTensor(arr, requires_grad=False)


def _threshold_backward(grad_out: Any, inp: Any, threshold: float) -> Any:
    if _is_torch_tensor(grad_out) or _is_torch_tensor(inp):
        grad_out_t = _to_torch(grad_out)
        inp_t = _to_torch(inp)
        return torch.where(inp_t > threshold, grad_out_t, torch.zeros_like(grad_out_t))
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
            env[node.name] = next(input_iter)
        elif node.op == "call_function":
            if node.target not in _SUPPORTED_FUNCTIONS:
                raise RuntimeError(f"Unsupported call_function: {node.target}")
            args = torch.fx.node.map_arg(node.args, lambda n: env[n.name] if hasattr(n, "name") else n)
            if node.target is operator.getitem:
                env[node.name] = args[0][args[1]]
            elif node.target in (torch.add, operator.add):
                left = args[0]
                right = args[1]
                if _is_torch_tensor(left) or _is_torch_tensor(right):
                    env[node.name] = _to_torch(left) + _to_torch(right)
                else:
                    env[node.name] = _to_pytensor(left).add(_to_pytensor(right))
            elif node.target in (torch.matmul, operator.matmul):
                left = args[0]
                right = args[1]
                if _is_torch_tensor(left) or _is_torch_tensor(right):
                    env[node.name] = _to_torch(left).matmul(_to_torch(right))
                else:
                    env[node.name] = _to_pytensor(left).matmul(_to_pytensor(right))
            elif node.target in (torch.relu, torch.nn.functional.relu):
                value = args[0]
                if _is_torch_tensor(value):
                    env[node.name] = F.relu(value)
                else:
                    env[node.name] = _to_pytensor(value).relu()
            elif node.target in (F.linear, getattr(torch._C._nn, "linear", None)):
                inp = args[0]
                weight = args[1]
                bias = args[2] if len(args) > 2 else None
                if _is_torch_tensor(inp) or _is_torch_tensor(weight):
                    env[node.name] = F.linear(_to_torch(inp), _to_torch(weight), _to_torch(bias) if bias is not None else None)
                else:
                    inp = _to_pytensor(inp)
                    weight_t = weight
                    if isinstance(weight_t, torch.Tensor):
                        weight_t = weight_t.t()
                    elif isinstance(weight_t, rustorch.PyTensor):
                        weight_t = _to_torch(weight_t).t()
                    else:
                        raise TypeError(f"Unsupported weight type for linear: {type(weight_t)}")
                    weight_t = _to_pytensor(weight_t)
                    out = inp.matmul(weight_t)
                    if bias is not None:
                        out = out.add(_to_pytensor(bias))
                    env[node.name] = out
            elif node.target in (torch.nn.functional.conv2d, torch.ops.aten.convolution.default, torch.ops.aten.conv2d.default):
                env[node.name] = _conv2d_from_args(args)
            elif node.target in (F.cross_entropy,):
                input_t = _to_torch(args[0])
                target_t = _to_torch(args[1])
                weight = _to_torch(args[2]) if len(args) > 2 and args[2] is not None else None
                ignore_index = _as_int(args[3]) if len(args) > 3 and args[3] is not None else -100
                reduction = args[4] if len(args) > 4 and args[4] is not None else "mean"
                label_smoothing = _as_float(args[5]) if len(args) > 5 and args[5] is not None else 0.0
                env[node.name] = F.cross_entropy(
                    input_t,
                    target_t,
                    weight=weight,
                    ignore_index=ignore_index,
                    reduction=reduction,
                    label_smoothing=label_smoothing,
                )
            elif node.target in (torch.nn.functional.batch_norm,):
                input_v = args[0]
                running_mean = args[1] if len(args) > 1 else None
                running_var = args[2] if len(args) > 2 else None
                weight = args[3] if len(args) > 3 else None
                bias = args[4] if len(args) > 4 else None
                training = bool(args[5]) if len(args) > 5 else False
                momentum = _as_float(args[6]) if len(args) > 6 and args[6] is not None else 0.1
                eps = _as_float(args[7]) if len(args) > 7 and args[7] is not None else 1e-5
                if any(
                    _is_torch_tensor(val)
                    for val in (input_v, running_mean, running_var, weight, bias)
                    if val is not None
                ):
                    env[node.name] = F.batch_norm(
                        _to_torch(input_v),
                        _to_torch(running_mean) if running_mean is not None else None,
                        _to_torch(running_var) if running_var is not None else None,
                        _to_torch(weight) if weight is not None else None,
                        _to_torch(bias) if bias is not None else None,
                        training=training,
                        momentum=momentum,
                        eps=eps,
                    )
                else:
                    env[node.name] = _to_pytensor(input_v).batch_norm(
                        _to_pytensor(running_mean) if running_mean is not None else None,
                        _to_pytensor(running_var) if running_var is not None else None,
                        _to_pytensor(weight) if weight is not None else None,
                        _to_pytensor(bias) if bias is not None else None,
                        training=training,
                        momentum=momentum,
                        eps=eps,
                    )
            elif node.target in (torch.nn.functional.dropout,):
                input_v = args[0]
                p = _as_float(args[1]) if len(args) > 1 and args[1] is not None else 0.5
                training = bool(args[2]) if len(args) > 2 and args[2] is not None else True
                inplace = bool(args[3]) if len(args) > 3 and args[3] is not None else False
                if _is_torch_tensor(input_v):
                    env[node.name] = F.dropout(_to_torch(input_v), p=p, training=training, inplace=inplace)
                else:
                    if inplace:
                        raise RuntimeError("inplace dropout not supported for rustorch tensors")
                    env[node.name] = _to_pytensor(input_v).dropout(p=p, training=training)
            elif node.target in (torch.nn.functional.log_softmax,):
                input_t = _to_torch(args[0])
                dim = _as_int(args[1]) if len(args) > 1 and args[1] is not None else -1
                env[node.name] = F.log_softmax(input_t, dim=dim)
            elif node.target in (torch.nn.functional.max_pool2d,):
                input_v = args[0]
                kernel_size = args[1]
                stride = args[2] if len(args) > 2 else None
                padding = args[3] if len(args) > 3 else 0
                dilation = args[4] if len(args) > 4 else 1
                ceil_mode = bool(args[5]) if len(args) > 5 else False
                if _is_torch_tensor(input_v):
                    env[node.name] = F.max_pool2d(
                        _to_torch(input_v),
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        ceil_mode=ceil_mode,
                    )
                else:
                    if not isinstance(kernel_size, int):
                        raise RuntimeError("rustorch max_pool2d expects integer kernel_size")
                    env[node.name] = _to_pytensor(input_v).max_pool2d(
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        ceil_mode=ceil_mode,
                    )
            elif node.target is torch.flatten:
                value = args[0]
                start_dim = _as_int(args[1]) if len(args) > 1 else 0
                end_dim = _as_int(args[2]) if len(args) > 2 else -1
                env[node.name] = _flatten_tensor(value, start_dim, end_dim)
            elif node.target in (torch.ops.aten.view.default, torch.ops.aten.reshape.default):
                value = args[0]
                shape = tuple(_shape_from_args(args))
                env[node.name] = _reshape_tensor(value, shape)
            elif node.target is torch.ops.aten.detach.default:
                value = args[0]
                env[node.name] = value.detach() if _is_torch_tensor(value) else _to_pytensor(value)
            elif node.target is torch.ops.aten.t.default:
                value = args[0]
                env[node.name] = _transpose_tensor(value)
            elif node.target is torch.ops.aten.mm.default:
                left = args[0]
                right = args[1]
                if _is_torch_tensor(left) or _is_torch_tensor(right):
                    env[node.name] = _to_torch(left).matmul(_to_torch(right))
                else:
                    env[node.name] = _to_pytensor(left).matmul(_to_pytensor(right))
            elif node.target is torch.ops.aten.addmm.default:
                inp = args[0]
                mat1 = args[1]
                mat2 = args[2]
                beta = _as_float(args[3]) if len(args) > 3 else 1.0
                alpha = _as_float(args[4]) if len(args) > 4 else 1.0
                if _is_torch_tensor(inp) or _is_torch_tensor(mat1) or _is_torch_tensor(mat2):
                    env[node.name] = torch.addmm(_to_torch(inp), _to_torch(mat1), _to_torch(mat2), beta=beta, alpha=alpha)
                else:
                    inp_pt = _to_pytensor(inp)
                    mat1_pt = _to_pytensor(mat1)
                    mat2_pt = _to_pytensor(mat2)
                    out = mat1_pt.matmul(mat2_pt)
                    out = _scale_tensor(out, alpha)
                    inp_pt = _scale_tensor(inp_pt, beta)
                    env[node.name] = out.add(inp_pt)
            elif node.target is torch.ops.aten.relu.default:
                value = args[0]
                if _is_torch_tensor(value):
                    env[node.name] = F.relu(value)
                else:
                    env[node.name] = _to_pytensor(value).relu()
            elif node.target is torch.ops.aten.batch_norm.default:
                converted_args = tuple(_to_torch_arg(arg) for arg in args)
                env[node.name] = torch.ops.aten.batch_norm.default(*converted_args)
            elif node.target is torch.ops.aten.native_batch_norm.default:
                converted_args = tuple(_to_torch_arg(arg) for arg in args)
                env[node.name] = torch.ops.aten.native_batch_norm.default(*converted_args)
            elif node.target is torch.ops.aten.native_batch_norm_backward.default:
                converted_args = tuple(_to_torch_arg(arg) for arg in args)
                env[node.name] = torch.ops.aten.native_batch_norm_backward.default(*converted_args)
            elif node.target is torch.ops.aten.dropout.default:
                converted_args = tuple(_to_torch_arg(arg) for arg in args)
                env[node.name] = torch.ops.aten.dropout.default(*converted_args)
            elif node.target is torch.ops.aten.native_dropout.default:
                converted_args = tuple(_to_torch_arg(arg) for arg in args)
                env[node.name] = torch.ops.aten.native_dropout.default(*converted_args)
            elif node.target is torch.ops.aten.native_dropout_backward.default:
                converted_args = tuple(_to_torch_arg(arg) for arg in args)
                env[node.name] = torch.ops.aten.native_dropout_backward.default(*converted_args)
            elif node.target is torch.ops.aten.log_softmax.default:
                converted_args = tuple(_to_torch_arg(arg) for arg in args)
                env[node.name] = torch.ops.aten.log_softmax.default(*converted_args)
            elif node.target is torch.ops.aten._log_softmax.default:
                converted_args = tuple(_to_torch_arg(arg) for arg in args)
                env[node.name] = torch.ops.aten._log_softmax.default(*converted_args)
            elif node.target is torch.ops.aten.max_pool2d_with_indices.default:
                converted_args = tuple(_to_torch_arg(arg) for arg in args)
                env[node.name] = torch.ops.aten.max_pool2d_with_indices.default(*converted_args)
            elif node.target is torch.ops.aten.max_pool2d_with_indices_backward.default:
                converted_args = tuple(_to_torch_arg(arg) for arg in args)
                env[node.name] = torch.ops.aten.max_pool2d_with_indices_backward.default(*converted_args)
            elif node.target is torch.ops.aten.sum.dim_IntList:
                value = args[0]
                dims = _shape_from_arg(args[1])
                keepdim = bool(args[2]) if len(args) > 2 else False
                env[node.name] = _sum_tensor(value, dims, keepdim)
            elif node.target is torch.ops.aten.threshold_backward.default:
                grad_out = args[0]
                inp = args[1]
                threshold = _as_float(args[2]) if len(args) > 2 else 0.0
                env[node.name] = _threshold_backward(grad_out, inp, threshold)
            elif node.target is torch.ops.aten.sym_size.int:
                value = args[0]
                dim = _as_int(args[1])
                env[node.name] = _sym_size_int(value, dim)
            elif node.target is torch.ops.aten.nll_loss_forward.default:
                input_t = _to_torch(args[0])
                target_t = _to_torch(args[1])
                weight = _to_torch(args[2]) if len(args) > 2 and args[2] is not None else None
                reduction = _as_int(args[3]) if len(args) > 3 and args[3] is not None else 1
                ignore_index = _as_int(args[4]) if len(args) > 4 and args[4] is not None else -100
                loss = F.nll_loss(input_t, target_t, weight=weight, ignore_index=ignore_index, reduction="mean")
                if reduction == 0:
                    loss = loss.unsqueeze(0)
                total_weight = weight.sum() if weight is not None else torch.tensor(target_t.numel(), device=target_t.device)
                env[node.name] = (loss, total_weight)
            elif node.target is torch.ops.aten.nll_loss_backward.default:
                converted_args = tuple(_to_torch_arg(arg) for arg in args)
                env[node.name] = torch.ops.aten.nll_loss_backward.default(*converted_args)
            elif node.target is torch.ops.aten.convolution_backward.default:
                converted_args = tuple(_to_torch_arg(arg) for arg in args)
                env[node.name] = torch.ops.aten.convolution_backward.default(*converted_args)
            elif node.target in (
                torch.ops.aten.log_softmax_backward_data.default,
                torch.ops.aten._log_softmax_backward_data.default,
            ):
                converted_args = tuple(_to_torch_arg(arg) for arg in args)
                env[node.name] = node.target(*converted_args)
            else:
                raise RuntimeError(f"Unhandled call_function: {node.target}")
        elif node.op == "call_method":
            if node.target not in _SUPPORTED_METHODS:
                raise RuntimeError(f"Unsupported call_method: {node.target}")
            args = torch.fx.node.map_arg(node.args, lambda n: env[n.name] if hasattr(n, "name") else n)
            if node.target in ("relu",):
                value = args[0]
                env[node.name] = F.relu(value) if _is_torch_tensor(value) else _to_pytensor(value).relu()
            elif node.target in ("add", "__add__"):
                left = args[0]
                right = args[1]
                if _is_torch_tensor(left) or _is_torch_tensor(right):
                    env[node.name] = _to_torch(left) + _to_torch(right)
                else:
                    env[node.name] = _to_pytensor(left).add(_to_pytensor(right))
            elif node.target in ("matmul", "__matmul__"):
                left = args[0]
                right = args[1]
                if _is_torch_tensor(left) or _is_torch_tensor(right):
                    env[node.name] = _to_torch(left).matmul(_to_torch(right))
                else:
                    env[node.name] = _to_pytensor(left).matmul(_to_pytensor(right))
            elif node.target in ("flatten",):
                value = args[0]
                start_dim = _as_int(args[1]) if len(args) > 1 else 0
                end_dim = _as_int(args[2]) if len(args) > 2 else -1
                env[node.name] = _flatten_tensor(value, start_dim, end_dim)
            elif node.target in ("view",):
                value = args[0]
                shape = tuple(_shape_from_args(args))
                env[node.name] = _reshape_tensor(value, shape)
            else:
                raise RuntimeError(f"Unhandled call_method: {node.target}")
        elif node.op == "call_module":
            module = gm.get_submodule(node.target)
            args = torch.fx.node.map_arg(node.args, lambda n: env[n.name] if hasattr(n, "name") else n)
            if isinstance(module, torch.nn.ReLU):
                value = args[0]
                env[node.name] = F.relu(value) if _is_torch_tensor(value) else _to_pytensor(value).relu()
            elif isinstance(module, torch.nn.Linear):
                inp = args[0]
                if _is_torch_tensor(inp):
                    env[node.name] = F.linear(inp, module.weight, module.bias)
                else:
                    inp_pt = _to_pytensor(inp)
                    weight = _to_pytensor(module.weight.t())
                    out = inp_pt.matmul(weight)
                    if module.bias is not None:
                        bias = _to_pytensor(module.bias)
                        out = out.add(bias)
                    env[node.name] = out
            elif isinstance(module, torch.nn.Flatten):
                value = args[0]
                env[node.name] = _flatten_tensor(value, module.start_dim, module.end_dim)
            elif isinstance(module, torch.nn.Conv2d):
                env[node.name] = module(_to_torch(args[0]))
            elif isinstance(module, torch.nn.BatchNorm2d):
                env[node.name] = module(_to_torch(args[0]))
            elif isinstance(module, torch.nn.CrossEntropyLoss):
                input_t = _to_torch(args[0])
                target_t = _to_torch(args[1])
                env[node.name] = module(input_t, target_t)
            elif isinstance(module, torch.nn.Dropout):
                env[node.name] = module(_to_torch(args[0]))
            elif isinstance(module, torch.nn.LogSoftmax):
                env[node.name] = module(_to_torch(args[0]))
            elif isinstance(module, torch.nn.MaxPool2d):
                env[node.name] = module(_to_torch(args[0]))
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
