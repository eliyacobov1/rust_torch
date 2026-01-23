from __future__ import annotations

from typing import Any, Callable, Dict
import logging
import operator

import numpy as np
import torch
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
}

_SUPPORTED_METHODS = {
    "relu",
    "add",
    "__add__",
    "matmul",
    "__matmul__",
}


def _to_pytensor(value: Any) -> rustorch.PyTensor:
    if isinstance(value, rustorch.PyTensor):
        return value
    if isinstance(value, torch.Tensor):
        data = value.detach().cpu().numpy()
        return rustorch.PyTensor(data, requires_grad=False)
    if isinstance(value, (int, float)):
        return rustorch.PyTensor(np.array(value, dtype=np.float32), requires_grad=False)
    raise TypeError(f"Unsupported input type for PyTensor conversion: {type(value)}")


def _to_torch(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, rustorch.PyTensor):
        return torch.from_numpy(value.numpy())
    if isinstance(value, (int, float)):
        return torch.tensor(value)
    raise TypeError(f"Unsupported output type for torch conversion: {type(value)}")


def _to_torch_tree(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        converted = [_to_torch_tree(item) for item in value]
        return type(value)(converted)
    return _to_torch(value)


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
            else:
                raise RuntimeError(f"Unsupported call_module: {type(module)}")
        elif node.op == "output":
            result = torch.fx.node.map_arg(node.args[0], lambda n: env[n.name] if hasattr(n, "name") else n)
            return _to_torch_tree(result)
        else:
            raise RuntimeError(f"Unsupported node op: {node.op}")

    raise RuntimeError("FX graph had no output node")


def run_fx(gm: GraphModule, example_inputs: list[torch.Tensor]) -> Callable:
    def compiled(*inputs: torch.Tensor) -> torch.Tensor:
        try:
            return _run_graph(gm, *inputs)
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            _LOG.warning("rustorch FX runner fallback to eager: %s", exc)
            return gm(*inputs)

    return compiled
