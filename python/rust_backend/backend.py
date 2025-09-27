from typing import List, Callable
import torch
from torch.fx import GraphModule
from torch._dynamo import register_backend

# The PyO3 module 'rustorch' exposes run_fx which returns a callable.
import rustorch

@register_backend
def rust_backend(gm: GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
    return rustorch.run_fx(gm, example_inputs)