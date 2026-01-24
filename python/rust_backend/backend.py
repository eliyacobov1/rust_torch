from typing import List, Callable
import torch
from torch.fx import GraphModule
from torch._dynamo import register_backend
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func

# The PyO3 module 'rustorch' exposes run_fx which returns a callable.
import rustorch

@register_backend
def rust_backend(gm: GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
    def _fw_compiler(fx_gm: GraphModule, fx_inputs: List[torch.Tensor]) -> Callable:
        return make_boxed_func(rustorch.run_fx(fx_gm, fx_inputs))

    compiler = aot_autograd(fw_compiler=_fw_compiler)
    return compiler(gm, example_inputs)
