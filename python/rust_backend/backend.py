from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch.fx import GraphModule
from torch._dynamo import register_backend
from torch._dynamo.backends.common import aot_autograd

try:
    from functorch.compile import make_boxed_func
except Exception:  # pragma: no cover - compatibility fallback for newer torch builds
    try:
        from torch._functorch.aot_autograd import make_boxed_func  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - last-resort fallback
        def make_boxed_func(fn: Callable) -> Callable:
            def boxed(args):
                return fn(*args)

            boxed._boxed_call = True  # type: ignore[attr-defined]
            return boxed

# The PyO3 module 'rustorch' exposes run_fx which returns a callable.
import rustorch


@register_backend
def rust_backend(gm: GraphModule, example_inputs: Sequence[torch.Tensor]) -> Callable:
    def _fw_compiler(fx_gm: GraphModule, fx_inputs: Sequence[torch.Tensor]) -> Callable:
        return make_boxed_func(rustorch.run_fx(fx_gm, fx_inputs))

    compiler = aot_autograd(fw_compiler=_fw_compiler)
    return compiler(gm, example_inputs)
