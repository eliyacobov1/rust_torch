import torch
import cpp_ext.build as build  # build extension (if not built)
ext = build.ext  # loaded module
# Create 'rustcpu' helpers
torch.utils.generate_methods_for_privateuse1_backend("rustcpu")
torch._register_device_module("rustcpu", type("RustCpuModule", (), {})())

# Now perform add on rustcpu device
a = torch.ones(4, device="rustcpu", dtype=torch.float32)
b = 2*torch.ones(4, device="rustcpu", dtype=torch.float32)
c = a + b  # should dispatch to our custom add
print("c.device:", c.device, "values:", c.cpu().numpy())