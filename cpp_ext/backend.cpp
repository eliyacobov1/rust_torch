#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/core/DeviceType.h>

// Forward declaration for Rust FFI (future work):
// extern "C" void rust_add_f32(float* out, const float* a, const float* b, size_t n);

// Simple native C++ kernel for demo purposes:
static at::Tensor rustcpu_add_kernel(const at::Tensor& a,
                                     const at::Tensor& b,
                                     const at::Scalar& alpha) {
  TORCH_CHECK(a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat, "f32 only");
  TORCH_CHECK(a.sizes() == b.sizes(), "shape mismatch (broadcasting elided)");
  auto out = at::empty_like(a, a.options().device(c10::DeviceType::PrivateUse1));
  auto n = a.numel();
  const float* ap = a.data_ptr<float>();
  const float* bp = b.data_ptr<float>();
  float* op = out.data_ptr<float>();
  for (int64_t i = 0; i < n; ++i) op[i] = ap[i] + bp[i]; // replace with rust_add_f32(...) later
  return out;
}

// Fallback: box to CPU for unimplemented ops
static void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  at::native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("add.Tensor", torch::CppFunction::makeUnboxedOnly(rustcpu_add_kernel));
}
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&cpu_fallback>());
}

// Utility to generate torch.rustcpu helpers (device() strings etc.):
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("register_device_helpers", [](){
    // In Python we call:
    // torch.utils.generate_methods_for_privateuse1_backend("rustcpu")
    // torch._register_device_module("rustcpu", type("RustCpuModule", (), {})())
    return true;
  });
}