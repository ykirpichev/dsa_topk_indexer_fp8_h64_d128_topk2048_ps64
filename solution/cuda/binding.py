"""
TVM FFI bindings template for the CUDA FP8 MQA Logits + Top-K kernel.

With `language = "cuda"` and the default TVM-FFI binding, ``flashinfer-bench``
compiles ``kernel.cu`` directly (via ``tvm_ffi.cpp.build``) and looks up the
exported symbol named ``kernel`` — see ``TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, kernel_fn)``
in ``kernel.cu``. This Python file is kept alongside the sources to match the
starter-kit layout but is not loaded by the CUDA builder.

See:
  https://github.com/flashinfer-ai/flashinfer-bench-starter-kit/blob/main/solution/cuda/binding.py
"""

from tvm.ffi import register_func


@register_func("flashinfer.kernel")
def kernel():
    """Placeholder: the actual entry point lives in kernel.cu (TVM-FFI C++)."""
    raise NotImplementedError(
        "binding.py::kernel is a placeholder; use kernel.cu::kernel "
        "(the compiled TVM-FFI symbol)."
    )
