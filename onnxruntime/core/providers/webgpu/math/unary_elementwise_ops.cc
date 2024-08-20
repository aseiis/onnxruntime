// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/unary_elementwise_ops.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_ELEMENTWISE_IMPL(OP_TYPE, ...)                              \
  class OP_TYPE final : public WebGpuKernel {                              \
   public:                                                                 \
    OP_TYPE(const OpKernelInfo& info) : WebGpuKernel{info} {}              \
                                                                           \
   protected:                                                              \
    Status ComputeInternal(ComputeContext& context) const override {       \
      const auto* input_tensor = context.Input(0);                         \
      auto* output_tensor = context.Output(0, input_tensor->Shape());      \
      UnaryElementwiseProgramInfo program{#OP_TYPE, __VA_ARGS__};          \
      return context.RunProgram(program, {input_tensor}, {output_tensor}); \
    }                                                                      \
  };

#define WEBGPU_ELEMENTWISE_KERNEL(OP_TYPE, VERSION, KERNEL_CLASS, TYPE) \
  ONNX_OPERATOR_KERNEL_EX(                                              \
      OP_TYPE, kOnnxDomain, VERSION, kWebGpuExecutionProvider,          \
      KernelDefBuilder().TypeConstraint("T", TYPE),                     \
      KERNEL_CLASS);

#define WEBGPU_ELEMENTWISE_VERSIONED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, KERNEL_CLASS, TYPE) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                               \
      OP_TYPE, kOnnxDomain, VERSION_FROM, VERSION_TO, kWebGpuExecutionProvider,                    \
      KernelDefBuilder().TypeConstraint("T", TYPE),                                                \
      KERNEL_CLASS);

WEBGPU_ELEMENTWISE_IMPL(Abs, "abs(x)")
WEBGPU_ELEMENTWISE_VERSIONED_KERNEL(Abs, 6, 12, Abs, WebGpuSupportedFloatTypes())
WEBGPU_ELEMENTWISE_KERNEL(Abs, 13, Abs, WebGpuSupportedFloatTypes())

}  // namespace webgpu
}  // namespace onnxruntime
