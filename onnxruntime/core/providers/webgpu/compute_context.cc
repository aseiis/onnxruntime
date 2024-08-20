// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace webgpu {
ComputeContext::ComputeContext(OpKernelContext& kernel_context)
    : webgpu_context_{WebGpuContextFactory::GetContext(kernel_context.GetDeviceId())},
      kernel_context_{kernel_context} {
}

wgpu::AdapterInfo ComputeContext::AdapterInfo() const {
  wgpu::AdapterInfo info{};
  ORT_ENFORCE(webgpu_context_.Adapter().GetInfo(&info));
  return info;
}

wgpu::Limits ComputeContext::DeviceLimits() const {
  wgpu::SupportedLimits limits{};
  ORT_ENFORCE(webgpu_context_.Device().GetLimits(&limits));
  return limits.limits;
}

int ComputeContext::InputCount() const {
  return kernel_context_.InputCount();
}

int ComputeContext::OutputCount() const {
  return kernel_context_.OutputCount();
}

Status ComputeContext::RunProgram(const ProgramInfo& program, std::initializer_list<const Tensor*> inputs, std::initializer_list<Tensor*> outputs) {
#ifndef NDEBUG
  ORT_ENFORCE(std::all_of(inputs.begin(), inputs.end(), [](const Tensor* tensor) {
                return tensor != nullptr &&
                       tensor->Location().mem_type == OrtMemType::OrtMemTypeDefault &&
                       tensor->Location().device.Type() == OrtDevice::GPU &&
                       tensor->Location().name == WEBGPU_BUFFER;
              }),
              "All inputs must be tensors on WebGPU buffers.");

  ORT_ENFORCE(std::all_of(outputs.begin(), outputs.end(), [](Tensor* tensor) {
                return tensor != nullptr &&
                       tensor->Location().mem_type == OrtMemType::OrtMemTypeDefault &&
                       tensor->Location().device.Type() == OrtDevice::GPU &&
                       tensor->Location().name == WEBGPU_BUFFER;
              }),
              "All outputs must be tensors on WebGPU buffers.");
#endif
  return webgpu_context_.Run(*this, program, std::forward<std::initializer_list<const Tensor*>>(inputs), std::forward<std::initializer_list<Tensor*>>(outputs));
}

}  // namespace webgpu
}  // namespace onnxruntime
