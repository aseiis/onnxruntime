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

}  // namespace webgpu
}  // namespace onnxruntime
