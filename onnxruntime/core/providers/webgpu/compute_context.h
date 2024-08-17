// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

#include <webgpu/webgpu_cpp.h>

#include "core/framework/execution_provider.h"

namespace onnxruntime {
namespace webgpu {

class WebGpuContext;

class ComputeContext {
 public:
  ComputeContext(const WebGpuContext& context) : context_{context} {}

  virtual ~ComputeContext() = default;

  virtual void Dispatch(WGPUComputePassEncoder pass) const = 0;

 protected:
  const WebGpuContext& context_;
};

}  // namespace webgpu
}  // namespace onnxruntime
