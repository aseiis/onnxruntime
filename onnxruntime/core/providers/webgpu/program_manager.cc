// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/common/safeint.h"

#include "core/providers/webgpu/program_manager.h"

namespace onnxruntime {
namespace webgpu {

ProgramManager::DispatchGroupSize ProgramManager::NormalizeDispatchGroupSize(ProgramManager::DispatchGroupSize dispatch) const {
  auto [x, y, z] = dispatch;

  auto limit_per_dimension = limits_.maxComputeWorkgroupsPerDimension;
  if (x <= limit_per_dimension && y <= limit_per_dimension && z <= limit_per_dimension) {
    return {x, y, z};
  }

  auto size = static_cast<double>(x) * static_cast<double>(y) * static_cast<double>(z);
  SafeInt<uint32_t> dispatch_avg = std::ceil(std::sqrt(size));
  if (dispatch_avg > limit_per_dimension) {
    dispatch_avg = std::ceil(std::cbrt(size));
    ORT_ENFORCE(dispatch_avg <= limit_per_dimension, "The dispatch group size exceeds WebGPU maximum.");
    return {dispatch_avg, dispatch_avg, dispatch_avg};
  } else {
    return {dispatch_avg, dispatch_avg, 1};
  }
}

}  // namespace webgpu
}  // namespace onnxruntime