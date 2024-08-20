// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>

#include "core/providers/webgpu/program_info.h"

onnxruntime::webgpu::ProgramInfo::ProgramInfo(const std::string& name)
    : name_{name},
      workgroup_dispatch_size_x_{1},
      workgroup_dispatch_size_y_{1},
      workgroup_dispatch_size_z_{1} {
}
