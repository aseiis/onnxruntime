// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program_info.h"

namespace onnxruntime {
namespace webgpu {

class UnaryElementwiseProgramInfo final : public ProgramInfo {
 public:
  UnaryElementwiseProgramInfo(const std::string& kernel_name, const std::string& expression, const std::string& additional_impl = "")
      : ProgramInfo{kernel_name}, expression_{expression}, additional_impl_{additional_impl} {
  }

  std::string GenerateShaderCode(ShaderHelper& /*sh*/) const override {
    return "";
  }

 private:
  std::string expression_;
  std::string additional_impl_;
};

}  // namespace webgpu
}  // namespace onnxruntime
