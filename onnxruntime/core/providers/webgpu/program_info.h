// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

namespace onnxruntime {
class Tensor;

namespace webgpu {
class ShaderHelper;

// TODO: revise this data type enum
enum class ProgramUniformVariableDataType {
  Float = 0,
  Int = 1,
  UInt = 2,
  Bool = 3,
  Float2 = 4,
  Int2 = 5,
  UInt2 = 6,
  Bool2 = 7,
  Float3 = 8,
  Int3 = 9,
  UInt3 = 10,
  Bool3 = 11,
  Float4 = 12,
  Int4 = 13,
  UInt4 = 14,
  Bool4 = 15,
  Float2x2 = 16,
  Int2x2 = 17,
  UInt2x2 = 18,
  Bool2x2 = 19,
  Float3x3 = 20,
  Int3x3 = 21,
  UInt3x3 = 22,
  Bool3x3 = 23,
  Float4x4 = 24,
  Int4x4 = 25,
  UInt4x4 = 26,
  Bool4x4 = 27,
};

struct ProgramUniformVariable {
  ProgramUniformVariableDataType data_type;
  gsl::span<uint8_t> data;
};

enum class ProgramInputTensorDependency {
  None = 0,
  Type = 1,
  Rank = 2,
  Shape = 4,
  TypeAndRank = Type | Rank,
  TypeAndShape = Type | Shape,
};

struct ProgramInput {
  const Tensor* tensor;
  ProgramInputTensorDependency dependency;
};

class ProgramInfo {
 public:
  ProgramInfo(const std::string& name);
  virtual ~ProgramInfo() = default;

  //
  // chain-style methods for setting properties
  //

  // set the cache hint for the program
  template <typename... T>
  ProgramInfo& CacheHint(T&&... args);

  ProgramInfo& Inputs(std::initializer_list<ProgramInput> inputs);
  ProgramInfo& Outputs(std::initializer_list<Tensor*> outputs);

  ProgramInfo& WorkgroupDispatchSize(uint32_t x, uint32_t y, uint32_t z);

  ProgramInfo& UniformVariables(std::initializer_list<ProgramUniformVariable> variables);

  //
  // shader code generation
  //

  virtual std::string GenerateShaderCode(ShaderHelper& sh) const = 0;

 private:
  std::string name_;
  std::string cache_hint_;
  std::vector<ProgramInputTensorDependency> cache_input_dependencies_;
  std::vector<const Tensor*> inputs_;
  std::vector<Tensor*> outputs_;

  uint32_t workgroup_dispatch_size_x_;
  uint32_t workgroup_dispatch_size_y_;
  uint32_t workgroup_dispatch_size_z_;

  std::vector<ProgramUniformVariable> variables_;
};

}  // namespace webgpu
}  // namespace onnxruntime
