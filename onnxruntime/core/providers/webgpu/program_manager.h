// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

#include <string>
#include <unordered_map>

#include <webgpu/webgpu_cpp.h>

#include "core/common/common.h"

#include "core/providers/webgpu/program_info.h"

namespace onnxruntime {
class Tensor;

namespace webgpu {
class ProgramArtifact {
 public:
  wgpu::ComputePipeline compute_pipeline;
  // TODO: add support for uniform buffers

 private:
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ProgramArtifact);
};

class ProgramManager {
  using DispatchGroupSize = std::tuple<uint32_t, uint32_t, uint32_t>;

 public:
  ProgramManager(const wgpu::Device& device, const wgpu::Limits& limits) : device_(device), limits_(limits) {}

  DispatchGroupSize NormalizeDispatchGroupSize(DispatchGroupSize dispatch) const;

  Status Build(const ProgramInfo& program, DispatchGroupSize normalized_dispatch, wgpu::ComputePipeline& compute_pipeline) const;
  const ProgramArtifact& Get(const std::string& key) const;
  void Set(const std::string& key, ProgramArtifact&& program);

  Status Run(const ProgramArtifact& artifact,
             gsl::span<WGPUBuffer> inputs,
             gsl::span<WGPUBuffer> outputs,
             DispatchGroupSize dispatch,
             WGPUBuffer uniform_buffer,
             uint64_t uniform_size) const;

 private:
  std::unordered_map<std::string, ProgramArtifact> programs_;
  const wgpu::Device& device_;
  const wgpu::Limits& limits_;
};

}  // namespace webgpu
}  // namespace onnxruntime
