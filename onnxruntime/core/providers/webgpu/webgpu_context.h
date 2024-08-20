// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

#include <memory>
#include <mutex>

#include <webgpu/webgpu_cpp.h>

#include "core/common/common.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/buffer_manager.h"

namespace onnxruntime {
class Tensor;

namespace webgpu {
class WebGpuContext;
class ComputeContext;
class ProgramInfo;

class WebGpuContextFactory {
 public:
  static WebGpuContext& CreateContext(int context_id, WGPUInstance instance, WGPUAdapter adapter, WGPUDevice device);
  static WebGpuContext& GetContext(int context_id);

 private:
  WebGpuContextFactory() {}

  static std::unordered_map<int32_t, std::unique_ptr<WebGpuContext>> contexts_;
  static std::mutex mutex_;
};

// Class WebGpuContext includes all necessary resources for the context.
class WebGpuContext final {
 public:
  void Initialize(const WebGpuExecutionProviderInfo& webgpu_ep_info);

  Status Wait(wgpu::Future f) const;

  const wgpu::Adapter& Adapter() const { return adapter_; }
  const wgpu::Device& Device() const { return device_; }

  const wgpu::CommandEncoder& GetCommandEncoder() const {
    if (!current_command_encoder_) {
      current_command_encoder_ = device_.CreateCommandEncoder();
    }
    return current_command_encoder_;
  }

  void EndComputePass() const {
    if (current_compute_pass_encoder_) {
      current_compute_pass_encoder_.End();
      current_compute_pass_encoder_ = nullptr;
    }
  }

  void Flush() const {
    if (!current_command_encoder_) {
      return;
    }

    EndComputePass();

    // TODO: add support for GPU Query

    auto command_buffer = current_command_encoder_.Finish();
    Device().GetQueue().Submit(1, &command_buffer);
    BufferManager().RefreshPendingBuffers();
    current_command_encoder_ = nullptr;
  }

  const IBufferManager& BufferManager() const { return *buffer_mgr_; }

  Status Run(const ComputeContext& context, const ProgramInfo& program, std::initializer_list<const Tensor*> inputs, std::initializer_list<Tensor*> outputs) const;

 private:
  WebGpuContext(WGPUInstance instance, WGPUAdapter adapter, WGPUDevice device) : instance_{instance}, adapter_{adapter}, device_{device} {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(WebGpuContext);

  std::once_flag init_flag_;

  mutable wgpu::Instance instance_;
  wgpu::Adapter adapter_;
  wgpu::Device device_;

  std::unique_ptr<IBufferManager> buffer_mgr_;
  mutable wgpu::CommandEncoder current_command_encoder_;
  mutable wgpu::ComputePassEncoder current_compute_pass_encoder_;

  friend class WebGpuContextFactory;
};

}  // namespace webgpu
}  // namespace onnxruntime
