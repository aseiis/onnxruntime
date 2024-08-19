// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/error_code_helper.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {

struct WebGpuProviderFactory : IExecutionProviderFactory {
  WebGpuProviderFactory(const int context_id, const webgpu::WebGpuContext& context, const WebGpuExecutionProviderInfo& webgpu_ep_info)
      : context_id_{context_id}, context_{context}, info_{webgpu_ep_info} {
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<WebGpuExecutionProvider>(context_id_, context_, info_);
  }

 private:
  int context_id_;
  const webgpu::WebGpuContext& context_;
  WebGpuExecutionProviderInfo info_;
};

std::shared_ptr<IExecutionProviderFactory> WebGpuProviderFactoryCreator::Create(const SessionOptions* session_options) {
  //
  // STEP.1 - prepare WebGpuExecutionProviderInfo
  //
  WebGpuExecutionProviderInfo webgpu_ep_info{
      // preferred layout is NHWC by default
      DataLayout::NHWC,
      // graph capture feature is disabled by default
      false,
      // buffer cache mode is None by default
      webgpu::BufferCacheMode::None,
  };

  std::string preferred_layout_str;
  if (session_options->config_options.TryGetConfigEntry("preferredLayout", preferred_layout_str) && preferred_layout_str == "NCHW") {
    webgpu_ep_info.data_layout = DataLayout::NCHW;
  }
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP preferred layout: " << int(webgpu_ep_info.data_layout) << " (parsed from \""
                        << preferred_layout_str << "\".";

  std::string enable_graph_capture_str;
  if (session_options->config_options.TryGetConfigEntry("enableGraphCapture", enable_graph_capture_str) &&
      (enable_graph_capture_str == "true" || enable_graph_capture_str == "1")) {
    webgpu_ep_info.enable_graph_capture = true;
  }
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP graph capture enable: " << webgpu_ep_info.enable_graph_capture;

  std::string buffer_cache_mode_str;
  if (session_options->config_options.TryGetConfigEntry("bufferCacheMode", buffer_cache_mode_str)) {
    if (buffer_cache_mode_str == "simple") {
      webgpu_ep_info.buffer_cache_mode = webgpu::BufferCacheMode::Simple;
    } else if (buffer_cache_mode_str == "bucket") {
      webgpu_ep_info.buffer_cache_mode = webgpu::BufferCacheMode::Bucket;
    }
  }
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP buffer cache mode: " << int(webgpu_ep_info.buffer_cache_mode) << " (parsed from \""
                        << buffer_cache_mode_str << "\".";

  //
  // STEP.2 - prepare WebGpuContext
  //
  int context_id = 0;
  std::string context_id_str;
  if (session_options->config_options.TryGetConfigEntry("contextId", context_id_str)) {
    context_id = static_cast<uint16_t>(std::stoi(context_id_str));
  }

  WGPUInstance webgpu_instance = nullptr;
  std::string webgpu_instance_str;
  if (session_options->config_options.TryGetConfigEntry("webgpuInstance", webgpu_instance_str)) {
    static_assert(sizeof(WGPUInstance) == sizeof(unsigned long long), "WGPUInstance size mismatch");
    webgpu_instance = reinterpret_cast<WGPUInstance>(std::stoull(webgpu_instance_str));
  }

  WGPUAdapter webgpu_adapter = nullptr;
  std::string webgpu_adapter_str;
  if (session_options->config_options.TryGetConfigEntry("webgpuAdapter", webgpu_adapter_str)) {
    static_assert(sizeof(WGPUAdapter) == sizeof(unsigned long long), "WGPUAdapter size mismatch");
    webgpu_adapter = reinterpret_cast<WGPUAdapter>(std::stoull(webgpu_adapter_str));
  }

  WGPUDevice webgpu_device = nullptr;
  std::string webgpu_device_str;
  if (session_options->config_options.TryGetConfigEntry("webgpuDevice", webgpu_device_str)) {
    static_assert(sizeof(WGPUDevice) == sizeof(unsigned long long), "WGPUDevice size mismatch");
    webgpu_device = reinterpret_cast<WGPUDevice>(std::stoull(webgpu_device_str));
  }

  auto& context = webgpu::WebGpuContextFactory::CreateContext(context_id, webgpu_instance, webgpu_adapter, webgpu_device);
  context.Initialize(webgpu_ep_info);

  return std::make_shared<WebGpuProviderFactory>(context_id, context, webgpu_ep_info);
}

}  // namespace onnxruntime
