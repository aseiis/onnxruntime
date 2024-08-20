// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <cmath>

#include "core/common/common.h"

#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/program_info.h"
#include "core/providers/webgpu/program_manager.h"

namespace onnxruntime {
namespace webgpu {

std::string CalculateProgramInfoUniqueKey(const ProgramInfo& program,
                                          bool is_1d_dispatch) {
  std::ostringstream ss;
  ss.imbue(std::locale::classic());

  // final key format:
  // <KEY>=<PROGRAM_NAME>[<PROGRAM_CUSTOM_CACHE_HINT>]:is1DimensionDispatch:<INPUTS_INFO_0>|<INPUTS_INFO_1>|...
  //
  // <PROGRAM_CUSTOM_CACHE_HINT>=<HINT_0>|<HINT_1>|...
  // <INPUTS_INFO_i>=<TENSOR_ELEMENT_TYPE_OR_EMPTY>;<TENSOR_SHAPE_OR_RANK_OR_EMPTY>
  ss << program.Name() << "[" << program.CacheHint() << "]:" << is_1d_dispatch << ":";
  bool first_input = true;
  for (const auto& input : program.Inputs()) {
    if (first_input) {
      first_input = false;
    } else {
      ss << "|";
    }
    if ((input.dependency & ProgramInputTensorDependency::Type) == ProgramInputTensorDependency::Type) {
      ss << input.tensor->GetElementType();
    }
    ss << ";";
    if ((input.dependency & ProgramInputTensorDependency::Rank) == ProgramInputTensorDependency::Rank) {
      ss << input.tensor->Shape().NumDimensions();
    } else if ((input.dependency & ProgramInputTensorDependency::Shape) == ProgramInputTensorDependency::Shape) {
      ss << input.tensor->Shape().ToString();
    }
  }

  return ss.str();
}

std::vector<wgpu::FeatureName> GetAvailableRequiredFeatures(const wgpu::Adapter& adapter) {
  std::vector<wgpu::FeatureName> required_features;
  constexpr wgpu::FeatureName features[]{
      wgpu::FeatureName::ChromiumExperimentalTimestampQueryInsidePasses,
      wgpu::FeatureName::TimestampQuery,
      wgpu::FeatureName::ShaderF16};
  for (auto feature : features) {
    if (adapter.HasFeature(feature)) {
      required_features.push_back(feature);
    }
  }
  return required_features;
}

wgpu::RequiredLimits GetAvailableRequiredLimits(const wgpu::Adapter& adapter) {
  wgpu::RequiredLimits required_limits{};
  wgpu::SupportedLimits adapter_limits;
  ORT_ENFORCE(adapter.GetLimits(&adapter_limits));

  required_limits.limits.maxBindGroups = adapter_limits.limits.maxBindGroups;
  required_limits.limits.maxComputeWorkgroupStorageSize = adapter_limits.limits.maxComputeWorkgroupStorageSize;
  required_limits.limits.maxComputeWorkgroupsPerDimension = adapter_limits.limits.maxComputeWorkgroupsPerDimension;
  required_limits.limits.maxStorageBufferBindingSize = adapter_limits.limits.maxStorageBufferBindingSize;
  required_limits.limits.maxBufferSize = adapter_limits.limits.maxBufferSize;
  required_limits.limits.maxComputeInvocationsPerWorkgroup = adapter_limits.limits.maxComputeInvocationsPerWorkgroup;
  required_limits.limits.maxComputeWorkgroupSizeX = adapter_limits.limits.maxComputeWorkgroupSizeX;
  required_limits.limits.maxComputeWorkgroupSizeY = adapter_limits.limits.maxComputeWorkgroupSizeY;
  required_limits.limits.maxComputeWorkgroupSizeZ = adapter_limits.limits.maxComputeWorkgroupSizeZ;

  return required_limits;
}

void WebGpuContext::Initialize(const WebGpuExecutionProviderInfo& webgpu_ep_info) {
  auto buffer_cache_mode = webgpu_ep_info.buffer_cache_mode;
  std::call_once(init_flag_, [this, buffer_cache_mode]() {
    // Initialization.Step.1 - Create wgpu::Instance
    if (instance_ == nullptr) {
      wgpu::InstanceDescriptor instance_desc{};
      instance_desc.features.timedWaitAnyEnable = true;
      instance_ = wgpu::CreateInstance(&instance_desc);

      ORT_ENFORCE(instance_ != nullptr, "Failed to create wgpu::Instance.");
    }

    // Initialization.Step.2 - Create wgpu::Adapter
    if (adapter_ == nullptr) {
      wgpu::RequestAdapterOptions req_adapter_options = {};
      wgpu::RequestAdapterCallbackInfo req_adapter_callback_info = {};
      req_adapter_callback_info.mode = wgpu::CallbackMode::WaitAnyOnly;
      req_adapter_callback_info.callback = [](WGPURequestAdapterStatus status,
                                              WGPUAdapter adapter, const char* message,
                                              void* userdata) {
        ORT_ENFORCE(status == WGPURequestAdapterStatus_Success, "Failed to get a WebGPU adapter: ", message);
        *static_cast<wgpu::Adapter*>(userdata) = wgpu::Adapter::Acquire(adapter);
      };
      req_adapter_callback_info.userdata = &adapter_;
      ORT_ENFORCE(wgpu::WaitStatus::Success == instance_.WaitAny(instance_.RequestAdapter(&req_adapter_options, req_adapter_callback_info), UINT64_MAX));
      ORT_ENFORCE(adapter_ != nullptr, "Failed to get a WebGPU adapter.");
    }

    // Initialization.Step.3 - Create wgpu::Device
    if (device_ == nullptr) {
      wgpu::DeviceDescriptor device_desc = {};
      std::vector<wgpu::FeatureName> required_features = GetAvailableRequiredFeatures(adapter_);
      if (required_features.size() > 0) {
        device_desc.requiredFeatures = required_features.data();
      }
      wgpu::RequiredLimits required_limits = GetAvailableRequiredLimits(adapter_);
      device_desc.requiredLimits = &required_limits;

      wgpu::RequestDeviceCallbackInfo req_device_callback_info = {};
      req_device_callback_info.mode = wgpu::CallbackMode::WaitAnyOnly;
      req_device_callback_info.callback = [](WGPURequestDeviceStatus status, WGPUDevice device, char const* message, void* userdata) {
        ORT_ENFORCE(status == WGPURequestAdapterStatus_Success, "Failed to get a WebGPU device: ", message);
        *static_cast<wgpu::Device*>(userdata) = wgpu::Device::Acquire(device);
      };
      req_device_callback_info.userdata = &device_;
      ORT_ENFORCE(wgpu::WaitStatus::Success == instance_.WaitAny(adapter_.RequestDevice(&device_desc, req_device_callback_info), UINT64_MAX));
      ORT_ENFORCE(device_ != nullptr, "Failed to get a WebGPU device.");
    }

    // cache adapter info
    ORT_ENFORCE(Adapter().GetInfo(&adapter_info_));
    // cache device limits
    wgpu::SupportedLimits device_supported_limits;
    ORT_ENFORCE(Device().GetLimits(&device_supported_limits));
    device_limits_ = device_supported_limits.limits;

    // create buffer manager
    buffer_mgr_ = BufferManagerFactory::Create(*this, buffer_cache_mode);

    // create program manager
    program_mgr_ = std::make_unique<ProgramManager>(Device(), DeviceLimits());
  });
}

Status WebGpuContext::Wait(wgpu::Future f) const {
  auto status = instance_.WaitAny(f, UINT64_MAX);
  if (status == wgpu::WaitStatus::Success) {
    return Status::OK();
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to wait for the operation:", uint32_t(status));
}

Status WebGpuContext::Run(const ComputeContext& /* context */, const ProgramInfo& program) const {
  const auto& inputs = program.Inputs();
  const auto& outputs = program.Outputs();

#ifndef NDEBUG
  ORT_ENFORCE(std::all_of(inputs.begin(), inputs.end(), [](const ProgramInput& input) {
                const auto* tensor = input.tensor;
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

  if (outputs.size() == 0) {
    return Status::OK();
  }

  ORT_RETURN_IF(program.Inputs().size() != inputs.size(), "The number of inputs does not match the program.");

  const auto [x, y, z] = program_mgr_->NormalizeDispatchGroupSize(program.WorkgroupDispatchSize());
  bool is_1d_dispatch = (y == 1 && z == 1);

  auto key = CalculateProgramInfoUniqueKey(program, is_1d_dispatch);

  return Status::OK();
}

std::unordered_map<int32_t, std::unique_ptr<WebGpuContext>> WebGpuContextFactory::contexts_;
std::mutex WebGpuContextFactory::mutex_;

WebGpuContext& WebGpuContextFactory::CreateContext(int context_id, WGPUInstance instance, WGPUAdapter adapter, WGPUDevice device) {
  if (context_id == 0) {
    // context ID is preserved for the default context. User cannot use context ID 0 as a custom context.
    ORT_ENFORCE(instance == nullptr && adapter == nullptr && device == nullptr,
                "WebGPU EP default context (contextId=0) must not have custom WebGPU instance, adapter or device.");
  } else {
    // for context ID > 0, user must provide custom WebGPU instance, adapter and device.
    ORT_ENFORCE(instance != nullptr && adapter != nullptr && device != nullptr,
                "WebGPU EP custom context (contextId>0) must have custom WebGPU instance, adapter and device.");
  }

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = contexts_.find(context_id);
  if (it == contexts_.end()) {
    auto context = std::unique_ptr<WebGpuContext>(new WebGpuContext(instance, adapter, device));
    it = contexts_.emplace(context_id, std::move(context)).first;
  } else if (context_id != 0) {
    ORT_ENFORCE(it->second->instance_.Get() == instance && it->second->adapter_.Get() == adapter && it->second->device_.Get() == device,
                "WebGPU EP context ID ", context_id, " is already created with different WebGPU instance, adapter or device.");
  }
  return *it->second;
}

WebGpuContext& WebGpuContextFactory::GetContext(int context_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = contexts_.find(context_id);
  ORT_ENFORCE(it != contexts_.end(), "WebGPU EP context ID ", context_id, " is not found.");

  return *it->second;
}

}  // namespace webgpu
}  // namespace onnxruntime
