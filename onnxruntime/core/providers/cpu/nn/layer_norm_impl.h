// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"

namespace onnxruntime {

class LayerNormImpl : public OpKernel {
 public:
  LayerNormImpl(const OpKernelInfo& op_kernel_info, bool simplified = false);
  Status Compute(OpKernelContext* p_op_kernel_context) const override;

 private:
  template <typename T>
  struct ComputeImpl;

  int64_t axis_;
  float epsilon_;
  const bool simplified_;
};

}  // namespace onnxruntime
