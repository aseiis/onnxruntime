// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/session/onnxruntime_c_api.h"
#include "core/framework/compute_capability.h"

namespace onnxruntime {
class ExecutionProviderAdapter : public IExecutionProvider {
public:
  ExecutionProviderAdapter(OrtExecutionProvider* ep) : IExecutionProvider(ep->type), ep_impl_(ep) {
    if (ep_impl_->RegisterKernels) {
      kernel_registry_ = std::make_shared<KernelRegistry>();
      ep_impl_->RegisterKernels(reinterpret_cast<OrtKernelRegistry*>(kernel_registry_.get()));
    }
  }
  virtual std::vector<std::unique_ptr<ComputeCapability>> GetCapability(const GraphViewer& graph_viewer, const IKernelLookup& kernel_lookup) const override {
    size_t cnt = 0;
    OrtIndexedSubGraph** indexed_subgraph = nullptr;
    if (ep_impl_->GetCapability) ep_impl_->GetCapability(ep_impl_, reinterpret_cast<const OrtGraphViewer*>(&graph_viewer), &cnt, &indexed_subgraph);

    if (cnt == 0) return IExecutionProvider::GetCapability(graph_viewer, kernel_lookup);

    std::vector<std::unique_ptr<ComputeCapability>> ret;
    for (size_t i = 0; i < cnt; i++) {
        std::unique_ptr<IndexedSubGraph> sb = std::make_unique<IndexedSubGraph>();
        sb->nodes.reserve(indexed_subgraph[i]->node_index_len);
        for (size_t j = 0; j < indexed_subgraph[i]->node_index_len; j++) sb->nodes.push_back((indexed_subgraph[i]->node_index)[j]);
        if (indexed_subgraph[i]->meta_def != nullptr) {
            std::unique_ptr<IndexedSubGraph::MetaDef> meta_def = std::make_unique<IndexedSubGraph::MetaDef>();
            meta_def->name = indexed_subgraph[i]->meta_def->name ? indexed_subgraph[i]->meta_def->name : "";
            meta_def->doc_string = indexed_subgraph[i]->meta_def->doc_string ? indexed_subgraph[i]->meta_def->doc_string : "";
            meta_def->domain = indexed_subgraph[i]->meta_def->domain ? indexed_subgraph[i]->meta_def->domain : "";
            meta_def->since_version = indexed_subgraph[i]->meta_def->since_version;

            meta_def->inputs.reserve(indexed_subgraph[i]->meta_def->input_len);
            for (size_t j = 0; j < indexed_subgraph[i]->meta_def->input_len; j++) meta_def->inputs.push_back(indexed_subgraph[i]->meta_def->inputs[j]);

            meta_def->outputs.reserve(indexed_subgraph[i]->meta_def->output_len);
            for (size_t j = 0; j < indexed_subgraph[i]->meta_def->output_len; j++) meta_def->outputs.push_back(indexed_subgraph[i]->meta_def->outputs[j]);

            meta_def->constant_initializers.reserve(indexed_subgraph[i]->meta_def->initializer_len);
            for (size_t j = 0; j < indexed_subgraph[i]->meta_def->initializer_len; j++) meta_def->constant_initializers.push_back(indexed_subgraph[i]->meta_def->constant_initializers[j]);

            sb->SetMetaDef(std::move(meta_def));
        }

        ret.push_back(std::make_unique<ComputeCapability>(std::move(sb)));
    }
    return ret;
  }

  virtual common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs, std::vector<NodeComputeInfo>& node_compute_funcs) override {
    std::vector<const OrtGraphViewer*> ortGraphs;
    std::vector<const OrtNode*> ortNodes;
    for (auto& fused_node_graph : fused_nodes_and_graphs) {
      const GraphViewer& graph_viewer = fused_node_graph.filtered_graph;
      const Node& fused_node = fused_node_graph.fused_node;
      ortGraphs.push_back(reinterpret_cast<const OrtGraphViewer*>(&graph_viewer));
      ortNodes.push_back(reinterpret_cast<const OrtNode*>(&fused_node));
    }
    size_t count = fused_nodes_and_graphs.size();
    std::vector<OrtNodeComputeInfo> cache;
    cache.resize(count);
    OrtNodeComputeInfo* cache_data = cache.data();
    ep_impl_->Compile(ep_impl_, ortGraphs.data(), ortNodes.data(), count, &cache_data);
    node_compute_funcs.reserve(count);
    for (size_t i = 0; i < count; i++) {
        NodeComputeInfo compute_info;
        compute_info.create_state_func = [&, cache, i](ComputeContext* context, void** state) {
            if (cache[i].CreateFunctionStateFunc) return cache[i].CreateFunctionStateFunc(reinterpret_cast<OrtComputeContext*>(context), state);
            return 0;
        };
        compute_info.compute_func = [&, cache, i](void* state, const OrtApi* api, OrtKernelContext* context) {
            return ToStatus(cache[i].ComputeFunc(state, api, context));
        };
        compute_info.release_state_func = [&, cache, i](void* state) {
            if (cache[i].DestroyFunctionStateFunc) {
                cache[i].DestroyFunctionStateFunc(state);
            }
        };
        node_compute_funcs.emplace_back(std::move(compute_info));
    }

    return Status::OK();
  }

  virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const override { return kernel_registry_; }
private:
  OrtExecutionProvider* ep_impl_;
  std::shared_ptr<KernelRegistry> kernel_registry_; // TODO(leca): should be static local
};
}