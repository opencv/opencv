// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owner

#ifndef OPENCV_DNN_CUDA4DNN_CSL_CUDNN_GRAPH_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_CUDNN_GRAPH_HPP

#include "cudnn.hpp"

#include "../pointer.hpp"
#include "../workspace.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <array>
#include <vector>

#ifdef HAVE_CUDNNJIT

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace cudnn {

    /** RAII wrapper for a cuDNN Graph-API backend descriptor */
    class BackendDescriptor {
    public:
        BackendDescriptor() noexcept : descriptor{ nullptr } { }
        BackendDescriptor(const BackendDescriptor&) = delete;
        BackendDescriptor(BackendDescriptor&& other) noexcept : descriptor{ other.descriptor } {
            other.descriptor = nullptr;
        }

        BackendDescriptor(cudnnBackendDescriptorType_t type) {
            CUDA4DNN_CHECK_CUDNN(cudnnBackendCreateDescriptor(type, &descriptor));
        }

        ~BackendDescriptor() noexcept {
            if (descriptor != nullptr)
                cudnnBackendDestroyDescriptor(descriptor);
        }

        BackendDescriptor& operator=(const BackendDescriptor&) = delete;
        BackendDescriptor& operator=(BackendDescriptor&& other) noexcept {
            if (&other != this) {
                if (descriptor != nullptr)
                    cudnnBackendDestroyDescriptor(descriptor);
                descriptor = other.descriptor;
                other.descriptor = nullptr;
            }
            return *this;
        }

        void set(cudnnBackendAttributeName_t name, cudnnBackendAttributeType_t type, int64_t count, const void* value) {
            CUDA4DNN_CHECK_CUDNN(cudnnBackendSetAttribute(descriptor, name, type, count, const_cast<void*>(value)));
        }

        void finalize() {
            CUDA4DNN_CHECK_CUDNN(cudnnBackendFinalize(descriptor));
        }

        cudnnBackendDescriptor_t get() const noexcept { return descriptor; }
        explicit operator bool() const noexcept { return descriptor != nullptr; }

    private:
        cudnnBackendDescriptor_t descriptor;
    };

    template <class T>
    BackendDescriptor makeTensorDescriptor(int64_t uid, const std::vector<int64_t>& dim, const std::vector<int64_t>& stride) {
        BackendDescriptor desc(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
        cudnnDataType_t dtype = detail::get_data_type<T>();
        int64_t alignment = 16;
        int64_t rank = static_cast<int64_t>(dim.size());
        desc.set(CUDNN_ATTR_TENSOR_DATA_TYPE,      CUDNN_TYPE_DATA_TYPE, 1, &dtype);
        desc.set(CUDNN_ATTR_TENSOR_DIMENSIONS,     CUDNN_TYPE_INT64, rank, dim.data());
        desc.set(CUDNN_ATTR_TENSOR_STRIDES,        CUDNN_TYPE_INT64, rank, stride.data());
        desc.set(CUDNN_ATTR_TENSOR_UNIQUE_ID,      CUDNN_TYPE_INT64, 1, &uid);
        desc.set(CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment);
        desc.finalize();
        return desc;
    }

    /** channel-last strides for a tensor whose dims stay in [N, C, spatial...] order */
    inline std::vector<int64_t> channelLastStrides(const std::vector<int64_t>& shape) {
        const int64_t rank = static_cast<int64_t>(shape.size());
        CV_Assert(rank >= 3);
        std::vector<int64_t> stride(rank);
        stride[1] = 1;
        int64_t running = shape[1];
        for (int64_t k = rank - 1; k >= 2; k--) {
            stride[k] = running;
            running *= shape[k];
        }
        stride[0] = running;
        return stride;
    }

    /** wraps one finalized op in an op graph and keeps the first heuristic engine config whose plan compiles */
    inline void buildExecutionPlan(const Handle& handle, const BackendDescriptor& op, const char* what,
                                   BackendDescriptor& opGraph, BackendDescriptor& engineConfig,
                                   BackendDescriptor& executionPlan, int64_t& workspace_size)
    {
        opGraph = BackendDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        {
            cudnnHandle_t h = handle.get();
            cudnnBackendDescriptor_t o = op.get();
            opGraph.set(CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE,             1, &h);
            opGraph.set(CUDNN_ATTR_OPERATIONGRAPH_OPS,    CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &o);
            opGraph.finalize();
        }

        BackendDescriptor heuristics(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
        {
            cudnnBackendDescriptor_t g = opGraph.get();
            cudnnBackendHeurMode_t mode = CUDNN_HEUR_MODE_A;
            heuristics.set(CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &g);
            heuristics.set(CUDNN_ATTR_ENGINEHEUR_MODE,            CUDNN_TYPE_HEUR_MODE,          1, &mode);
            heuristics.finalize();
        }

        constexpr int max_configs = 32;
        std::vector<BackendDescriptor> configs;
        configs.reserve(max_configs);
        std::array<cudnnBackendDescriptor_t, max_configs> configHandles{};
        for (int c = 0; c < max_configs; c++) {
            configs.emplace_back(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
            configHandles[c] = configs.back().get();
        }

        int64_t returned = 0;
        CUDA4DNN_CHECK_CUDNN(cudnnBackendGetAttribute(
            heuristics.get(), CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR,
            max_configs, &returned, configHandles.data()));

        /* keep the first engine config whose plan finalizes (i.e. compiles) */
        for (int c = 0; c < returned; c++) {
            BackendDescriptor plan(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
            cudnnHandle_t h = handle.get();
            plan.set(CUDNN_ATTR_EXECUTION_PLAN_HANDLE,        CUDNN_TYPE_HANDLE,             1, &h);
            plan.set(CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &configHandles[c]);
            if (cudnnBackendFinalize(plan.get()) == CUDNN_STATUS_SUCCESS) {
                engineConfig = std::move(configs[c]);
                executionPlan = std::move(plan);
                break;
            }
        }

        if (!executionPlan)
            CV_Error(cv::Error::GpuApiCallError, cv::format("cuDNN JIT did not produce an execution plan for the %s.", what));

        int64_t count = 0;
        CUDA4DNN_CHECK_CUDNN(cudnnBackendGetAttribute(
            executionPlan.get(), CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64,
            1, &count, &workspace_size));
    }

    /** cuDNN graph convolution (NHWC data, KRSC filter); the plan is built once at construction */
    template <class T>
    class ConvolutionGraph {
    public:
        struct params_type {
            std::vector<int64_t> input_shape;   /* N, C, spatial... */
            std::vector<int64_t> output_shape;  /* N, C, spatial... */
            std::vector<int64_t> filter_shape;  /* OC, IC, kernel... */
            std::vector<int64_t> padding_pre, padding_post, stride, dilation; /* one entry per spatial dim */
        };

        ConvolutionGraph() = default;
        ConvolutionGraph(const ConvolutionGraph&) = delete;
        ConvolutionGraph(ConvolutionGraph&&) = default;
        ConvolutionGraph& operator=(const ConvolutionGraph&) = delete;
        ConvolutionGraph& operator=(ConvolutionGraph&&) = default;

        ConvolutionGraph(const Handle& handle, const params_type& params) {
            const auto& i = params.input_shape;
            const auto& o = params.output_shape;
            const auto& f = params.filter_shape;

            const int64_t rank = static_cast<int64_t>(i.size());
            CV_Assert(o.size() == static_cast<std::size_t>(rank));
            CV_Assert(f.size() == static_cast<std::size_t>(rank));

            const int64_t spatial_dims = rank - 2;
            CV_Assert(params.padding_pre.size() == static_cast<std::size_t>(spatial_dims));
            CV_Assert(params.padding_post.size() == static_cast<std::size_t>(spatial_dims));
            CV_Assert(params.stride.size() == static_cast<std::size_t>(spatial_dims));
            CV_Assert(params.dilation.size() == static_cast<std::size_t>(spatial_dims));

            /* channel-last strides, e.g. NHWC: {C*H*W, 1, W*C, C} */
            const std::vector<int64_t> xStride = channelLastStrides(i);
            const std::vector<int64_t> yStride = channelLastStrides(o);
            const std::vector<int64_t> wStride = channelLastStrides(f);

            xDesc = makeTensorDescriptor<T>('x', i, xStride);
            yDesc = makeTensorDescriptor<T>('y', o, yStride);
            wDesc = makeTensorDescriptor<T>('w', f, wStride);

            convDesc = BackendDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
            {
                cudnnDataType_t comp = CUDNN_DATA_FLOAT;
                cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
                convDesc.set(CUDNN_ATTR_CONVOLUTION_COMP_TYPE,      CUDNN_TYPE_DATA_TYPE,        1, &comp);
                convDesc.set(CUDNN_ATTR_CONVOLUTION_CONV_MODE,      CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode);
                convDesc.set(CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS,   CUDNN_TYPE_INT64, 1, &spatial_dims);
                convDesc.set(CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,   CUDNN_TYPE_INT64, spatial_dims, params.padding_pre.data());
                convDesc.set(CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,  CUDNN_TYPE_INT64, spatial_dims, params.padding_post.data());
                convDesc.set(CUDNN_ATTR_CONVOLUTION_DILATIONS,      CUDNN_TYPE_INT64, spatial_dims, params.dilation.data());
                convDesc.set(CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, spatial_dims, params.stride.data());
                convDesc.finalize();
            }

            convOp = BackendDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
            {
                float alpha = 1.0f, beta = 0.0f;
                cudnnBackendDescriptor_t x = xDesc.get(), w = wDesc.get(), y = yDesc.get(), c = convDesc.get();
                convOp.set(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,         CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &x);
                convOp.set(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,         CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &w);
                convOp.set(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,         CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &y);
                convOp.set(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &c);
                convOp.set(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,     CUDNN_TYPE_FLOAT, 1, &alpha);
                convOp.set(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,      CUDNN_TYPE_FLOAT, 1, &beta);
                convOp.finalize();
            }

            buildExecutionPlan(handle, convOp, "convolution", opGraph, engineConfig, executionPlan, workspace_size);
        }

        std::size_t get_workspace_size() const noexcept { return static_cast<std::size_t>(workspace_size); }

        void convolve(
            const Handle& handle,
            DevicePtr<const T> input,
            DevicePtr<const T> filter,
            DevicePtr<T> output,
            WorkspaceInstance scratchpad)
        {
            CV_Assert(executionPlan);

            void* pointers[3] = { const_cast<T*>(input.get()), const_cast<T*>(filter.get()), output.get() };
            void* workspace = static_cast<void*>(scratchpad.get());

            if (!variantPack || pointers[0] != cachedPointers[0] || pointers[1] != cachedPointers[1] ||
                pointers[2] != cachedPointers[2] || workspace != cachedPointers[3])
            {
                int64_t uids[3] = { 'x', 'w', 'y' };

                BackendDescriptor newVariantPack(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
                newVariantPack.set(CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, pointers);
                newVariantPack.set(CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,    CUDNN_TYPE_INT64,    3, uids);
                newVariantPack.set(CUDNN_ATTR_VARIANT_PACK_WORKSPACE,     CUDNN_TYPE_VOID_PTR, 1, &workspace);
                newVariantPack.finalize();

                variantPack = std::move(newVariantPack);
                cachedPointers[0] = pointers[0];
                cachedPointers[1] = pointers[1];
                cachedPointers[2] = pointers[2];
                cachedPointers[3] = workspace;
            }

            CUDA4DNN_CHECK_CUDNN(cudnnBackendExecute(handle.get(), executionPlan.get(), variantPack.get()));
        }

    private:
        /* the plan references this chain; all of it must outlive the plan */
        BackendDescriptor xDesc, yDesc, wDesc;
        BackendDescriptor convDesc, convOp, opGraph;
        BackendDescriptor engineConfig, executionPlan;
        int64_t workspace_size = 0;

        /* cached variant pack: rebuilt only when the bound pointers actually change */
        BackendDescriptor variantPack;
        void* cachedPointers[4] = { nullptr, nullptr, nullptr, nullptr };
    };

}}}}} /* namespace cv::dnn::cuda4dnn::csl::cudnn */

#endif /* HAVE_CUDNNJIT */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_CUDNN_GRAPH_HPP */
