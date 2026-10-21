// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CONV_TRANSPOSE2_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CONV_TRANSPOSE2_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/cublas.hpp"
#include "../csl/span.hpp"
#include "../csl/tensor.hpp"
#include "../csl/workspace.hpp"

#include "../kernels/col2im.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    struct ConvTranspose2Configuration {
        MatShape input_shape, weight_shape, output_shape;  // NCHW, [Cin, Cout/group, kh, kw], NCHW
        int groups;
        std::vector<int> strides, dilations, pads_begin;   // two spatial entries each
    };

    // ConvTranspose without legacy cuDNN: per group, cols = W^T * X (cuBLAS), then col2im adds into Y
    template <class T>
    class ConvTranspose2Op final : public CUDABackendNode {
    public:
        ConvTranspose2Op(csl::Stream stream_, csl::cublas::Handle handle, const ConvTranspose2Configuration& config)
            : stream(std::move(stream_)), cublasHandle(std::move(handle))
        {
            const MatShape& in = config.input_shape;
            const MatShape& w = config.weight_shape;
            const MatShape& out = config.output_shape;
            CV_Assert(in.dims == 4 && w.dims == 4 && out.dims == 4);
            CV_Assert(config.groups > 0 && in[1] % config.groups == 0 && w[0] == in[1]);

            groups = config.groups;
            batch = in[0];
            cin = in[1];
            cin_g = cin / groups;
            cout_g = w[1];
            cout = out[1];
            CV_Assert(cout == cout_g * groups);
            in_h = in[2]; in_w = in[3];
            out_h = out[2]; out_w = out[3];
            kernel_h = w[2]; kernel_w = w[3];
            strides = config.strides;
            dilations = config.dilations;
            pads_begin = config.pads_begin;

            // one group's columns for the whole batch; groups reuse it in stream order
            csl::WorkspaceBuilder builder;
            builder.require<T>(colsSize());
            scratch_mem_in_bytes = builder.required_workspace_size();
        }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(inputs.size() >= 2 && outputs.size() == 1);

            auto x = csl::viewOf<T>(inputs[0]);
            auto w = csl::viewOf<T>(inputs[1]);
            csl::View<T> bias;
            if (inputs.size() >= 3 && !inputs[2].empty()) {
                auto b = csl::viewOf<T>(inputs[2]);
                bias = csl::View<T>(b.get(), b.size());
            }
            auto y = csl::spanOf<T>(outputs[0]);

            auto ws_allocator = csl::WorkspaceAllocator(workspace);
            auto cols = ws_allocator.get_span<T>(colsSize());

            const std::size_t hw = (std::size_t)in_h * in_w;
            const std::size_t m = (std::size_t)cout_g * kernel_h * kernel_w;
            for (int g = 0; g < groups; g++) {
                // row-major cols(m x hw) = W_g^T(m x cin_g) * X_g(cin_g x hw); in column-major terms cols^T = X_g^T * W_g
                csl::cublas::gemmStridedBatched<T>(cublasHandle,
                    false, true, hw, m, cin_g,
                    T(1), x.get() + (std::ptrdiff_t)(g * cin_g * hw), hw, (std::size_t)cin * hw,
                    w.get() + (std::ptrdiff_t)(g * cin_g * m), m, 0,
                    T(0), cols.data(), hw, m * hw,
                    batch);

                kernels::Col2ImParams p;
                p.batch = batch; p.channels = cout_g; p.total_channels = cout; p.channel_offset = g * cout_g;
                p.in_h = in_h; p.in_w = in_w; p.out_h = out_h; p.out_w = out_w;
                p.kernel_h = kernel_h; p.kernel_w = kernel_w;
                p.stride_h = strides[0]; p.stride_w = strides[1];
                p.pad_h = pads_begin[0]; p.pad_w = pads_begin[1];
                p.dil_h = dilations[0]; p.dil_w = dilations[1];
                kernels::col2im_2d<T>(stream, csl::Span<T>(y.get(), y.size()),
                                      csl::View<T>(cols.data(), cols.size()), bias, p);
            }
        }

        std::size_t get_workspace_memory_in_bytes() const noexcept override { return scratch_mem_in_bytes; }

    private:
        std::size_t colsSize() const { return (std::size_t)batch * cout_g * kernel_h * kernel_w * in_h * in_w; }

        csl::Stream stream;
        csl::cublas::Handle cublasHandle;
        int groups, batch, cin, cin_g, cout, cout_g;
        int in_h, in_w, out_h, out_w, kernel_h, kernel_w;
        std::vector<int> strides, dilations, pads_begin;
        std::size_t scratch_mem_in_bytes;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CONV_TRANSPOSE2_HPP */
