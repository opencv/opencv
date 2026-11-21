// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "types.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"
#include "../cuda4dnn/kernels/col2im.hpp"

#include <opencv2/core.hpp>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        template <class T>
        __global__ void col2im_2d(Span<T> output, View<T> cols, View<T> bias, Col2ImParams p, size_type count)
        {
            const index_type in_plane = p.in_h * p.in_w;
            const index_type out_plane = p.out_h * p.out_w;
            for (auto i : grid_stride_range(count)) {
                const index_type ox = i % p.out_w;
                const index_type oy = (i / p.out_w) % p.out_h;
                const index_type c = (i / out_plane) % p.channels;
                const index_type n = i / (out_plane * p.channels);

                float acc = bias.size() ? static_cast<float>(bias[p.channel_offset + c]) : 0.f;
                const index_type batch_base = n * p.channels * p.kernel_h * p.kernel_w * in_plane;
                for (index_type ky = 0; ky < p.kernel_h; ky++) {
                    // input row iy feeds oy when iy * stride == oy + pad - ky * dilation
                    const index_type ty = oy + p.pad_h - ky * p.dil_h;
                    if (ty < 0 || ty % p.stride_h != 0)
                        continue;
                    const index_type iy = ty / p.stride_h;
                    if (iy >= p.in_h)
                        continue;
                    for (index_type kx = 0; kx < p.kernel_w; kx++) {
                        const index_type tx = ox + p.pad_w - kx * p.dil_w;
                        if (tx < 0 || tx % p.stride_w != 0)
                            continue;
                        const index_type ix = tx / p.stride_w;
                        if (ix >= p.in_w)
                            continue;
                        const index_type row = (c * p.kernel_h + ky) * p.kernel_w + kx;
                        acc += static_cast<float>(cols[batch_base + row * in_plane + iy * p.in_w + ix]);
                    }
                }
                output[(n * p.total_channels + p.channel_offset + c) * out_plane + oy * p.out_w + ox] = static_cast<T>(acc);
            }
        }
    }

    template <class T>
    void col2im_2d(const Stream& stream, Span<T> output, View<T> cols, View<T> bias, const Col2ImParams& params)
    {
        const size_type count = params.batch * params.channels * params.out_h * params.out_w;
        if (count == 0)
            return;
        auto kernel = raw::col2im_2d<T>;
        auto policy = make_policy(kernel, count, 0, stream);
        launch_kernel(kernel, policy, output, cols, bias, params, count);
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void col2im_2d(const Stream&, Span<__half>, View<__half>, View<__half>, const Col2ImParams&);
#endif
    template void col2im_2d(const Stream&, Span<float>, View<float>, View<float>, const Col2ImParams&);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
