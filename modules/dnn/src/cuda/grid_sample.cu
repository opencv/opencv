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

#include "../cuda4dnn/kernels/grid_sample.hpp"

#include <opencv2/core.hpp>

#include <cstdint>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {

        enum { M_NEAREST = 0, M_BILINEAR = 1, M_BICUBIC = 2 };
        enum { P_ZEROS = 0, P_BORDER = 1, P_REFLECTION = 2 };

        __device__ inline float reflect_coord(float x, int limit, bool align_corners) {
            if (limit <= 1) return 0.f;

            const float minv = align_corners ? 0.f : -0.5f;
            const float maxv = align_corners ? float(limit - 1) : float(limit) - 0.5f;
            const float m = maxv - minv;
            const float two_m = 2.f * m;

            float t = fmodf(x - minv, two_m);
            if (t < 0) t += two_m;
            if (t > m) t = two_m - t;
            return t + minv;
        }

        __device__ inline void cubic_coeffs(float x, float A, float* c) {
            c[0] = ((A * (x + 1.0f) - 5.0f * A) * (x + 1.0f) + 8.0f * A) * (x + 1.0f) - 4.0f * A;
            c[1] = ((A + 2.0f) * x - (A + 3.0f)) * x * x + 1.0f;
            c[2] = ((A + 2.0f) * (1.0f - x) - (A + 3.0f)) * (1.0f - x) * (1.0f - x) + 1.0f;
            c[3] = 1.0f - c[0] - c[1] - c[2];
        }

        /* The padding mode is decided per layer, so this branch is warp-uniform. */
        template <class T>
        __device__ inline float fetch(const T* baseNC, int yy, int xx, const GridSampleParams& p) {
            int px = xx, py = yy;
            if (p.padding == P_BORDER) {
                px = static_cast<int>(fminf(float(p.width - 1),  fmaxf(0.f, float(px))));
                py = static_cast<int>(fminf(float(p.height - 1), fmaxf(0.f, float(py))));
            } else if (p.padding == P_REFLECTION) {
                px = static_cast<int>(floorf(reflect_coord(float(px), p.width,  p.align_corners) + 0.5f));
                py = static_cast<int>(floorf(reflect_coord(float(py), p.height, p.align_corners) + 0.5f));
            }
            if (px < 0 || py < 0 || px >= p.width || py >= p.height) return 0.f;
            return static_cast<float>(baseNC[static_cast<std::int64_t>(py) * p.width + px]);
        }

        template <class T>
        __global__ void grid_sample(Span<T> output, View<T> input, View<float> grid, GridSampleParams p)
        {
            const float delta = p.align_corners ? 1.f : 0.f;
            const float xscale = 0.5f * (p.width - delta);
            const float yscale = 0.5f * (p.height - delta);
            const float xdelta = 0.5f * (p.width - delta)  + 0.5f * (delta - 1.f);
            const float ydelta = 0.5f * (p.height - delta) + 0.5f * (delta - 1.f);

            for (auto id : grid_stride_range(output.size())) {
                int rem = id;
                const int w = rem % p.output_width;  rem /= p.output_width;
                const int h = rem % p.output_height; rem /= p.output_height;
                const int c = rem % p.channels;      rem /= p.channels;
                const int n = rem;

                const T* baseNC = input.data().get() +
                                  (static_cast<std::int64_t>(n) * p.channels + c) *
                                  static_cast<std::int64_t>(p.height) * p.width;

                const std::int64_t g = ((static_cast<std::int64_t>(n) * p.output_height + h) *
                                        p.output_width + w) * 2;
                const float nx = grid.data().get()[g + 0];
                const float ny = grid.data().get()[g + 1];

                const float xf = nx * xscale + xdelta;
                const float yf = ny * yscale + ydelta;

                float outv = 0.f;
                if (p.mode == M_NEAREST) {
                    outv = fetch<T>(baseNC, __float2int_rn(yf), __float2int_rn(xf), p);
                } else if (p.mode == M_BILINEAR) {
                    const int x0 = static_cast<int>(floorf(xf));
                    const int y0 = static_cast<int>(floorf(yf));
                    const float dx = xf - x0, dy = yf - y0;

                    const float v00 = fetch<T>(baseNC, y0,     x0,     p);
                    const float v01 = fetch<T>(baseNC, y0,     x0 + 1, p);
                    const float v10 = fetch<T>(baseNC, y0 + 1, x0,     p);
                    const float v11 = fetch<T>(baseNC, y0 + 1, x0 + 1, p);

                    const float vx0 = v00 * (1.f - dx) + v01 * dx;
                    const float vx1 = v10 * (1.f - dx) + v11 * dx;
                    outv = vx0 * (1.f - dy) + vx1 * dy;
                } else {
                    const int x1 = static_cast<int>(floorf(xf));
                    const int y1 = static_cast<int>(floorf(yf));
                    const float tx = xf - x1, ty = yf - y1;

                    float wx[4], wy[4];
                    cubic_coeffs(tx, p.cubic_alpha, wx);
                    cubic_coeffs(ty, p.cubic_alpha, wy);

                    outv = 0.f;
                    for (int j = 0; j < 4; ++j) {
                        float rowv = 0.f;
                        for (int i = 0; i < 4; ++i)
                            rowv += fetch<T>(baseNC, y1 - 1 + j, x1 - 1 + i, p) * wx[i];
                        outv += rowv * wy[j];
                    }
                }

                output[id] = static_cast<T>(outv);
            }
        }
    }

    template <class T>
    void grid_sample(const Stream& stream,
        Span<T> output, View<T> input, View<float> grid,
        const GridSampleParams& params)
    {
        CV_Assert(params.channels > 0 && params.height > 0 && params.width > 0);
        CV_Assert(params.output_height > 0 && params.output_width > 0);
        CV_Assert(params.mode >= 0 && params.mode <= 2);
        CV_Assert(params.padding >= 0 && params.padding <= 2);

        if (output.size() == 0)
            return;

        auto kernel = raw::grid_sample<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input, grid, params);
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void grid_sample(const Stream&, Span<__half>, View<__half>, View<float>, const GridSampleParams&);
#endif
    template void grid_sample(const Stream&, Span<float>, View<float>, View<float>, const GridSampleParams&);
    template void grid_sample(const Stream&, Span<int8_t>, View<int8_t>, View<float>, const GridSampleParams&);
    template void grid_sample(const Stream&, Span<uint8_t>, View<uint8_t>, View<float>, const GridSampleParams&);
    template void grid_sample(const Stream&, Span<int32_t>, View<int32_t>, View<float>, const GridSampleParams&);
    template void grid_sample(const Stream&, Span<int64_t>, View<int64_t>, View<float>, const GridSampleParams&);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
