/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#if !defined CUDA_DISABLER

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/limits.hpp"

#include "cuda/disparity_bilateral_filter.hpp"

namespace cv { namespace cuda { namespace device
{
    namespace disp_bilateral_filter
    {
        template <int channels>
        struct DistRgbMax
        {
            static __device__ __forceinline__ uchar calc(const uchar* a, const uchar* b)
            {
                uchar x = ::abs(a[0] - b[0]);
                uchar y = ::abs(a[1] - b[1]);
                uchar z = ::abs(a[2] - b[2]);
                return (::max(::max(x, y), z));
            }
        };

        template <>
        struct DistRgbMax<1>
        {
            static __device__ __forceinline__ uchar calc(const uchar* a, const uchar* b)
            {
                return ::abs(a[0] - b[0]);
            }
        };

        template <int channels, typename T>
        __global__ void disp_bilateral_filter(int t, T* disp, size_t disp_step,
            const uchar* img, size_t img_step, int h, int w,
            const float* ctable_color, const float * ctable_space, size_t ctable_space_step,
            int cradius,
            short cedge_disc, short cmax_disc)
        {
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            const int x = ((blockIdx.x * blockDim.x + threadIdx.x) << 1) + ((y + t) & 1);

            T dp[5];

            if (y > 0 && y < h - 1 && x > 0 && x < w - 1)
            {
                dp[0] = *(disp + (y  ) * disp_step + x + 0);
                dp[1] = *(disp + (y-1) * disp_step + x + 0);
                dp[2] = *(disp + (y  ) * disp_step + x - 1);
                dp[3] = *(disp + (y+1) * disp_step + x + 0);
                dp[4] = *(disp + (y  ) * disp_step + x + 1);

                if(::abs(dp[1] - dp[0]) >= cedge_disc || ::abs(dp[2] - dp[0]) >= cedge_disc || ::abs(dp[3] - dp[0]) >= cedge_disc || ::abs(dp[4] - dp[0]) >= cedge_disc)
                {
                    const int ymin = ::max(0, y - cradius);
                    const int xmin = ::max(0, x - cradius);
                    const int ymax = ::min(h - 1, y + cradius);
                    const int xmax = ::min(w - 1, x + cradius);

                    float cost[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

                    const uchar* ic = img + y * img_step + channels * x;

                    for(int yi = ymin; yi <= ymax; yi++)
                    {
                        const T* disp_y = disp + yi * disp_step;

                        for(int xi = xmin; xi <= xmax; xi++)
                        {
                            const uchar* in = img + yi * img_step + channels * xi;

                            uchar dist_rgb = DistRgbMax<channels>::calc(in, ic);

                            const float weight = ctable_color[dist_rgb] * (ctable_space + ::abs(y-yi)* ctable_space_step)[::abs(x-xi)];

                            const T disp_reg = disp_y[xi];

                            cost[0] += ::min(cmax_disc, ::abs(disp_reg - dp[0])) * weight;
                            cost[1] += ::min(cmax_disc, ::abs(disp_reg - dp[1])) * weight;
                            cost[2] += ::min(cmax_disc, ::abs(disp_reg - dp[2])) * weight;
                            cost[3] += ::min(cmax_disc, ::abs(disp_reg - dp[3])) * weight;
                            cost[4] += ::min(cmax_disc, ::abs(disp_reg - dp[4])) * weight;
                        }
                    }

                    float minimum = numeric_limits<float>::max();
                    int id = 0;

                    if (cost[0] < minimum)
                    {
                        minimum = cost[0];
                        id = 0;
                    }
                    if (cost[1] < minimum)
                    {
                        minimum = cost[1];
                        id = 1;
                    }
                    if (cost[2] < minimum)
                    {
                        minimum = cost[2];
                        id = 2;
                    }
                    if (cost[3] < minimum)
                    {
                        minimum = cost[3];
                        id = 3;
                    }
                    if (cost[4] < minimum)
                    {
                        minimum = cost[4];
                        id = 4;
                    }

                    *(disp + y * disp_step + x) = dp[id];
                }
            }
        }

        template <typename T>
        void disp_bilateral_filter(PtrStepSz<T> disp, PtrStepSzb img, int channels, int iters, const float *table_color, const float* table_space, size_t table_step, int radius, short edge_disc, short max_disc, cudaStream_t stream)
        {
            dim3 threads(32, 8, 1);
            dim3 grid(1, 1, 1);
            grid.x = divUp(disp.cols, threads.x << 1);
            grid.y = divUp(disp.rows, threads.y);

            switch (channels)
            {
            case 1:
                for (int i = 0; i < iters; ++i)
                {
                    disp_bilateral_filter<1><<<grid, threads, 0, stream>>>(0, disp.data, disp.step/sizeof(T), img.data, img.step, disp.rows, disp.cols, table_color, table_space, table_step, radius, edge_disc, max_disc);
                    cudaSafeCall( cudaGetLastError() );

                    disp_bilateral_filter<1><<<grid, threads, 0, stream>>>(1, disp.data, disp.step/sizeof(T), img.data, img.step, disp.rows, disp.cols, table_color, table_space, table_step, radius, edge_disc, max_disc);
                    cudaSafeCall( cudaGetLastError() );
                }
                break;
            case 3:
                for (int i = 0; i < iters; ++i)
                {
                    disp_bilateral_filter<3><<<grid, threads, 0, stream>>>(0, disp.data, disp.step/sizeof(T), img.data, img.step, disp.rows, disp.cols, table_color, table_space, table_step, radius, edge_disc, max_disc);
                    cudaSafeCall( cudaGetLastError() );

                    disp_bilateral_filter<3><<<grid, threads, 0, stream>>>(1, disp.data, disp.step/sizeof(T), img.data, img.step, disp.rows, disp.cols, table_color, table_space, table_step, radius, edge_disc, max_disc);
                    cudaSafeCall( cudaGetLastError() );
                }
                break;
            default:
                CV_Error(cv::Error::BadNumChannels, "Unsupported channels count");
            }

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template void disp_bilateral_filter<uchar>(PtrStepSz<uchar> disp, PtrStepSzb img, int channels, int iters, const float *table_color, const float *table_space, size_t table_step, int radius, short, short, cudaStream_t stream);
        template void disp_bilateral_filter<short>(PtrStepSz<short> disp, PtrStepSzb img, int channels, int iters, const float *table_color, const float *table_space, size_t table_step, int radius, short, short, cudaStream_t stream);
    } // namespace bilateral_filter
}}} // namespace cv { namespace cuda { namespace cudev

#endif /* CUDA_DISABLER */
