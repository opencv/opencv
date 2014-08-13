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
#include "opencv2/core/cuda/saturate_cast.hpp"
#include "opencv2/core/cuda/limits.hpp"
#include "opencv2/core/cuda/reduce.hpp"
#include "opencv2/core/cuda/functional.hpp"

#include "cuda/stereocsbp.hpp"

namespace cv { namespace cuda { namespace device
{
    namespace stereocsbp
    {
        ///////////////////////////////////////////////////////////////
        /////////////////////// init data cost ////////////////////////
        ///////////////////////////////////////////////////////////////

        template <int channels> static float __device__ pixeldiff(const uchar* left, const uchar* right, float max_data_term);
        template<> __device__ __forceinline__ static float pixeldiff<1>(const uchar* left, const uchar* right, float max_data_term)
        {
            return fmin( ::abs((int)*left - *right), max_data_term);
        }
        template<> __device__ __forceinline__ static float pixeldiff<3>(const uchar* left, const uchar* right, float max_data_term)
        {
            float tb = 0.114f * ::abs((int)left[0] - right[0]);
            float tg = 0.587f * ::abs((int)left[1] - right[1]);
            float tr = 0.299f * ::abs((int)left[2] - right[2]);

            return fmin(tr + tg + tb, max_data_term);
        }
        template<> __device__ __forceinline__ static float pixeldiff<4>(const uchar* left, const uchar* right, float max_data_term)
        {
            uchar4 l = *((const uchar4*)left);
            uchar4 r = *((const uchar4*)right);

            float tb = 0.114f * ::abs((int)l.x - r.x);
            float tg = 0.587f * ::abs((int)l.y - r.y);
            float tr = 0.299f * ::abs((int)l.z - r.z);

            return fmin(tr + tg + tb, max_data_term);
        }

        template <typename T>
        __global__ void get_first_k_initial_global(uchar *ctemp, T* data_cost_selected_, T *selected_disp_pyr, int h, int w, int nr_plane, int ndisp,
            size_t msg_step, size_t disp_step)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (y < h && x < w)
            {
                T* selected_disparity = selected_disp_pyr + y * msg_step + x;
                T* data_cost_selected = data_cost_selected_ + y * msg_step + x;
                T* data_cost = (T*)ctemp + y * msg_step + x;

                for(int i = 0; i < nr_plane; i++)
                {
                    T minimum = device::numeric_limits<T>::max();
                    int id = 0;
                    for(int d = 0; d < ndisp; d++)
                    {
                        T cur = data_cost[d * disp_step];
                        if(cur < minimum)
                        {
                            minimum = cur;
                            id = d;
                        }
                    }

                    data_cost_selected[i  * disp_step] = minimum;
                    selected_disparity[i  * disp_step] = id;
                    data_cost         [id * disp_step] = numeric_limits<T>::max();
                }
            }
        }


        template <typename T>
        __global__ void get_first_k_initial_local(uchar *ctemp, T* data_cost_selected_, T* selected_disp_pyr, int h, int w, int nr_plane, int ndisp,
            size_t msg_step, size_t disp_step)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (y < h && x < w)
            {
                T* selected_disparity = selected_disp_pyr + y * msg_step + x;
                T* data_cost_selected = data_cost_selected_ + y * msg_step + x;
                T* data_cost = (T*)ctemp + y * msg_step + x;

                int nr_local_minimum = 0;

                T prev = data_cost[0 * disp_step];
                T cur  = data_cost[1 * disp_step];
                T next = data_cost[2 * disp_step];

                for (int d = 1; d < ndisp - 1 && nr_local_minimum < nr_plane; d++)
                {
                    if (cur < prev && cur < next)
                    {
                        data_cost_selected[nr_local_minimum * disp_step] = cur;
                        selected_disparity[nr_local_minimum * disp_step] = d;

                        data_cost[d * disp_step] = numeric_limits<T>::max();

                        nr_local_minimum++;
                    }
                    prev = cur;
                    cur = next;
                    next = data_cost[(d + 1) * disp_step];
                }

                for (int i = nr_local_minimum; i < nr_plane; i++)
                {
                    T minimum = numeric_limits<T>::max();
                    int id = 0;

                    for (int d = 0; d < ndisp; d++)
                    {
                        cur = data_cost[d * disp_step];
                        if (cur < minimum)
                        {
                            minimum = cur;
                            id = d;
                        }
                    }
                    data_cost_selected[i * disp_step] = minimum;
                    selected_disparity[i * disp_step] = id;

                    data_cost[id * disp_step] = numeric_limits<T>::max();
                }
            }
        }

        template <typename T, int channels>
        __global__ void init_data_cost(const uchar *cleft, const uchar *cright, uchar *ctemp, size_t cimg_step,
                                      int h, int w, int level, int ndisp, float data_weight, float max_data_term,
                                      int min_disp, size_t msg_step, size_t disp_step)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (y < h && x < w)
            {
                int y0 = y << level;
                int yt = (y + 1) << level;

                int x0 = x << level;
                int xt = (x + 1) << level;

                T* data_cost = (T*)ctemp + y * msg_step + x;

                for(int d = 0; d < ndisp; ++d)
                {
                    float val = 0.0f;
                    for(int yi = y0; yi < yt; yi++)
                    {
                        for(int xi = x0; xi < xt; xi++)
                        {
                            int xr = xi - d;
                            if(d < min_disp || xr < 0)
                                val += data_weight * max_data_term;
                            else
                            {
                                const uchar* lle = cleft + yi * cimg_step + xi * channels;
                                const uchar* lri = cright + yi * cimg_step + xr * channels;

                                val += data_weight * pixeldiff<channels>(lle, lri, max_data_term);
                            }
                        }
                    }
                    data_cost[disp_step * d] = saturate_cast<T>(val);
                }
            }
        }

        template <typename T, int winsz, int channels>
        __global__ void init_data_cost_reduce(const uchar *cleft, const uchar *cright, uchar *ctemp, size_t cimg_step,
                                              int level, int rows, int cols, int h, int ndisp, float data_weight, float max_data_term,
                                              int min_disp, size_t msg_step, size_t disp_step)
        {
            int x_out = blockIdx.x;
            int y_out = blockIdx.y % h;
            int d = (blockIdx.y / h) * blockDim.z + threadIdx.z;

            int tid = threadIdx.x;

            if (d < ndisp)
            {
                int x0 = x_out << level;
                int y0 = y_out << level;

                int len = ::min(y0 + winsz, rows) - y0;

                float val = 0.0f;
                if (x0 + tid < cols)
                {
                    if (x0 + tid - d < 0 || d < min_disp)
                        val = data_weight * max_data_term * len;
                    else
                    {
                        const uchar* lle =  cleft + y0 * cimg_step + channels * (x0 + tid    );
                        const uchar* lri = cright + y0 * cimg_step + channels * (x0 + tid - d);

                        for(int y = 0; y < len; ++y)
                        {
                            val += data_weight * pixeldiff<channels>(lle, lri, max_data_term);

                            lle += cimg_step;
                            lri += cimg_step;
                        }
                    }
                }

                extern __shared__ float smem[];

                reduce<winsz>(smem + winsz * threadIdx.z, val, tid, plus<float>());

                T* data_cost = (T*)ctemp + y_out * msg_step + x_out;

                if (tid == 0)
                    data_cost[disp_step * d] = saturate_cast<T>(val);
            }
        }


        template <typename T>
        void init_data_cost_caller_(const uchar *cleft, const uchar *cright, uchar *ctemp, size_t cimg_step, int /*rows*/, int /*cols*/, int h, int w, int level, int ndisp, int channels, float data_weight, float max_data_term, int min_disp, size_t msg_step, size_t disp_step, cudaStream_t stream)
        {
            dim3 threads(32, 8, 1);
            dim3 grid(1, 1, 1);

            grid.x = divUp(w, threads.x);
            grid.y = divUp(h, threads.y);

            switch (channels)
            {
            case 1: init_data_cost<T, 1><<<grid, threads, 0, stream>>>(cleft, cright, ctemp, cimg_step, h, w, level, ndisp, data_weight, max_data_term, min_disp, msg_step, disp_step); break;
            case 3: init_data_cost<T, 3><<<grid, threads, 0, stream>>>(cleft, cright, ctemp, cimg_step, h, w, level, ndisp, data_weight, max_data_term, min_disp, msg_step, disp_step); break;
            case 4: init_data_cost<T, 4><<<grid, threads, 0, stream>>>(cleft, cright, ctemp, cimg_step, h, w, level, ndisp, data_weight, max_data_term, min_disp, msg_step, disp_step); break;
            default: CV_Error(cv::Error::BadNumChannels, "Unsupported channels count");
            }
        }

        template <typename T, int winsz>
        void init_data_cost_reduce_caller_(const uchar *cleft, const uchar *cright, uchar *ctemp, size_t cimg_step, int rows, int cols, int h, int w, int level, int ndisp, int channels, float data_weight, float max_data_term, int min_disp, size_t msg_step, size_t disp_step, cudaStream_t stream)
        {
            const int threadsNum = 256;
            const size_t smem_size = threadsNum * sizeof(float);

            dim3 threads(winsz, 1, threadsNum / winsz);
            dim3 grid(w, h, 1);
            grid.y *= divUp(ndisp, threads.z);

            switch (channels)
            {
            case 1: init_data_cost_reduce<T, winsz, 1><<<grid, threads, smem_size, stream>>>(cleft, cright, ctemp, cimg_step, level, rows, cols, h, ndisp, data_weight, max_data_term, min_disp, msg_step, disp_step); break;
            case 3: init_data_cost_reduce<T, winsz, 3><<<grid, threads, smem_size, stream>>>(cleft, cright, ctemp, cimg_step, level, rows, cols, h, ndisp, data_weight, max_data_term, min_disp, msg_step, disp_step); break;
            case 4: init_data_cost_reduce<T, winsz, 4><<<grid, threads, smem_size, stream>>>(cleft, cright, ctemp, cimg_step, level, rows, cols, h, ndisp, data_weight, max_data_term, min_disp, msg_step, disp_step); break;
            default: CV_Error(cv::Error::BadNumChannels, "Unsupported channels count");
            }
        }

        template<class T>
        void init_data_cost(const uchar *cleft, const uchar *cright, uchar *ctemp, size_t cimg_step, int rows, int cols, T* disp_selected_pyr, T* data_cost_selected, size_t msg_step,
                    int h, int w, int level, int nr_plane, int ndisp, int channels, float data_weight, float max_data_term, int min_disp, bool use_local_init_data_cost, cudaStream_t stream)
        {

            typedef void (*InitDataCostCaller)(const uchar *cleft, const uchar *cright, uchar *ctemp, size_t cimg_step, int cols, int rows, int w, int h, int level, int ndisp, int channels, float data_weight, float max_data_term, int min_disp, size_t msg_step, size_t disp_step, cudaStream_t stream);

            static const InitDataCostCaller init_data_cost_callers[] =
            {
                init_data_cost_caller_<T>, init_data_cost_caller_<T>, init_data_cost_reduce_caller_<T, 4>,
                init_data_cost_reduce_caller_<T, 8>, init_data_cost_reduce_caller_<T, 16>, init_data_cost_reduce_caller_<T, 32>,
                init_data_cost_reduce_caller_<T, 64>, init_data_cost_reduce_caller_<T, 128>, init_data_cost_reduce_caller_<T, 256>
            };

            size_t disp_step = msg_step * h;

            init_data_cost_callers[level](cleft, cright, ctemp, cimg_step, rows, cols, h, w, level, ndisp, channels, data_weight, max_data_term, min_disp, msg_step, disp_step, stream);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );

            dim3 threads(32, 8, 1);
            dim3 grid(1, 1, 1);

            grid.x = divUp(w, threads.x);
            grid.y = divUp(h, threads.y);

            if (use_local_init_data_cost == true)
                get_first_k_initial_local<<<grid, threads, 0, stream>>> (ctemp, data_cost_selected, disp_selected_pyr, h, w, nr_plane, ndisp, msg_step, disp_step);
            else
                get_first_k_initial_global<<<grid, threads, 0, stream>>>(ctemp, data_cost_selected, disp_selected_pyr, h, w, nr_plane, ndisp, msg_step, disp_step);

            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template void init_data_cost<short>(const uchar *cleft, const uchar *cright, uchar *ctemp, size_t cimg_step, int rows, int cols, short* disp_selected_pyr, short* data_cost_selected, size_t msg_step,
                    int h, int w, int level, int nr_plane, int ndisp, int channels, float data_weight, float max_data_term, int min_disp, bool use_local_init_data_cost, cudaStream_t stream);

        template void init_data_cost<float>(const uchar *cleft, const uchar *cright, uchar *ctemp, size_t cimg_step, int rows, int cols, float* disp_selected_pyr, float* data_cost_selected, size_t msg_step,
                    int h, int w, int level, int nr_plane, int ndisp, int channels, float data_weight, float max_data_term, int min_disp, bool use_local_init_data_cost, cudaStream_t stream);

        ///////////////////////////////////////////////////////////////
        ////////////////////// compute data cost //////////////////////
        ///////////////////////////////////////////////////////////////

        template <typename T, int channels>
        __global__ void compute_data_cost(const uchar *cleft, const uchar *cright, size_t cimg_step, const T* selected_disp_pyr, T* data_cost_, int h, int w, int level, int nr_plane, float data_weight, float max_data_term, int min_disp, size_t msg_step, size_t disp_step1, size_t disp_step2)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (y < h && x < w)
            {
                int y0 = y << level;
                int yt = (y + 1) << level;

                int x0 = x << level;
                int xt = (x + 1) << level;

                const T* selected_disparity = selected_disp_pyr + y/2 * msg_step + x/2;
                T* data_cost = data_cost_ + y * msg_step + x;

                for(int d = 0; d < nr_plane; d++)
                {
                    float val = 0.0f;
                    for(int yi = y0; yi < yt; yi++)
                    {
                        for(int xi = x0; xi < xt; xi++)
                        {
                            int sel_disp = selected_disparity[d * disp_step2];
                            int xr = xi - sel_disp;

                            if (xr < 0 || sel_disp < min_disp)
                                val += data_weight * max_data_term;
                            else
                            {
                                const uchar* left_x = cleft + yi * cimg_step + xi * channels;
                                const uchar* right_x = cright + yi * cimg_step + xr * channels;

                                val += data_weight * pixeldiff<channels>(left_x, right_x, max_data_term);
                            }
                        }
                    }
                    data_cost[disp_step1 * d] = saturate_cast<T>(val);
                }
            }
        }

        template <typename T, int winsz, int channels>
        __global__ void compute_data_cost_reduce(const uchar *cleft, const uchar *cright, size_t cimg_step, const T* selected_disp_pyr, T* data_cost_, int level, int rows, int cols, int h, int nr_plane, float data_weight, float max_data_term, int min_disp, size_t msg_step, size_t disp_step1, size_t disp_step2)
        {
            int x_out = blockIdx.x;
            int y_out = blockIdx.y % h;
            int d = (blockIdx.y / h) * blockDim.z + threadIdx.z;

            int tid = threadIdx.x;

            const T* selected_disparity = selected_disp_pyr + y_out/2 * msg_step + x_out/2;
            T* data_cost = data_cost_ + y_out * msg_step + x_out;

            if (d < nr_plane)
            {
                int sel_disp = selected_disparity[d * disp_step2];

                int x0 = x_out << level;
                int y0 = y_out << level;

                int len = ::min(y0 + winsz, rows) - y0;

                float val = 0.0f;
                if (x0 + tid < cols)
                {
                    if (x0 + tid - sel_disp < 0 || sel_disp < min_disp)
                        val = data_weight * max_data_term * len;
                    else
                    {
                        const uchar* lle =  cleft + y0 * cimg_step + channels * (x0 + tid    );
                        const uchar* lri = cright + y0 * cimg_step + channels * (x0 + tid - sel_disp);

                        for(int y = 0; y < len; ++y)
                        {
                            val += data_weight * pixeldiff<channels>(lle, lri, max_data_term);

                            lle += cimg_step;
                            lri += cimg_step;
                        }
                    }
                }

                extern __shared__ float smem[];

                reduce<winsz>(smem + winsz * threadIdx.z, val, tid, plus<float>());

                if (tid == 0)
                    data_cost[disp_step1 * d] = saturate_cast<T>(val);
            }
        }

        template <typename T>
        void compute_data_cost_caller_(const uchar *cleft, const uchar *cright, size_t cimg_step, const T* disp_selected_pyr, T* data_cost, int /*rows*/, int /*cols*/,
                                      int h, int w, int level, int nr_plane, int channels, float data_weight, float max_data_term, int min_disp, size_t msg_step, size_t disp_step1, size_t disp_step2, cudaStream_t stream)
        {
            dim3 threads(32, 8, 1);
            dim3 grid(1, 1, 1);

            grid.x = divUp(w, threads.x);
            grid.y = divUp(h, threads.y);

            switch(channels)
            {
            case 1: compute_data_cost<T, 1><<<grid, threads, 0, stream>>>(cleft, cright, cimg_step, disp_selected_pyr, data_cost, h, w, level, nr_plane, data_weight, max_data_term, min_disp, msg_step, disp_step1, disp_step2); break;
            case 3: compute_data_cost<T, 3><<<grid, threads, 0, stream>>>(cleft, cright, cimg_step, disp_selected_pyr, data_cost, h, w, level, nr_plane, data_weight, max_data_term, min_disp, msg_step, disp_step1, disp_step2); break;
            case 4: compute_data_cost<T, 4><<<grid, threads, 0, stream>>>(cleft, cright, cimg_step, disp_selected_pyr, data_cost, h, w, level, nr_plane, data_weight, max_data_term, min_disp, msg_step, disp_step1, disp_step2); break;
            default: CV_Error(cv::Error::BadNumChannels, "Unsupported channels count");
            }
        }

        template <typename T, int winsz>
        void compute_data_cost_reduce_caller_(const uchar *cleft, const uchar *cright, size_t cimg_step, const T* disp_selected_pyr, T* data_cost, int rows, int cols,
                                      int h, int w, int level, int nr_plane, int channels, float data_weight, float max_data_term, int min_disp, size_t msg_step, size_t disp_step1, size_t disp_step2, cudaStream_t stream)
        {
            const int threadsNum = 256;
            const size_t smem_size = threadsNum * sizeof(float);

            dim3 threads(winsz, 1, threadsNum / winsz);
            dim3 grid(w, h, 1);
            grid.y *= divUp(nr_plane, threads.z);

            switch (channels)
            {
            case 1: compute_data_cost_reduce<T, winsz, 1><<<grid, threads, smem_size, stream>>>(cleft, cright, cimg_step, disp_selected_pyr, data_cost, level, rows, cols, h, nr_plane, data_weight, max_data_term, min_disp, msg_step, disp_step1, disp_step2); break;
            case 3: compute_data_cost_reduce<T, winsz, 3><<<grid, threads, smem_size, stream>>>(cleft, cright, cimg_step, disp_selected_pyr, data_cost, level, rows, cols, h, nr_plane, data_weight, max_data_term, min_disp, msg_step, disp_step1, disp_step2); break;
            case 4: compute_data_cost_reduce<T, winsz, 4><<<grid, threads, smem_size, stream>>>(cleft, cright, cimg_step, disp_selected_pyr, data_cost, level, rows, cols, h, nr_plane, data_weight, max_data_term, min_disp, msg_step, disp_step1, disp_step2); break;
            default: CV_Error(cv::Error::BadNumChannels, "Unsupported channels count");
            }
        }

        template<class T>
        void compute_data_cost(const uchar *cleft, const uchar *cright, size_t cimg_step, const T* disp_selected_pyr, T* data_cost, size_t msg_step,
                               int rows, int cols, int h, int w, int h2, int level, int nr_plane, int channels, float data_weight, float max_data_term,
                               int min_disp, cudaStream_t stream)
        {
            typedef void (*ComputeDataCostCaller)(const uchar *cleft, const uchar *cright, size_t cimg_step, const T* disp_selected_pyr, T* data_cost, int rows, int cols,
                int h, int w, int level, int nr_plane, int channels, float data_weight, float max_data_term, int min_disp, size_t msg_step, size_t disp_step1, size_t disp_step2, cudaStream_t stream);

            static const ComputeDataCostCaller callers[] =
            {
                compute_data_cost_caller_<T>, compute_data_cost_caller_<T>, compute_data_cost_reduce_caller_<T, 4>,
                compute_data_cost_reduce_caller_<T, 8>, compute_data_cost_reduce_caller_<T, 16>, compute_data_cost_reduce_caller_<T, 32>,
                compute_data_cost_reduce_caller_<T, 64>, compute_data_cost_reduce_caller_<T, 128>, compute_data_cost_reduce_caller_<T, 256>
            };

            size_t disp_step1 = msg_step * h;
            size_t disp_step2 = msg_step * h2;

            callers[level](cleft, cright, cimg_step, disp_selected_pyr, data_cost, rows, cols, h, w, level, nr_plane, channels, data_weight, max_data_term, min_disp, msg_step, disp_step1, disp_step2, stream);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template void compute_data_cost(const uchar *cleft, const uchar *cright, size_t cimg_step, const short* disp_selected_pyr, short* data_cost, size_t msg_step,
                               int rows, int cols, int h, int w, int h2, int level, int nr_plane, int channels, float data_weight, float max_data_term, int min_disp, cudaStream_t stream);

        template void compute_data_cost(const uchar *cleft, const uchar *cright, size_t cimg_step, const float* disp_selected_pyr, float* data_cost, size_t msg_step,
                               int rows, int cols, int h, int w, int h2, int level, int nr_plane, int channels, float data_weight, float max_data_term, int min_disp, cudaStream_t stream);


        ///////////////////////////////////////////////////////////////
        //////////////////////// init message /////////////////////////
        ///////////////////////////////////////////////////////////////


         template <typename T>
        __device__ void get_first_k_element_increase(T* u_new, T* d_new, T* l_new, T* r_new,
                                                     const T* u_cur, const T* d_cur, const T* l_cur, const T* r_cur,
                                                     T* data_cost_selected, T* disparity_selected_new, T* data_cost_new,
                                                     const T* data_cost_cur, const T* disparity_selected_cur,
                                                     int nr_plane, int nr_plane2, size_t disp_step1, size_t disp_step2)
        {
            for(int i = 0; i < nr_plane; i++)
            {
                T minimum = numeric_limits<T>::max();
                int id = 0;
                for(int j = 0; j < nr_plane2; j++)
                {
                    T cur = data_cost_new[j * disp_step1];
                    if(cur < minimum)
                    {
                        minimum = cur;
                        id = j;
                    }
                }

                data_cost_selected[i * disp_step1] = data_cost_cur[id * disp_step1];
                disparity_selected_new[i * disp_step1] = disparity_selected_cur[id * disp_step2];

                u_new[i * disp_step1] = u_cur[id * disp_step2];
                d_new[i * disp_step1] = d_cur[id * disp_step2];
                l_new[i * disp_step1] = l_cur[id * disp_step2];
                r_new[i * disp_step1] = r_cur[id * disp_step2];

                data_cost_new[id * disp_step1] = numeric_limits<T>::max();
            }
        }

        template <typename T>
        __global__ void init_message(uchar *ctemp, T* u_new_, T* d_new_, T* l_new_, T* r_new_,
                                     const T* u_cur_, const T* d_cur_, const T* l_cur_, const T* r_cur_,
                                     T* selected_disp_pyr_new, const T* selected_disp_pyr_cur,
                                     T* data_cost_selected_, const T* data_cost_,
                                     int h, int w, int nr_plane, int h2, int w2, int nr_plane2,
                                     size_t msg_step, size_t disp_step1, size_t disp_step2)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (y < h && x < w)
            {
                const T* u_cur = u_cur_ + ::min(h2-1, y/2 + 1) * msg_step + x/2;
                const T* d_cur = d_cur_ + ::max(0, y/2 - 1)    * msg_step + x/2;
                const T* l_cur = l_cur_ + (y/2)                * msg_step + ::min(w2-1, x/2 + 1);
                const T* r_cur = r_cur_ + (y/2)                * msg_step + ::max(0, x/2 - 1);

                T* data_cost_new = (T*)ctemp + y * msg_step + x;

                const T* disparity_selected_cur = selected_disp_pyr_cur + y/2 * msg_step + x/2;
                const T* data_cost = data_cost_ + y * msg_step + x;

                for(int d = 0; d < nr_plane2; d++)
                {
                    int idx2 = d * disp_step2;

                    T val  = data_cost[d * disp_step1] + u_cur[idx2] + d_cur[idx2] + l_cur[idx2] + r_cur[idx2];
                    data_cost_new[d * disp_step1] = val;
                }

                T* data_cost_selected = data_cost_selected_ + y * msg_step + x;
                T* disparity_selected_new = selected_disp_pyr_new + y * msg_step + x;

                T* u_new = u_new_ + y * msg_step + x;
                T* d_new = d_new_ + y * msg_step + x;
                T* l_new = l_new_ + y * msg_step + x;
                T* r_new = r_new_ + y * msg_step + x;

                u_cur = u_cur_ + y/2 * msg_step + x/2;
                d_cur = d_cur_ + y/2 * msg_step + x/2;
                l_cur = l_cur_ + y/2 * msg_step + x/2;
                r_cur = r_cur_ + y/2 * msg_step + x/2;

                get_first_k_element_increase(u_new, d_new, l_new, r_new, u_cur, d_cur, l_cur, r_cur,
                                             data_cost_selected, disparity_selected_new, data_cost_new,
                                             data_cost, disparity_selected_cur, nr_plane, nr_plane2,
                                             disp_step1, disp_step2);
            }
        }


        template<class T>
        void init_message(uchar *ctemp, T* u_new, T* d_new, T* l_new, T* r_new,
                          const T* u_cur, const T* d_cur, const T* l_cur, const T* r_cur,
                          T* selected_disp_pyr_new, const T* selected_disp_pyr_cur,
                          T* data_cost_selected, const T* data_cost, size_t msg_step,
                          int h, int w, int nr_plane, int h2, int w2, int nr_plane2, cudaStream_t stream)
        {

            size_t disp_step1 = msg_step * h;
            size_t disp_step2 = msg_step * h2;

            dim3 threads(32, 8, 1);
            dim3 grid(1, 1, 1);

            grid.x = divUp(w, threads.x);
            grid.y = divUp(h, threads.y);

            init_message<<<grid, threads, 0, stream>>>(ctemp, u_new, d_new, l_new, r_new,
                                                       u_cur, d_cur, l_cur, r_cur,
                                                       selected_disp_pyr_new, selected_disp_pyr_cur,
                                                       data_cost_selected, data_cost,
                                                       h, w, nr_plane, h2, w2, nr_plane2,
                                                       msg_step, disp_step1, disp_step2);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }


        template void init_message(uchar *ctemp, short* u_new, short* d_new, short* l_new, short* r_new,
                          const short* u_cur, const short* d_cur, const short* l_cur, const short* r_cur,
                          short* selected_disp_pyr_new, const short* selected_disp_pyr_cur,
                          short* data_cost_selected, const short* data_cost, size_t msg_step,
                          int h, int w, int nr_plane, int h2, int w2, int nr_plane2, cudaStream_t stream);

        template void init_message(uchar *ctemp, float* u_new, float* d_new, float* l_new, float* r_new,
                          const float* u_cur, const float* d_cur, const float* l_cur, const float* r_cur,
                          float* selected_disp_pyr_new, const float* selected_disp_pyr_cur,
                          float* data_cost_selected, const float* data_cost, size_t msg_step,
                          int h, int w, int nr_plane, int h2, int w2, int nr_plane2, cudaStream_t stream);

        ///////////////////////////////////////////////////////////////
        ////////////////////  calc all iterations /////////////////////
        ///////////////////////////////////////////////////////////////

        template <typename T>
        __device__ void message_per_pixel(const T* data, T* msg_dst, const T* msg1, const T* msg2, const T* msg3,
                                          const T* dst_disp, const T* src_disp, int nr_plane, int max_disc_term, float disc_single_jump, volatile T* temp,
                                          size_t disp_step)
        {
            T minimum = numeric_limits<T>::max();

            for(int d = 0; d < nr_plane; d++)
            {
                int idx = d * disp_step;
                T val  = data[idx] + msg1[idx] + msg2[idx] + msg3[idx];

                if(val < minimum)
                    minimum = val;

                msg_dst[idx] = val;
            }

            float sum = 0;
            for(int d = 0; d < nr_plane; d++)
            {
                float cost_min = minimum + max_disc_term;
                T src_disp_reg = src_disp[d * disp_step];

                for(int d2 = 0; d2 < nr_plane; d2++)
                    cost_min = fmin(cost_min, msg_dst[d2 * disp_step] + disc_single_jump * ::abs(dst_disp[d2 * disp_step] - src_disp_reg));

                temp[d * disp_step] = saturate_cast<T>(cost_min);
                sum += cost_min;
            }
            sum /= nr_plane;

            for(int d = 0; d < nr_plane; d++)
                msg_dst[d * disp_step] = saturate_cast<T>(temp[d * disp_step] - sum);
        }

        template <typename T>
        __global__ void compute_message(uchar *ctemp, T* u_, T* d_, T* l_, T* r_, const T* data_cost_selected, const T* selected_disp_pyr_cur, int h, int w, int nr_plane, int i, int max_disc_term, float disc_single_jump, size_t msg_step, size_t disp_step)
        {
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            int x = ((blockIdx.x * blockDim.x + threadIdx.x) << 1) + ((y + i) & 1);

            if (y > 0 && y < h - 1 && x > 0 && x < w - 1)
            {
                const T* data = data_cost_selected + y * msg_step + x;

                T* u = u_ + y * msg_step + x;
                T* d = d_ + y * msg_step + x;
                T* l = l_ + y * msg_step + x;
                T* r = r_ + y * msg_step + x;

                const T* disp = selected_disp_pyr_cur + y * msg_step + x;

                T* temp = (T*)ctemp + y * msg_step + x;

                message_per_pixel(data, u, r - 1, u + msg_step, l + 1, disp, disp - msg_step, nr_plane, max_disc_term, disc_single_jump, temp, disp_step);
                message_per_pixel(data, d, d - msg_step, r - 1, l + 1, disp, disp + msg_step, nr_plane, max_disc_term, disc_single_jump, temp, disp_step);
                message_per_pixel(data, l, u + msg_step, d - msg_step, l + 1, disp, disp - 1, nr_plane, max_disc_term, disc_single_jump, temp, disp_step);
                message_per_pixel(data, r, u + msg_step, d - msg_step, r - 1, disp, disp + 1, nr_plane, max_disc_term, disc_single_jump, temp, disp_step);
            }
        }


        template<class T>
        void calc_all_iterations(uchar *ctemp, T* u, T* d, T* l, T* r, const T* data_cost_selected,
            const T* selected_disp_pyr_cur, size_t msg_step, int h, int w, int nr_plane, int iters, int max_disc_term, float disc_single_jump, cudaStream_t stream)
        {
            size_t disp_step = msg_step * h;

            dim3 threads(32, 8, 1);
            dim3 grid(1, 1, 1);

            grid.x = divUp(w, threads.x << 1);
            grid.y = divUp(h, threads.y);

            for(int t = 0; t < iters; ++t)
            {
                compute_message<<<grid, threads, 0, stream>>>(ctemp, u, d, l, r, data_cost_selected, selected_disp_pyr_cur, h, w, nr_plane, t & 1, max_disc_term, disc_single_jump, msg_step, disp_step);
                cudaSafeCall( cudaGetLastError() );
            }
            if (stream == 0)
                    cudaSafeCall( cudaDeviceSynchronize() );
        };

        template void calc_all_iterations(uchar *ctemp, short* u, short* d, short* l, short* r, const short* data_cost_selected, const short* selected_disp_pyr_cur, size_t msg_step,
            int h, int w, int nr_plane, int iters, int max_disc_term, float disc_single_jump, cudaStream_t stream);

        template void calc_all_iterations(uchar *ctemp, float* u, float* d, float* l, float* r, const float* data_cost_selected, const float* selected_disp_pyr_cur, size_t msg_step,
            int h, int w, int nr_plane, int iters, int max_disc_term, float disc_single_jump, cudaStream_t stream);


        ///////////////////////////////////////////////////////////////
        /////////////////////////// output ////////////////////////////
        ///////////////////////////////////////////////////////////////


        template <typename T>
        __global__ void compute_disp(const T* u_, const T* d_, const T* l_, const T* r_,
                                     const T* data_cost_selected, const T* disp_selected_pyr,
                                     PtrStepSz<short> disp, int nr_plane, size_t msg_step, size_t disp_step)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (y > 0 && y < disp.rows - 1 && x > 0 && x < disp.cols - 1)
            {
                const T* data = data_cost_selected + y * msg_step + x;
                const T* disp_selected = disp_selected_pyr + y * msg_step + x;

                const T* u = u_ + (y+1) * msg_step + (x+0);
                const T* d = d_ + (y-1) * msg_step + (x+0);
                const T* l = l_ + (y+0) * msg_step + (x+1);
                const T* r = r_ + (y+0) * msg_step + (x-1);

                int best = 0;
                T best_val = numeric_limits<T>::max();
                for (int i = 0; i < nr_plane; ++i)
                {
                    int idx = i * disp_step;
                    T val = data[idx]+ u[idx] + d[idx] + l[idx] + r[idx];

                    if (val < best_val)
                    {
                        best_val = val;
                        best = saturate_cast<short>(disp_selected[idx]);
                    }
                }
                disp(y, x) = best;
            }
        }

        template<class T>
        void compute_disp(const T* u, const T* d, const T* l, const T* r, const T* data_cost_selected, const T* disp_selected, size_t msg_step,
            const PtrStepSz<short>& disp, int nr_plane, cudaStream_t stream)
        {
            size_t disp_step = disp.rows * msg_step;

            dim3 threads(32, 8, 1);
            dim3 grid(1, 1, 1);

            grid.x = divUp(disp.cols, threads.x);
            grid.y = divUp(disp.rows, threads.y);

            compute_disp<<<grid, threads, 0, stream>>>(u, d, l, r, data_cost_selected, disp_selected, disp, nr_plane, msg_step, disp_step);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template void compute_disp(const short* u, const short* d, const short* l, const short* r, const short* data_cost_selected, const short* disp_selected, size_t msg_step,
            const PtrStepSz<short>& disp, int nr_plane, cudaStream_t stream);

        template void compute_disp(const float* u, const float* d, const float* l, const float* r, const float* data_cost_selected, const float* disp_selected, size_t msg_step,
            const PtrStepSz<short>& disp, int nr_plane, cudaStream_t stream);
    } // namespace stereocsbp
}}} // namespace cv { namespace cuda { namespace cudev {

#endif /* CUDA_DISABLER */
