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

#include "opencv2/gpu/devmem2d.hpp"
#include "opencv2/gpu/device/saturate_cast.hpp"
#include "opencv2/gpu/device/limits_gpu.hpp"
#include "safe_call.hpp"

using namespace cv::gpu;
using namespace cv::gpu::device;


namespace cv { namespace gpu { namespace csbp
{  

///////////////////////////////////////////////////////////////
/////////////////////// load constants ////////////////////////
///////////////////////////////////////////////////////////////

    __constant__ int cndisp;

    __constant__ float cmax_data_term;
    __constant__ float cdata_weight;
    __constant__ float cmax_disc_term;
    __constant__ float cdisc_single_jump;

    __constant__ int cth;

    __constant__ size_t cimg_step;
    __constant__ size_t cmsg_step1;
    __constant__ size_t cmsg_step2;
    __constant__ size_t cdisp_step1;
    __constant__ size_t cdisp_step2;

    __constant__ uchar* cleft;
    __constant__ uchar* cright;
    __constant__ uchar* ctemp;


    void load_constants(int ndisp, float max_data_term, float data_weight, float max_disc_term, float disc_single_jump, int min_disp_th,
                        const DevMem2D& left, const DevMem2D& right, const DevMem2D& temp)
    {
        cudaSafeCall( cudaMemcpyToSymbol(cndisp, &ndisp, sizeof(int)) );

        cudaSafeCall( cudaMemcpyToSymbol(cmax_data_term,    &max_data_term,    sizeof(float)) );
        cudaSafeCall( cudaMemcpyToSymbol(cdata_weight,      &data_weight,      sizeof(float)) );
        cudaSafeCall( cudaMemcpyToSymbol(cmax_disc_term,    &max_disc_term,    sizeof(float)) );
        cudaSafeCall( cudaMemcpyToSymbol(cdisc_single_jump, &disc_single_jump, sizeof(float)) );

        cudaSafeCall( cudaMemcpyToSymbol(cth, &min_disp_th, sizeof(int)) );

        cudaSafeCall( cudaMemcpyToSymbol(cimg_step, &left.step, sizeof(size_t)) );

        cudaSafeCall( cudaMemcpyToSymbol(cleft,  &left.data,  sizeof(left.data)) );
        cudaSafeCall( cudaMemcpyToSymbol(cright, &right.data, sizeof(right.data)) );
        cudaSafeCall( cudaMemcpyToSymbol(ctemp, &temp.data, sizeof(temp.data)) );
    }

///////////////////////////////////////////////////////////////
/////////////////////// init data cost ////////////////////////
///////////////////////////////////////////////////////////////

    template <int channels> struct DataCostPerPixel;
    template <> struct DataCostPerPixel<1>
    {
        static __device__ float compute(const uchar* left, const uchar* right)
        {
            return fmin(cdata_weight * abs((int)*left - *right), cdata_weight * cmax_data_term);
        }
    };
    template <> struct DataCostPerPixel<3>
    {
        static __device__ float compute(const uchar* left, const uchar* right)
        {
            float tb = 0.114f * abs((int)left[0] - right[0]);
            float tg = 0.587f * abs((int)left[1] - right[1]);
            float tr = 0.299f * abs((int)left[2] - right[2]);

            return fmin(cdata_weight * (tr + tg + tb), cdata_weight * cmax_data_term);
        }
    };
    template <> struct DataCostPerPixel<4>
    {
        static __device__ float compute(const uchar* left, const uchar* right)
        {
            uchar4 l = *((const uchar4*)left);
            uchar4 r = *((const uchar4*)right);

            float tb = 0.114f * abs((int)l.x - r.x);
            float tg = 0.587f * abs((int)l.y - r.y);
            float tr = 0.299f * abs((int)l.z - r.z);

            return fmin(cdata_weight * (tr + tg + tb), cdata_weight * cmax_data_term);
        }
    };

    template <typename T>
    __global__ void get_first_k_initial_global(T* data_cost_selected_, T *selected_disp_pyr, int h, int w, int nr_plane)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (y < h && x < w)
        {
            T* selected_disparity = selected_disp_pyr + y * cmsg_step1 + x;
            T* data_cost_selected = data_cost_selected_ + y * cmsg_step1 + x;
            T* data_cost = (T*)ctemp + y * cmsg_step1 + x;

            for(int i = 0; i < nr_plane; i++)
            {
                T minimum = numeric_limits_gpu<T>::max();
                int id = 0;
                for(int d = 0; d < cndisp; d++)
                {
                    T cur = data_cost[d * cdisp_step1];
                    if(cur < minimum)
                    {
                        minimum = cur;
                        id = d;
                    }
                }

                data_cost_selected[i  * cdisp_step1] = minimum;
                selected_disparity[i  * cdisp_step1] = id;
                data_cost         [id * cdisp_step1] = numeric_limits_gpu<T>::max();
            }
        }
    }


    template <typename T>
    __global__ void get_first_k_initial_local(T* data_cost_selected_, T* selected_disp_pyr, int h, int w, int nr_plane)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (y < h && x < w)
        {
            T* selected_disparity = selected_disp_pyr + y * cmsg_step1 + x;
            T* data_cost_selected = data_cost_selected_ + y * cmsg_step1 + x;
            T* data_cost = (T*)ctemp + y * cmsg_step1 + x;

            int nr_local_minimum = 0;

            T prev = data_cost[0 * cdisp_step1];
            T cur  = data_cost[1 * cdisp_step1];
            T next = data_cost[2 * cdisp_step1];

            for (int d = 1; d < cndisp - 1 && nr_local_minimum < nr_plane; d++)
            {
                if (cur < prev && cur < next)
                {
                    data_cost_selected[nr_local_minimum * cdisp_step1] = cur;
                    selected_disparity[nr_local_minimum * cdisp_step1] = d;

                    data_cost[d * cdisp_step1] = numeric_limits_gpu<T>::max();

                    nr_local_minimum++;
                }
                prev = cur;
                cur = next;
                next = data_cost[(d + 1) * cdisp_step1];
            }

            for (int i = nr_local_minimum; i < nr_plane; i++)
            {
                T minimum = numeric_limits_gpu<T>::max();
                int id = 0;

                for (int d = 0; d < cndisp; d++)
                {
                    cur = data_cost[d * cdisp_step1];
                    if (cur < minimum)
                    {
                        minimum = cur;
                        id = d;
                    }
                }
                data_cost_selected[i * cdisp_step1] = minimum;
                selected_disparity[i * cdisp_step1] = id;

                data_cost[id * cdisp_step1] = numeric_limits_gpu<T>::max();
            }
        }
    }

    template <typename T, int channels>
    __global__ void init_data_cost(int h, int w, int level)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (y < h && x < w)
        {
            int y0 = y << level;
            int yt = (y + 1) << level;

            int x0 = x << level;
            int xt = (x + 1) << level;

            T* data_cost = (T*)ctemp + y * cmsg_step1 + x;

            for(int d = 0; d < cndisp; ++d)
            {
                float val = 0.0f;
                for(int yi = y0; yi < yt; yi++)
                {
                    for(int xi = x0; xi < xt; xi++)
                    {
                        int xr = xi - d;
                        if(d < cth || xr < 0)
                            val += cdata_weight * cmax_data_term;
                        else
                        {
                            const uchar* lle = cleft + yi * cimg_step + xi * channels;
                            const uchar* lri = cright + yi * cimg_step + xr * channels;

                            val += DataCostPerPixel<channels>::compute(lle, lri);
                        }
                    }
                }
                data_cost[cdisp_step1 * d] = saturate_cast<T>(val);
            }
        }
    }

    template <typename T, int winsz, int channels>
    __global__ void init_data_cost_reduce(int level, int rows, int cols, int h)
    {
        int x_out = blockIdx.x;
        int y_out = blockIdx.y % h;
        int d = (blockIdx.y / h) * blockDim.z + threadIdx.z;

        int tid = threadIdx.x;

        if (d < cndisp)
        {
            int x0 = x_out << level;
            int y0 = y_out << level;

            int len = min(y0 + winsz, rows) - y0;

            float val = 0.0f;
            if (x0 + tid < cols)
            {
                if (x0 + tid - d < 0 || d < cth)
                    val = cdata_weight * cmax_data_term * len;
                else
                {
                    const uchar* lle =  cleft + y0 * cimg_step + channels * (x0 + tid    );
                    const uchar* lri = cright + y0 * cimg_step + channels * (x0 + tid - d);

                    for(int y = 0; y < len; ++y)
                    {
                        val += DataCostPerPixel<channels>::compute(lle, lri);

                        lle += cimg_step;
                        lri += cimg_step;
                    }
                }
            }

            extern __shared__ float smem[];
            float* dline = smem + winsz * threadIdx.z;

            dline[tid] = val;

            __syncthreads();

            if (winsz >= 256) { if (tid < 128) { dline[tid] += dline[tid + 128]; } __syncthreads(); }
            if (winsz >= 128) { if (tid <  64) { dline[tid] += dline[tid + 64]; } __syncthreads(); }

			volatile float* vdline = smem + winsz * threadIdx.z;

            if (winsz >= 64) if (tid < 32) vdline[tid] += vdline[tid + 32];
            if (winsz >= 32) if (tid < 16) vdline[tid] += vdline[tid + 16];
            if (winsz >= 16) if (tid <  8) vdline[tid] += vdline[tid + 8];
            if (winsz >=  8) if (tid <  4) vdline[tid] += vdline[tid + 4];
            if (winsz >=  4) if (tid <  2) vdline[tid] += vdline[tid + 2];
            if (winsz >=  2) if (tid <  1) vdline[tid] += vdline[tid + 1];

            T* data_cost = (T*)ctemp + y_out * cmsg_step1 + x_out;

            if (tid == 0)
                data_cost[cdisp_step1 * d] = saturate_cast<T>(dline[0]);
        }
    }


    template <typename T>
    void init_data_cost_caller_(int /*rows*/, int /*cols*/, int h, int w, int level, int /*ndisp*/, int channels, cudaStream_t stream)
    {
        dim3 threads(32, 8, 1);
        dim3 grid(1, 1, 1);

        grid.x = divUp(w, threads.x);
        grid.y = divUp(h, threads.y);

        switch (channels)
        {
        case 1: init_data_cost<T, 1><<<grid, threads, 0, stream>>>(h, w, level); break;
        case 3: init_data_cost<T, 3><<<grid, threads, 0, stream>>>(h, w, level); break;
        case 4: init_data_cost<T, 4><<<grid, threads, 0, stream>>>(h, w, level); break;
        default: cv::gpu::error("Unsupported channels count", __FILE__, __LINE__);
        }
    }

    template <typename T, int winsz>
    void init_data_cost_reduce_caller_(int rows, int cols, int h, int w, int level, int ndisp, int channels, cudaStream_t stream)
    {
        const int threadsNum = 256;
        const size_t smem_size = threadsNum * sizeof(float);

        dim3 threads(winsz, 1, threadsNum / winsz);
        dim3 grid(w, h, 1);
        grid.y *= divUp(ndisp, threads.z);

        switch (channels)
        {
        case 1: init_data_cost_reduce<T, winsz, 1><<<grid, threads, smem_size, stream>>>(level, rows, cols, h); break;
        case 3: init_data_cost_reduce<T, winsz, 3><<<grid, threads, smem_size, stream>>>(level, rows, cols, h); break;
        case 4: init_data_cost_reduce<T, winsz, 4><<<grid, threads, smem_size, stream>>>(level, rows, cols, h); break;
        default: cv::gpu::error("Unsupported channels count", __FILE__, __LINE__);
        }
    }

    template<class T>
    void init_data_cost(int rows, int cols, T* disp_selected_pyr, T* data_cost_selected, size_t msg_step,
                int h, int w, int level, int nr_plane, int ndisp, int channels, bool use_local_init_data_cost, cudaStream_t stream)
    {

        typedef void (*InitDataCostCaller)(int cols, int rows, int w, int h, int level, int ndisp, int channels, cudaStream_t stream);

        static const InitDataCostCaller init_data_cost_callers[] =
        {
            init_data_cost_caller_<T>, init_data_cost_caller_<T>, init_data_cost_reduce_caller_<T, 4>,
            init_data_cost_reduce_caller_<T, 8>, init_data_cost_reduce_caller_<T, 16>, init_data_cost_reduce_caller_<T, 32>,
            init_data_cost_reduce_caller_<T, 64>, init_data_cost_reduce_caller_<T, 128>, init_data_cost_reduce_caller_<T, 256>
        };

        size_t disp_step = msg_step * h;
        cudaSafeCall( cudaMemcpyToSymbol(cdisp_step1, &disp_step, sizeof(size_t)) );
        cudaSafeCall( cudaMemcpyToSymbol(cmsg_step1,  &msg_step,  sizeof(size_t)) );

        init_data_cost_callers[level](rows, cols, h, w, level, ndisp, channels, stream);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaThreadSynchronize() );

        dim3 threads(32, 8, 1);
        dim3 grid(1, 1, 1);

        grid.x = divUp(w, threads.x);
        grid.y = divUp(h, threads.y);

        if (use_local_init_data_cost == true)
            get_first_k_initial_local<<<grid, threads, 0, stream>>> (data_cost_selected, disp_selected_pyr, h, w, nr_plane);
        else
            get_first_k_initial_global<<<grid, threads, 0, stream>>>(data_cost_selected, disp_selected_pyr, h, w, nr_plane);
        
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaThreadSynchronize() );
    }

    template void init_data_cost(int rows, int cols, short* disp_selected_pyr, short* data_cost_selected, size_t msg_step,
                int h, int w, int level, int nr_plane, int ndisp, int channels, bool use_local_init_data_cost, cudaStream_t stream);

    template void init_data_cost(int rows, int cols, float* disp_selected_pyr, float* data_cost_selected, size_t msg_step,
                int h, int w, int level, int nr_plane, int ndisp, int channels, bool use_local_init_data_cost, cudaStream_t stream);

///////////////////////////////////////////////////////////////
////////////////////// compute data cost //////////////////////
///////////////////////////////////////////////////////////////

    template <typename T, int channels>
    __global__ void compute_data_cost(const T* selected_disp_pyr, T* data_cost_, int h, int w, int level, int nr_plane)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (y < h && x < w)
        {
            int y0 = y << level;
            int yt = (y + 1) << level;

            int x0 = x << level;
            int xt = (x + 1) << level;

            const T* selected_disparity = selected_disp_pyr + y/2 * cmsg_step2 + x/2;
            T* data_cost = data_cost_ + y * cmsg_step1 + x;

            for(int d = 0; d < nr_plane; d++)
            {
                float val = 0.0f;
                for(int yi = y0; yi < yt; yi++)
                {
                    for(int xi = x0; xi < xt; xi++)
                    {
                        int sel_disp = selected_disparity[d * cdisp_step2];
                        int xr = xi - sel_disp;

                        if (xr < 0 || sel_disp < cth)
                            val += cdata_weight * cmax_data_term;
                        else
                        {
                            const uchar* left_x = cleft + yi * cimg_step + xi * channels;
                            const uchar* right_x = cright + yi * cimg_step + xr * channels;

                            val += DataCostPerPixel<channels>::compute(left_x, right_x);
                        }
                    }
                }
                data_cost[cdisp_step1 * d] = saturate_cast<T>(val);
            }
        }
    }

    template <typename T, int winsz, int channels>
    __global__ void compute_data_cost_reduce(const T* selected_disp_pyr, T* data_cost_, int level, int rows, int cols, int h, int nr_plane)
    {
        int x_out = blockIdx.x;
        int y_out = blockIdx.y % h;
        int d = (blockIdx.y / h) * blockDim.z + threadIdx.z;

        int tid = threadIdx.x;

        const T* selected_disparity = selected_disp_pyr + y_out/2 * cmsg_step2 + x_out/2;
        T* data_cost = data_cost_ + y_out * cmsg_step1 + x_out;

        if (d < nr_plane)
        {
            int sel_disp = selected_disparity[d * cdisp_step2];

            int x0 = x_out << level;
            int y0 = y_out << level;

            int len = min(y0 + winsz, rows) - y0;

            float val = 0.0f;
            if (x0 + tid < cols)
            {
                if (x0 + tid - sel_disp < 0 || sel_disp < cth)
                    val = cdata_weight * cmax_data_term * len;
                else
                {
                    const uchar* lle =  cleft + y0 * cimg_step + channels * (x0 + tid    );
                    const uchar* lri = cright + y0 * cimg_step + channels * (x0 + tid - sel_disp);

                    for(int y = 0; y < len; ++y)
                    {
                        val += DataCostPerPixel<channels>::compute(lle, lri);

                        lle += cimg_step;
                        lri += cimg_step;
                    }
                }
            }

            extern __shared__ float smem[];
            float* dline = smem + winsz * threadIdx.z;

            dline[tid] = val;

            __syncthreads();

            if (winsz >= 256) { if (tid < 128) { dline[tid] += dline[tid + 128]; } __syncthreads(); }
            if (winsz >= 128) { if (tid <  64) { dline[tid] += dline[tid +  64]; } __syncthreads(); }

			volatile float* vdline = smem + winsz * threadIdx.z;

            if (winsz >= 64) if (tid < 32) vdline[tid] += vdline[tid + 32];
            if (winsz >= 32) if (tid < 16) vdline[tid] += vdline[tid + 16];
            if (winsz >= 16) if (tid <  8) vdline[tid] += vdline[tid + 8];
            if (winsz >=  8) if (tid <  4) vdline[tid] += vdline[tid + 4];
            if (winsz >=  4) if (tid <  2) vdline[tid] += vdline[tid + 2];
            if (winsz >=  2) if (tid <  1) vdline[tid] += vdline[tid + 1];

            if (tid == 0)
                data_cost[cdisp_step1 * d] = saturate_cast<T>(dline[0]);
        }
    }

    template <typename T>
    void compute_data_cost_caller_(const T* disp_selected_pyr, T* data_cost, int /*rows*/, int /*cols*/,
                                  int h, int w, int level, int nr_plane, int channels, cudaStream_t stream)
    {
        dim3 threads(32, 8, 1);
        dim3 grid(1, 1, 1);

        grid.x = divUp(w, threads.x);
        grid.y = divUp(h, threads.y);

        switch(channels)
        {
        case 1: compute_data_cost<T, 1><<<grid, threads, 0, stream>>>(disp_selected_pyr, data_cost, h, w, level, nr_plane); break;
        case 3: compute_data_cost<T, 3><<<grid, threads, 0, stream>>>(disp_selected_pyr, data_cost, h, w, level, nr_plane); break;
        case 4: compute_data_cost<T, 4><<<grid, threads, 0, stream>>>(disp_selected_pyr, data_cost, h, w, level, nr_plane); break;
        default: cv::gpu::error("Unsupported channels count", __FILE__, __LINE__);
        }
    }

    template <typename T, int winsz>
    void compute_data_cost_reduce_caller_(const T* disp_selected_pyr, T* data_cost, int rows, int cols,
                                  int h, int w, int level, int nr_plane, int channels, cudaStream_t stream)
    {
        const int threadsNum = 256;
        const size_t smem_size = threadsNum * sizeof(float);

        dim3 threads(winsz, 1, threadsNum / winsz);
        dim3 grid(w, h, 1);
        grid.y *= divUp(nr_plane, threads.z);

        switch (channels)
        {
        case 1: compute_data_cost_reduce<T, winsz, 1><<<grid, threads, smem_size, stream>>>(disp_selected_pyr, data_cost, level, rows, cols, h, nr_plane); break;
        case 3: compute_data_cost_reduce<T, winsz, 3><<<grid, threads, smem_size, stream>>>(disp_selected_pyr, data_cost, level, rows, cols, h, nr_plane); break;
        case 4: compute_data_cost_reduce<T, winsz, 4><<<grid, threads, smem_size, stream>>>(disp_selected_pyr, data_cost, level, rows, cols, h, nr_plane); break;
        default: cv::gpu::error("Unsupported channels count", __FILE__, __LINE__);
        }
    }

    template<class T>
    void compute_data_cost(const T* disp_selected_pyr, T* data_cost, size_t msg_step1, size_t msg_step2,
                           int rows, int cols, int h, int w, int h2, int level, int nr_plane, int channels, cudaStream_t stream)
    {
        typedef void (*ComputeDataCostCaller)(const T* disp_selected_pyr, T* data_cost, int rows, int cols,
            int h, int w, int level, int nr_plane, int channels, cudaStream_t stream);

        static const ComputeDataCostCaller callers[] =
        {
            compute_data_cost_caller_<T>, compute_data_cost_caller_<T>, compute_data_cost_reduce_caller_<T, 4>,
            compute_data_cost_reduce_caller_<T, 8>, compute_data_cost_reduce_caller_<T, 16>, compute_data_cost_reduce_caller_<T, 32>,
            compute_data_cost_reduce_caller_<T, 64>, compute_data_cost_reduce_caller_<T, 128>, compute_data_cost_reduce_caller_<T, 256>
        };

        size_t disp_step1 = msg_step1 * h;
        size_t disp_step2 = msg_step2 * h2;
        cudaSafeCall( cudaMemcpyToSymbol(cdisp_step1, &disp_step1, sizeof(size_t)) );
        cudaSafeCall( cudaMemcpyToSymbol(cdisp_step2, &disp_step2, sizeof(size_t)) );
        cudaSafeCall( cudaMemcpyToSymbol(cmsg_step1,  &msg_step1,  sizeof(size_t)) );
        cudaSafeCall( cudaMemcpyToSymbol(cmsg_step2,  &msg_step2,  sizeof(size_t)) );

        callers[level](disp_selected_pyr, data_cost, rows, cols, h, w, level, nr_plane, channels, stream);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaThreadSynchronize() );
    }

    template void compute_data_cost(const short* disp_selected_pyr, short* data_cost, size_t msg_step1, size_t msg_step2,
                           int rows, int cols, int h, int w, int h2, int level, int nr_plane, int channels, cudaStream_t stream);

    template void compute_data_cost(const float* disp_selected_pyr, float* data_cost, size_t msg_step1, size_t msg_step2,
                           int rows, int cols, int h, int w, int h2, int level, int nr_plane, int channels, cudaStream_t stream);
     

///////////////////////////////////////////////////////////////
//////////////////////// init message /////////////////////////
///////////////////////////////////////////////////////////////

 
     template <typename T>
    __device__ void get_first_k_element_increase(T* u_new, T* d_new, T* l_new, T* r_new,
                                                 const T* u_cur, const T* d_cur, const T* l_cur, const T* r_cur,
                                                 T* data_cost_selected, T* disparity_selected_new, T* data_cost_new,
                                                 const T* data_cost_cur, const T* disparity_selected_cur,
                                                 int nr_plane, int nr_plane2)
    {
        for(int i = 0; i < nr_plane; i++)
        {
            T minimum = numeric_limits_gpu<T>::max();
            int id = 0;
            for(int j = 0; j < nr_plane2; j++)
            {
                T cur = data_cost_new[j * cdisp_step1];
                if(cur < minimum)
                {
                    minimum = cur;
                    id = j;
                }
            }

            data_cost_selected[i * cdisp_step1] = data_cost_cur[id * cdisp_step1];
            disparity_selected_new[i * cdisp_step1] = disparity_selected_cur[id * cdisp_step2];

            u_new[i * cdisp_step1] = u_cur[id * cdisp_step2];
            d_new[i * cdisp_step1] = d_cur[id * cdisp_step2];
            l_new[i * cdisp_step1] = l_cur[id * cdisp_step2];
            r_new[i * cdisp_step1] = r_cur[id * cdisp_step2];

            data_cost_new[id * cdisp_step1] = numeric_limits_gpu<T>::max();
        }
    }

    template <typename T>
    __global__ void init_message(T* u_new_, T* d_new_, T* l_new_, T* r_new_,
                                 const T* u_cur_, const T* d_cur_, const T* l_cur_, const T* r_cur_,
                                 T* selected_disp_pyr_new, const T* selected_disp_pyr_cur,
                                 T* data_cost_selected_, const T* data_cost_,
                                 int h, int w, int nr_plane, int h2, int w2, int nr_plane2)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (y < h && x < w)
        {
            const T* u_cur = u_cur_ + min(h2-1, y/2 + 1) * cmsg_step2 + x/2;
            const T* d_cur = d_cur_ + max(0, y/2 - 1)    * cmsg_step2 + x/2;
            const T* l_cur = l_cur_ + y/2                * cmsg_step2 + min(w2-1, x/2 + 1);
            const T* r_cur = r_cur_ + y/2                * cmsg_step2 + max(0, x/2 - 1);

            T* data_cost_new = (T*)ctemp + y * cmsg_step1 + x;

            const T* disparity_selected_cur = selected_disp_pyr_cur + y/2 * cmsg_step2 + x/2;
            const T* data_cost = data_cost_ + y * cmsg_step1 + x;

            for(int d = 0; d < nr_plane2; d++)
            {
                int idx2 = d * cdisp_step2;

                T val  = data_cost[d * cdisp_step1] + u_cur[idx2] + d_cur[idx2] + l_cur[idx2] + r_cur[idx2];
                data_cost_new[d * cdisp_step1] = val;
            }

            T* data_cost_selected = data_cost_selected_ + y * cmsg_step1 + x;
            T* disparity_selected_new = selected_disp_pyr_new + y * cmsg_step1 + x;

            T* u_new = u_new_ + y * cmsg_step1 + x;
            T* d_new = d_new_ + y * cmsg_step1 + x;
            T* l_new = l_new_ + y * cmsg_step1 + x;
            T* r_new = r_new_ + y * cmsg_step1 + x;

            u_cur = u_cur_ + y/2 * cmsg_step2 + x/2;
            d_cur = d_cur_ + y/2 * cmsg_step2 + x/2;
            l_cur = l_cur_ + y/2 * cmsg_step2 + x/2;
            r_cur = r_cur_ + y/2 * cmsg_step2 + x/2;

            get_first_k_element_increase(u_new, d_new, l_new, r_new, u_cur, d_cur, l_cur, r_cur,
                                         data_cost_selected, disparity_selected_new, data_cost_new,
                                         data_cost, disparity_selected_cur, nr_plane, nr_plane2);
        }
    }


    template<class T>
    void init_message(T* u_new, T* d_new, T* l_new, T* r_new,
                      const T* u_cur, const T* d_cur, const T* l_cur, const T* r_cur,
                      T* selected_disp_pyr_new, const T* selected_disp_pyr_cur,
                      T* data_cost_selected, const T* data_cost, size_t msg_step1, size_t msg_step2,
                      int h, int w, int nr_plane, int h2, int w2, int nr_plane2, cudaStream_t stream)
    {

        size_t disp_step1 = msg_step1 * h;
        size_t disp_step2 = msg_step2 * h2;
        cudaSafeCall( cudaMemcpyToSymbol(cdisp_step1, &disp_step1, sizeof(size_t)) );
        cudaSafeCall( cudaMemcpyToSymbol(cdisp_step2, &disp_step2, sizeof(size_t)) );
        cudaSafeCall( cudaMemcpyToSymbol(cmsg_step1,   &msg_step1, sizeof(size_t)) );
        cudaSafeCall( cudaMemcpyToSymbol(cmsg_step2,   &msg_step2, sizeof(size_t)) );

        dim3 threads(32, 8, 1);
        dim3 grid(1, 1, 1);

        grid.x = divUp(w, threads.x);
        grid.y = divUp(h, threads.y);

        init_message<<<grid, threads, 0, stream>>>(u_new, d_new, l_new, r_new,
                                                   u_cur, d_cur, l_cur, r_cur,
                                                   selected_disp_pyr_new, selected_disp_pyr_cur,
                                                   data_cost_selected, data_cost,
                                                   h, w, nr_plane, h2, w2, nr_plane2);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaThreadSynchronize() );
    }


    template void init_message(short* u_new, short* d_new, short* l_new, short* r_new,
                      const short* u_cur, const short* d_cur, const short* l_cur, const short* r_cur,
                      short* selected_disp_pyr_new, const short* selected_disp_pyr_cur,
                      short* data_cost_selected, const short* data_cost, size_t msg_step1, size_t msg_step2,
                      int h, int w, int nr_plane, int h2, int w2, int nr_plane2, cudaStream_t stream);

    template void init_message(float* u_new, float* d_new, float* l_new, float* r_new,
                      const float* u_cur, const float* d_cur, const float* l_cur, const float* r_cur,
                      float* selected_disp_pyr_new, const float* selected_disp_pyr_cur,
                      float* data_cost_selected, const float* data_cost, size_t msg_step1, size_t msg_step2,
                      int h, int w, int nr_plane, int h2, int w2, int nr_plane2, cudaStream_t stream);        

///////////////////////////////////////////////////////////////
////////////////////  calc all iterations /////////////////////
///////////////////////////////////////////////////////////////

    template <typename T>
    __device__ void message_per_pixel(const T* data, T* msg_dst, const T* msg1, const T* msg2, const T* msg3,
                                      const T* dst_disp, const T* src_disp, int nr_plane, T* temp)
    {
        T minimum = numeric_limits_gpu<T>::max();

        for(int d = 0; d < nr_plane; d++)
        {
            int idx = d * cdisp_step1;
            T val  = data[idx] + msg1[idx] + msg2[idx] + msg3[idx];

            if(val < minimum)
                minimum = val;

            msg_dst[idx] = val;
        }

        float sum = 0;
        for(int d = 0; d < nr_plane; d++)
        {
            float cost_min = minimum + cmax_disc_term;
            T src_disp_reg = src_disp[d * cdisp_step1];

            for(int d2 = 0; d2 < nr_plane; d2++)
                cost_min = fmin(cost_min, msg_dst[d2 * cdisp_step1] + cdisc_single_jump * abs(dst_disp[d2 * cdisp_step1] - src_disp_reg));

            temp[d * cdisp_step1] = saturate_cast<T>(cost_min);
            sum += cost_min;
        }
        sum /= nr_plane;

        for(int d = 0; d < nr_plane; d++)
            msg_dst[d * cdisp_step1] = saturate_cast<T>(temp[d * cdisp_step1] - sum);
    }

    template <typename T>
    __global__ void compute_message(T* u_, T* d_, T* l_, T* r_, const T* data_cost_selected, const T* selected_disp_pyr_cur, int h, int w, int nr_plane, int i)
    {
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int x = ((blockIdx.x * blockDim.x + threadIdx.x) << 1) + ((y + i) & 1);

        if (y > 0 && y < h - 1 && x > 0 && x < w - 1)
        {
            const T* data = data_cost_selected + y * cmsg_step1 + x;

            T* u = u_ + y * cmsg_step1 + x;
            T* d = d_ + y * cmsg_step1 + x;
            T* l = l_ + y * cmsg_step1 + x;
            T* r = r_ + y * cmsg_step1 + x;

            const T* disp = selected_disp_pyr_cur + y * cmsg_step1 + x;

            T* temp = (T*)ctemp + y * cmsg_step1 + x;

            message_per_pixel(data, u, r - 1, u + cmsg_step1, l + 1, disp, disp - cmsg_step1, nr_plane, temp);
            message_per_pixel(data, d, d - cmsg_step1, r - 1, l + 1, disp, disp + cmsg_step1, nr_plane, temp);
            message_per_pixel(data, l, u + cmsg_step1, d - cmsg_step1, l + 1, disp, disp - 1, nr_plane, temp);
            message_per_pixel(data, r, u + cmsg_step1, d - cmsg_step1, r - 1, disp, disp + 1, nr_plane, temp);
        }
    }


    template<class T>
    void calc_all_iterations(T* u, T* d, T* l, T* r, const T* data_cost_selected,
        const T* selected_disp_pyr_cur, size_t msg_step, int h, int w, int nr_plane, int iters, cudaStream_t stream)
    {
        size_t disp_step = msg_step * h;
        cudaSafeCall( cudaMemcpyToSymbol(cdisp_step1, &disp_step, sizeof(size_t)) );
        cudaSafeCall( cudaMemcpyToSymbol(cmsg_step1,  &msg_step,  sizeof(size_t)) );

        dim3 threads(32, 8, 1);
        dim3 grid(1, 1, 1);

        grid.x = divUp(w, threads.x << 1);
        grid.y = divUp(h, threads.y);

        for(int t = 0; t < iters; ++t)
        {
            compute_message<<<grid, threads, 0, stream>>>(u, d, l, r, data_cost_selected, selected_disp_pyr_cur, h, w, nr_plane, t & 1);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaThreadSynchronize() );
        }
    };
    
    template void calc_all_iterations(short* u, short* d, short* l, short* r, const short* data_cost_selected, const short* selected_disp_pyr_cur, size_t msg_step,
        int h, int w, int nr_plane, int iters, cudaStream_t stream);

    template void calc_all_iterations(float* u, float* d, float* l, float* r, const float* data_cost_selected, const float* selected_disp_pyr_cur, size_t msg_step, 
        int h, int w, int nr_plane, int iters, cudaStream_t stream);


///////////////////////////////////////////////////////////////
/////////////////////////// output ////////////////////////////
///////////////////////////////////////////////////////////////


    template <typename T>
    __global__ void compute_disp(const T* u_, const T* d_, const T* l_, const T* r_,
                                 const T* data_cost_selected, const T* disp_selected_pyr,
                                 short* disp, size_t res_step, int cols, int rows, int nr_plane)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (y > 0 && y < rows - 1 && x > 0 && x < cols - 1)
        {
            const T* data = data_cost_selected + y * cmsg_step1 + x;
            const T* disp_selected = disp_selected_pyr + y * cmsg_step1 + x;

            const T* u = u_ + (y+1) * cmsg_step1 + (x+0);
            const T* d = d_ + (y-1) * cmsg_step1 + (x+0);
            const T* l = l_ + (y+0) * cmsg_step1 + (x+1);
            const T* r = r_ + (y+0) * cmsg_step1 + (x-1);

            int best = 0;
            T best_val = numeric_limits_gpu<T>::max();
            for (int i = 0; i < nr_plane; ++i)
            {
                int idx = i * cdisp_step1;
                T val = data[idx]+ u[idx] + d[idx] + l[idx] + r[idx];

                if (val < best_val)
                {
                    best_val = val;
                    best = saturate_cast<short>(disp_selected[idx]);
                }
            }
            disp[res_step * y + x] = best;
        }
    }

    template<class T>
    void compute_disp(const T* u, const T* d, const T* l, const T* r, const T* data_cost_selected, const T* disp_selected, size_t msg_step,
        const DevMem2D_<short>& disp, int nr_plane, cudaStream_t stream)
    {
        size_t disp_step = disp.rows * msg_step;
        cudaSafeCall( cudaMemcpyToSymbol(cdisp_step1, &disp_step, sizeof(size_t)) );
        cudaSafeCall( cudaMemcpyToSymbol(cmsg_step1,  &msg_step,  sizeof(size_t)) );

        dim3 threads(32, 8, 1);
        dim3 grid(1, 1, 1);

        grid.x = divUp(disp.cols, threads.x);
        grid.y = divUp(disp.rows, threads.y);

        compute_disp<<<grid, threads, 0, stream>>>(u, d, l, r, data_cost_selected, disp_selected,
                                                   disp.data, disp.step / disp.elemSize(), disp.cols, disp.rows, nr_plane);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaThreadSynchronize() );
    }

    template void compute_disp(const short* u, const short* d, const short* l, const short* r, const short* data_cost_selected, const short* disp_selected, size_t msg_step, 
        const DevMem2D_<short>& disp, int nr_plane, cudaStream_t stream);

    template void compute_disp(const float* u, const float* d, const float* l, const float* r, const float* data_cost_selected, const float* disp_selected, size_t msg_step,
        const DevMem2D_<short>& disp, int nr_plane, cudaStream_t stream);
}}}
