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

#include "opencv2/gpu/device/limits_gpu.hpp"
#include "opencv2/gpu/device/saturate_cast.hpp"
#include "transform.hpp"
#include "internal_shared.hpp"

using namespace cv::gpu;
using namespace cv::gpu::device;

#ifndef CV_PI
#define CV_PI   3.1415926535897932384626433832795f
#endif

//////////////////////////////////////////////////////////////////////////////////////
// Cart <-> Polar

namespace cv { namespace gpu { namespace mathfunc
{
    template <int size, typename T>
    __device__ void sum_in_smem(volatile T* data, const unsigned int tid)
    {
        T sum = data[tid];

        if (size >= 512) { if (tid < 256) { data[tid] = sum = sum + data[tid + 256]; } __syncthreads(); }
        if (size >= 256) { if (tid < 128) { data[tid] = sum = sum + data[tid + 128]; } __syncthreads(); }
        if (size >= 128) { if (tid < 64) { data[tid] = sum = sum + data[tid + 64]; } __syncthreads(); }

        if (tid < 32)
        {
            if (size >= 64) data[tid] = sum = sum + data[tid + 32];
            if (size >= 32) data[tid] = sum = sum + data[tid + 16];
            if (size >= 16) data[tid] = sum = sum + data[tid + 8];
            if (size >= 8) data[tid] = sum = sum + data[tid + 4];
            if (size >= 4) data[tid] = sum = sum + data[tid + 2];
            if (size >= 2) data[tid] = sum = sum + data[tid + 1];
        }
    }

    struct Nothing
    {
        static __device__ void calc(int, int, float, float, float*, size_t, float)
        {
        }
    };
    struct Magnitude
    {
        static __device__ void calc(int x, int y, float x_data, float y_data, float* dst, size_t dst_step, float)
        {
            dst[y * dst_step + x] = sqrtf(x_data * x_data + y_data * y_data);
        }
    };
    struct MagnitudeSqr
    {
        static __device__ void calc(int x, int y, float x_data, float y_data, float* dst, size_t dst_step, float)
        {
            dst[y * dst_step + x] = x_data * x_data + y_data * y_data;
        }
    };
    struct Atan2
    {
        static __device__ void calc(int x, int y, float x_data, float y_data, float* dst, size_t dst_step, float scale)
        {
            dst[y * dst_step + x] = scale * atan2f(y_data, x_data);
        }
    };
    template <typename Mag, typename Angle>
    __global__ void cartToPolar(const float* xptr, size_t x_step, const float* yptr, size_t y_step, 
                                float* mag, size_t mag_step, float* angle, size_t angle_step, float scale, int width, int height)
    {
		const int x = blockDim.x * blockIdx.x + threadIdx.x;
		const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x < width && y < height)
        {
            float x_data = xptr[y * x_step + x];
            float y_data = yptr[y * y_step + x];

            Mag::calc(x, y, x_data, y_data, mag, mag_step, scale);
            Angle::calc(x, y, x_data, y_data, angle, angle_step, scale);
        }
    }

    struct NonEmptyMag
    {
        static __device__ float get(const float* mag, size_t mag_step, int x, int y)
        {
            return mag[y * mag_step + x];
        }
    };
    struct EmptyMag
    {
        static __device__ float get(const float*, size_t, int, int)
        {
            return 1.0f;
        }
    };
    template <typename Mag>
    __global__ void polarToCart(const float* mag, size_t mag_step, const float* angle, size_t angle_step, float scale,
        float* xptr, size_t x_step, float* yptr, size_t y_step, int width, int height)
    {
		const int x = blockDim.x * blockIdx.x + threadIdx.x;
		const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x < width && y < height)
        {
            float mag_data = Mag::get(mag, mag_step, x, y);
            float angle_data = angle[y * angle_step + x];
            float sin_a, cos_a;

            sincosf(scale * angle_data, &sin_a, &cos_a);

            xptr[y * x_step + x] = mag_data * cos_a;
            yptr[y * y_step + x] = mag_data * sin_a;
        }
    }

    template <typename Mag, typename Angle>
    void cartToPolar_caller(const DevMem2Df& x, const DevMem2Df& y, const DevMem2Df& mag, const DevMem2Df& angle, bool angleInDegrees, cudaStream_t stream)
    {
        dim3 threads(16, 16, 1);
        dim3 grid(1, 1, 1);

        grid.x = divUp(x.cols, threads.x);
        grid.y = divUp(x.rows, threads.y);
        
        const float scale = angleInDegrees ? (float)(180.0f / CV_PI) : 1.f;

        cartToPolar<Mag, Angle><<<grid, threads, 0, stream>>>(
            x.data, x.step/x.elemSize(), y.data, y.step/y.elemSize(), 
            mag.data, mag.step/mag.elemSize(), angle.data, angle.step/angle.elemSize(), scale, x.cols, x.rows);

        if (stream == 0)
            cudaSafeCall( cudaThreadSynchronize() );
    }

    void cartToPolar_gpu(const DevMem2Df& x, const DevMem2Df& y, const DevMem2Df& mag, bool magSqr, const DevMem2Df& angle, bool angleInDegrees, cudaStream_t stream)
    {
        typedef void (*caller_t)(const DevMem2Df& x, const DevMem2Df& y, const DevMem2Df& mag, const DevMem2Df& angle, bool angleInDegrees, cudaStream_t stream);
        static const caller_t callers[2][2][2] = 
        {
            {
                {
                    cartToPolar_caller<Magnitude, Atan2>,
                    cartToPolar_caller<Magnitude, Nothing>
                },
                {
                    cartToPolar_caller<MagnitudeSqr, Atan2>,
                    cartToPolar_caller<MagnitudeSqr, Nothing>,
                }
            },
            {
                {
                    cartToPolar_caller<Nothing, Atan2>,
                    cartToPolar_caller<Nothing, Nothing>
                },
                {
                    cartToPolar_caller<Nothing, Atan2>,
                    cartToPolar_caller<Nothing, Nothing>,
                }
            }
        };

        callers[mag.data == 0][magSqr][angle.data == 0](x, y, mag, angle, angleInDegrees, stream);
    }

    template <typename Mag>
    void polarToCart_caller(const DevMem2Df& mag, const DevMem2Df& angle, const DevMem2Df& x, const DevMem2Df& y, bool angleInDegrees, cudaStream_t stream)
    {
        dim3 threads(16, 16, 1);
        dim3 grid(1, 1, 1);

        grid.x = divUp(mag.cols, threads.x);
        grid.y = divUp(mag.rows, threads.y);
        
        const float scale = angleInDegrees ? (float)(CV_PI / 180.0f) : 1.0f;

        polarToCart<Mag><<<grid, threads, 0, stream>>>(mag.data, mag.step/mag.elemSize(), 
            angle.data, angle.step/angle.elemSize(), scale, x.data, x.step/x.elemSize(), y.data, y.step/y.elemSize(), mag.cols, mag.rows);

        if (stream == 0)
            cudaSafeCall( cudaThreadSynchronize() );
    }

    void polarToCart_gpu(const DevMem2Df& mag, const DevMem2Df& angle, const DevMem2Df& x, const DevMem2Df& y, bool angleInDegrees, cudaStream_t stream)
    {
        typedef void (*caller_t)(const DevMem2Df& mag, const DevMem2Df& angle, const DevMem2Df& x, const DevMem2Df& y, bool angleInDegrees, cudaStream_t stream);
        static const caller_t callers[2] = 
        {
            polarToCart_caller<NonEmptyMag>,
            polarToCart_caller<EmptyMag>
        };

        callers[mag.data == 0](mag, angle, x, y, angleInDegrees, stream);
    }

//////////////////////////////////////////////////////////////////////////////////////
// Compare

    template <typename T1, typename T2>
    struct NotEqual
    {
        __device__ uchar operator()(const T1& src1, const T2& src2)
        {
            return static_cast<uchar>(static_cast<int>(src1 != src2) * 255);
        }
    };

    template <typename T1, typename T2>
    inline void compare_ne(const DevMem2D& src1, const DevMem2D& src2, const DevMem2D& dst)
    {
        NotEqual<T1, T2> op;
        transform(static_cast< DevMem2D_<T1> >(src1), static_cast< DevMem2D_<T2> >(src2), dst, op, 0);
    }

    void compare_ne_8uc4(const DevMem2D& src1, const DevMem2D& src2, const DevMem2D& dst)
    {
        compare_ne<uint, uint>(src1, src2, dst);
    }
    void compare_ne_32f(const DevMem2D& src1, const DevMem2D& src2, const DevMem2D& dst)
    {
        compare_ne<float, float>(src1, src2, dst);
    }


//////////////////////////////////////////////////////////////////////////////
// Per-element bit-wise logical matrix operations

    struct Mask8U
    {
        explicit Mask8U(PtrStep mask): mask(mask) {}
        __device__ bool operator()(int y, int x) const { return mask.ptr(y)[x]; }
        PtrStep mask;
    };
    struct MaskTrue { __device__ bool operator()(int y, int x) const { return true; } };

    // Unary operations

    enum { UN_OP_NOT };

    template <typename T, int opid>
    struct UnOp { __device__ T operator()(T lhs, T rhs); };

    template <typename T>
    struct UnOp<T, UN_OP_NOT>{ __device__ T operator()(T x) { return ~x; } };

    template <typename T, int cn, typename UnOp, typename Mask>
    __global__ void bitwise_un_op(int rows, int cols, const PtrStep src, PtrStep dst, UnOp op, Mask mask)
    {
        const int x = blockDim.x * blockIdx.x + threadIdx.x;
        const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x < cols && y < rows && mask(y, x)) 
        {
            T* dsty = (T*)dst.ptr(y);
            const T* srcy = (const T*)src.ptr(y);

            #pragma unroll
            for (int i = 0; i < cn; ++i)
                dsty[cn * x + i] = op(srcy[cn * x + i]);
        }
    }

    template <int opid, typename Mask>
    void bitwise_un_op(int rows, int cols, const PtrStep src, PtrStep dst, int elem_size, Mask mask, cudaStream_t stream)
    {
        dim3 threads(16, 16);
        dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));
        switch (elem_size)
        {
        case 1: bitwise_un_op<unsigned char, 1><<<grid, threads>>>(rows, cols, src, dst, UnOp<unsigned char, opid>(), mask); break;
        case 2: bitwise_un_op<unsigned short, 1><<<grid, threads>>>(rows, cols, src, dst, UnOp<unsigned short, opid>(), mask); break;
        case 3: bitwise_un_op<unsigned char, 3><<<grid, threads>>>(rows, cols, src, dst, UnOp<unsigned char, opid>(), mask); break;
        case 4: bitwise_un_op<unsigned int, 1><<<grid, threads>>>(rows, cols, src, dst, UnOp<unsigned int, opid>(), mask); break;
        case 6: bitwise_un_op<unsigned short, 3><<<grid, threads>>>(rows, cols, src, dst, UnOp<unsigned short, opid>(), mask); break;
        case 8: bitwise_un_op<unsigned int, 2><<<grid, threads>>>(rows, cols, src, dst, UnOp<unsigned int, opid>(), mask); break;       
        case 12: bitwise_un_op<unsigned int, 3><<<grid, threads>>>(rows, cols, src, dst, UnOp<unsigned int, opid>(), mask); break;
        case 16: bitwise_un_op<unsigned int, 4><<<grid, threads>>>(rows, cols, src, dst, UnOp<unsigned int, opid>(), mask); break;
        case 24: bitwise_un_op<unsigned int, 6><<<grid, threads>>>(rows, cols, src, dst, UnOp<unsigned int, opid>(), mask); break;
        case 32: bitwise_un_op<unsigned int, 8><<<grid, threads>>>(rows, cols, src, dst, UnOp<unsigned int, opid>(), mask); break;
        }
        if (stream == 0) cudaSafeCall(cudaThreadSynchronize());        
    }

    void bitwise_not_caller(int rows, int cols,const PtrStep src, int elem_size, PtrStep dst, cudaStream_t stream)
    {
        bitwise_un_op<UN_OP_NOT>(rows, cols, src, dst, elem_size, MaskTrue(), stream);
    }

    void bitwise_not_caller(int rows, int cols,const PtrStep src, int elem_size, PtrStep dst, const PtrStep mask, cudaStream_t stream)
    {
        bitwise_un_op<UN_OP_NOT>(rows, cols, src, dst, elem_size, Mask8U(mask), stream);
    }

    // Binary operations

    enum { BIN_OP_OR, BIN_OP_AND, BIN_OP_XOR };

    template <typename T, int opid>
    struct BinOp { __device__ T operator()(T lhs, T rhs); };

    template <typename T>
    struct BinOp<T, BIN_OP_OR>{ __device__ T operator()(T lhs, T rhs) { return lhs | rhs; } };

    template <typename T>
    struct BinOp<T, BIN_OP_AND>{ __device__ T operator()(T lhs, T rhs) { return lhs & rhs; } };

    template <typename T>
    struct BinOp<T, BIN_OP_XOR>{ __device__ T operator()(T lhs, T rhs) { return lhs ^ rhs; } };

    template <typename T, int cn, typename BinOp, typename Mask>
    __global__ void bitwise_bin_op(int rows, int cols, const PtrStep src1, const PtrStep src2, PtrStep dst, BinOp op, Mask mask)
    {
        const int x = blockDim.x * blockIdx.x + threadIdx.x;
        const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x < cols && y < rows && mask(y, x)) 
        {
            T* dsty = (T*)dst.ptr(y);
            const T* src1y = (const T*)src1.ptr(y);
            const T* src2y = (const T*)src2.ptr(y);

            #pragma unroll
            for (int i = 0; i < cn; ++i)
                dsty[cn * x + i] = op(src1y[cn * x + i], src2y[cn * x + i]);
        }
    }

    template <int opid, typename Mask>
    void bitwise_bin_op(int rows, int cols, const PtrStep src1, const PtrStep src2, PtrStep dst, int elem_size, Mask mask, cudaStream_t stream)
    {
        dim3 threads(16, 16);
        dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));
        switch (elem_size)
        {
        case 1: bitwise_bin_op<unsigned char, 1><<<grid, threads>>>(rows, cols, src1, src2, dst, BinOp<unsigned char, opid>(), mask); break;
        case 2: bitwise_bin_op<unsigned short, 1><<<grid, threads>>>(rows, cols, src1, src2, dst, BinOp<unsigned short, opid>(), mask); break;
        case 3: bitwise_bin_op<unsigned char, 3><<<grid, threads>>>(rows, cols, src1, src2, dst, BinOp<unsigned char, opid>(), mask); break;
        case 4: bitwise_bin_op<unsigned int, 1><<<grid, threads>>>(rows, cols, src1, src2, dst, BinOp<unsigned int, opid>(), mask); break;
        case 6: bitwise_bin_op<unsigned short, 3><<<grid, threads>>>(rows, cols, src1, src2, dst, BinOp<unsigned short, opid>(), mask); break;
        case 8: bitwise_bin_op<unsigned int, 2><<<grid, threads>>>(rows, cols, src1, src2, dst, BinOp<unsigned int, opid>(), mask); break;       
        case 12: bitwise_bin_op<unsigned int, 3><<<grid, threads>>>(rows, cols, src1, src2, dst, BinOp<unsigned int, opid>(), mask); break;
        case 16: bitwise_bin_op<unsigned int, 4><<<grid, threads>>>(rows, cols, src1, src2, dst, BinOp<unsigned int, opid>(), mask); break;
        case 24: bitwise_bin_op<unsigned int, 6><<<grid, threads>>>(rows, cols, src1, src2, dst, BinOp<unsigned int, opid>(), mask); break;
        case 32: bitwise_bin_op<unsigned int, 8><<<grid, threads>>>(rows, cols, src1, src2, dst, BinOp<unsigned int, opid>(), mask); break;
        }
        if (stream == 0) cudaSafeCall(cudaThreadSynchronize());        
    }

    void bitwise_or_caller(int rows, int cols, const PtrStep src1, const PtrStep src2, int elem_size, PtrStep dst, cudaStream_t stream)
    {
        bitwise_bin_op<BIN_OP_OR>(rows, cols, src1, src2, dst, elem_size, MaskTrue(), stream);
    }

    void bitwise_or_caller(int rows, int cols, const PtrStep src1, const PtrStep src2, int elem_size, PtrStep dst, const PtrStep mask, cudaStream_t stream)
    {
        bitwise_bin_op<BIN_OP_OR>(rows, cols, src1, src2, dst, elem_size, Mask8U(mask), stream);
    }

    void bitwise_and_caller(int rows, int cols, const PtrStep src1, const PtrStep src2, int elem_size, PtrStep dst, cudaStream_t stream)
    {
        bitwise_bin_op<BIN_OP_AND>(rows, cols, src1, src2, dst, elem_size, MaskTrue(), stream);
    }

    void bitwise_and_caller(int rows, int cols, const PtrStep src1, const PtrStep src2, int elem_size, PtrStep dst, const PtrStep mask, cudaStream_t stream)
    {
        bitwise_bin_op<BIN_OP_AND>(rows, cols, src1, src2, dst, elem_size, Mask8U(mask), stream);
    }

    void bitwise_xor_caller(int rows, int cols, const PtrStep src1, const PtrStep src2, int elem_size, PtrStep dst, cudaStream_t stream)
    {
        bitwise_bin_op<BIN_OP_XOR>(rows, cols, src1, src2, dst, elem_size, MaskTrue(), stream);
    }

    void bitwise_xor_caller(int rows, int cols, const PtrStep src1, const PtrStep src2, int elem_size, PtrStep dst, const PtrStep mask, cudaStream_t stream)
    {
        bitwise_bin_op<BIN_OP_XOR>(rows, cols, src1, src2, dst, elem_size, Mask8U(mask), stream);
    }  



//////////////////////////////////////////////////////////////////////////////
// Min max

    // To avoid shared bank conflicts we convert each value into value of 
    // appropriate type (32 bits minimum)
    template <typename T> struct MinMaxTypeTraits {};
    template <> struct MinMaxTypeTraits<unsigned char> { typedef int best_type; };
    template <> struct MinMaxTypeTraits<char> { typedef int best_type; };
    template <> struct MinMaxTypeTraits<unsigned short> { typedef int best_type; };
    template <> struct MinMaxTypeTraits<short> { typedef int best_type; };
    template <> struct MinMaxTypeTraits<int> { typedef int best_type; };
    template <> struct MinMaxTypeTraits<float> { typedef float best_type; };
    template <> struct MinMaxTypeTraits<double> { typedef double best_type; };


    namespace minmax 
    {

    __constant__ int ctwidth;
    __constant__ int ctheight;

    // Global counter of blocks finished its work
    __device__ unsigned int blocks_finished = 0;


    // Estimates good thread configuration
    //  - threads variable satisfies to threads.x * threads.y == 256
    void estimate_thread_cfg(int cols, int rows, dim3& threads, dim3& grid)
    {
        threads = dim3(32, 8);
        grid = dim3(divUp(cols, threads.x * 8), divUp(rows, threads.y * 32));
        grid.x = min(grid.x, threads.x);
        grid.y = min(grid.y, threads.y);
    }


    // Returns required buffer sizes
    void get_buf_size_required(int cols, int rows, int elem_size, int& bufcols, int& bufrows)
    {
        dim3 threads, grid;
        estimate_thread_cfg(cols, rows, threads, grid);
        bufcols = grid.x * grid.y * elem_size; 
        bufrows = 2;
    }


    // Estimates device constants which are used in the kernels using specified thread configuration
    void set_kernel_consts(int cols, int rows, const dim3& threads, const dim3& grid)
    {        
        int twidth = divUp(divUp(cols, grid.x), threads.x);
        int theight = divUp(divUp(rows, grid.y), threads.y);
        cudaSafeCall(cudaMemcpyToSymbol(ctwidth, &twidth, sizeof(ctwidth))); 
        cudaSafeCall(cudaMemcpyToSymbol(ctheight, &theight, sizeof(ctheight))); 
    }  


    // Does min and max in shared memory
    template <typename T>
    __device__ void merge(unsigned int tid, unsigned int offset, volatile T* minval, volatile T* maxval)
    {
        minval[tid] = min(minval[tid], minval[tid + offset]);
        maxval[tid] = max(maxval[tid], maxval[tid + offset]);
    }


    template <int size, typename T>
    __device__ void find_min_max_in_smem(volatile T* minval, volatile T* maxval, const unsigned int tid)
    {
        if (size >= 512) { if (tid < 256) { merge(tid, 256, minval, maxval); } __syncthreads(); }
        if (size >= 256) { if (tid < 128) { merge(tid, 128, minval, maxval); }  __syncthreads(); }
        if (size >= 128) { if (tid < 64) { merge(tid, 64, minval, maxval); } __syncthreads(); }

        if (tid < 32)
        {
            if (size >= 64) merge(tid, 32, minval, maxval);
            if (size >= 32) merge(tid, 16, minval, maxval);
            if (size >= 16) merge(tid, 8, minval, maxval);
            if (size >= 8) merge(tid, 4, minval, maxval);
            if (size >= 4) merge(tid, 2, minval, maxval);
            if (size >= 2) merge(tid, 1, minval, maxval);
        }
    }


    template <int nthreads, typename T, typename Mask>
    __global__ void min_max_kernel(const DevMem2D src, Mask mask, T* minval, T* maxval)
    {
        typedef typename MinMaxTypeTraits<T>::best_type best_type;
        __shared__ best_type sminval[nthreads];
        __shared__ best_type smaxval[nthreads];

        unsigned int x0 = blockIdx.x * blockDim.x * ctwidth + threadIdx.x;
        unsigned int y0 = blockIdx.y * blockDim.y * ctheight + threadIdx.y;
        unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

        T mymin = numeric_limits_gpu<T>::max();
        T mymax = numeric_limits_gpu<T>::is_signed ? -numeric_limits_gpu<T>::max() : numeric_limits_gpu<T>::min();
        unsigned int y_end = min(y0 + (ctheight - 1) * blockDim.y + 1, src.rows);
        unsigned int x_end = min(x0 + (ctwidth - 1) * blockDim.x + 1, src.cols);
        for (unsigned int y = y0; y < y_end; y += blockDim.y)
        {
            const T* src_row = (const T*)src.ptr(y);
            for (unsigned int x = x0; x < x_end; x += blockDim.x)
            {
                T val = src_row[x];
                if (mask(y, x)) 
                { 
                    mymin = min(mymin, val); 
                    mymax = max(mymax, val); 
                }
            }
        }

        sminval[tid] = mymin;
        smaxval[tid] = mymax;
        __syncthreads();

        find_min_max_in_smem<nthreads, best_type>(sminval, smaxval, tid);

        if (tid == 0) 
        {
            minval[blockIdx.y * gridDim.x + blockIdx.x] = (T)sminval[0];
            maxval[blockIdx.y * gridDim.x + blockIdx.x] = (T)smaxval[0];
        }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 110
		__shared__ bool is_last;

		if (tid == 0)
		{
			minval[blockIdx.y * gridDim.x + blockIdx.x] = (T)sminval[0];
            maxval[blockIdx.y * gridDim.x + blockIdx.x] = (T)smaxval[0];
			__threadfence();

			unsigned int ticket = atomicInc(&blocks_finished, gridDim.x * gridDim.y);
			is_last = ticket == gridDim.x * gridDim.y - 1;
		}

		__syncthreads();

		if (is_last)
		{
            unsigned int idx = min(tid, gridDim.x * gridDim.y - 1);

            sminval[tid] = minval[idx];
            smaxval[tid] = maxval[idx];
            __syncthreads();

			find_min_max_in_smem<nthreads, best_type>(sminval, smaxval, tid);

            if (tid == 0) 
            {
                minval[0] = (T)sminval[0];
                maxval[0] = (T)smaxval[0];
                blocks_finished = 0;
            }
		}
#else
        if (tid == 0) 
        {
            minval[blockIdx.y * gridDim.x + blockIdx.x] = (T)sminval[0];
            maxval[blockIdx.y * gridDim.x + blockIdx.x] = (T)smaxval[0];
        }
#endif
    }

   
    template <typename T>
    void min_max_mask_caller(const DevMem2D src, const PtrStep mask, double* minval, double* maxval, PtrStep buf)
    {
        dim3 threads, grid;
        estimate_thread_cfg(src.cols, src.rows, threads, grid);
        set_kernel_consts(src.cols, src.rows, threads, grid);

        T* minval_buf = (T*)buf.ptr(0);
        T* maxval_buf = (T*)buf.ptr(1);

        min_max_kernel<256, T, Mask8U><<<grid, threads>>>(src, Mask8U(mask), minval_buf, maxval_buf);
        cudaSafeCall(cudaThreadSynchronize());

        T minval_, maxval_;
        cudaSafeCall(cudaMemcpy(&minval_, minval_buf, sizeof(T), cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaMemcpy(&maxval_, maxval_buf, sizeof(T), cudaMemcpyDeviceToHost));
        *minval = minval_;
        *maxval = maxval_;
    }  

    template void min_max_mask_caller<unsigned char>(const DevMem2D, const PtrStep, double*, double*, PtrStep);
    template void min_max_mask_caller<char>(const DevMem2D, const PtrStep, double*, double*, PtrStep);
    template void min_max_mask_caller<unsigned short>(const DevMem2D, const PtrStep, double*, double*, PtrStep);
    template void min_max_mask_caller<short>(const DevMem2D, const PtrStep, double*, double*, PtrStep);
    template void min_max_mask_caller<int>(const DevMem2D, const PtrStep, double*, double*, PtrStep);
    template void min_max_mask_caller<float>(const DevMem2D, const PtrStep, double*, double*, PtrStep);
    template void min_max_mask_caller<double>(const DevMem2D, const PtrStep, double*, double*, PtrStep);


    template <typename T>
    void min_max_caller(const DevMem2D src, double* minval, double* maxval, PtrStep buf)
    {
        dim3 threads, grid;
        estimate_thread_cfg(src.cols, src.rows, threads, grid);
        set_kernel_consts(src.cols, src.rows, threads, grid);

        T* minval_buf = (T*)buf.ptr(0);
        T* maxval_buf = (T*)buf.ptr(1);

        min_max_kernel<256, T, MaskTrue><<<grid, threads>>>(src, MaskTrue(), minval_buf, maxval_buf);
        cudaSafeCall(cudaThreadSynchronize());

        T minval_, maxval_;
        cudaSafeCall(cudaMemcpy(&minval_, minval_buf, sizeof(T), cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaMemcpy(&maxval_, maxval_buf, sizeof(T), cudaMemcpyDeviceToHost));
        *minval = minval_;
        *maxval = maxval_;
    }  

    template void min_max_caller<unsigned char>(const DevMem2D, double*, double*, PtrStep);
    template void min_max_caller<char>(const DevMem2D, double*, double*, PtrStep);
    template void min_max_caller<unsigned short>(const DevMem2D, double*, double*, PtrStep);
    template void min_max_caller<short>(const DevMem2D, double*, double*, PtrStep);
    template void min_max_caller<int>(const DevMem2D, double*, double*, PtrStep);
    template void min_max_caller<float>(const DevMem2D, double*,double*, PtrStep);
    template void min_max_caller<double>(const DevMem2D, double*, double*, PtrStep);


    template <int nthreads, typename T>
    __global__ void min_max_pass2_kernel(T* minval, T* maxval, int size)
    {
        typedef typename MinMaxTypeTraits<T>::best_type best_type;
        __shared__ best_type sminval[nthreads];
        __shared__ best_type smaxval[nthreads];
        
        unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
        unsigned int idx = min(tid, gridDim.x * gridDim.y - 1);

        sminval[tid] = minval[idx];
        smaxval[tid] = maxval[idx];
        __syncthreads();

		find_min_max_in_smem<nthreads, best_type>(sminval, smaxval, tid);

        if (tid == 0) 
        {
            minval[0] = (T)sminval[0];
            maxval[0] = (T)smaxval[0];
        }
    }


    template <typename T>
    void min_max_mask_multipass_caller(const DevMem2D src, const PtrStep mask, double* minval, double* maxval, PtrStep buf)
    {
        dim3 threads, grid;
        estimate_thread_cfg(src.cols, src.rows, threads, grid);
        set_kernel_consts(src.cols, src.rows, threads, grid);

        T* minval_buf = (T*)buf.ptr(0);
        T* maxval_buf = (T*)buf.ptr(1);

        min_max_kernel<256, T, Mask8U><<<grid, threads>>>(src, Mask8U(mask), minval_buf, maxval_buf);
        min_max_pass2_kernel<256, T><<<1, 256>>>(minval_buf, maxval_buf, grid.x * grid.y);
        cudaSafeCall(cudaThreadSynchronize());

        T minval_, maxval_;
        cudaSafeCall(cudaMemcpy(&minval_, minval_buf, sizeof(T), cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaMemcpy(&maxval_, maxval_buf, sizeof(T), cudaMemcpyDeviceToHost));
        *minval = minval_;
        *maxval = maxval_;
    }

    template void min_max_mask_multipass_caller<unsigned char>(const DevMem2D, const PtrStep, double*, double*, PtrStep);
    template void min_max_mask_multipass_caller<char>(const DevMem2D, const PtrStep, double*, double*, PtrStep);
    template void min_max_mask_multipass_caller<unsigned short>(const DevMem2D, const PtrStep, double*, double*, PtrStep);
    template void min_max_mask_multipass_caller<short>(const DevMem2D, const PtrStep, double*, double*, PtrStep);
    template void min_max_mask_multipass_caller<int>(const DevMem2D, const PtrStep, double*, double*, PtrStep);
    template void min_max_mask_multipass_caller<float>(const DevMem2D, const PtrStep, double*, double*, PtrStep);


    template <typename T>
    void min_max_multipass_caller(const DevMem2D src, double* minval, double* maxval, PtrStep buf)
    {
        dim3 threads, grid;
        estimate_thread_cfg(src.cols, src.rows, threads, grid);
        set_kernel_consts(src.cols, src.rows, threads, grid);

        T* minval_buf = (T*)buf.ptr(0);
        T* maxval_buf = (T*)buf.ptr(1);

        min_max_kernel<256, T, MaskTrue><<<grid, threads>>>(src, MaskTrue(), minval_buf, maxval_buf);
        min_max_pass2_kernel<256, T><<<1, 256>>>(minval_buf, maxval_buf, grid.x * grid.y);
        cudaSafeCall(cudaThreadSynchronize());

        T minval_, maxval_;
        cudaSafeCall(cudaMemcpy(&minval_, minval_buf, sizeof(T), cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaMemcpy(&maxval_, maxval_buf, sizeof(T), cudaMemcpyDeviceToHost));
        *minval = minval_;
        *maxval = maxval_;
    }

    template void min_max_multipass_caller<unsigned char>(const DevMem2D, double*, double*, PtrStep);
    template void min_max_multipass_caller<char>(const DevMem2D, double*, double*, PtrStep);
    template void min_max_multipass_caller<unsigned short>(const DevMem2D, double*, double*, PtrStep);
    template void min_max_multipass_caller<short>(const DevMem2D, double*, double*, PtrStep);
    template void min_max_multipass_caller<int>(const DevMem2D, double*, double*, PtrStep);
    template void min_max_multipass_caller<float>(const DevMem2D, double*, double*, PtrStep);

    } // namespace minmax

///////////////////////////////////////////////////////////////////////////////
// minMaxLoc

    namespace minmaxloc {

    __constant__ int ctwidth;
    __constant__ int ctheight;

    // Global counter of blocks finished its work
    __device__ unsigned int blocks_finished = 0;


    // Estimates good thread configuration
    //  - threads variable satisfies to threads.x * threads.y == 256
    void estimate_thread_cfg(int cols, int rows, dim3& threads, dim3& grid)
    {
        threads = dim3(32, 8);
        grid = dim3(divUp(cols, threads.x * 8), divUp(rows, threads.y * 32));
        grid.x = min(grid.x, threads.x);
        grid.y = min(grid.y, threads.y);
    }


    // Returns required buffer sizes
    void get_buf_size_required(int cols, int rows, int elem_size, int& b1cols, 
                               int& b1rows, int& b2cols, int& b2rows)
    {
        dim3 threads, grid;
        estimate_thread_cfg(cols, rows, threads, grid);
        b1cols = grid.x * grid.y * elem_size; // For values
        b1rows = 2;
        b2cols = grid.x * grid.y * sizeof(int); // For locations
        b2rows = 2;
    }


    // Estimates device constants which are used in the kernels using specified thread configuration
    void set_kernel_consts(int cols, int rows, const dim3& threads, const dim3& grid)
    {        
        int twidth = divUp(divUp(cols, grid.x), threads.x);
        int theight = divUp(divUp(rows, grid.y), threads.y);
        cudaSafeCall(cudaMemcpyToSymbol(ctwidth, &twidth, sizeof(ctwidth))); 
        cudaSafeCall(cudaMemcpyToSymbol(ctheight, &theight, sizeof(ctheight))); 
    }  


    template <typename T>
    __device__ void merge(unsigned int tid, unsigned int offset, volatile T* minval, volatile T* maxval, 
                          volatile unsigned int* minloc, volatile unsigned int* maxloc)
    {
        T val = minval[tid + offset];
        if (val < minval[tid])
        {
            minval[tid] = val;
            minloc[tid] = minloc[tid + offset];
        }
        val = maxval[tid + offset];
        if (val > maxval[tid])
        {
            maxval[tid] = val;
            maxloc[tid] = maxloc[tid + offset];
        }
    }


    template <int size, typename T>
    __device__ void find_min_max_loc_in_smem(volatile T* minval, volatile T* maxval, volatile unsigned int* minloc, 
                                             volatile unsigned int* maxloc, const unsigned int tid)
    {
        if (size >= 512) { if (tid < 256) { merge(tid, 256, minval, maxval, minloc, maxloc); } __syncthreads(); }
        if (size >= 256) { if (tid < 128) { merge(tid, 128, minval, maxval, minloc, maxloc); }  __syncthreads(); }
        if (size >= 128) { if (tid < 64) { merge(tid, 64, minval, maxval, minloc, maxloc); } __syncthreads(); }

        if (tid < 32)
        {
            if (size >= 64) merge(tid, 32, minval, maxval, minloc, maxloc);
            if (size >= 32) merge(tid, 16, minval, maxval, minloc, maxloc);
            if (size >= 16) merge(tid, 8, minval, maxval, minloc, maxloc);
            if (size >= 8) merge(tid, 4, minval, maxval, minloc, maxloc);
            if (size >= 4) merge(tid, 2, minval, maxval, minloc, maxloc);
            if (size >= 2) merge(tid, 1, minval, maxval, minloc, maxloc);
        }
    }


    template <int nthreads, typename T, typename Mask>
    __global__ void min_max_loc_kernel(const DevMem2D src, Mask mask, T* minval, T* maxval, 
                                       unsigned int* minloc, unsigned int* maxloc)
    {
        typedef typename MinMaxTypeTraits<T>::best_type best_type;
        __shared__ best_type sminval[nthreads];
        __shared__ best_type smaxval[nthreads];
        __shared__ unsigned int sminloc[nthreads];
        __shared__ unsigned int smaxloc[nthreads];

        unsigned int x0 = blockIdx.x * blockDim.x * ctwidth + threadIdx.x;
        unsigned int y0 = blockIdx.y * blockDim.y * ctheight + threadIdx.y;
        unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

        T mymin = numeric_limits_gpu<T>::max();
        T mymax = numeric_limits_gpu<T>::is_signed ? -numeric_limits_gpu<T>::max() : numeric_limits_gpu<T>::min(); 
        unsigned int myminloc = 0;
        unsigned int mymaxloc = 0;
        unsigned int y_end = min(y0 + (ctheight - 1) * blockDim.y + 1, src.rows);
        unsigned int x_end = min(x0 + (ctwidth - 1) * blockDim.x + 1, src.cols);

        for (unsigned int y = y0; y < y_end; y += blockDim.y)
        {
            const T* ptr = (const T*)src.ptr(y);
            for (unsigned int x = x0; x < x_end; x += blockDim.x)
            {
                if (mask(y, x))
                {
                    T val = ptr[x];
                    if (val <= mymin) { mymin = val; myminloc = y * src.cols + x; }
                    if (val >= mymax) { mymax = val; mymaxloc = y * src.cols + x; }
                }
            }
        }

        sminval[tid] = mymin; 
        smaxval[tid] = mymax;
        sminloc[tid] = myminloc;
        smaxloc[tid] = mymaxloc;
        __syncthreads();

        find_min_max_loc_in_smem<nthreads, best_type>(sminval, smaxval, sminloc, smaxloc, tid);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 110
		__shared__ bool is_last;

		if (tid == 0)
		{
			minval[blockIdx.y * gridDim.x + blockIdx.x] = (T)sminval[0];
            maxval[blockIdx.y * gridDim.x + blockIdx.x] = (T)smaxval[0];
            minloc[blockIdx.y * gridDim.x + blockIdx.x] = sminloc[0];
            maxloc[blockIdx.y * gridDim.x + blockIdx.x] = smaxloc[0];
			__threadfence();

			unsigned int ticket = atomicInc(&blocks_finished, gridDim.x * gridDim.y);
			is_last = ticket == gridDim.x * gridDim.y - 1;
		}

		__syncthreads();

		if (is_last)
		{
            unsigned int idx = min(tid, gridDim.x * gridDim.y - 1);

            sminval[tid] = minval[idx];
            smaxval[tid] = maxval[idx];
            sminloc[tid] = minloc[idx];
            smaxloc[tid] = maxloc[idx];
            __syncthreads();

			find_min_max_loc_in_smem<nthreads, best_type>(sminval, smaxval, sminloc, smaxloc, tid);

            if (tid == 0) 
            {
                minval[0] = (T)sminval[0];
                maxval[0] = (T)smaxval[0];
                minloc[0] = sminloc[0];
                maxloc[0] = smaxloc[0];
                blocks_finished = 0;
            }
		}
#else
        if (tid == 0) 
        {
            minval[blockIdx.y * gridDim.x + blockIdx.x] = (T)sminval[0];
            maxval[blockIdx.y * gridDim.x + blockIdx.x] = (T)smaxval[0];
            minloc[blockIdx.y * gridDim.x + blockIdx.x] = sminloc[0];
            maxloc[blockIdx.y * gridDim.x + blockIdx.x] = smaxloc[0];
        }
#endif
    }


    template <typename T>
    void min_max_loc_mask_caller(const DevMem2D src, const PtrStep mask, double* minval, double* maxval, 
                                 int minloc[2], int maxloc[2], PtrStep valbuf, PtrStep locbuf)
    {
        dim3 threads, grid;
        estimate_thread_cfg(src.cols, src.rows, threads, grid);
        set_kernel_consts(src.cols, src.rows, threads, grid);

        T* minval_buf = (T*)valbuf.ptr(0);
        T* maxval_buf = (T*)valbuf.ptr(1);
        unsigned int* minloc_buf = (unsigned int*)locbuf.ptr(0);
        unsigned int* maxloc_buf = (unsigned int*)locbuf.ptr(1);

        min_max_loc_kernel<256, T, Mask8U><<<grid, threads>>>(src, Mask8U(mask), minval_buf, maxval_buf, minloc_buf, maxloc_buf);
        cudaSafeCall(cudaThreadSynchronize());

        T minval_, maxval_;
        cudaSafeCall(cudaMemcpy(&minval_, minval_buf, sizeof(T), cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaMemcpy(&maxval_, maxval_buf, sizeof(T), cudaMemcpyDeviceToHost));
        *minval = minval_;
        *maxval = maxval_;

        unsigned int minloc_, maxloc_;
        cudaSafeCall(cudaMemcpy(&minloc_, minloc_buf, sizeof(int), cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaMemcpy(&maxloc_, maxloc_buf, sizeof(int), cudaMemcpyDeviceToHost));
        minloc[1] = minloc_ / src.cols; minloc[0] = minloc_ - minloc[1] * src.cols;
        maxloc[1] = maxloc_ / src.cols; maxloc[0] = maxloc_ - maxloc[1] * src.cols;
    }

    template void min_max_loc_mask_caller<unsigned char>(const DevMem2D, const PtrStep, double*, double*, int[2], int[2], PtrStep, PtrStep);
    template void min_max_loc_mask_caller<char>(const DevMem2D, const PtrStep, double*, double*, int[2], int[2], PtrStep, PtrStep);
    template void min_max_loc_mask_caller<unsigned short>(const DevMem2D, const PtrStep, double*, double*, int[2], int[2], PtrStep, PtrStep);
    template void min_max_loc_mask_caller<short>(const DevMem2D, const PtrStep, double*, double*, int[2], int[2], PtrStep, PtrStep);
    template void min_max_loc_mask_caller<int>(const DevMem2D, const PtrStep, double*, double*, int[2], int[2], PtrStep, PtrStep);
    template void min_max_loc_mask_caller<float>(const DevMem2D, const PtrStep, double*, double*, int[2], int[2], PtrStep, PtrStep);
    template void min_max_loc_mask_caller<double>(const DevMem2D, const PtrStep, double*, double*, int[2], int[2], PtrStep, PtrStep);


    template <typename T>
    void min_max_loc_caller(const DevMem2D src, double* minval, double* maxval, 
                            int minloc[2], int maxloc[2], PtrStep valbuf, PtrStep locbuf)
    {
        dim3 threads, grid;
        estimate_thread_cfg(src.cols, src.rows, threads, grid);
        set_kernel_consts(src.cols, src.rows, threads, grid);

        T* minval_buf = (T*)valbuf.ptr(0);
        T* maxval_buf = (T*)valbuf.ptr(1);
        unsigned int* minloc_buf = (unsigned int*)locbuf.ptr(0);
        unsigned int* maxloc_buf = (unsigned int*)locbuf.ptr(1);

        min_max_loc_kernel<256, T, MaskTrue><<<grid, threads>>>(src, MaskTrue(), minval_buf, maxval_buf, minloc_buf, maxloc_buf);
        cudaSafeCall(cudaThreadSynchronize());

        T minval_, maxval_;
        cudaSafeCall(cudaMemcpy(&minval_, minval_buf, sizeof(T), cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaMemcpy(&maxval_, maxval_buf, sizeof(T), cudaMemcpyDeviceToHost));
        *minval = minval_;
        *maxval = maxval_;

        unsigned int minloc_, maxloc_;
        cudaSafeCall(cudaMemcpy(&minloc_, minloc_buf, sizeof(int), cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaMemcpy(&maxloc_, maxloc_buf, sizeof(int), cudaMemcpyDeviceToHost));
        minloc[1] = minloc_ / src.cols; minloc[0] = minloc_ - minloc[1] * src.cols;
        maxloc[1] = maxloc_ / src.cols; maxloc[0] = maxloc_ - maxloc[1] * src.cols;
    }

    template void min_max_loc_caller<unsigned char>(const DevMem2D, double*, double*, int[2], int[2], PtrStep, PtrStep);
    template void min_max_loc_caller<char>(const DevMem2D, double*, double*, int[2], int[2], PtrStep, PtrStep);
    template void min_max_loc_caller<unsigned short>(const DevMem2D, double*, double*, int[2], int[2], PtrStep, PtrStep);
    template void min_max_loc_caller<short>(const DevMem2D, double*, double*, int[2], int[2], PtrStep, PtrStep);
    template void min_max_loc_caller<int>(const DevMem2D, double*, double*, int[2], int[2], PtrStep, PtrStep);
    template void min_max_loc_caller<float>(const DevMem2D, double*, double*, int[2], int[2], PtrStep, PtrStep);
    template void min_max_loc_caller<double>(const DevMem2D, double*, double*, int[2], int[2], PtrStep, PtrStep);


    // This kernel will be used only when compute capability is 1.0
    template <int nthreads, typename T>
    __global__ void min_max_loc_pass2_kernel(T* minval, T* maxval, unsigned int* minloc, unsigned int* maxloc, int size)
    {
        typedef typename MinMaxTypeTraits<T>::best_type best_type;
        __shared__ best_type sminval[nthreads];
        __shared__ best_type smaxval[nthreads];
        __shared__ unsigned int sminloc[nthreads];
        __shared__ unsigned int smaxloc[nthreads];

        unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
        unsigned int idx = min(tid, gridDim.x * gridDim.y - 1);

        sminval[tid] = minval[idx];
        smaxval[tid] = maxval[idx];
        sminloc[tid] = minloc[idx];
        smaxloc[tid] = maxloc[idx];
        __syncthreads();

		find_min_max_loc_in_smem<nthreads, best_type>(sminval, smaxval, sminloc, smaxloc, tid);

        if (tid == 0) 
        {
            minval[0] = (T)sminval[0];
            maxval[0] = (T)smaxval[0];
            minloc[0] = sminloc[0];
            maxloc[0] = smaxloc[0];
        }
    }


    template <typename T>
    void min_max_loc_mask_multipass_caller(const DevMem2D src, const PtrStep mask, double* minval, double* maxval, 
                                           int minloc[2], int maxloc[2], PtrStep valbuf, PtrStep locbuf)
    {
        dim3 threads, grid;
        estimate_thread_cfg(src.cols, src.rows, threads, grid);
        set_kernel_consts(src.cols, src.rows, threads, grid);

        T* minval_buf = (T*)valbuf.ptr(0);
        T* maxval_buf = (T*)valbuf.ptr(1);
        unsigned int* minloc_buf = (unsigned int*)locbuf.ptr(0);
        unsigned int* maxloc_buf = (unsigned int*)locbuf.ptr(1);

        min_max_loc_kernel<256, T, Mask8U><<<grid, threads>>>(src, Mask8U(mask), minval_buf, maxval_buf, minloc_buf, maxloc_buf);
        min_max_loc_pass2_kernel<256, T><<<1, 256>>>(minval_buf, maxval_buf, minloc_buf, maxloc_buf, grid.x * grid.y);
        cudaSafeCall(cudaThreadSynchronize());

        T minval_, maxval_;
        cudaSafeCall(cudaMemcpy(&minval_, minval_buf, sizeof(T), cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaMemcpy(&maxval_, maxval_buf, sizeof(T), cudaMemcpyDeviceToHost));
        *minval = minval_;
        *maxval = maxval_;

        unsigned int minloc_, maxloc_;
        cudaSafeCall(cudaMemcpy(&minloc_, minloc_buf, sizeof(int), cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaMemcpy(&maxloc_, maxloc_buf, sizeof(int), cudaMemcpyDeviceToHost));
        minloc[1] = minloc_ / src.cols; minloc[0] = minloc_ - minloc[1] * src.cols;
        maxloc[1] = maxloc_ / src.cols; maxloc[0] = maxloc_ - maxloc[1] * src.cols;
    }

    template void min_max_loc_mask_multipass_caller<unsigned char>(const DevMem2D, const PtrStep, double*, double*, int[2], int[2], PtrStep, PtrStep);
    template void min_max_loc_mask_multipass_caller<char>(const DevMem2D, const PtrStep, double*, double*, int[2], int[2], PtrStep, PtrStep);
    template void min_max_loc_mask_multipass_caller<unsigned short>(const DevMem2D, const PtrStep, double*, double*, int[2], int[2], PtrStep, PtrStep);
    template void min_max_loc_mask_multipass_caller<short>(const DevMem2D, const PtrStep, double*, double*, int[2], int[2], PtrStep, PtrStep);
    template void min_max_loc_mask_multipass_caller<int>(const DevMem2D, const PtrStep, double*, double*, int[2], int[2], PtrStep, PtrStep);
    template void min_max_loc_mask_multipass_caller<float>(const DevMem2D, const PtrStep, double*, double*, int[2], int[2], PtrStep, PtrStep);


    template <typename T>
    void min_max_loc_multipass_caller(const DevMem2D src, double* minval, double* maxval, 
                                      int minloc[2], int maxloc[2], PtrStep valbuf, PtrStep locbuf)
    {
        dim3 threads, grid;
        estimate_thread_cfg(src.cols, src.rows, threads, grid);
        set_kernel_consts(src.cols, src.rows, threads, grid);

        T* minval_buf = (T*)valbuf.ptr(0);
        T* maxval_buf = (T*)valbuf.ptr(1);
        unsigned int* minloc_buf = (unsigned int*)locbuf.ptr(0);
        unsigned int* maxloc_buf = (unsigned int*)locbuf.ptr(1);

        min_max_loc_kernel<256, T, MaskTrue><<<grid, threads>>>(src, MaskTrue(), minval_buf, maxval_buf, minloc_buf, maxloc_buf);
        min_max_loc_pass2_kernel<256, T><<<1, 256>>>(minval_buf, maxval_buf, minloc_buf, maxloc_buf, grid.x * grid.y);
        cudaSafeCall(cudaThreadSynchronize());

        T minval_, maxval_;
        cudaSafeCall(cudaMemcpy(&minval_, minval_buf, sizeof(T), cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaMemcpy(&maxval_, maxval_buf, sizeof(T), cudaMemcpyDeviceToHost));
        *minval = minval_;
        *maxval = maxval_;

        unsigned int minloc_, maxloc_;
        cudaSafeCall(cudaMemcpy(&minloc_, minloc_buf, sizeof(int), cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaMemcpy(&maxloc_, maxloc_buf, sizeof(int), cudaMemcpyDeviceToHost));
        minloc[1] = minloc_ / src.cols; minloc[0] = minloc_ - minloc[1] * src.cols;
        maxloc[1] = maxloc_ / src.cols; maxloc[0] = maxloc_ - maxloc[1] * src.cols;
    }

    template void min_max_loc_multipass_caller<unsigned char>(const DevMem2D, double*, double*, int[2], int[2], PtrStep, PtrStep);
    template void min_max_loc_multipass_caller<char>(const DevMem2D, double*, double*, int[2], int[2], PtrStep, PtrStep);
    template void min_max_loc_multipass_caller<unsigned short>(const DevMem2D, double*, double*, int[2], int[2], PtrStep, PtrStep);
    template void min_max_loc_multipass_caller<short>(const DevMem2D, double*, double*, int[2], int[2], PtrStep, PtrStep);
    template void min_max_loc_multipass_caller<int>(const DevMem2D, double*, double*, int[2], int[2], PtrStep, PtrStep);
    template void min_max_loc_multipass_caller<float>(const DevMem2D, double*, double*, int[2], int[2], PtrStep, PtrStep);

    } // namespace minmaxloc

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// countNonZero

    namespace countnonzero 
    {

    __constant__ int ctwidth;
    __constant__ int ctheight;

    __device__ unsigned int blocks_finished = 0;

    void estimate_thread_cfg(int cols, int rows, dim3& threads, dim3& grid)
    {
        threads = dim3(32, 8);
        grid = dim3(divUp(cols, threads.x * 8), divUp(rows, threads.y * 32));
        grid.x = min(grid.x, threads.x);
        grid.y = min(grid.y, threads.y);
    }


    void get_buf_size_required(int cols, int rows, int& bufcols, int& bufrows)
    {
        dim3 threads, grid;
        estimate_thread_cfg(cols, rows, threads, grid);
        bufcols = grid.x * grid.y * sizeof(int);
        bufrows = 1;
    }


    void set_kernel_consts(int cols, int rows, const dim3& threads, const dim3& grid)
    {        
        int twidth = divUp(divUp(cols, grid.x), threads.x);
        int theight = divUp(divUp(rows, grid.y), threads.y);
        cudaSafeCall(cudaMemcpyToSymbol(ctwidth, &twidth, sizeof(twidth))); 
        cudaSafeCall(cudaMemcpyToSymbol(ctheight, &theight, sizeof(theight))); 
    }


    template <int nthreads, typename T>
    __global__ void count_non_zero_kernel(const DevMem2D src, volatile unsigned int* count)
    {
        __shared__ unsigned int scount[nthreads];

        unsigned int x0 = blockIdx.x * blockDim.x * ctwidth + threadIdx.x;
        unsigned int y0 = blockIdx.y * blockDim.y * ctheight + threadIdx.y;
        unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

		unsigned int cnt = 0;
        for (unsigned int y = 0; y < ctheight && y0 + y * blockDim.y < src.rows; ++y)
        {
            const T* ptr = (const T*)src.ptr(y0 + y * blockDim.y);
            for (unsigned int x = 0; x < ctwidth && x0 + x * blockDim.x < src.cols; ++x)
				cnt += ptr[x0 + x * blockDim.x] != 0;
		}

		scount[tid] = cnt;
		__syncthreads();

        sum_in_smem<nthreads, unsigned int>(scount, tid);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 110
		__shared__ bool is_last;

		if (tid == 0)
		{
			count[blockIdx.y * gridDim.x + blockIdx.x] = scount[0];
			__threadfence();

			unsigned int ticket = atomicInc(&blocks_finished, gridDim.x * gridDim.y);
			is_last = ticket == gridDim.x * gridDim.y - 1;
		}

		__syncthreads();

		if (is_last)
		{
            scount[tid] = tid < gridDim.x * gridDim.y ? count[tid] : 0;
            __syncthreads();

			sum_in_smem<nthreads, unsigned int>(scount, tid);

			if (tid == 0) 
            {
                count[0] = scount[0];
                blocks_finished = 0;
            }
		}
#else
        if (tid == 0) count[blockIdx.y * gridDim.x + blockIdx.x] = scount[0];
#endif
    }

   
    template <typename T>
    int count_non_zero_caller(const DevMem2D src, PtrStep buf)
    {
        dim3 threads, grid;
        estimate_thread_cfg(src.cols, src.rows, threads, grid);
        set_kernel_consts(src.cols, src.rows, threads, grid);

        unsigned int* count_buf = (unsigned int*)buf.ptr(0);

        count_non_zero_kernel<256, T><<<grid, threads>>>(src, count_buf);
        cudaSafeCall(cudaThreadSynchronize());

        unsigned int count;
        cudaSafeCall(cudaMemcpy(&count, count_buf, sizeof(int), cudaMemcpyDeviceToHost));
        
        return count;
    }  

    template int count_non_zero_caller<unsigned char>(const DevMem2D, PtrStep);
    template int count_non_zero_caller<char>(const DevMem2D, PtrStep);
    template int count_non_zero_caller<unsigned short>(const DevMem2D, PtrStep);
    template int count_non_zero_caller<short>(const DevMem2D, PtrStep);
    template int count_non_zero_caller<int>(const DevMem2D, PtrStep);
    template int count_non_zero_caller<float>(const DevMem2D, PtrStep);
    template int count_non_zero_caller<double>(const DevMem2D, PtrStep);


    template <int nthreads, typename T>
    __global__ void count_non_zero_pass2_kernel(unsigned int* count, int size)
    {
        __shared__ unsigned int scount[nthreads];
        unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

        scount[tid] = tid < size ? count[tid] : 0;
        __syncthreads();

        sum_in_smem<nthreads, unsigned int>(scount, tid);

        if (tid == 0) 
            count[0] = scount[0];
    }


    template <typename T>
    int count_non_zero_multipass_caller(const DevMem2D src, PtrStep buf)
    {
        dim3 threads, grid;
        estimate_thread_cfg(src.cols, src.rows, threads, grid);
        set_kernel_consts(src.cols, src.rows, threads, grid);

        unsigned int* count_buf = (unsigned int*)buf.ptr(0);

        count_non_zero_kernel<256, T><<<grid, threads>>>(src, count_buf);
        count_non_zero_pass2_kernel<256, T><<<1, 256>>>(count_buf, grid.x * grid.y);
        cudaSafeCall(cudaThreadSynchronize());

        unsigned int count;
        cudaSafeCall(cudaMemcpy(&count, count_buf, sizeof(int), cudaMemcpyDeviceToHost));
        
        return count;
    }  

    template int count_non_zero_multipass_caller<unsigned char>(const DevMem2D, PtrStep);
    template int count_non_zero_multipass_caller<char>(const DevMem2D, PtrStep);
    template int count_non_zero_multipass_caller<unsigned short>(const DevMem2D, PtrStep);
    template int count_non_zero_multipass_caller<short>(const DevMem2D, PtrStep);
    template int count_non_zero_multipass_caller<int>(const DevMem2D, PtrStep);
    template int count_non_zero_multipass_caller<float>(const DevMem2D, PtrStep);

    } // namespace countnonzero

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// transpose

    __global__ void transpose(const DevMem2Di src, PtrStepi dst)
    {
    	__shared__ int s_mem[16 * 17];

    	int x = blockIdx.x * blockDim.x + threadIdx.x;
    	int y = blockIdx.y * blockDim.y + threadIdx.y;
	    int smem_idx = threadIdx.y * blockDim.x + threadIdx.x + threadIdx.y;

	    if (y < src.rows && x < src.cols)
	    {
            s_mem[smem_idx] = src.ptr(y)[x];
	    }
	    __syncthreads();

	    smem_idx = threadIdx.x * blockDim.x + threadIdx.y + threadIdx.x;

	    x = blockIdx.y * blockDim.x + threadIdx.x;
	    y = blockIdx.x * blockDim.y + threadIdx.y;

	    if (y < src.cols && x < src.rows)
	    {
		    dst.ptr(y)[x] = s_mem[smem_idx];
	    }
    }

    void transpose_gpu(const DevMem2Di& src, const DevMem2Di& dst)
    {
	    dim3 threads(16, 16, 1);
	    dim3 grid(divUp(src.cols, 16), divUp(src.rows, 16), 1);

	    transpose<<<grid, threads>>>(src, dst);
        cudaSafeCall( cudaThreadSynchronize() );
    }
    
//////////////////////////////////////////////////////////////////////////////////////////////////////////
// min/max

    struct MinOp
    {        
        template <typename T>
        __device__ T operator()(T a, T b)
        {
            return min(a, b);
        }
        __device__ float operator()(float a, float b)
        {
            return fmin(a, b);
        }
        __device__ double operator()(double a, double b)
        {
            return fmin(a, b);
        }
    };

    struct MaxOp
    {        
        template <typename T>
        __device__ T operator()(T a, T b)
        {
            return max(a, b);
        }
        __device__ float operator()(float a, float b)
        {
            return fmax(a, b);
        }
        __device__ double operator()(double a, double b)
        {
            return fmax(a, b);
        }
    };
    
    struct ScalarMinOp
    {
        double s;

        explicit ScalarMinOp(double s_) : s(s_) {}

        template <typename T>
        __device__ T operator()(T a)
        {
            return saturate_cast<T>(fmin((double)a, s));
        }
    };
    
    struct ScalarMaxOp
    {
        double s;

        explicit ScalarMaxOp(double s_) : s(s_) {}

        template <typename T>
        __device__ T operator()(T a)
        {
            return saturate_cast<T>(fmax((double)a, s));
        }
    };
    
    template <typename T>
    void min_gpu(const DevMem2D_<T>& src1, const DevMem2D_<T>& src2, const DevMem2D_<T>& dst, cudaStream_t stream)
    {
        MinOp op;
        transform(src1, src2, dst, op, stream);    
    }

    template void min_gpu<uchar >(const DevMem2D& src1, const DevMem2D& src2, const DevMem2D& dst, cudaStream_t stream);
    template void min_gpu<char  >(const DevMem2D_<char>& src1, const DevMem2D_<char>& src2, const DevMem2D_<char>& dst, cudaStream_t stream);
    template void min_gpu<ushort>(const DevMem2D_<ushort>& src1, const DevMem2D_<ushort>& src2, const DevMem2D_<ushort>& dst, cudaStream_t stream);
    template void min_gpu<short >(const DevMem2D_<short>& src1, const DevMem2D_<short>& src2, const DevMem2D_<short>& dst, cudaStream_t stream);
    template void min_gpu<int   >(const DevMem2D_<int>& src1, const DevMem2D_<int>& src2, const DevMem2D_<int>& dst, cudaStream_t stream);
    template void min_gpu<float >(const DevMem2D_<float>& src1, const DevMem2D_<float>& src2, const DevMem2D_<float>& dst, cudaStream_t stream);
    template void min_gpu<double>(const DevMem2D_<double>& src1, const DevMem2D_<double>& src2, const DevMem2D_<double>& dst, cudaStream_t stream);

    template <typename T>
    void max_gpu(const DevMem2D_<T>& src1, const DevMem2D_<T>& src2, const DevMem2D_<T>& dst, cudaStream_t stream)
    {
        MaxOp op;
        transform(src1, src2, dst, op, stream);    
    }
    
    template void max_gpu<uchar >(const DevMem2D& src1, const DevMem2D& src2, const DevMem2D& dst, cudaStream_t stream);
    template void max_gpu<char  >(const DevMem2D_<char>& src1, const DevMem2D_<char>& src2, const DevMem2D_<char>& dst, cudaStream_t stream);
    template void max_gpu<ushort>(const DevMem2D_<ushort>& src1, const DevMem2D_<ushort>& src2, const DevMem2D_<ushort>& dst, cudaStream_t stream);
    template void max_gpu<short >(const DevMem2D_<short>& src1, const DevMem2D_<short>& src2, const DevMem2D_<short>& dst, cudaStream_t stream);
    template void max_gpu<int   >(const DevMem2D_<int>& src1, const DevMem2D_<int>& src2, const DevMem2D_<int>& dst, cudaStream_t stream);
    template void max_gpu<float >(const DevMem2D_<float>& src1, const DevMem2D_<float>& src2, const DevMem2D_<float>& dst, cudaStream_t stream);
    template void max_gpu<double>(const DevMem2D_<double>& src1, const DevMem2D_<double>& src2, const DevMem2D_<double>& dst, cudaStream_t stream);

    template <typename T>
    void min_gpu(const DevMem2D_<T>& src1, double src2, const DevMem2D_<T>& dst, cudaStream_t stream)
    {
        ScalarMinOp op(src2);
        transform(src1, dst, op, stream);    
    }

    template void min_gpu<uchar >(const DevMem2D& src1, double src2, const DevMem2D& dst, cudaStream_t stream);
    template void min_gpu<char  >(const DevMem2D_<char>& src1, double src2, const DevMem2D_<char>& dst, cudaStream_t stream);
    template void min_gpu<ushort>(const DevMem2D_<ushort>& src1, double src2, const DevMem2D_<ushort>& dst, cudaStream_t stream);
    template void min_gpu<short >(const DevMem2D_<short>& src1, double src2, const DevMem2D_<short>& dst, cudaStream_t stream);
    template void min_gpu<int   >(const DevMem2D_<int>& src1, double src2, const DevMem2D_<int>& dst, cudaStream_t stream);
    template void min_gpu<float >(const DevMem2D_<float>& src1, double src2, const DevMem2D_<float>& dst, cudaStream_t stream);
    template void min_gpu<double>(const DevMem2D_<double>& src1, double src2, const DevMem2D_<double>& dst, cudaStream_t stream);
    
    template <typename T>
    void max_gpu(const DevMem2D_<T>& src1, double src2, const DevMem2D_<T>& dst, cudaStream_t stream)
    {
        ScalarMaxOp op(src2);
        transform(src1, dst, op, stream);    
    }

    template void max_gpu<uchar >(const DevMem2D& src1, double src2, const DevMem2D& dst, cudaStream_t stream);
    template void max_gpu<char  >(const DevMem2D_<char>& src1, double src2, const DevMem2D_<char>& dst, cudaStream_t stream);
    template void max_gpu<ushort>(const DevMem2D_<ushort>& src1, double src2, const DevMem2D_<ushort>& dst, cudaStream_t stream);
    template void max_gpu<short >(const DevMem2D_<short>& src1, double src2, const DevMem2D_<short>& dst, cudaStream_t stream);
    template void max_gpu<int   >(const DevMem2D_<int>& src1, double src2, const DevMem2D_<int>& dst, cudaStream_t stream);
    template void max_gpu<float >(const DevMem2D_<float>& src1, double src2, const DevMem2D_<float>& dst, cudaStream_t stream);
    template void max_gpu<double>(const DevMem2D_<double>& src1, double src2, const DevMem2D_<double>& dst, cudaStream_t stream);

//////////////////////////////////////////////////////////////////////////////
// Sum

    namespace sum 
    {

    template <typename T> struct SumType {};
    template <> struct SumType<unsigned char> { typedef unsigned int R; };
    template <> struct SumType<char> { typedef int R; };
    template <> struct SumType<unsigned short> { typedef unsigned int R; };
    template <> struct SumType<short> { typedef int R; };
    template <> struct SumType<int> { typedef int R; };
    template <> struct SumType<float> { typedef float R; };
    template <> struct SumType<double> { typedef double R; };

    __constant__ int ctwidth;
    __constant__ int ctheight;
    __device__ unsigned int blocks_finished = 0;

    const int threads_x = 32;
    const int threads_y = 8;

    void estimate_thread_cfg(int cols, int rows, dim3& threads, dim3& grid)
    {
        threads = dim3(threads_x, threads_y);
        grid = dim3(divUp(cols, threads.x * threads.y), 
                    divUp(rows, threads.y * threads.x));
        grid.x = min(grid.x, threads.x);
        grid.y = min(grid.y, threads.y);
    }


    void get_buf_size_required(int cols, int rows, int& bufcols, int& bufrows)
    {
        dim3 threads, grid;
        estimate_thread_cfg(cols, rows, threads, grid);
        bufcols = grid.x * grid.y * sizeof(double);
        bufrows = 1;
    }


    void set_kernel_consts(int cols, int rows, const dim3& threads, const dim3& grid)
    {        
        int twidth = divUp(divUp(cols, grid.x), threads.x);
        int theight = divUp(divUp(rows, grid.y), threads.y);
        cudaSafeCall(cudaMemcpyToSymbol(ctwidth, &twidth, sizeof(twidth))); 
        cudaSafeCall(cudaMemcpyToSymbol(ctheight, &theight, sizeof(theight))); 
    }

    template <typename T, typename R, int nthreads>
    __global__ void sum_kernel(const DevMem2D_<T> src, R* result)
    {
        __shared__ R smem[nthreads];

        const int x0 = blockIdx.x * blockDim.x * ctwidth + threadIdx.x;
        const int y0 = blockIdx.y * blockDim.y * ctheight + threadIdx.y;
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int bid = blockIdx.y * gridDim.x + blockIdx.x;

        R sum = 0;
        for (int y = 0; y < ctheight && y0 + y * blockDim.y < src.rows; ++y)
        {
            const T* ptr = src.ptr(y0 + y * blockDim.y);
            for (int x = 0; x < ctwidth && x0 + x * blockDim.x < src.cols; ++x)
                sum += ptr[x0 + x * blockDim.x];
        }

        smem[tid] = sum;
        __syncthreads();

        sum_in_smem<nthreads, R>(smem, tid);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 110
        __shared__ bool is_last;

        if (tid == 0)
        {
            result[bid] = smem[0];
            __threadfence();

            unsigned int ticket = atomicInc(&blocks_finished, gridDim.x * gridDim.y);
            is_last = (ticket == gridDim.x * gridDim.y - 1);
        }

        __syncthreads();

        if (is_last)
        {
            smem[tid] = tid < gridDim.x * gridDim.y ? result[tid] : 0;
            __syncthreads();

            sum_in_smem<nthreads, R>(smem, tid);

            if (tid == 0) 
            {
                result[0] = smem[0];
                blocks_finished = 0;
            }
        }
#else
        if (tid == 0) result[bid] = smem[0];
#endif
    }


    template <typename T, typename R, int nthreads>
    __global__ void sum_pass2_kernel(R* result, int size)
    {
        __shared__ R smem[nthreads];
        int tid = threadIdx.y * blockDim.x + threadIdx.x;

        smem[tid] = tid < size ? result[tid] : 0;
        __syncthreads();

        sum_in_smem<nthreads, R>(smem, tid);

        if (tid == 0) 
            result[0] = smem[0];
    }

    } // namespace sum


    template <typename T>
    void sum_multipass_caller(const DevMem2D src, PtrStep buf, double* sum)
    {
        using namespace sum;
        typedef typename SumType<T>::R R;

        dim3 threads, grid;
        estimate_thread_cfg(src.cols, src.rows, threads, grid);
        set_kernel_consts(src.cols, src.rows, threads, grid);

        R* buf_ = (R*)buf.ptr(0);

        sum_kernel<T, R, threads_x * threads_y><<<grid, threads>>>((const DevMem2D_<T>)src, buf_);
        sum_pass2_kernel<T, R, threads_x * threads_y><<<1, threads_x * threads_y>>>(
                buf_, grid.x * grid.y);
        cudaSafeCall(cudaThreadSynchronize());

        R result = 0;
        cudaSafeCall(cudaMemcpy(&result, buf_, result, cudaMemcpyDeviceToHost));
        sum[0] = result;
    }  

    template void sum_multipass_caller<unsigned char>(const DevMem2D, PtrStep, double*);
    template void sum_multipass_caller<char>(const DevMem2D, PtrStep, double*);
    template void sum_multipass_caller<unsigned short>(const DevMem2D, PtrStep, double*);
    template void sum_multipass_caller<short>(const DevMem2D, PtrStep, double*);
    template void sum_multipass_caller<int>(const DevMem2D, PtrStep, double*);
    template void sum_multipass_caller<float>(const DevMem2D, PtrStep, double*);


    template <typename T>
    void sum_caller(const DevMem2D src, PtrStep buf, double* sum)
    {
        using namespace sum;
        typedef typename SumType<T>::R R;

        dim3 threads, grid;
        estimate_thread_cfg(src.cols, src.rows, threads, grid);
        set_kernel_consts(src.cols, src.rows, threads, grid);

        R* buf_ = (R*)buf.ptr(0);

        sum_kernel<T, R, threads_x * threads_y><<<grid, threads>>>((const DevMem2D_<T>)src, buf_);
        cudaSafeCall(cudaThreadSynchronize());

        R result = 0;
        cudaSafeCall(cudaMemcpy(&result, buf_, sizeof(result), cudaMemcpyDeviceToHost));
        sum[0] = result;
    }  

    template void sum_caller<unsigned char>(const DevMem2D, PtrStep, double*);
    template void sum_caller<char>(const DevMem2D, PtrStep, double*);
    template void sum_caller<unsigned short>(const DevMem2D, PtrStep, double*);
    template void sum_caller<short>(const DevMem2D, PtrStep, double*);
    template void sum_caller<int>(const DevMem2D, PtrStep, double*);
    template void sum_caller<float>(const DevMem2D, PtrStep, double*);
    template void sum_caller<double>(const DevMem2D, PtrStep, double*);
}}}

