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

#include "cuda_shared.hpp"
#include "transform.hpp"
#include "limits_gpu.hpp"

using namespace cv::gpu;
using namespace cv::gpu::device;

#ifndef CV_PI
#define CV_PI   3.1415926535897932384626433832795f
#endif

//////////////////////////////////////////////////////////////////////////////////////
// Cart <-> Polar

namespace cv { namespace gpu { namespace mathfunc
{
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
        __device__ bool operator()(int y, int x) { return mask.ptr(y)[x]; }
        PtrStep mask;
    };
    struct MaskTrue { __device__ bool operator()(int y, int x) { return true; } };

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
    template <> struct MinMaxTypeTraits<signed char> { typedef int best_type; };
    template <> struct MinMaxTypeTraits<unsigned short> { typedef int best_type; };
    template <> struct MinMaxTypeTraits<signed short> { typedef int best_type; };
    template <> struct MinMaxTypeTraits<int> { typedef int best_type; };
    template <> struct MinMaxTypeTraits<float> { typedef float best_type; };
    template <> struct MinMaxTypeTraits<double> { typedef double best_type; };

    // Available optimization operations
    enum { OP_MIN, OP_MAX };

    namespace minmax 
    {

    __constant__ int ctwidth;
    __constant__ int ctheight;

    static const unsigned int czero = 0;

    // Global counter of blocks finished its work
    __device__ unsigned int blocks_finished;


    // Estimates good thread configuration
    //  - threads variable satisfies to threads.x * threads.y == 256
    void estimate_thread_cfg(dim3& threads, dim3& grid)
    {
        threads = dim3(64, 4);
        grid = dim3(6, 5);
    }


    // Returns required buffer sizes
    void get_buf_size_required(int elem_size, int& cols, int& rows)
    {
        dim3 threads, grid;
        estimate_thread_cfg(threads, grid);
        cols = grid.x * grid.y * elem_size; 
        rows = 2;
    }


    // Estimates device constants which are used in the kernels using specified thread configuration
    void estimate_kernel_consts(int cols, int rows, const dim3& threads, const dim3& grid)
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


    template <int nthreads, typename T>
    __global__ void min_max_kernel(int cols, int rows, const PtrStep src, T* minval, T* maxval)
    {
        typedef typename MinMaxTypeTraits<T>::best_type best_type;
        __shared__ best_type sminval[nthreads];
        __shared__ best_type smaxval[nthreads];

        unsigned int x0 = blockIdx.x * blockDim.x * ctwidth + threadIdx.x;
        unsigned int y0 = blockIdx.y * blockDim.y * ctheight + threadIdx.y;
        unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

        T val;
        T mymin = numeric_limits_gpu<T>::max();
        T mymax = numeric_limits_gpu<T>::min();
        for (unsigned int y = 0; y < ctheight && y0 + y * blockDim.y < rows; ++y)
        {
            const T* ptr = (const T*)src.ptr(y0 + y * blockDim.y);
            for (unsigned int x = 0; x < ctwidth && x0 + x * blockDim.x < cols; ++x)
            {
                val = ptr[x0 + x * blockDim.x];
                mymin = min(mymin, val);
                mymax = max(mymax, val);
            }
        }

        sminval[tid] = mymin;
        smaxval[tid] = mymax;

        __syncthreads();

        if (nthreads >= 512) if (tid < 256) { merge(tid, 256, sminval, smaxval); __syncthreads(); }
        if (nthreads >= 256) if (tid < 128) { merge(tid, 128, sminval, smaxval); __syncthreads(); }
        if (nthreads >= 128) if (tid < 64) { merge(tid, 64, sminval, smaxval); __syncthreads(); }

        if (tid < 32)
        {
            if (nthreads >= 64) merge(tid, 32, sminval, smaxval);
            if (nthreads >= 32) merge(tid, 16, sminval, smaxval);
            if (nthreads >= 16) merge(tid, 8, sminval, smaxval);
            if (nthreads >= 8) merge(tid, 4, sminval, smaxval);
            if (nthreads >= 4) merge(tid, 2, sminval, smaxval);
            if (nthreads >= 2) merge(tid, 1, sminval, smaxval);
        }

        __syncthreads();

        if (tid == 0) 
        {
            minval[blockIdx.y * gridDim.x + blockIdx.x] = (T)sminval[0];
            maxval[blockIdx.y * gridDim.x + blockIdx.x] = (T)smaxval[0];
        }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 110
        
        // Process partial results in the first thread of the last block      
        if ((gridDim.x > 1 || gridDim.y > 1) && tid == 0)
        {
            __threadfence();
            if (atomicInc(&blocks_finished, gridDim.x * gridDim.y) == gridDim.x * gridDim.y - 1)
            {
                mymin = numeric_limits_gpu<T>::max();
                mymax = numeric_limits_gpu<T>::min();
                for (unsigned int i = 0; i < gridDim.x * gridDim.y; ++i)
                {                    
                    mymin = min(mymin, minval[i]);
                    mymax = max(mymax, maxval[i]);
                }
                minval[0] = mymin;
                maxval[0] = mymax;
            }
        }

#endif
    }

   
    template <typename T>
    void min_max_caller(const DevMem2D src, double* minval, double* maxval, PtrStep buf)
    {
        dim3 threads, grid;
        estimate_thread_cfg(threads, grid);
        estimate_kernel_consts(src.cols, src.rows, threads, grid);

        T* minval_buf = (T*)buf.ptr(0);
        T* maxval_buf = (T*)buf.ptr(1);

        cudaSafeCall(cudaMemcpyToSymbol(blocks_finished, &czero, sizeof(blocks_finished)));
        min_max_kernel<256, T><<<grid, threads>>>(src.cols, src.rows, src, minval_buf, maxval_buf);
        cudaSafeCall(cudaThreadSynchronize());

        T minval_, maxval_;
        cudaSafeCall(cudaMemcpy(&minval_, minval_buf, sizeof(T), cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaMemcpy(&maxval_, maxval_buf, sizeof(T), cudaMemcpyDeviceToHost));
        *minval = minval_;
        *maxval = maxval_;
    }  

    template void min_max_caller<unsigned char>(const DevMem2D, double*, double*, PtrStep);
    template void min_max_caller<signed char>(const DevMem2D, double*, double*, PtrStep);
    template void min_max_caller<unsigned short>(const DevMem2D, double*, double*, PtrStep);
    template void min_max_caller<signed short>(const DevMem2D, double*, double*, PtrStep);
    template void min_max_caller<int>(const DevMem2D, double*, double*, PtrStep);
    template void min_max_caller<float>(const DevMem2D, double*, double*, PtrStep);
    template void min_max_caller<double>(const DevMem2D, double*, double*, PtrStep);


    // This kernel will be used only when compute capability is 1.0
    template <typename T>
    __global__ void min_max_kernel_2ndstep(T* minval, T* maxval, int size)
    {
        T val;
        T mymin = numeric_limits_gpu<T>::max();
        T mymax = numeric_limits_gpu<T>::min();
        for (unsigned int i = 0; i < size; ++i)
        {     
            val = minval[i]; if (val < mymin) mymin = val;
            val = maxval[i]; if (val > mymax) mymax = val;
        }
        minval[0] = mymin;
        maxval[0] = mymax;
    }


    template <typename T>
    void min_max_caller_2steps(const DevMem2D src, double* minval, double* maxval, PtrStep buf)
    {
        dim3 threads, grid;
        estimate_thread_cfg(threads, grid);
        estimate_kernel_consts(src.cols, src.rows, threads, grid);

        T* minval_buf = (T*)buf.ptr(0);
        T* maxval_buf = (T*)buf.ptr(1);

        cudaSafeCall(cudaMemcpyToSymbol(blocks_finished, &czero, sizeof(blocks_finished)));
        min_max_kernel<256, T><<<grid, threads>>>(src.cols, src.rows, src, minval_buf, maxval_buf);
        min_max_kernel_2ndstep<T><<<1, 1>>>(minval_buf, maxval_buf, grid.x * grid.y);
        cudaSafeCall(cudaThreadSynchronize());

        T minval_, maxval_;
        cudaSafeCall(cudaMemcpy(&minval_, minval_buf, sizeof(T), cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaMemcpy(&maxval_, maxval_buf, sizeof(T), cudaMemcpyDeviceToHost));
        *minval = minval_;
        *maxval = maxval_;
    }

    template void min_max_caller_2steps<unsigned char>(const DevMem2D, double*, double*, PtrStep);
    template void min_max_caller_2steps<signed char>(const DevMem2D, double*, double*, PtrStep);
    template void min_max_caller_2steps<unsigned short>(const DevMem2D, double*, double*, PtrStep);
    template void min_max_caller_2steps<signed short>(const DevMem2D, double*, double*, PtrStep);
    template void min_max_caller_2steps<int>(const DevMem2D, double*, double*, PtrStep);
    template void min_max_caller_2steps<float>(const DevMem2D, double*, double*, PtrStep);

    } // namespace minmax


    namespace minmaxloc {

    template <typename T, int op> struct OptLoc {};
    
    template <typename T>
    struct OptLoc<T, OP_MIN> 
    {
        static __device__ void call(unsigned int tid, unsigned int offset, volatile T* optval, volatile unsigned int* optloc)
        {
            T val = optval[tid + offset];
            if (val < optval[tid])
            {
                optval[tid] = val;
                optloc[tid] = optloc[tid + offset];
            }
        }
    };

    template <typename T>
    struct OptLoc<T, OP_MAX> 
    {
        static __device__ void call(unsigned int tid, unsigned int offset, volatile T* optval, volatile unsigned int* optloc)
        {
            T val = optval[tid + offset];
            if (val > optval[tid])
            {
                optval[tid] = val;
                optloc[tid] = optloc[tid + offset];
            }
        }
    };

    template <int nthreads, int op, typename T>
    __global__ void opt_loc_init_kernel(int cols, int rows, const PtrStep src, PtrStep optval, PtrStep optloc)
    {
        typedef typename MinMaxTypeTraits<T>::best_type best_type;
        __shared__ best_type soptval[nthreads];
        __shared__ unsigned int soptloc[nthreads];

        unsigned int x0 = blockIdx.x * blockDim.x;
        unsigned int y0 = blockIdx.y * blockDim.y;
        unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

        if (x0 + threadIdx.x < cols && y0 + threadIdx.y < rows)
        {
            soptval[tid] = ((const T*)src.ptr(y0 + threadIdx.y))[x0 + threadIdx.x];
            soptloc[tid] = (y0 + threadIdx.y) * cols + x0 + threadIdx.x;
        }
        else
        {
            soptval[tid] = ((const T*)src.ptr(y0))[x0];
            soptloc[tid] = y0 * cols + x0;
        }

        __syncthreads();

        if (nthreads >= 512) if (tid < 256) { OptLoc<best_type, op>::call(tid, 256, soptval, soptloc); __syncthreads(); }
        if (nthreads >= 256) if (tid < 128) { OptLoc<best_type, op>::call(tid, 128, soptval, soptloc); __syncthreads(); }
        if (nthreads >= 128) if (tid < 64) { OptLoc<best_type, op>::call(tid, 64, soptval, soptloc); __syncthreads(); }

        if (tid < 32)
        {
            if (nthreads >= 64) OptLoc<best_type, op>::call(tid, 32, soptval, soptloc);
            if (nthreads >= 32) OptLoc<best_type, op>::call(tid, 16, soptval, soptloc);
            if (nthreads >= 16) OptLoc<best_type, op>::call(tid, 8, soptval, soptloc);
            if (nthreads >= 8) OptLoc<best_type, op>::call(tid, 4, soptval, soptloc);
            if (nthreads >= 4) OptLoc<best_type, op>::call(tid, 2, soptval, soptloc);
            if (nthreads >= 2) OptLoc<best_type, op>::call(tid, 1, soptval, soptloc);
        }

        if (tid == 0) 
        {
            ((T*)optval.ptr(blockIdx.y))[blockIdx.x] = (T)soptval[0];
            ((unsigned int*)optloc.ptr(blockIdx.y))[blockIdx.x] = soptloc[0];
        }
    }

    template <int nthreads, int op, typename T>
    __global__ void opt_loc_kernel(int cols, int rows, const PtrStep src, const PtrStep loc, PtrStep optval, PtrStep optloc)
    {
        typedef typename MinMaxTypeTraits<T>::best_type best_type;
        __shared__ best_type soptval[nthreads];
        __shared__ unsigned int soptloc[nthreads];

        unsigned int x0 = blockIdx.x * blockDim.x;
        unsigned int y0 = blockIdx.y * blockDim.y;
        unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

        if (x0 + threadIdx.x < cols && y0 + threadIdx.y < rows)
        {
            soptval[tid] = ((const T*)src.ptr(y0 + threadIdx.y))[x0 + threadIdx.x];
            soptloc[tid] = ((const unsigned int*)loc.ptr(y0 + threadIdx.y))[x0 + threadIdx.x];
        }
        else
        {
            soptval[tid] = ((const T*)src.ptr(y0))[x0];
            soptloc[tid] = ((const unsigned int*)loc.ptr(y0))[x0];
        }

        __syncthreads();

        if (nthreads >= 512) if (tid < 256) { OptLoc<best_type, op>::call(tid, 256, soptval, soptloc); __syncthreads(); }
        if (nthreads >= 256) if (tid < 128) { OptLoc<best_type, op>::call(tid, 128, soptval, soptloc); __syncthreads(); }
        if (nthreads >= 128) if (tid < 64) { OptLoc<best_type, op>::call(tid, 64, soptval, soptloc); __syncthreads(); }

        if (tid < 32)
        {
            if (nthreads >= 64) OptLoc<best_type, op>::call(tid, 32, soptval, soptloc);
            if (nthreads >= 32) OptLoc<best_type, op>::call(tid, 16, soptval, soptloc);
            if (nthreads >= 16) OptLoc<best_type, op>::call(tid, 8, soptval, soptloc);
            if (nthreads >= 8) OptLoc<best_type, op>::call(tid, 4, soptval, soptloc);
            if (nthreads >= 4) OptLoc<best_type, op>::call(tid, 2, soptval, soptloc);
            if (nthreads >= 2) OptLoc<best_type, op>::call(tid, 1, soptval, soptloc);
        }

        if (tid == 0) 
        {
            ((T*)optval.ptr(blockIdx.y))[blockIdx.x] = (T)soptval[0];
            ((unsigned int*)optloc.ptr(blockIdx.y))[blockIdx.x] = soptloc[0];
        }
    }

    template <typename T>
    void min_max_loc_caller(const DevMem2D src, double* minval, double* maxval, int* minlocx, int* minlocy, 
                                                                                int* maxlocx, int* maxlocy)
    {
        dim3 threads(32, 8);

        // Allocate memory for aux. buffers

        DevMem2D minval_buf[2]; 
        minval_buf[0].cols = divUp(src.cols, threads.x); 
        minval_buf[0].rows = divUp(src.rows, threads.y);
        minval_buf[1].cols = divUp(minval_buf[0].cols, threads.x); 
        minval_buf[1].rows = divUp(minval_buf[0].rows, threads.y);
        cudaSafeCall(cudaMallocPitch(&minval_buf[0].data, &minval_buf[0].step, minval_buf[0].cols * sizeof(T), minval_buf[0].rows));
        cudaSafeCall(cudaMallocPitch(&minval_buf[1].data, &minval_buf[1].step, minval_buf[1].cols * sizeof(T), minval_buf[1].rows));

        DevMem2D maxval_buf[2];        
        maxval_buf[0].cols = divUp(src.cols, threads.x); 
        maxval_buf[0].rows = divUp(src.rows, threads.y);
        maxval_buf[1].cols = divUp(maxval_buf[0].cols, threads.x); 
        maxval_buf[1].rows = divUp(maxval_buf[0].rows, threads.y);
        cudaSafeCall(cudaMallocPitch(&maxval_buf[0].data, &maxval_buf[0].step, maxval_buf[0].cols * sizeof(T), maxval_buf[0].rows));
        cudaSafeCall(cudaMallocPitch(&maxval_buf[1].data, &maxval_buf[1].step, maxval_buf[1].cols * sizeof(T), maxval_buf[1].rows));

        DevMem2D minloc_buf[2]; 
        minloc_buf[0].cols = divUp(src.cols, threads.x); 
        minloc_buf[0].rows = divUp(src.rows, threads.y);
        minloc_buf[1].cols = divUp(minloc_buf[0].cols, threads.x); 
        minloc_buf[1].rows = divUp(minloc_buf[0].rows, threads.y);
        cudaSafeCall(cudaMallocPitch(&minloc_buf[0].data, &minloc_buf[0].step, minloc_buf[0].cols * sizeof(int), minloc_buf[0].rows));
        cudaSafeCall(cudaMallocPitch(&minloc_buf[1].data, &minloc_buf[1].step, minloc_buf[1].cols * sizeof(int), minloc_buf[1].rows));

        DevMem2D maxloc_buf[2]; 
        maxloc_buf[0].cols = divUp(src.cols, threads.x); 
        maxloc_buf[0].rows = divUp(src.rows, threads.y);
        maxloc_buf[1].cols = divUp(maxloc_buf[0].cols, threads.x); 
        maxloc_buf[1].rows = divUp(maxloc_buf[0].rows, threads.y);
        cudaSafeCall(cudaMallocPitch(&maxloc_buf[0].data, &maxloc_buf[0].step, maxloc_buf[0].cols * sizeof(int), maxloc_buf[0].rows));
        cudaSafeCall(cudaMallocPitch(&maxloc_buf[1].data, &maxloc_buf[1].step, maxloc_buf[1].cols * sizeof(int), maxloc_buf[1].rows));

        int curbuf = 0;
        dim3 cursize(src.cols, src.rows);
        dim3 grid(divUp(cursize.x, threads.x), divUp(cursize.y, threads.y));

        opt_loc_init_kernel<256, OP_MIN, T><<<grid, threads>>>(cursize.x, cursize.y, src, minval_buf[curbuf], minloc_buf[curbuf]);
        opt_loc_init_kernel<256, OP_MAX, T><<<grid, threads>>>(cursize.x, cursize.y, src, maxval_buf[curbuf], maxloc_buf[curbuf]);
        cursize = grid;
      
        while (cursize.x > 1 || cursize.y > 1)
        {
            grid.x = divUp(cursize.x, threads.x); 
            grid.y = divUp(cursize.y, threads.y);  
            opt_loc_kernel<256, OP_MIN, T><<<grid, threads>>>(cursize.x, cursize.y, minval_buf[curbuf], minloc_buf[curbuf], 
                                                              minval_buf[1 - curbuf], minloc_buf[1 - curbuf]);
            opt_loc_kernel<256, OP_MAX, T><<<grid, threads>>>(cursize.x, cursize.y, maxval_buf[curbuf], maxloc_buf[curbuf], 
                                                              maxval_buf[1 - curbuf], maxloc_buf[1 - curbuf]);
            curbuf = 1 - curbuf;
            cursize = grid;
        }

        cudaSafeCall(cudaThreadSynchronize());

        // Copy results from device to host

        T minval_, maxval_;
        cudaSafeCall(cudaMemcpy(&minval_, minval_buf[curbuf].ptr(0), sizeof(T), cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaMemcpy(&maxval_, maxval_buf[curbuf].ptr(0), sizeof(T), cudaMemcpyDeviceToHost));
        *minval = minval_;
        *maxval = maxval_;

        unsigned int minloc, maxloc;
        cudaSafeCall(cudaMemcpy(&minloc, minloc_buf[curbuf].ptr(0), sizeof(int), cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaMemcpy(&maxloc, maxloc_buf[curbuf].ptr(0), sizeof(int), cudaMemcpyDeviceToHost));
        *minlocy = minloc / src.cols; *minlocx = minloc - *minlocy * src.cols;
        *maxlocy = maxloc / src.cols; *maxlocx = maxloc - *maxlocy * src.cols;

        // Release aux. buffers
        cudaSafeCall(cudaFree(minval_buf[0].data));
        cudaSafeCall(cudaFree(minval_buf[1].data));
        cudaSafeCall(cudaFree(maxval_buf[0].data));
        cudaSafeCall(cudaFree(maxval_buf[1].data));
        cudaSafeCall(cudaFree(minloc_buf[0].data));
        cudaSafeCall(cudaFree(minloc_buf[1].data));
        cudaSafeCall(cudaFree(maxloc_buf[0].data));
        cudaSafeCall(cudaFree(maxloc_buf[1].data));
    }

    template void min_max_loc_caller<unsigned char>(const DevMem2D, double*, double*, int*, int*, int*, int*);
    template void min_max_loc_caller<signed char>(const DevMem2D, double*, double*, int*, int*, int*, int*);
    template void min_max_loc_caller<unsigned short>(const DevMem2D, double*, double*, int*, int*, int*, int*);
    template void min_max_loc_caller<signed short>(const DevMem2D, double*, double*, int*, int*, int*, int*);
    template void min_max_loc_caller<int>(const DevMem2D, double*, double*, int*, int*, int*, int*);
    template void min_max_loc_caller<float>(const DevMem2D, double*, double*, int*, int*, int*, int*);
    template void min_max_loc_caller<double>(const DevMem2D, double*, double*, int*, int*, int*, int*);

    } // namespace minmaxloc

}}}
