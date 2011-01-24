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

#include "opencv2/gpu/device/vecmath.hpp"
#include "opencv2/gpu/device/transform.hpp"
#include "opencv2/gpu/device/saturate_cast.hpp"
#include "internal_shared.hpp"

using namespace cv::gpu;
using namespace cv::gpu::device;

namespace cv { namespace gpu { namespace mathfunc
{

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


    //////////////////////////////////////////////////////////////////////////
    // Unary bitwise logical matrix operations

    enum { UN_OP_NOT };

    template <typename T, int opid>
    struct UnOp;

    template <typename T>
    struct UnOp<T, UN_OP_NOT>
    { 
        static __device__ T call(T v) { return ~v; }
    };


    template <int opid>
    __global__ void bitwiseUnOpKernel(int rows, int width, const PtrStep src, PtrStep dst)
    {
        const int x = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
        const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (y < rows) 
        {
            uchar* dst_ptr = dst.ptr(y) + x;
            const uchar* src_ptr = src.ptr(y) + x;
            if (x + sizeof(uint) - 1 < width)
            {
                *(uint*)dst_ptr = UnOp<uint, opid>::call(*(uint*)src_ptr);
            }
            else
            {
                const uchar* src_end = src.ptr(y) + width;
                while (src_ptr < src_end)
                {
                    *dst_ptr++ = UnOp<uchar, opid>::call(*src_ptr++);
                }
            }
        }
    }


    template <int opid>
    void bitwiseUnOp(int rows, int width, const PtrStep src, PtrStep dst, 
                     cudaStream_t stream)
    {
        dim3 threads(16, 16);
        dim3 grid(divUp(width, threads.x * sizeof(uint)), 
                  divUp(rows, threads.y));

        bitwiseUnOpKernel<opid><<<grid, threads>>>(rows, width, src, dst);

        if (stream == 0) 
            cudaSafeCall(cudaThreadSynchronize());
    }


    template <typename T, int opid>
    __global__ void bitwiseUnOpKernel(int rows, int cols, int cn, const PtrStep src, 
                                      const PtrStep mask, PtrStep dst)
    {
        const int x = blockDim.x * blockIdx.x + threadIdx.x;
        const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x < cols && y < rows && mask.ptr(y)[x / cn]) 
        {
            T* dst_row = (T*)dst.ptr(y);
            const T* src_row = (const T*)src.ptr(y);

            dst_row[x] = UnOp<T, opid>::call(src_row[x]);
        }
    }


    template <typename T, int opid>
    void bitwiseUnOp(int rows, int cols, int cn, const PtrStep src, 
                     const PtrStep mask, PtrStep dst, cudaStream_t stream)
    {
        dim3 threads(16, 16);
        dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));

        bitwiseUnOpKernel<T, opid><<<grid, threads>>>(rows, cols, cn, src, mask, dst); 

        if (stream == 0) 
            cudaSafeCall(cudaThreadSynchronize());
    }


    void bitwiseNotCaller(int rows, int cols, int elem_size1, int cn, 
                          const PtrStep src, PtrStep dst, cudaStream_t stream)
    {
        bitwiseUnOp<UN_OP_NOT>(rows, cols * elem_size1 * cn, src, dst, stream);
    }


    template <typename T>
    void bitwiseMaskNotCaller(int rows, int cols, int cn, const PtrStep src, 
                              const PtrStep mask, PtrStep dst, cudaStream_t stream)
    {
        bitwiseUnOp<T, UN_OP_NOT>(rows, cols * cn, cn, src, mask, dst, stream);
    }

    template void bitwiseMaskNotCaller<uchar>(int, int, int, const PtrStep, const PtrStep, PtrStep, cudaStream_t);
    template void bitwiseMaskNotCaller<ushort>(int, int, int, const PtrStep, const PtrStep, PtrStep, cudaStream_t);
    template void bitwiseMaskNotCaller<uint>(int, int, int, const PtrStep, const PtrStep, PtrStep, cudaStream_t);


    //////////////////////////////////////////////////////////////////////////
    // Binary bitwise logical matrix operations

    enum { BIN_OP_OR, BIN_OP_AND, BIN_OP_XOR };

    template <typename T, int opid>
    struct BinOp;

    template <typename T>
    struct BinOp<T, BIN_OP_OR>
    { 
        static __device__ T call(T a, T b) { return a | b; } 
    };


    template <typename T>
    struct BinOp<T, BIN_OP_AND>
    { 
        static __device__ T call(T a, T b) { return a & b; } 
    };

    template <typename T>
    struct BinOp<T, BIN_OP_XOR>
    { 
        static __device__ T call(T a, T b) { return a ^ b; } 
    };


    template <int opid>
    __global__ void bitwiseBinOpKernel(int rows, int width, const PtrStep src1, 
                                       const PtrStep src2, PtrStep dst)
    {
        const int x = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
        const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (y < rows) 
        {
            uchar* dst_ptr = dst.ptr(y) + x;
            const uchar* src1_ptr = src1.ptr(y) + x;
            const uchar* src2_ptr = src2.ptr(y) + x;

            if (x + sizeof(uint) - 1 < width)
            {
                *(uint*)dst_ptr = BinOp<uint, opid>::call(*(uint*)src1_ptr, *(uint*)src2_ptr);
            }
            else
            {
                const uchar* src1_end = src1.ptr(y) + width;
                while (src1_ptr < src1_end)
                {
                    *dst_ptr++ = BinOp<uchar, opid>::call(*src1_ptr++, *src2_ptr++);
                }
            }
        }
    }


    template <int opid>
    void bitwiseBinOp(int rows, int width, const PtrStep src1, const PtrStep src2, 
                      PtrStep dst, cudaStream_t stream)
    {
        dim3 threads(16, 16);
        dim3 grid(divUp(width, threads.x * sizeof(uint)), divUp(rows, threads.y));

        bitwiseBinOpKernel<opid><<<grid, threads>>>(rows, width, src1, src2, dst);

        if (stream == 0) 
            cudaSafeCall(cudaThreadSynchronize());
    }


    template <typename T, int opid>
    __global__ void bitwiseBinOpKernel(
            int rows, int cols, int cn, const PtrStep src1, const PtrStep src2, 
            const PtrStep mask, PtrStep dst)
    {
        const int x = blockDim.x * blockIdx.x + threadIdx.x;
        const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x < cols && y < rows && mask.ptr(y)[x / cn]) 
        {
            T* dst_row = (T*)dst.ptr(y);
            const T* src1_row = (const T*)src1.ptr(y);
            const T* src2_row = (const T*)src2.ptr(y);

            dst_row[x] = BinOp<T, opid>::call(src1_row[x], src2_row[x]);
        }
    }


    template <typename T, int opid>
    void bitwiseBinOp(int rows, int cols, int cn, const PtrStep src1, const PtrStep src2, 
                        const PtrStep mask, PtrStep dst, cudaStream_t stream)
    {
        dim3 threads(16, 16);
        dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));

        bitwiseBinOpKernel<T, opid><<<grid, threads>>>(rows, cols, cn, src1, src2, mask, dst); 

        if (stream == 0) 
            cudaSafeCall(cudaThreadSynchronize());
    }


    void bitwiseOrCaller(int rows, int cols, int elem_size1, int cn, const PtrStep src1, 
                         const PtrStep src2, PtrStep dst, cudaStream_t stream)
    {
        bitwiseBinOp<BIN_OP_OR>(rows, cols * elem_size1 * cn, src1, src2, dst, stream);
    }


    template <typename T>
    void bitwiseMaskOrCaller(int rows, int cols, int cn, const PtrStep src1, const PtrStep src2, 
                             const PtrStep mask, PtrStep dst, cudaStream_t stream)
    {
        bitwiseBinOp<T, BIN_OP_OR>(rows, cols * cn, cn, src1, src2, mask, dst, stream);
    }

    template void bitwiseMaskOrCaller<uchar>(int, int, int, const PtrStep, const PtrStep, const PtrStep, PtrStep, cudaStream_t);
    template void bitwiseMaskOrCaller<ushort>(int, int, int, const PtrStep, const PtrStep, const PtrStep, PtrStep, cudaStream_t);
    template void bitwiseMaskOrCaller<uint>(int, int, int, const PtrStep, const PtrStep, const PtrStep, PtrStep, cudaStream_t);


    void bitwiseAndCaller(int rows, int cols, int elem_size1, int cn, const PtrStep src1, 
                          const PtrStep src2, PtrStep dst, cudaStream_t stream)
    {
        bitwiseBinOp<BIN_OP_AND>(rows, cols * elem_size1 * cn, src1, src2, dst, stream);
    }


    template <typename T>
    void bitwiseMaskAndCaller(int rows, int cols, int cn, const PtrStep src1, const PtrStep src2, 
                              const PtrStep mask, PtrStep dst, cudaStream_t stream)
    {
        bitwiseBinOp<T, BIN_OP_AND>(rows, cols * cn, cn, src1, src2, mask, dst, stream);
    }

    template void bitwiseMaskAndCaller<uchar>(int, int, int, const PtrStep, const PtrStep, const PtrStep, PtrStep, cudaStream_t);
    template void bitwiseMaskAndCaller<ushort>(int, int, int, const PtrStep, const PtrStep, const PtrStep, PtrStep, cudaStream_t);
    template void bitwiseMaskAndCaller<uint>(int, int, int, const PtrStep, const PtrStep, const PtrStep, PtrStep, cudaStream_t);


    void bitwiseXorCaller(int rows, int cols, int elem_size1, int cn, const PtrStep src1, 
                          const PtrStep src2, PtrStep dst, cudaStream_t stream)
    {
        bitwiseBinOp<BIN_OP_XOR>(rows, cols * elem_size1 * cn, src1, src2, dst, stream);
    }


    template <typename T>
    void bitwiseMaskXorCaller(int rows, int cols, int cn, const PtrStep src1, const PtrStep src2, 
                              const PtrStep mask, PtrStep dst, cudaStream_t stream)
    {
        bitwiseBinOp<T, BIN_OP_XOR>(rows, cols * cn, cn, src1, src2, mask, dst, stream);
    }

    template void bitwiseMaskXorCaller<uchar>(int, int, int, const PtrStep, const PtrStep, const PtrStep, PtrStep, cudaStream_t);
    template void bitwiseMaskXorCaller<ushort>(int, int, int, const PtrStep, const PtrStep, const PtrStep, PtrStep, cudaStream_t);
    template void bitwiseMaskXorCaller<uint>(int, int, int, const PtrStep, const PtrStep, const PtrStep, PtrStep, cudaStream_t);


    //////////////////////////////////////////////////////////////////////////
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

    
    //////////////////////////////////////////////////////////////////////////
    // threshold

    class ThreshOp
    {
    public:
        ThreshOp(float thresh_, float maxVal_) : thresh(thresh_), maxVal(maxVal_) {}

    protected:
        float thresh;
        float maxVal;
    };

    class ThreshBinary : public ThreshOp
    {
    public:
        ThreshBinary(float thresh_, float maxVal_) : ThreshOp(thresh_, maxVal_) {}

        template<typename T>
        __device__ T operator()(const T& src) const
        {
            return (float)src > thresh ? saturate_cast<T>(maxVal) : 0;
        }
    };

    class ThreshBinaryInv : public ThreshOp
    {
    public:
        ThreshBinaryInv(float thresh_, float maxVal_) : ThreshOp(thresh_, maxVal_) {}

        template<typename T>
        __device__ T operator()(const T& src) const
        {
            return (float)src > thresh ? 0 : saturate_cast<T>(maxVal);
        }
    };

    class ThreshTrunc : public ThreshOp
    {
    public:
        ThreshTrunc(float thresh_, float maxVal_) : ThreshOp(thresh_, maxVal_) {}

        template<typename T>
        __device__ T operator()(const T& src) const
        {
            return saturate_cast<T>(fmin((float)src, thresh));
        }
    };

    class ThreshToZero : public ThreshOp
    {
    public:
        ThreshToZero(float thresh_, float maxVal_) : ThreshOp(thresh_, maxVal_) {}

        template<typename T>
        __device__ T operator()(const T& src) const
        {
            return (float)src > thresh ? src : 0;
        }
    };

    class ThreshToZeroInv : public ThreshOp
    {
    public:
        ThreshToZeroInv(float thresh_, float maxVal_) : ThreshOp(thresh_, maxVal_) {}

        template<typename T>
        __device__ T operator()(const T& src) const
        {
            return (float)src > thresh ? 0 : src;
        }
    };

    template <class Op, typename T>
    void threshold_caller(const DevMem2D_<T>& src, const DevMem2D_<T>& dst, float thresh, float maxVal, 
        cudaStream_t stream)
    {
        Op op(thresh, maxVal);
        transform(src, dst, op, stream);
    }

    template <typename T>
    void threshold_gpu(const DevMem2D& src, const DevMem2D& dst, float thresh, float maxVal, int type,
        cudaStream_t stream)
    {
        typedef void (*caller_t)(const DevMem2D_<T>& src, const DevMem2D_<T>& dst, float thresh, float maxVal, 
            cudaStream_t stream);

        static const caller_t callers[] = 
        {
            threshold_caller<ThreshBinary, T>, 
            threshold_caller<ThreshBinaryInv, T>, 
            threshold_caller<ThreshTrunc, T>, 
            threshold_caller<ThreshToZero, T>, 
            threshold_caller<ThreshToZeroInv, T>
        };

        callers[type]((DevMem2D_<T>)src, (DevMem2D_<T>)dst, thresh, maxVal, stream);
    }

    template void threshold_gpu<uchar>(const DevMem2D& src, const DevMem2D& dst, float thresh, float maxVal, int type, cudaStream_t stream);
    template void threshold_gpu<schar>(const DevMem2D& src, const DevMem2D& dst, float thresh, float maxVal, int type, cudaStream_t stream);
    template void threshold_gpu<ushort>(const DevMem2D& src, const DevMem2D& dst, float thresh, float maxVal, int type, cudaStream_t stream);
    template void threshold_gpu<short>(const DevMem2D& src, const DevMem2D& dst, float thresh, float maxVal, int type, cudaStream_t stream);
    template void threshold_gpu<int>(const DevMem2D& src, const DevMem2D& dst, float thresh, float maxVal, int type, cudaStream_t stream);
    template void threshold_gpu<float>(const DevMem2D& src, const DevMem2D& dst, float thresh, float maxVal, int type, cudaStream_t stream);
}}}
