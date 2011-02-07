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

#ifndef __OPENCV_GPU_TRANSFORM_HPP__
#define __OPENCV_GPU_TRANSFORM_HPP__

#include "internal_shared.hpp"
#include "vecmath.hpp"

namespace cv { namespace gpu { namespace device
{
    //! Mask accessor

    class MaskReader
    {
    public:
        explicit MaskReader(const PtrStep& mask_): mask(mask_) {}

        __device__ bool operator()(int y, int x) const { return mask.ptr(y)[x]; }

    private:
        PtrStep mask;
    };

    struct NoMask 
    {
        __device__ bool operator()(int y, int x) const { return true; } 
    };

    //! Read Write Traits

    template <size_t src_elem_size, size_t dst_elem_size>
    struct UnReadWriteTraits_
    {
        enum {shift=1};
    };
    template <size_t src_elem_size>
    struct UnReadWriteTraits_<src_elem_size, 1>
    {
        enum {shift=4};
    };
    template <size_t src_elem_size>
    struct UnReadWriteTraits_<src_elem_size, 2>
    {
        enum {shift=2};
    };
    template <typename T, typename D> struct UnReadWriteTraits
    {
        enum {shift=UnReadWriteTraits_<sizeof(T), sizeof(D)>::shift};
        
        typedef typename TypeVec<T, shift>::vec_t read_type;
        typedef typename TypeVec<D, shift>::vec_t write_type;
    };

    template <size_t src_elem_size1, size_t src_elem_size2, size_t dst_elem_size>
    struct BinReadWriteTraits_
    {
        enum {shift=1};
    };
    template <size_t src_elem_size1, size_t src_elem_size2>
    struct BinReadWriteTraits_<src_elem_size1, src_elem_size2, 1>
    {
        enum {shift=4};
    };
    template <size_t src_elem_size1, size_t src_elem_size2>
    struct BinReadWriteTraits_<src_elem_size1, src_elem_size2, 2>
    {
        enum {shift=2};
    };
    template <typename T1, typename T2, typename D> struct BinReadWriteTraits
    {
        enum {shift=BinReadWriteTraits_<sizeof(T1), sizeof(T2), sizeof(D)>::shift};

        typedef typename TypeVec<T1, shift>::vec_t read_type1;
        typedef typename TypeVec<T2, shift>::vec_t read_type2;
        typedef typename TypeVec<D , shift>::vec_t write_type;
    };

    //! Transform kernels

    template <int shift> struct OpUnroller;
    template <> struct OpUnroller<1>
    {
        template <typename T, typename D, typename UnOp, typename Mask>
        static __device__ void unroll(const T& src, D& dst, const Mask& mask, UnOp& op, int x_shifted, int y)
        {
            if (mask(y, x_shifted))
                dst.x = op(src.x);
        }

        template <typename T1, typename T2, typename D, typename BinOp, typename Mask>
        static __device__ void unroll(const T1& src1, const T2& src2, D& dst, const Mask& mask, BinOp& op, int x_shifted, int y)
        {
            if (mask(y, x_shifted))
                dst.x = op(src1.x, src2.x);
        }
    };
    template <> struct OpUnroller<2>
    {
        template <typename T, typename D, typename UnOp, typename Mask>
        static __device__ void unroll(const T& src, D& dst, const Mask& mask, UnOp& op, int x_shifted, int y)
        {
            if (mask(y, x_shifted))
                dst.x = op(src.x);
            if (mask(y, x_shifted + 1))
                dst.y = op(src.y);
        }

        template <typename T1, typename T2, typename D, typename BinOp, typename Mask>
        static __device__ void unroll(const T1& src1, const T2& src2, D& dst, const Mask& mask, BinOp& op, int x_shifted, int y)
        {
            if (mask(y, x_shifted))
                dst.x = op(src1.x, src2.x);
            if (mask(y, x_shifted + 1))
                dst.y = op(src1.y, src2.y);
        }
    };
    template <> struct OpUnroller<3>
    {
        template <typename T, typename D, typename UnOp, typename Mask>
        static __device__ void unroll(const T& src, D& dst, const Mask& mask, UnOp& op, int x_shifted, int y)
        {
            if (mask(y, x_shifted))
                dst.x = op(src.x);
            if (mask(y, x_shifted + 1))
                dst.y = op(src.y);
            if (mask(y, x_shifted + 2))
                dst.z = op(src.z);
        }

        template <typename T1, typename T2, typename D, typename BinOp, typename Mask>
        static __device__ void unroll(const T1& src1, const T2& src2, D& dst, const Mask& mask, BinOp& op, int x_shifted, int y)
        {
            if (mask(y, x_shifted))
                dst.x = op(src1.x, src2.x);
            if (mask(y, x_shifted + 1))
                dst.y = op(src1.y, src2.y);
            if (mask(y, x_shifted + 2))
                dst.z = op(src1.z, src2.z);
        }
    };
    template <> struct OpUnroller<4>
    {
        template <typename T, typename D, typename UnOp, typename Mask>
        static __device__ void unroll(const T& src, D& dst, const Mask& mask, UnOp& op, int x_shifted, int y)
        {
            if (mask(y, x_shifted))
                dst.x = op(src.x);
            if (mask(y, x_shifted + 1))
                dst.y = op(src.y);
            if (mask(y, x_shifted + 2))
                dst.z = op(src.z);
            if (mask(y, x_shifted + 3))
                dst.w = op(src.w);
        }

        template <typename T1, typename T2, typename D, typename BinOp, typename Mask>
        static __device__ void unroll(const T1& src1, const T2& src2, D& dst, const Mask& mask, BinOp& op, int x_shifted, int y)
        {
            if (mask(y, x_shifted))
                dst.x = op(src1.x, src2.x);
            if (mask(y, x_shifted + 1))
                dst.y = op(src1.y, src2.y);
            if (mask(y, x_shifted + 2))
                dst.z = op(src1.z, src2.z);
            if (mask(y, x_shifted + 3))
                dst.w = op(src1.w, src2.w);
        }
    };

    template <typename T, typename D, typename UnOp, typename Mask>
    __global__ static void transformSmart(const DevMem2D_<T> src_, PtrStep_<D> dst_, const Mask mask, UnOp op)
    {
        typedef typename UnReadWriteTraits<T, D>::read_type read_type;
        typedef typename UnReadWriteTraits<T, D>::write_type write_type;
        const int shift = UnReadWriteTraits<T, D>::shift;

        const int x = threadIdx.x + blockIdx.x * blockDim.x;
        const int y = threadIdx.y + blockIdx.y * blockDim.y;
        const int x_shifted = x * shift;

        if (y < src_.rows)
        {
            const T* src = src_.ptr(y);
            D* dst = dst_.ptr(y);

            if (x_shifted + shift - 1 < src_.cols)
            {
                read_type src_n_el = ((const read_type*)src)[x];
                write_type dst_n_el;

                OpUnroller<shift>::unroll(src_n_el, dst_n_el, mask, op, x_shifted, y);

                ((write_type*)dst)[x] = dst_n_el;
            }
            else
            {
                for (int real_x = x_shifted; real_x < src_.cols; ++real_x)
                {
                    if (mask(y, real_x))
                        dst[real_x] = op(src[real_x]);
                }
            }
        }
    }

    template <typename T, typename D, typename UnOp, typename Mask>
    static __global__ void transformSimple(const DevMem2D_<T> src, PtrStep_<D> dst, const Mask mask, UnOp op)
    {
		const int x = blockDim.x * blockIdx.x + threadIdx.x;
		const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x < src.cols && y < src.rows && mask(y, x))
        {
            dst.ptr(y)[x] = op(src.ptr(y)[x]);
        }
    }

    template <typename T1, typename T2, typename D, typename BinOp, typename Mask>
    __global__ static void transformSmart(const DevMem2D_<T1> src1_, const PtrStep_<T2> src2_, PtrStep_<D> dst_, 
        const Mask mask, BinOp op)
    {
        typedef typename BinReadWriteTraits<T1, T2, D>::read_type1 read_type1;
        typedef typename BinReadWriteTraits<T1, T2, D>::read_type2 read_type2;
        typedef typename BinReadWriteTraits<T1, T2, D>::write_type write_type;
        const int shift = BinReadWriteTraits<T1, T2, D>::shift;

        const int x = threadIdx.x + blockIdx.x * blockDim.x;
        const int y = threadIdx.y + blockIdx.y * blockDim.y;
        const int x_shifted = x * shift;

        if (y < src1_.rows)
        {
            const T1* src1 = src1_.ptr(y);
            const T2* src2 = src2_.ptr(y);
            D* dst = dst_.ptr(y);

            if (x_shifted + shift - 1 < src1_.cols)
            {
                read_type1 src1_n_el = ((const read_type1*)src1)[x];
                read_type2 src2_n_el = ((const read_type2*)src2)[x];
                write_type dst_n_el;
                
                OpUnroller<shift>::unroll(src1_n_el, src2_n_el, dst_n_el, mask, op, x_shifted, y);

                ((write_type*)dst)[x] = dst_n_el;
            }
            else
            {
                for (int real_x = x_shifted; real_x < src1_.cols; ++real_x)
                {
                    if (mask(y, real_x))
                        dst[real_x] = op(src1[real_x], src2[real_x]);
                }
            }
        }
    }

    template <typename T1, typename T2, typename D, typename BinOp, typename Mask>
    static __global__ void transformSimple(const DevMem2D_<T1> src1, const PtrStep_<T2> src2, PtrStep_<D> dst, 
        const Mask mask, BinOp op)
    {
		const int x = blockDim.x * blockIdx.x + threadIdx.x;
		const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x < src1.cols && y < src1.rows && mask(y, x))
        {
            T1 src1_data = src1.ptr(y)[x];
            T2 src2_data = src2.ptr(y)[x];
            dst.ptr(y)[x] = op(src1_data, src2_data);
        }
    }  
}}}

namespace cv 
{ 
    namespace gpu 
    {
        template <bool UseSmart> struct TransformDispatcher;
        template<> struct TransformDispatcher<false>
        {
            template <typename T, typename D, typename UnOp, typename Mask>
            static void call(const DevMem2D_<T>& src, const DevMem2D_<D>& dst, UnOp op, const Mask& mask, 
                             cudaStream_t stream = 0)
            {
                dim3 threads(16, 16, 1);
                dim3 grid(1, 1, 1);

                grid.x = divUp(src.cols, threads.x);
                grid.y = divUp(src.rows, threads.y);        

                device::transformSimple<T, D><<<grid, threads, 0, stream>>>(src, dst, mask, op);

                if (stream == 0)
                    cudaSafeCall( cudaThreadSynchronize() ); 
            }

            template <typename T1, typename T2, typename D, typename BinOp, typename Mask>
            static void call(const DevMem2D_<T1>& src1, const DevMem2D_<T2>& src2, const DevMem2D_<D>& dst, 
                             BinOp op, const Mask& mask, cudaStream_t stream = 0)
            {
                dim3 threads(16, 16, 1);
                dim3 grid(1, 1, 1);

                grid.x = divUp(src1.cols, threads.x);
                grid.y = divUp(src1.rows, threads.y);        

                device::transformSimple<T1, T2, D><<<grid, threads, 0, stream>>>(src1, src2, dst, mask, op);

                if (stream == 0)
                    cudaSafeCall( cudaThreadSynchronize() );            
            }
        };
        template<> struct TransformDispatcher<true>
        {
            template <typename T, typename D, typename UnOp, typename Mask>
            static void call(const DevMem2D_<T>& src, const DevMem2D_<D>& dst, UnOp op, const Mask& mask, 
                             cudaStream_t stream = 0)
            {
                const int shift = device::UnReadWriteTraits<T, D>::shift;

                dim3 threads(16, 16, 1);
                dim3 grid(1, 1, 1);            

                grid.x = divUp(src.cols, threads.x * shift);
                grid.y = divUp(src.rows, threads.y);        

                device::transformSmart<T, D><<<grid, threads, 0, stream>>>(src, dst, mask, op);

                if (stream == 0)
                    cudaSafeCall( cudaThreadSynchronize() );
            }

            template <typename T1, typename T2, typename D, typename BinOp, typename Mask>
            static void call(const DevMem2D_<T1>& src1, const DevMem2D_<T2>& src2, const DevMem2D_<D>& dst, 
                             BinOp op, const Mask& mask, cudaStream_t stream = 0)
            {
                const int shift = device::BinReadWriteTraits<T1, T2, D>::shift;

                dim3 threads(16, 16, 1);
                dim3 grid(1, 1, 1);

                grid.x = divUp(src1.cols, threads.x * shift);
                grid.y = divUp(src1.rows, threads.y);        

                device::transformSmart<T1, T2, D><<<grid, threads, 0, stream>>>(src1, src2, dst, mask, op);

                if (stream == 0)
                    cudaSafeCall( cudaThreadSynchronize() );            
            }
        };

        template <typename T, typename D, typename UnOp, typename Mask>
        static void transform_caller(const DevMem2D_<T>& src, const DevMem2D_<D>& dst, UnOp op, const Mask& mask, 
            cudaStream_t stream = 0)
        {
            TransformDispatcher<device::VecTraits<T>::cn == 1 && device::VecTraits<D>::cn == 1 && device::UnReadWriteTraits<T, D>::shift != 1>::call(src, dst, op, mask, stream);
        }

        template <typename T, typename D, typename UnOp>
        static void transform(const DevMem2D_<T>& src, const DevMem2D_<D>& dst, UnOp op, cudaStream_t stream = 0)
        {
            transform_caller(src, dst, op, device::NoMask(), stream);
        }
        template <typename T, typename D, typename UnOp>
        static void transform(const DevMem2D_<T>& src, const DevMem2D_<D>& dst, const PtrStep& mask, UnOp op, 
            cudaStream_t stream = 0)
        {
            transform_caller(src, dst, op, device::MaskReader(mask), stream);
        }

        template <typename T1, typename T2, typename D, typename BinOp, typename Mask>
        static void transform_caller(const DevMem2D_<T1>& src1, const DevMem2D_<T2>& src2, const DevMem2D_<D>& dst, 
            BinOp op, const Mask& mask, cudaStream_t stream = 0)
        {
            TransformDispatcher<device::VecTraits<T1>::cn == 1 && device::VecTraits<T2>::cn == 1 && device::VecTraits<D>::cn == 1 && device::BinReadWriteTraits<T1, T2, D>::shift != 1>::call(src1, src2, dst, op, mask, stream);
        }

        template <typename T1, typename T2, typename D, typename BinOp>
        static void transform(const DevMem2D_<T1>& src1, const DevMem2D_<T2>& src2, const DevMem2D_<D>& dst, 
            BinOp op, cudaStream_t stream = 0)
        {
            transform_caller(src1, src2, dst, op, device::NoMask(), stream);
        }
        template <typename T1, typename T2, typename D, typename BinOp>
        static void transform(const DevMem2D_<T1>& src1, const DevMem2D_<T2>& src2, const DevMem2D_<D>& dst, 
            const PtrStep& mask, BinOp op, cudaStream_t stream = 0)
        {
            transform_caller(src1, src2, dst, op, device::MaskReader(mask), stream);
        }
    }
}

#endif // __OPENCV_GPU_TRANSFORM_HPP__
