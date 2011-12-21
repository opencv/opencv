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

#ifndef __OPENCV_GPU_TRANSFORM_DETAIL_HPP__
#define __OPENCV_GPU_TRANSFORM_DETAIL_HPP__

#include "../common.hpp"
#include "../vec_traits.hpp"
#include "../functional.hpp"

namespace cv { namespace gpu { namespace device 
{
    namespace transform_detail
    {
        //! Read Write Traits

        template <typename T, typename D, int shift> struct UnaryReadWriteTraits
        {
            typedef typename TypeVec<T, shift>::vec_type read_type;
            typedef typename TypeVec<D, shift>::vec_type write_type;
        };

        template <typename T1, typename T2, typename D, int shift> struct BinaryReadWriteTraits
        {
            typedef typename TypeVec<T1, shift>::vec_type read_type1;
            typedef typename TypeVec<T2, shift>::vec_type read_type2;
            typedef typename TypeVec<D, shift>::vec_type write_type;
        };

        //! Transform kernels

        template <int shift> struct OpUnroller;
        template <> struct OpUnroller<1>
        {
            template <typename T, typename D, typename UnOp, typename Mask>
            static __device__ __forceinline__ void unroll(const T& src, D& dst, const Mask& mask, UnOp& op, int x_shifted, int y)
            {
                if (mask(y, x_shifted))
                    dst.x = op(src.x);
            }

            template <typename T1, typename T2, typename D, typename BinOp, typename Mask>
            static __device__ __forceinline__ void unroll(const T1& src1, const T2& src2, D& dst, const Mask& mask, BinOp& op, int x_shifted, int y)
            {
                if (mask(y, x_shifted))
                    dst.x = op(src1.x, src2.x);
            }
        };
        template <> struct OpUnroller<2>
        {
            template <typename T, typename D, typename UnOp, typename Mask>
            static __device__ __forceinline__ void unroll(const T& src, D& dst, const Mask& mask, UnOp& op, int x_shifted, int y)
            {
                if (mask(y, x_shifted))
                    dst.x = op(src.x);
                if (mask(y, x_shifted + 1))
                    dst.y = op(src.y);
            }

            template <typename T1, typename T2, typename D, typename BinOp, typename Mask>
            static __device__ __forceinline__ void unroll(const T1& src1, const T2& src2, D& dst, const Mask& mask, BinOp& op, int x_shifted, int y)
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
            static __device__ __forceinline__ void unroll(const T& src, D& dst, const Mask& mask, const UnOp& op, int x_shifted, int y)
            {
                if (mask(y, x_shifted))
                    dst.x = op(src.x);
                if (mask(y, x_shifted + 1))
                    dst.y = op(src.y);
                if (mask(y, x_shifted + 2))
                    dst.z = op(src.z);
            }

            template <typename T1, typename T2, typename D, typename BinOp, typename Mask>
            static __device__ __forceinline__ void unroll(const T1& src1, const T2& src2, D& dst, const Mask& mask, const BinOp& op, int x_shifted, int y)
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
            static __device__ __forceinline__ void unroll(const T& src, D& dst, const Mask& mask, const UnOp& op, int x_shifted, int y)
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
            static __device__ __forceinline__ void unroll(const T1& src1, const T2& src2, D& dst, const Mask& mask, const BinOp& op, int x_shifted, int y)
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
        template <> struct OpUnroller<8>
        {
            template <typename T, typename D, typename UnOp, typename Mask>
            static __device__ __forceinline__ void unroll(const T& src, D& dst, const Mask& mask, const UnOp& op, int x_shifted, int y)
            {
                if (mask(y, x_shifted))
                    dst.a0 = op(src.a0);
                if (mask(y, x_shifted + 1))
                    dst.a1 = op(src.a1);
                if (mask(y, x_shifted + 2))
                    dst.a2 = op(src.a2);
                if (mask(y, x_shifted + 3))
                    dst.a3 = op(src.a3);
                if (mask(y, x_shifted + 4))
                    dst.a4 = op(src.a4);
                if (mask(y, x_shifted + 5))
                    dst.a5 = op(src.a5);
                if (mask(y, x_shifted + 6))
                    dst.a6 = op(src.a6);
                if (mask(y, x_shifted + 7))
                    dst.a7 = op(src.a7);
            }

            template <typename T1, typename T2, typename D, typename BinOp, typename Mask>
            static __device__ __forceinline__ void unroll(const T1& src1, const T2& src2, D& dst, const Mask& mask, const BinOp& op, int x_shifted, int y)
            {
                if (mask(y, x_shifted))
                    dst.a0 = op(src1.a0, src2.a0);
                if (mask(y, x_shifted + 1))
                    dst.a1 = op(src1.a1, src2.a1);
                if (mask(y, x_shifted + 2))
                    dst.a2 = op(src1.a2, src2.a2);
                if (mask(y, x_shifted + 3))
                    dst.a3 = op(src1.a3, src2.a3);
                if (mask(y, x_shifted + 4))
                    dst.a4 = op(src1.a4, src2.a4);
                if (mask(y, x_shifted + 5))
                    dst.a5 = op(src1.a5, src2.a5);
                if (mask(y, x_shifted + 6))
                    dst.a6 = op(src1.a6, src2.a6);
                if (mask(y, x_shifted + 7))
                    dst.a7 = op(src1.a7, src2.a7);
            }
        };

        template <typename T, typename D, typename UnOp, typename Mask>
        __global__ static void transformSmart(const DevMem2D_<T> src_, PtrStep<D> dst_, const Mask mask, const UnOp op)
        {
            typedef TransformFunctorTraits<UnOp> ft;
            typedef typename UnaryReadWriteTraits<T, D, ft::smart_shift>::read_type read_type;
            typedef typename UnaryReadWriteTraits<T, D, ft::smart_shift>::write_type write_type;

            const int x = threadIdx.x + blockIdx.x * blockDim.x;
            const int y = threadIdx.y + blockIdx.y * blockDim.y;
            const int x_shifted = x * ft::smart_shift;

            if (y < src_.rows)
            {
                const T* src = src_.ptr(y);
                D* dst = dst_.ptr(y);

                if (x_shifted + ft::smart_shift - 1 < src_.cols)
                {
                    const read_type src_n_el = ((const read_type*)src)[x];
                    write_type dst_n_el;

                    OpUnroller<ft::smart_shift>::unroll(src_n_el, dst_n_el, mask, op, x_shifted, y);

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
        static __global__ void transformSimple(const DevMem2D_<T> src, PtrStep<D> dst, const Mask mask, const UnOp op)
        {
	        const int x = blockDim.x * blockIdx.x + threadIdx.x;
	        const int y = blockDim.y * blockIdx.y + threadIdx.y;

            if (x < src.cols && y < src.rows && mask(y, x))
            {
                dst.ptr(y)[x] = op(src.ptr(y)[x]);
            }
        }

        template <typename T1, typename T2, typename D, typename BinOp, typename Mask>
        __global__ static void transformSmart(const DevMem2D_<T1> src1_, const PtrStep<T2> src2_, PtrStep<D> dst_, 
            const Mask mask, const BinOp op)
        {
            typedef TransformFunctorTraits<BinOp> ft;
            typedef typename BinaryReadWriteTraits<T1, T2, D, ft::smart_shift>::read_type1 read_type1;
            typedef typename BinaryReadWriteTraits<T1, T2, D, ft::smart_shift>::read_type2 read_type2;
            typedef typename BinaryReadWriteTraits<T1, T2, D, ft::smart_shift>::write_type write_type;

            const int x = threadIdx.x + blockIdx.x * blockDim.x;
            const int y = threadIdx.y + blockIdx.y * blockDim.y;
            const int x_shifted = x * ft::smart_shift;

            if (y < src1_.rows)
            {
                const T1* src1 = src1_.ptr(y);
                const T2* src2 = src2_.ptr(y);
                D* dst = dst_.ptr(y);

                if (x_shifted + ft::smart_shift - 1 < src1_.cols)
                {
                    const read_type1 src1_n_el = ((const read_type1*)src1)[x];
                    const read_type2 src2_n_el = ((const read_type2*)src2)[x];
                    write_type dst_n_el;
                    
                    OpUnroller<ft::smart_shift>::unroll(src1_n_el, src2_n_el, dst_n_el, mask, op, x_shifted, y);

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
        static __global__ void transformSimple(const DevMem2D_<T1> src1, const PtrStep<T2> src2, PtrStep<D> dst, 
            const Mask mask, const BinOp op)
        {
	        const int x = blockDim.x * blockIdx.x + threadIdx.x;
	        const int y = blockDim.y * blockIdx.y + threadIdx.y;

            if (x < src1.cols && y < src1.rows && mask(y, x))
            {
                const T1 src1_data = src1.ptr(y)[x];
                const T2 src2_data = src2.ptr(y)[x];
                dst.ptr(y)[x] = op(src1_data, src2_data);
            }
        }

        template <bool UseSmart> struct TransformDispatcher;
        template<> struct TransformDispatcher<false>
        {
            template <typename T, typename D, typename UnOp, typename Mask>
            static void call(DevMem2D_<T> src, DevMem2D_<D> dst, UnOp op, Mask mask, cudaStream_t stream)
            {
                typedef TransformFunctorTraits<UnOp> ft;

                const dim3 threads(ft::simple_block_dim_x, ft::simple_block_dim_y, 1);
                const dim3 grid(divUp(src.cols, threads.x), divUp(src.rows, threads.y), 1);     

                transformSimple<T, D><<<grid, threads, 0, stream>>>(src, dst, mask, op);
                cudaSafeCall( cudaGetLastError() );

                if (stream == 0)
                    cudaSafeCall( cudaDeviceSynchronize() ); 
            }

            template <typename T1, typename T2, typename D, typename BinOp, typename Mask>
            static void call(DevMem2D_<T1> src1, DevMem2D_<T2> src2, DevMem2D_<D> dst, BinOp op, Mask mask, cudaStream_t stream)
            {
                typedef TransformFunctorTraits<BinOp> ft;

                const dim3 threads(ft::simple_block_dim_x, ft::simple_block_dim_y, 1);
                const dim3 grid(divUp(src1.cols, threads.x), divUp(src1.rows, threads.y), 1);     

                transformSimple<T1, T2, D><<<grid, threads, 0, stream>>>(src1, src2, dst, mask, op);
                cudaSafeCall( cudaGetLastError() );

                if (stream == 0)
                    cudaSafeCall( cudaDeviceSynchronize() );            
            }
        };
        template<> struct TransformDispatcher<true>
        {
            template <typename T, typename D, typename UnOp, typename Mask>
            static void call(DevMem2D_<T> src, DevMem2D_<D> dst, UnOp op, Mask mask, cudaStream_t stream)
            {
                typedef TransformFunctorTraits<UnOp> ft;

                StaticAssert<ft::smart_shift != 1>::check();

                if (!isAligned(src.data, ft::smart_shift * sizeof(T)) || !isAligned(dst.data, ft::smart_shift * sizeof(D)))
                {
                    TransformDispatcher<false>::call(src, dst, op, mask, stream);
                    return;
                }

                const dim3 threads(ft::smart_block_dim_x, ft::smart_block_dim_y, 1);
                const dim3 grid(divUp(src.cols, threads.x * ft::smart_shift), divUp(src.rows, threads.y), 1);      

                transformSmart<T, D><<<grid, threads, 0, stream>>>(src, dst, mask, op);
                cudaSafeCall( cudaGetLastError() );

                if (stream == 0)
                    cudaSafeCall( cudaDeviceSynchronize() );
            }

            template <typename T1, typename T2, typename D, typename BinOp, typename Mask>
            static void call(DevMem2D_<T1> src1, DevMem2D_<T2> src2, DevMem2D_<D> dst, BinOp op, Mask mask, cudaStream_t stream)
            {
                typedef TransformFunctorTraits<BinOp> ft;

                StaticAssert<ft::smart_shift != 1>::check();

                if (!isAligned(src1.data, ft::smart_shift * sizeof(T1)) || !isAligned(src2.data, ft::smart_shift * sizeof(T2)) || !isAligned(dst.data, ft::smart_shift * sizeof(D)))
                {
                    TransformDispatcher<false>::call(src1, src2, dst, op, mask, stream);
                    return;
                }

                const dim3 threads(ft::smart_block_dim_x, ft::smart_block_dim_y, 1);
                const dim3 grid(divUp(src1.cols, threads.x * ft::smart_shift), divUp(src1.rows, threads.y), 1);    

                transformSmart<T1, T2, D><<<grid, threads, 0, stream>>>(src1, src2, dst, mask, op);
                cudaSafeCall( cudaGetLastError() );

                if (stream == 0)
                    cudaSafeCall( cudaDeviceSynchronize() );            
            }
        };        

        template <typename T, typename D, typename UnOp, typename Mask>
        static inline void transform_caller(DevMem2D_<T> src, DevMem2D_<D> dst, UnOp op, Mask mask, cudaStream_t stream)
        {
            typedef TransformFunctorTraits<UnOp> ft;
            TransformDispatcher<VecTraits<T>::cn == 1 && VecTraits<D>::cn == 1 && ft::smart_shift != 1>::call(src, dst, op, mask, stream);
        }

        template <typename T1, typename T2, typename D, typename BinOp, typename Mask>
        static inline void transform_caller(DevMem2D_<T1> src1, DevMem2D_<T2> src2, DevMem2D_<D> dst, BinOp op, Mask mask, cudaStream_t stream)
        {
            typedef TransformFunctorTraits<BinOp> ft;
            TransformDispatcher<VecTraits<T1>::cn == 1 && VecTraits<T2>::cn == 1 && VecTraits<D>::cn == 1 && ft::smart_shift != 1>::call(src1, src2, dst, op, mask, stream);
        }
    } // namespace transform_detail
}}} // namespace cv { namespace gpu { namespace device

#endif // __OPENCV_GPU_TRANSFORM_DETAIL_HPP__
