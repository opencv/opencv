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

#include <stddef.h>
#include <stdio.h>
#include <iostream>
#include "cuda_shared.hpp"
#include "cuda_runtime.h"

using namespace cv::gpu;
using namespace cv::gpu::impl;

__constant__ __align__(16) double scalar_d[4];

namespace mat_operators
{
    //////////////////////////////////////////////////////////
    // SetTo
    //////////////////////////////////////////////////////////

    template<typename T>
    __global__ void kernel_set_to_without_mask(T * mat, int cols, int rows, int step, int channels)
    {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if ((x < cols * channels ) && (y < rows))
        {
            size_t idx = y * (step / sizeof(T)) + x;
            mat[idx] = scalar_d[ x % channels ];
        }
    }

    template<typename T>
    __global__ void kernel_set_to_with_mask(T * mat, const unsigned char * mask, int cols, int rows, int step, int channels, int step_mask)
    {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if ((x < cols * channels ) && (y < rows))
            if (mask[y * step_mask + x / channels] != 0)
            {
                size_t idx = y * (step / sizeof(T)) + x;
                mat[idx] = scalar_d[ x % channels ];
            }
    }


    //////////////////////////////////////////////////////////
    // ConvertTo
    //////////////////////////////////////////////////////////

    template <typename T, typename DT, size_t src_elem_size, size_t dst_elem_size>
    struct Converter
    {
        __device__ static void convert(uchar* srcmat, size_t src_step, uchar* dstmat, size_t dst_step, size_t width, size_t height, double alpha, double beta)
        {
            size_t x = threadIdx.x + blockIdx.x * blockDim.x;
            size_t y = threadIdx.y + blockIdx.y * blockDim.y;
            if (x < width && y < height)
            {
                const T* src = (const T*)(srcmat + src_step * y);
                DT* dst = (DT*)(dstmat + dst_step * y);

                dst[x] = (DT)__double2int_rn(alpha * src[x] + beta);
            }
        }
        __host__ static inline dim3 calcGrid(size_t width, size_t height, dim3 block)
        {
            return dim3(divUp(width, block.x), divUp(height, block.y));
        }
    };

    template <typename T, typename DT>
    struct Converter<T, DT, 1, 1>
    {
        __device__ static void convert(uchar* srcmat, size_t src_step, uchar* dstmat, size_t dst_step, size_t width, size_t height, double alpha, double beta)
        {
            size_t x = threadIdx.x + blockIdx.x * blockDim.x;
            size_t y = threadIdx.y + blockIdx.y * blockDim.y;
            if (y < height)
            {
                const T* src = (const T*)(srcmat + src_step * y);
                DT* dst = (DT*)(dstmat + dst_step * y);
                if ((x << 2) + 3 < width)
                {
                    uchar4 src4b = ((const uchar4*)src)[x];
                    uchar4 dst4b;

                    const T* src1b = (const T*) &src4b.x;
                    DT* dst1b = (DT*) &dst4b.x;

                    dst1b[0] = (DT)__double2int_rn(alpha * src1b[0] + beta);
                    dst1b[1] = (DT)__double2int_rn(alpha * src1b[1] + beta);
                    dst1b[2] = (DT)__double2int_rn(alpha * src1b[2] + beta);
                    dst1b[3] = (DT)__double2int_rn(alpha * src1b[3] + beta);

                    ((uchar4*)dst)[x] = dst4b;
                }
                else
                {
                    if ((x << 2) + 0 < width)
                        dst[(x << 2) + 0] = (DT)__double2int_rn(alpha * src[(x << 2) + 0] + beta);

                    if ((x << 2) + 1 < width)
                        dst[(x << 2) + 1] = (DT)__double2int_rn(alpha * src[(x << 2) + 1] + beta);

                    if ((x << 2) + 2 < width)
                        dst[(x << 2) + 2] = (DT)__double2int_rn(alpha * src[(x << 2) + 2] + beta);
                }
            }
        }
        __host__ static inline dim3 calcGrid(size_t width, size_t height, dim3 block)
        {
            return dim3(divUp(width, block.x << 2), divUp(height, block.y));
        }
    };/**/

    template <typename T, typename DT>
    struct Converter<T, DT, 1, 2>
    {
        __device__ static void convert(uchar* srcmat, size_t src_step, uchar* dstmat, size_t dst_step, size_t width, size_t height, double alpha, double beta)
        {
            size_t x = threadIdx.x + blockIdx.x * blockDim.x;
            size_t y = threadIdx.y + blockIdx.y * blockDim.y;
            if (y < height)
            {
                const T* src = (const T*)(srcmat + src_step * y);
                DT* dst = (DT*)(dstmat + dst_step * y);
                if ((x << 1) + 1 < width)
                {
                    uchar2 src2b = ((const uchar2*)src)[x];
                    ushort2 dst2s;

                    const T* src1b = (const T*) &src2b;
                    DT* dst1s = (DT*) &dst2s;
                    dst1s[0] = (DT)__double2int_rn(alpha * src1b[0] + beta);
                    dst1s[1] = (DT)__double2int_rn(alpha * src1b[1] + beta);

                    ((ushort2*)(dst))[x] = dst2s;
                }
                else
                {
                    if ((x << 1) < width)
                        dst[(x << 1)] = (DT)__double2int_rn(alpha * src[(x << 1)] + beta);
                }
            }
        }
        __host__ static inline dim3 calcGrid(size_t width, size_t height, dim3 block)
        {
            return dim3(divUp(width, block.x << 1), divUp(height, block.y));
        }
    };/**/

    template <typename T, typename DT>
    struct Converter<T, DT, 2, 1>
    {
        __device__ static void convert(uchar* srcmat, size_t src_step, uchar* dstmat, size_t dst_step, size_t width, size_t height, double alpha, double beta)
        {
            size_t x = threadIdx.x + blockIdx.x * blockDim.x;
            size_t y = threadIdx.y + blockIdx.y * blockDim.y;
            if (y < height)
            {
                const T* src = (const T*)(srcmat + src_step * y);
                DT* dst = (DT*)(dstmat + dst_step * y);
                if ((x << 2) + 3 < width)
                {
                    ushort4 src4s = ((const ushort4*)src)[x];
                    uchar4 dst4b;

                    const T* src1s = (const T*) &src4s.x;
                    DT* dst1b = (DT*) &dst4b.x;
                    dst1b[0] = (DT)__double2int_rn(alpha * src1s[0] + beta);
                    dst1b[1] = (DT)__double2int_rn(alpha * src1s[1] + beta);
                    dst1b[2] = (DT)__double2int_rn(alpha * src1s[2] + beta);
                    dst1b[3] = (DT)__double2int_rn(alpha * src1s[3] + beta);

                    ((uchar4*)(dst))[x] = dst4b;
                }
                else
                {
                    if ((x << 2) + 0 < width)
                        dst[(x << 2) + 0] = (DT)__double2int_rn(alpha * src[(x << 2) + 0] + beta);
                    if ((x << 2) + 1 < width)
                        dst[(x << 2) + 1] = (DT)__double2int_rn(alpha * src[(x << 2) + 1] + beta);
                    if ((x << 2) + 2 < width)
                        dst[(x << 2) + 2] = (DT)__double2int_rn(alpha * src[(x << 2) + 2] + beta);
                }
            }
        }
        __host__ static inline dim3 calcGrid(size_t width, size_t height, dim3 block)
        {
            return dim3(divUp(width, block.x << 2), divUp(height, block.y));
        }
    };/**/

    template <typename T, typename DT>
    struct Converter<T, DT, 2, 2>
    {
        __device__ static void convert(uchar* srcmat, size_t src_step, uchar* dstmat, size_t dst_step, size_t width, size_t height, double alpha, double beta)
        {
            size_t x = threadIdx.x + blockIdx.x * blockDim.x;
            size_t y = threadIdx.y + blockIdx.y * blockDim.y;
            if (y < height)
            {
                const T* src = (const T*)(srcmat + src_step * y);
                DT* dst = (DT*)(dstmat + dst_step * y);
                if ((x << 1) + 1 < width)
                {
                    ushort2 src2s = ((const ushort2*)src)[x];
                    ushort2 dst2s;

                    const T* src1s = (const T*) &src2s.x;
                    DT* dst1s = (DT*) &dst2s.x;
                    dst1s[0] = (DT)__double2int_rn(alpha * src1s[0] + beta);
                    dst1s[1] = (DT)__double2int_rn(alpha * src1s[1] + beta);

                    ((ushort2*)dst)[x] = dst2s;
                }
                else
                {
                    if ((x << 1) < width)
                        dst[(x << 1)] = (DT)__double2int_rn(alpha * src[(x << 1)] + beta);
                }
            }
        }
        __host__ static inline dim3 calcGrid(size_t width, size_t height, dim3 block)
        {
            return dim3(divUp(width, block.x << 1), divUp(height, block.y));
        }
    };/**/

    template <typename T, size_t src_elem_size, size_t dst_elem_size>
    struct Converter<T, float, src_elem_size, dst_elem_size>
    {
        __device__ static void convert(uchar* srcmat, size_t src_step, uchar* dstmat, size_t dst_step, size_t width, size_t height, double alpha, double beta)
        {
            size_t x = threadIdx.x + blockIdx.x * blockDim.x;
            size_t y = threadIdx.y + blockIdx.y * blockDim.y;
            if (x < width && y < height)
            {
                const T* src = (const T*)(srcmat + src_step * y);
                float* dst = (float*)(dstmat + dst_step * y);

                dst[x] = (float)(alpha * src[x] + beta);
            }
        }
        __host__ static inline dim3 calcGrid(size_t width, size_t height, dim3 block)
        {
            return dim3(divUp(width, block.x), divUp(height, block.y));
        }
    };

    template <typename T, size_t src_elem_size, size_t dst_elem_size>
    struct Converter<T, double, src_elem_size, dst_elem_size>
    {
        __device__ static void convert(uchar* srcmat, size_t src_step, uchar* dstmat, size_t dst_step, size_t width, size_t height, double alpha, double beta)
        {
            size_t x = threadIdx.x + blockIdx.x * blockDim.x;
            size_t y = threadIdx.y + blockIdx.y * blockDim.y;
            if (x < width && y < height)
            {
                const T* src = (const T*)(srcmat + src_step * y);
                double* dst = (double*)(dstmat + dst_step * y);

                dst[x] = (double)(alpha * src[x] + beta);
            }
        }
        __host__ static inline dim3 calcGrid(size_t width, size_t height, dim3 block)
        {
            return dim3(divUp(width, block.x), divUp(height, block.y));
        }
    };

    template <typename T, typename DT>
    __global__ static void kernel_convert_to(uchar* srcmat, size_t src_step, uchar* dstmat, size_t dst_step, size_t width, size_t height, double alpha, double beta)
    {
        Converter<T, DT, sizeof(T), sizeof(DT)>::convert(srcmat, src_step, dstmat, dst_step, width, height, alpha, beta);
    }

} // namespace mat_operators

namespace cv
{
	namespace gpu
	{
		namespace impl
		{

                        //////////////////////////////////////////////////////////////
                        // SetTo
                        //////////////////////////////////////////////////////////////

                        typedef void (*SetToFunc_with_mask)(const DevMem2D& mat, const DevMem2D& mask, int channels);
                        typedef void (*SetToFunc_without_mask)(const DevMem2D& mat, int channels);

                        template <typename T>
                        void set_to_with_mask_run(const DevMem2D& mat, const DevMem2D& mask, int channels)
                        {
                            dim3 threadsPerBlock(32, 8, 1);
                            dim3 numBlocks (mat.cols * channels / threadsPerBlock.x + 1, mat.rows / threadsPerBlock.y + 1, 1);
                            ::mat_operators::kernel_set_to_with_mask<T><<<numBlocks,threadsPerBlock>>>((T*)mat.ptr, (unsigned char *)mask.ptr, mat.cols, mat.rows, mat.step, channels, mask.step);
                            cudaSafeCall ( cudaThreadSynchronize() );
                        }

                        template <typename T>
                        void set_to_without_mask_run(const DevMem2D& mat, int channels)
                        {
                            dim3 threadsPerBlock(32, 8, 1);
                            dim3 numBlocks (mat.cols * channels / threadsPerBlock.x + 1, mat.rows / threadsPerBlock.y + 1, 1);
                            ::mat_operators::kernel_set_to_without_mask<T><<<numBlocks,threadsPerBlock>>>((T*)mat.ptr, mat.cols, mat.rows, mat.step, channels);
                            cudaSafeCall ( cudaThreadSynchronize() );
                        }

                        extern "C" void set_to_without_mask(const DevMem2D& mat, int depth, const double * scalar, int channels)
                        {
                            double data[4];
                            data[0] = scalar[0];
                            data[1] = scalar[1];
                            data[2] = scalar[2];
                            data[3] = scalar[3];
                            cudaSafeCall( cudaMemcpyToSymbol(scalar_d, &data, sizeof(data)));

                            static SetToFunc_without_mask tab[8] =
                            {
                                set_to_without_mask_run<unsigned char>,
                                set_to_without_mask_run<char>,
                                set_to_without_mask_run<unsigned short>,
                                set_to_without_mask_run<short>,
                                set_to_without_mask_run<int>,
                                set_to_without_mask_run<float>,
                                set_to_without_mask_run<double>,
                                0
                            };

                            SetToFunc_without_mask func = tab[depth];

                            if (func == 0) error("Operation \'ConvertTo\' doesn't supported on your GPU model", __FILE__, __LINE__);

                            func(mat, channels);
                        }


                        extern "C" void set_to_with_mask(const DevMem2D& mat, int depth, const double * scalar, const DevMem2D& mask, int channels)
                        {
                            double data[4];
                            data[0] = scalar[0];
                            data[1] = scalar[1];
                            data[2] = scalar[2];
                            data[3] = scalar[3];
                            cudaSafeCall( cudaMemcpyToSymbol(scalar_d, &data, sizeof(data)));

                            static SetToFunc_with_mask tab[8] =
                            {
                                set_to_with_mask_run<unsigned char>,
                                set_to_with_mask_run<char>,
                                set_to_with_mask_run<unsigned short>,
                                set_to_with_mask_run<short>,
                                set_to_with_mask_run<int>,
                                set_to_with_mask_run<float>,
                                set_to_with_mask_run<double>,
                                0
                            };

                            SetToFunc_with_mask func = tab[depth];

                            if (func == 0) error("Operation \'ConvertTo\' doesn't supported on your GPU model", __FILE__, __LINE__);

                            func(mat, mask, channels);
                        }

                        //////////////////////////////////////////////////////////////
                        // ConvertTo
                        //////////////////////////////////////////////////////////////



			typedef void (*CvtFunc)(const DevMem2D& src, DevMem2D& dst, size_t width, size_t height, double alpha, double beta);

			//#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 130)

			template<typename T, typename DT>
			void cvt_(const DevMem2D& src, DevMem2D& dst, size_t width, size_t height, double alpha, double beta)
			{
				dim3 block(32, 8);
				dim3 grid = ::mat_operators::Converter<T, DT, sizeof(T), sizeof(DT)>::calcGrid(width, height, block);
				::mat_operators::kernel_convert_to<T, DT><<<grid, block>>>(src.ptr, src.step, dst.ptr, dst.step, width, height, alpha, beta);
				cudaSafeCall( cudaThreadSynchronize() );
			}
			//#endif

			extern "C" void convert_to(const DevMem2D& src, int sdepth, DevMem2D dst, int ddepth, size_t width, size_t height, double alpha, double beta)
			{
				static CvtFunc tab[8][8] =
				{
					{cvt_<uchar, uchar>, cvt_<uchar, schar>, cvt_<uchar, ushort>, cvt_<uchar, short>,
					cvt_<uchar, int>, cvt_<uchar, float>, cvt_<uchar, double>, 0},

					{cvt_<schar, uchar>, cvt_<schar, schar>, cvt_<schar, ushort>, cvt_<schar, short>,
					cvt_<schar, int>, cvt_<schar, float>, cvt_<schar, double>, 0},

					{cvt_<ushort, uchar>, cvt_<ushort, schar>, cvt_<ushort, ushort>, cvt_<ushort, short>,
					cvt_<ushort, int>, cvt_<ushort, float>, cvt_<ushort, double>, 0},

					{cvt_<short, uchar>, cvt_<short, schar>, cvt_<short, ushort>, cvt_<short, short>,
					cvt_<short, int>, cvt_<short, float>, cvt_<short, double>, 0},

					{cvt_<int, uchar>, cvt_<int, schar>, cvt_<int, ushort>,
					cvt_<int, short>, cvt_<int, int>, cvt_<int, float>, cvt_<int, double>, 0},

					{cvt_<float, uchar>, cvt_<float, schar>, cvt_<float, ushort>,
					cvt_<float, short>, cvt_<float, int>, cvt_<float, float>, cvt_<float, double>, 0},

					{cvt_<double, uchar>, cvt_<double, schar>, cvt_<double, ushort>,
					cvt_<double, short>, cvt_<double, int>, cvt_<double, float>, cvt_<double, double>, 0},

					{0,0,0,0,0,0,0,0}
				};

				CvtFunc func = tab[sdepth][ddepth];
				if (func == 0)
					error("Operation \'ConvertTo\' doesn't supported on your GPU model", __FILE__, __LINE__);
				func(src, dst, width, height, alpha, beta);
			}
		}


	}
}
