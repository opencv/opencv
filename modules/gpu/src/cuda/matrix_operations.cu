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
#include "cuda_shared.hpp"
#include "cuda_runtime.h"
#include "saturate_cast.hpp"

using namespace cv::gpu;
using namespace cv::gpu::impl;


namespace mat_operators
{
    __constant__ double scalar_d[4];


    template <typename T>
    class shift_and_sizeof;

    template <>
    class shift_and_sizeof<char>
    {
        public:
        enum { shift = 0 };
    };

    template <>
    class shift_and_sizeof<unsigned char>
    {
        public:
        enum { shift = 0 };
    };

    template <>
    class shift_and_sizeof<short>
    {
        public:
        enum { shift = 1 };
    };

    template <>
    class shift_and_sizeof<unsigned short>
    {
        public:
        enum { shift = 1 };
    };

    template <>
    class shift_and_sizeof<int>
    {
        public:
        enum { shift = 2 };
    };

    template <>
    class shift_and_sizeof<float>
    {
        public:
        enum { shift = 2 };
    };

    template <>
    class shift_and_sizeof<double>
    {
        public:
        enum { shift = 3 };
    };


    ///////////////////////////////////////////////////////////////////////////
    ////////////////////////////////// CopyTo /////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    template<typename T>
    __global__ void kernel_copy_to_with_mask(T * mat_src, T * mat_dst, const unsigned char * mask, int cols, int rows, int step_mat, int step_mask, int channels)
    {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if ((x < cols * channels ) && (y < rows))
            if (mask[y * step_mask + x / channels] != 0)
            {
                size_t idx = y * ( step_mat >> shift_and_sizeof<T>::shift ) + x;
                mat_dst[idx] = mat_src[idx];
            }
    }

    ///////////////////////////////////////////////////////////////////////////
    ////////////////////////////////// SetTo //////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    template<typename T>
    __global__ void kernel_set_to_without_mask(T * mat, int cols, int rows, int step, int channels)
    {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if ((x < cols * channels ) && (y < rows))
        {
            size_t idx = y * ( step >> shift_and_sizeof<T>::shift ) + x;
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
                size_t idx = y * ( step >> shift_and_sizeof<T>::shift ) + x;
                mat[idx] = scalar_d[ x % channels ];
            }
    }


    ///////////////////////////////////////////////////////////////////////////
    //////////////////////////////// ConvertTo ////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    template <typename T, typename DT, size_t src_elem_size, size_t dst_elem_size>
    struct ReadWriteTraits
    {
        enum {shift=1};

        typedef T read_type;
        typedef DT write_type;
    };
    template <typename T, typename DT>
    struct ReadWriteTraits<T, DT, 1, 1>
    {
        enum {shift=4};

        typedef char4 read_type;
        typedef char4 write_type;
    };
    template <typename T, typename DT>
    struct ReadWriteTraits<T, DT, 2, 1>
    {
        enum {shift=4};

        typedef short4 read_type;
        typedef char4 write_type;
    };
    template <typename T, typename DT>
    struct ReadWriteTraits<T, DT, 4, 1>
    {
        enum {shift=4};

        typedef int4 read_type;
        typedef char4 write_type;
    };
    template <typename T, typename DT>
    struct ReadWriteTraits<T, DT, 1, 2>
    {
        enum {shift=2};

        typedef char2 read_type;
        typedef short2 write_type;
    };
    template <typename T, typename DT>
    struct ReadWriteTraits<T, DT, 2, 2>
    {
        enum {shift=2};

        typedef short2 read_type;
        typedef short2 write_type;
    };
    template <typename T, typename DT>
    struct ReadWriteTraits<T, DT, 4, 2>
    {
        enum {shift=2};

        typedef int2 read_type;
        typedef short2 write_type;
    };

    template <typename T, typename DT>
    __global__ static void kernel_convert_to(uchar* srcmat, size_t src_step, uchar* dstmat, size_t dst_step, size_t width, size_t height, double alpha, double beta)
    {
        typedef typename ReadWriteTraits<T, DT, sizeof(T), sizeof(DT)>::read_type read_type;
        typedef typename ReadWriteTraits<T, DT, sizeof(T), sizeof(DT)>::write_type write_type;
        const int shift = ReadWriteTraits<T, DT, sizeof(T), sizeof(DT)>::shift;

        const size_t x = threadIdx.x + blockIdx.x * blockDim.x;
        const size_t y = threadIdx.y + blockIdx.y * blockDim.y;

        if (y < height)
        {
            const T* src = (const T*)(srcmat + src_step * y);
            DT* dst = (DT*)(dstmat + dst_step * y);
            if ((x * shift) + shift - 1 < width)
            {
                read_type srcn_el = ((read_type*)src)[x];
                write_type dstn_el;

                const T* src1_el = (const T*) &srcn_el;
                DT* dst1_el = (DT*) &dstn_el;

                for (int i = 0; i < shift; ++i)
                    dst1_el[i] =  saturate_cast<DT>(alpha * src1_el[i] + beta);

                ((write_type*)dst)[x] = dstn_el;
            }
            else
            {
                for (int i = 0; i < shift - 1; ++i)
                    if ((x * shift) + i < width)
                        dst[(x * shift) + i] = saturate_cast<DT>(alpha * src[(x * shift) + i] + beta);
            }
        }
    }

} // namespace mat_operators

namespace cv
{
    namespace gpu
    {
        namespace impl
        {

            ///////////////////////////////////////////////////////////////////////////
            ////////////////////////////////// CopyTo /////////////////////////////////
            ///////////////////////////////////////////////////////////////////////////

            typedef void (*CopyToFunc)(const DevMem2D& mat_src, const DevMem2D& mat_dst, const DevMem2D& mask, int channels, const cudaStream_t & stream);

            template<typename T>
            void copy_to_with_mask_run(const DevMem2D& mat_src, const DevMem2D& mat_dst, const DevMem2D& mask, int channels, const cudaStream_t & stream)
            {
                dim3 threadsPerBlock(16,16, 1);
                dim3 numBlocks ( divUp(mat_src.cols * channels , threadsPerBlock.x) , divUp(mat_src.rows , threadsPerBlock.y), 1);
                if (stream == 0)
                {
                    ::mat_operators::kernel_copy_to_with_mask<T><<<numBlocks,threadsPerBlock>>>
                        ((T*)mat_src.ptr, (T*)mat_dst.ptr, (unsigned char*)mask.ptr, mat_src.cols, mat_src.rows, mat_src.step, mask.step, channels);
                    cudaSafeCall ( cudaThreadSynchronize() );
                }
                else
                {
                    ::mat_operators::kernel_copy_to_with_mask<T><<<numBlocks,threadsPerBlock, 0, stream>>>
                        ((T*)mat_src.ptr, (T*)mat_dst.ptr, (unsigned char*)mask.ptr, mat_src.cols, mat_src.rows, mat_src.step, mask.step, channels);
                }
            }

            extern "C" void copy_to_with_mask(const DevMem2D& mat_src, DevMem2D mat_dst, int depth, const DevMem2D& mask, int channels, const cudaStream_t & stream)
            {
                static CopyToFunc tab[8] =
                {
                    copy_to_with_mask_run<unsigned char>,
                    copy_to_with_mask_run<char>,
                    copy_to_with_mask_run<unsigned short>,
                    copy_to_with_mask_run<short>,
                    copy_to_with_mask_run<int>,
                    copy_to_with_mask_run<float>,
                    copy_to_with_mask_run<double>,
                    0
                };

                CopyToFunc func = tab[depth];

                if (func == 0) cv::gpu::error("Unsupported copyTo operation", __FILE__, __LINE__);

                func(mat_src, mat_dst, mask, channels, stream);
            }


            ///////////////////////////////////////////////////////////////////////////
            ////////////////////////////////// SetTo //////////////////////////////////
            ///////////////////////////////////////////////////////////////////////////

            typedef void (*SetToFunc_with_mask)(const DevMem2D& mat, const DevMem2D& mask, int channels, const cudaStream_t & stream);
            typedef void (*SetToFunc_without_mask)(const DevMem2D& mat, int channels, const cudaStream_t & stream);

            template <typename T>
            void set_to_with_mask_run(const DevMem2D& mat, const DevMem2D& mask, int channels, const cudaStream_t & stream)
            {
                dim3 threadsPerBlock(32, 8, 1);
                dim3 numBlocks (mat.cols * channels / threadsPerBlock.x + 1, mat.rows / threadsPerBlock.y + 1, 1);

                if (stream == 0)
                {
                    ::mat_operators::kernel_set_to_with_mask<T><<<numBlocks,threadsPerBlock>>>((T*)mat.ptr, (unsigned char *)mask.ptr, mat.cols, mat.rows, mat.step, channels, mask.step);
                    cudaSafeCall ( cudaThreadSynchronize() );
                }
                else
                {
                    ::mat_operators::kernel_set_to_with_mask<T><<<numBlocks,threadsPerBlock, 0, stream>>>((T*)mat.ptr, (unsigned char *)mask.ptr, mat.cols, mat.rows, mat.step, channels, mask.step);
                }

            }

            template <typename T>
            void set_to_without_mask_run(const DevMem2D& mat, int channels, const cudaStream_t & stream)
            {
                dim3 threadsPerBlock(32, 8, 1);
                dim3 numBlocks (mat.cols * channels / threadsPerBlock.x + 1, mat.rows / threadsPerBlock.y + 1, 1);

                if (stream == 0)
                {
                    mat_operators::kernel_set_to_without_mask<T><<<numBlocks,threadsPerBlock>>>((T*)mat.ptr, mat.cols, mat.rows, mat.step, channels);
                    cudaSafeCall ( cudaThreadSynchronize() );
                }
                else
                {
                    mat_operators::kernel_set_to_without_mask<T><<<numBlocks,threadsPerBlock, 0, stream>>>((T*)mat.ptr, mat.cols, mat.rows, mat.step, channels);
                }
            }

            extern "C" void set_to_without_mask(DevMem2D mat, int depth, const double *scalar, int channels, const cudaStream_t & stream)
            {
                cudaSafeCall( cudaMemcpyToSymbol(mat_operators::scalar_d, scalar, sizeof(double) * 4));

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

                if (func == 0)
                    cv::gpu::error("Unsupported setTo operation", __FILE__, __LINE__);

                func(mat, channels, stream);
            }


            extern "C" void set_to_with_mask(DevMem2D mat, int depth, const double * scalar, const DevMem2D& mask, int channels, const cudaStream_t & stream)
            {
                cudaSafeCall( cudaMemcpyToSymbol(mat_operators::scalar_d, scalar, sizeof(double) * 4));

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

                if (func == 0)
                    cv::gpu::error("Unsupported setTo operation", __FILE__, __LINE__);

                func(mat, mask, channels, stream);
            }


            ///////////////////////////////////////////////////////////////////////////
            //////////////////////////////// ConvertTo ////////////////////////////////
            ///////////////////////////////////////////////////////////////////////////

            typedef void (*CvtFunc)(const DevMem2D& src, DevMem2D& dst, size_t width, size_t height, double alpha, double beta, const cudaStream_t & stream);

            template<typename T, typename DT>
            void cvt_(const DevMem2D& src, DevMem2D& dst, size_t width, size_t height, double alpha, double beta, const cudaStream_t & stream)
            {
                const int shift = ::mat_operators::ReadWriteTraits<T, DT, sizeof(T), sizeof(DT)>::shift;

                dim3 block(32, 8);
                dim3 grid(divUp(width, block.x * shift), divUp(height, block.y));

                if (stream == 0)
                {
                    mat_operators::kernel_convert_to<T, DT><<<grid, block>>>(src.ptr, src.step, dst.ptr, dst.step, width, height, alpha, beta);
                    cudaSafeCall( cudaThreadSynchronize() );
                }
                else
                {
                    mat_operators::kernel_convert_to<T, DT><<<grid, block, 0, stream>>>(src.ptr, src.step, dst.ptr, dst.step, width, height, alpha, beta);
                }
            }

            extern "C" void convert_to(const DevMem2D& src, int sdepth, DevMem2D dst, int ddepth, int channels, double alpha, double beta, const cudaStream_t & stream)
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
                    cv::gpu::error("Unsupported convert operation", __FILE__, __LINE__);
                func(src, dst, src.cols * channels, src.rows, alpha, beta, stream);
            }
        } // namespace impl
    } // namespace gpu
} // namespace cv
