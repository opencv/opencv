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

#include "internal_shared.hpp"
#include "opencv2/gpu/device/saturate_cast.hpp"
#include "opencv2/gpu/device/transform.hpp"

using namespace cv::gpu::device;

namespace cv { namespace gpu { namespace matrix_operations {

    template <typename T> struct shift_and_sizeof;
    template <> struct shift_and_sizeof<char> { enum { shift = 0 }; };
    template <> struct shift_and_sizeof<unsigned char> { enum { shift = 0 }; };
    template <> struct shift_and_sizeof<short> { enum { shift = 1 }; };
    template <> struct shift_and_sizeof<unsigned short> { enum { shift = 1 }; };
    template <> struct shift_and_sizeof<int> { enum { shift = 2 }; };
    template <> struct shift_and_sizeof<float> { enum { shift = 2 }; };
    template <> struct shift_and_sizeof<double> { enum { shift = 3 }; };

///////////////////////////////////////////////////////////////////////////
////////////////////////////////// CopyTo /////////////////////////////////
///////////////////////////////////////////////////////////////////////////

    template<typename T>
    __global__ void copy_to_with_mask(T * mat_src, T * mat_dst, const unsigned char * mask, int cols, int rows, int step_mat, int step_mask, int channels)
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
    typedef void (*CopyToFunc)(const DevMem2D& mat_src, const DevMem2D& mat_dst, const DevMem2D& mask, int channels, const cudaStream_t & stream);

    template<typename T>
    void copy_to_with_mask_run(const DevMem2D& mat_src, const DevMem2D& mat_dst, const DevMem2D& mask, int channels, const cudaStream_t & stream)
    {
        dim3 threadsPerBlock(16,16, 1);
        dim3 numBlocks ( divUp(mat_src.cols * channels , threadsPerBlock.x) , divUp(mat_src.rows , threadsPerBlock.y), 1);

        copy_to_with_mask<T><<<numBlocks,threadsPerBlock, 0, stream>>>
                ((T*)mat_src.data, (T*)mat_dst.data, (unsigned char*)mask.data, mat_src.cols, mat_src.rows, mat_src.step, mask.step, channels);

        if (stream == 0)
            cudaSafeCall ( cudaThreadSynchronize() );        
    }

    void copy_to_with_mask(const DevMem2D& mat_src, DevMem2D mat_dst, int depth, const DevMem2D& mask, int channels, const cudaStream_t & stream)
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

    __constant__ double scalar_d[4]; 

    template<typename T>
    __global__ void set_to_without_mask(T * mat, int cols, int rows, int step, int channels)
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
    __global__ void set_to_with_mask(T * mat, const unsigned char * mask, int cols, int rows, int step, int channels, int step_mask)
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
    typedef void (*SetToFunc_with_mask)(const DevMem2D& mat, const DevMem2D& mask, int channels, const cudaStream_t & stream);
    typedef void (*SetToFunc_without_mask)(const DevMem2D& mat, int channels, const cudaStream_t & stream);

    template <typename T>
    void set_to_with_mask_run(const DevMem2D& mat, const DevMem2D& mask, int channels, const cudaStream_t & stream)
    {
        dim3 threadsPerBlock(32, 8, 1);
        dim3 numBlocks (mat.cols * channels / threadsPerBlock.x + 1, mat.rows / threadsPerBlock.y + 1, 1);

        set_to_with_mask<T><<<numBlocks,threadsPerBlock, 0, stream>>>((T*)mat.data, (unsigned char *)mask.data, mat.cols, mat.rows, mat.step, channels, mask.step);
        if (stream == 0)
            cudaSafeCall ( cudaThreadSynchronize() );
    }

    template <typename T>
    void set_to_without_mask_run(const DevMem2D& mat, int channels, const cudaStream_t & stream)
    {
        dim3 threadsPerBlock(32, 8, 1);
        dim3 numBlocks (mat.cols * channels / threadsPerBlock.x + 1, mat.rows / threadsPerBlock.y + 1, 1);

        set_to_without_mask<T><<<numBlocks,threadsPerBlock, 0, stream>>>((T*)mat.data, mat.cols, mat.rows, mat.step, channels);

        if (stream == 0)
            cudaSafeCall ( cudaThreadSynchronize() );
    }

    void set_to_without_mask(DevMem2D mat, int depth, const double *scalar, int channels, const cudaStream_t & stream)
    {
        cudaSafeCall( cudaMemcpyToSymbol(scalar_d, scalar, sizeof(double) * 4));

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

    void set_to_with_mask(DevMem2D mat, int depth, const double * scalar, const DevMem2D& mask, int channels, const cudaStream_t & stream)
    {
        cudaSafeCall( cudaMemcpyToSymbol(scalar_d, scalar, sizeof(double) * 4));

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

    template <typename T, typename D>
    class Convertor
    {
    public:
        Convertor(double alpha_, double beta_): alpha(alpha_), beta(beta_) {}

        __device__ D operator()(const T& src)
        {
            return saturate_cast<D>(alpha * src + beta);
        }

    private:
        double alpha, beta;
    };
    
    template<typename T, typename D>
    void cvt_(const DevMem2D& src, const DevMem2D& dst, double alpha, double beta, cudaStream_t stream)
    {
        Convertor<T, D> op(alpha, beta);
        transform((DevMem2D_<T>)src, (DevMem2D_<D>)dst, op, stream);
    }

    void convert_gpu(const DevMem2D& src, int sdepth, const DevMem2D& dst, int ddepth, double alpha, double beta, 
        cudaStream_t stream = 0)
    {
        typedef void (*caller_t)(const DevMem2D& src, const DevMem2D& dst, double alpha, double beta, 
            cudaStream_t stream);

        static const caller_t tab[8][8] =
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

        caller_t func = tab[sdepth][ddepth];
        if (!func)
            cv::gpu::error("Unsupported convert operation", __FILE__, __LINE__);

        func(src, dst, alpha, beta, stream);
    }
}}}
