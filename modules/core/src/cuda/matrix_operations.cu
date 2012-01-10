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

#include "opencv2/gpu/device/saturate_cast.hpp"
#include "opencv2/gpu/device/transform.hpp"
#include "opencv2/gpu/device/functional.hpp"

namespace cv { namespace gpu { namespace device 
{
    template <typename T> struct shift_and_sizeof;
    template <> struct shift_and_sizeof<signed char> { enum { shift = 0 }; };
    template <> struct shift_and_sizeof<unsigned char> { enum { shift = 0 }; };
    template <> struct shift_and_sizeof<short> { enum { shift = 1 }; };
    template <> struct shift_and_sizeof<unsigned short> { enum { shift = 1 }; };
    template <> struct shift_and_sizeof<int> { enum { shift = 2 }; };
    template <> struct shift_and_sizeof<float> { enum { shift = 2 }; };
    template <> struct shift_and_sizeof<double> { enum { shift = 3 }; };

    ///////////////////////////////////////////////////////////////////////////
    ////////////////////////////////// CopyTo /////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    template <typename T> void copyToWithMask(DevMem2Db src, DevMem2Db dst, DevMem2Db mask, int channels, cudaStream_t stream)
    {        
        cv::gpu::device::transform((DevMem2D_<T>)src, (DevMem2D_<T>)dst, identity<T>(), SingleMaskChannels(mask, channels), stream);
    }

    void copyToWithMask_gpu(DevMem2Db src, DevMem2Db dst, int depth, int channels, DevMem2Db mask, cudaStream_t stream)
    {
        typedef void (*func_t)(DevMem2Db src, DevMem2Db dst, DevMem2Db mask, int channels, cudaStream_t stream);

        static func_t tab[] =
        {
            copyToWithMask<unsigned char>,
            copyToWithMask<signed char>,
            copyToWithMask<unsigned short>,
            copyToWithMask<short>,
            copyToWithMask<int>,
            copyToWithMask<float>,
            copyToWithMask<double>
        };

        tab[depth](src, dst, mask, channels, stream);
    }

    ///////////////////////////////////////////////////////////////////////////
    ////////////////////////////////// SetTo //////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    __constant__ uchar scalar_8u[4];
    __constant__ schar scalar_8s[4];
    __constant__ ushort scalar_16u[4];
    __constant__ short scalar_16s[4];
    __constant__ int scalar_32s[4];
    __constant__ float scalar_32f[4]; 
    __constant__ double scalar_64f[4];

    template <typename T> __device__ __forceinline__ T readScalar(int i);
    template <> __device__ __forceinline__ uchar readScalar<uchar>(int i) {return scalar_8u[i];}
    template <> __device__ __forceinline__ schar readScalar<schar>(int i) {return scalar_8s[i];}
    template <> __device__ __forceinline__ ushort readScalar<ushort>(int i) {return scalar_16u[i];}
    template <> __device__ __forceinline__ short readScalar<short>(int i) {return scalar_16s[i];}
    template <> __device__ __forceinline__ int readScalar<int>(int i) {return scalar_32s[i];}
    template <> __device__ __forceinline__ float readScalar<float>(int i) {return scalar_32f[i];}
    template <> __device__ __forceinline__ double readScalar<double>(int i) {return scalar_64f[i];}

    void writeScalar(const uchar* vals)
    {
        cudaSafeCall( cudaMemcpyToSymbol(scalar_8u, vals, sizeof(uchar) * 4) );
    }
    void writeScalar(const schar* vals)
    {
        cudaSafeCall( cudaMemcpyToSymbol(scalar_8s, vals, sizeof(schar) * 4) );
    }
    void writeScalar(const ushort* vals)
    {
        cudaSafeCall( cudaMemcpyToSymbol(scalar_16u, vals, sizeof(ushort) * 4) );
    }
    void writeScalar(const short* vals)
    {
        cudaSafeCall( cudaMemcpyToSymbol(scalar_16s, vals, sizeof(short) * 4) );
    }
    void writeScalar(const int* vals)
    {
        cudaSafeCall( cudaMemcpyToSymbol(scalar_32s, vals, sizeof(int) * 4) );
    }
    void writeScalar(const float* vals)
    {
        cudaSafeCall( cudaMemcpyToSymbol(scalar_32f, vals, sizeof(float) * 4) );
    }
    void writeScalar(const double* vals)
    {
        cudaSafeCall( cudaMemcpyToSymbol(scalar_64f, vals, sizeof(double) * 4) );
    }

    template<typename T>
    __global__ void set_to_without_mask(T* mat, int cols, int rows, size_t step, int channels)
    {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if ((x < cols * channels ) && (y < rows))
        {
            size_t idx = y * ( step >> shift_and_sizeof<T>::shift ) + x;
            mat[idx] = readScalar<T>(x % channels);
        }
    }

    template<typename T>
    __global__ void set_to_with_mask(T* mat, const uchar* mask, int cols, int rows, size_t step, int channels, size_t step_mask)
    {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if ((x < cols * channels ) && (y < rows))
            if (mask[y * step_mask + x / channels] != 0)
            {
                size_t idx = y * ( step >> shift_and_sizeof<T>::shift ) + x;
                mat[idx] = readScalar<T>(x % channels);
            }
    }
    template <typename T>
    void set_to_gpu(DevMem2Db mat, const T* scalar, DevMem2Db mask, int channels, cudaStream_t stream)
    {
        writeScalar(scalar);

        dim3 threadsPerBlock(32, 8, 1);
        dim3 numBlocks (mat.cols * channels / threadsPerBlock.x + 1, mat.rows / threadsPerBlock.y + 1, 1);

        set_to_with_mask<T><<<numBlocks, threadsPerBlock, 0, stream>>>((T*)mat.data, (uchar*)mask.data, mat.cols, mat.rows, mat.step, channels, mask.step);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall ( cudaDeviceSynchronize() );
    }

    template void set_to_gpu<uchar >(DevMem2Db mat, const uchar*  scalar, DevMem2Db mask, int channels, cudaStream_t stream);
    template void set_to_gpu<schar >(DevMem2Db mat, const schar*  scalar, DevMem2Db mask, int channels, cudaStream_t stream);
    template void set_to_gpu<ushort>(DevMem2Db mat, const ushort* scalar, DevMem2Db mask, int channels, cudaStream_t stream);
    template void set_to_gpu<short >(DevMem2Db mat, const short*  scalar, DevMem2Db mask, int channels, cudaStream_t stream);
    template void set_to_gpu<int   >(DevMem2Db mat, const int*    scalar, DevMem2Db mask, int channels, cudaStream_t stream);
    template void set_to_gpu<float >(DevMem2Db mat, const float*  scalar, DevMem2Db mask, int channels, cudaStream_t stream);
    template void set_to_gpu<double>(DevMem2Db mat, const double* scalar, DevMem2Db mask, int channels, cudaStream_t stream);

    template <typename T>
    void set_to_gpu(DevMem2Db mat, const T* scalar, int channels, cudaStream_t stream)
    {
        writeScalar(scalar);

        dim3 threadsPerBlock(32, 8, 1);
        dim3 numBlocks (mat.cols * channels / threadsPerBlock.x + 1, mat.rows / threadsPerBlock.y + 1, 1);

        set_to_without_mask<T><<<numBlocks, threadsPerBlock, 0, stream>>>((T*)mat.data, mat.cols, mat.rows, mat.step, channels);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall ( cudaDeviceSynchronize() );
    }

    template void set_to_gpu<uchar >(DevMem2Db mat, const uchar*  scalar, int channels, cudaStream_t stream);
    template void set_to_gpu<schar >(DevMem2Db mat, const schar*  scalar, int channels, cudaStream_t stream);
    template void set_to_gpu<ushort>(DevMem2Db mat, const ushort* scalar, int channels, cudaStream_t stream);
    template void set_to_gpu<short >(DevMem2Db mat, const short*  scalar, int channels, cudaStream_t stream);
    template void set_to_gpu<int   >(DevMem2Db mat, const int*    scalar, int channels, cudaStream_t stream);
    template void set_to_gpu<float >(DevMem2Db mat, const float*  scalar, int channels, cudaStream_t stream);
    template void set_to_gpu<double>(DevMem2Db mat, const double* scalar, int channels, cudaStream_t stream);

    ///////////////////////////////////////////////////////////////////////////
    //////////////////////////////// ConvertTo ////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    template <typename T, typename D> struct Convertor : unary_function<T, D>
    {
        Convertor(double alpha_, double beta_) : alpha(alpha_), beta(beta_) {}

        __device__ __forceinline__ D operator()(const T& src) const
        {
            return saturate_cast<D>(alpha * src + beta);
        }

        double alpha, beta;
    };

    namespace detail
    {
        template <size_t src_size, size_t dst_size, typename F> struct ConvertTraitsDispatcher : DefaultTransformFunctorTraits<F>
        {
        };
        template <typename F> struct ConvertTraitsDispatcher<1, 1, F> : DefaultTransformFunctorTraits<F>
        {
            enum { smart_shift = 8 };
        };
        template <typename F> struct ConvertTraitsDispatcher<1, 2, F> : DefaultTransformFunctorTraits<F>
        {
            enum { smart_shift = 4 };
        };
        template <typename F> struct ConvertTraitsDispatcher<1, 4, F> : DefaultTransformFunctorTraits<F>
        {
            enum { smart_block_dim_y = 8 };
            enum { smart_shift = 4 };
        };

        template <typename F> struct ConvertTraitsDispatcher<2, 2, F> : DefaultTransformFunctorTraits<F>
        {
            enum { smart_shift = 4 };
        };
        template <typename F> struct ConvertTraitsDispatcher<2, 4, F> : DefaultTransformFunctorTraits<F>
        {
            enum { smart_shift = 2 };
        };

        template <typename F> struct ConvertTraitsDispatcher<4, 2, F> : DefaultTransformFunctorTraits<F>
        {
            enum { smart_block_dim_y = 8 };
            enum { smart_shift = 4 };
        };
        template <typename F> struct ConvertTraitsDispatcher<4, 4, F> : DefaultTransformFunctorTraits<F>
        {
            enum { smart_block_dim_y = 8 };
            enum { smart_shift = 2 };
        };

        template <typename F> struct ConvertTraits : ConvertTraitsDispatcher<sizeof(typename F::argument_type), sizeof(typename F::result_type), F>
        {
        };
    }

    template <typename T, typename D> struct TransformFunctorTraits< Convertor<T, D> > : detail::ConvertTraits< Convertor<T, D> >
    {
    };
        
    template<typename T, typename D>
    void cvt_(DevMem2Db src, DevMem2Db dst, double alpha, double beta, cudaStream_t stream)
    {
        cudaSafeCall( cudaSetDoubleForDevice(&alpha) );
        cudaSafeCall( cudaSetDoubleForDevice(&beta) );
        Convertor<T, D> op(alpha, beta);
        cv::gpu::device::transform((DevMem2D_<T>)src, (DevMem2D_<D>)dst, op, WithOutMask(), stream);
    }

    void convert_gpu(DevMem2Db src, int sdepth, DevMem2Db dst, int ddepth, double alpha, double beta, cudaStream_t stream)
    {
        typedef void (*caller_t)(DevMem2Db src, DevMem2Db dst, double alpha, double beta, cudaStream_t stream);

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
            cv::gpu::error("Unsupported convert operation", __FILE__, __LINE__, "convert_gpu");

        func(src, dst, alpha, beta, stream);
    }
}}} // namespace cv { namespace gpu { namespace device
