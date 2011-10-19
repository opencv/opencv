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
//     and/or other GpuMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or bpied warranties, including, but not limited to, the bpied
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

#include "precomp.hpp"

using namespace cv;
using namespace cv::gpu;
using namespace std;

#if !defined (HAVE_CUDA)

void cv::gpu::gemm(const GpuMat&, const GpuMat&, double, const GpuMat&, double, GpuMat&, int, Stream&) { throw_nogpu(); }
void cv::gpu::transpose(const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::flip(const GpuMat&, GpuMat&, int, Stream&) { throw_nogpu(); }
void cv::gpu::LUT(const GpuMat&, const Mat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::exp(const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::log(const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::magnitude(const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::magnitudeSqr(const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::magnitude(const GpuMat&, const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::magnitudeSqr(const GpuMat&, const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::phase(const GpuMat&, const GpuMat&, GpuMat&, bool, Stream&) { throw_nogpu(); }
void cv::gpu::cartToPolar(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, bool, Stream&) { throw_nogpu(); }
void cv::gpu::polarToCart(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, bool, Stream&) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

////////////////////////////////////////////////////////////////////////
// gemm

void cv::gpu::gemm(const GpuMat& src1, const GpuMat& src2, double alpha, const GpuMat& src3, double beta, GpuMat& dst, int flags, Stream& stream)
{
#ifndef HAVE_CUBLAS

    OPENCV_GPU_UNUSED(src1);
    OPENCV_GPU_UNUSED(src2);
    OPENCV_GPU_UNUSED(alpha);
    OPENCV_GPU_UNUSED(src3);
    OPENCV_GPU_UNUSED(beta);
    OPENCV_GPU_UNUSED(dst);
    OPENCV_GPU_UNUSED(flags);
    OPENCV_GPU_UNUSED(stream);

    throw_nogpu();

#else

    // CUBLAS works with column-major matrices

    CV_Assert(src1.type() == CV_32FC1 || src1.type() == CV_32FC2 || src1.type() == CV_64FC1 || src1.type() == CV_64FC2);
    CV_Assert(src2.type() == src1.type() && (src3.empty() || src3.type() == src1.type()));

    bool tr1 = flags & GEMM_1_T;
    bool tr2 = flags & GEMM_2_T;
    bool tr3 = flags & GEMM_3_T;

    Size src1Size = tr1 ? Size(src1.rows, src1.cols) : src1.size();
    Size src2Size = tr2 ? Size(src2.rows, src2.cols) : src2.size();
    Size src3Size = tr3 ? Size(src3.rows, src3.cols) : src3.size();
    Size dstSize(src2Size.width, src1Size.height);

    CV_Assert(src1Size.width == src2Size.height);
    CV_Assert(src3.empty() || src3Size == dstSize);

    dst.create(dstSize, CV_32FC1);

    if (beta != 0)
    {
        if (src3.empty())
        {
            if (stream)
                stream.enqueueMemSet(dst, Scalar::all(0));
            else
                dst.setTo(Scalar::all(0));
        }
        else
        {
            if (tr3)
            {
                transpose(src3, dst, stream);
            }
            else
            {
                if (stream)
                    stream.enqueueCopy(src3, dst);
                else
                    src3.copyTo(dst);
            }
        }
    }

    cublasHandle_t handle;
    cublasSafeCall( cublasCreate_v2(&handle) );

    cublasSafeCall( cublasSetStream_v2(handle, StreamAccessor::getStream(stream)) );

    cublasSafeCall( cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_HOST) );

    const float alphaf = static_cast<float>(alpha);
    const float betaf = static_cast<float>(beta);

    const cuComplex alphacf = make_cuComplex(alphaf, 0);
    const cuComplex betacf = make_cuComplex(betaf, 0);

    const cuDoubleComplex alphac = make_cuDoubleComplex(alpha, 0);
    const cuDoubleComplex betac = make_cuDoubleComplex(beta, 0);

    cublasOperation_t transa = tr2 ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = tr1 ? CUBLAS_OP_T : CUBLAS_OP_N;

    switch (src1.type())
    {
    case CV_32FC1:
        cublasSafeCall( cublasSgemm_v2(handle, transa, transb, tr2 ? src2.rows : src2.cols, tr1 ? src1.cols : src1.rows, tr2 ? src2.cols : src2.rows,
            &alphaf, 
            src2.ptr<float>(), static_cast<int>(src2.step / sizeof(float)),
            src1.ptr<float>(), static_cast<int>(src1.step / sizeof(float)),
            &betaf,
            dst.ptr<float>(), static_cast<int>(dst.step / sizeof(float))) );
        break;

    case CV_64FC1:
        cublasSafeCall( cublasDgemm_v2(handle, transa, transb, tr2 ? src2.rows : src2.cols, tr1 ? src1.cols : src1.rows, tr2 ? src2.cols : src2.rows,
            &alpha, 
            src2.ptr<double>(), static_cast<int>(src2.step / sizeof(double)),
            src1.ptr<double>(), static_cast<int>(src1.step / sizeof(double)),
            &beta,
            dst.ptr<double>(), static_cast<int>(dst.step / sizeof(double))) );
        break;

    case CV_32FC2:
        cublasSafeCall( cublasCgemm_v2(handle, transa, transb, tr2 ? src2.rows : src2.cols, tr1 ? src1.cols : src1.rows, tr2 ? src2.cols : src2.rows,
            &alphacf, 
            src2.ptr<cuComplex>(), static_cast<int>(src2.step / sizeof(cuComplex)),
            src1.ptr<cuComplex>(), static_cast<int>(src1.step / sizeof(cuComplex)),
            &betacf,
            dst.ptr<cuComplex>(), static_cast<int>(dst.step / sizeof(cuComplex))) );
        break;

    case CV_64FC2:
        cublasSafeCall( cublasZgemm_v2(handle, transa, transb, tr2 ? src2.rows : src2.cols, tr1 ? src1.cols : src1.rows, tr2 ? src2.cols : src2.rows,
            &alphac, 
            src2.ptr<cuDoubleComplex>(), static_cast<int>(src2.step / sizeof(cuDoubleComplex)),
            src1.ptr<cuDoubleComplex>(), static_cast<int>(src1.step / sizeof(cuDoubleComplex)),
            &betac,
            dst.ptr<cuDoubleComplex>(), static_cast<int>(dst.step / sizeof(cuDoubleComplex))) );
        break;
    }

    cublasSafeCall( cublasDestroy_v2(handle) );

#endif
}

////////////////////////////////////////////////////////////////////////
// transpose

void cv::gpu::transpose(const GpuMat& src, GpuMat& dst, Stream& s)
{
    CV_Assert(src.elemSize() == 1 || src.elemSize() == 4 || src.elemSize() == 8);

    dst.create( src.cols, src.rows, src.type() );

    cudaStream_t stream = StreamAccessor::getStream(s);

    if (src.elemSize() == 1)
    {
        NppStreamHandler h(stream);

        NppiSize sz;
        sz.width  = src.cols;
        sz.height = src.rows;

        nppSafeCall( nppiTranspose_8u_C1R(src.ptr<Npp8u>(), static_cast<int>(src.step), 
            dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz) );		
    }
    else if (src.elemSize() == 4)
    {
        NppStStreamHandler h(stream);

        NcvSize32u sz;
        sz.width  = src.cols;
        sz.height = src.rows;

        ncvSafeCall( nppiStTranspose_32u_C1R(const_cast<Ncv32u*>(src.ptr<Ncv32u>()), static_cast<int>(src.step), 
            dst.ptr<Ncv32u>(), static_cast<int>(dst.step), sz) );
    }
    else // if (src.elemSize() == 8)
    {
        NppStStreamHandler h(stream);

        NcvSize32u sz;
        sz.width  = src.cols;
        sz.height = src.rows;

        ncvSafeCall( nppiStTranspose_64u_C1R(const_cast<Ncv64u*>(src.ptr<Ncv64u>()), static_cast<int>(src.step), 
            dst.ptr<Ncv64u>(), static_cast<int>(dst.step), sz) );		
    }

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}

////////////////////////////////////////////////////////////////////////
// flip

void cv::gpu::flip(const GpuMat& src, GpuMat& dst, int flipCode, Stream& s)
{
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC4);

    dst.create( src.size(), src.type() );

    NppiSize sz;
    sz.width  = src.cols;
    sz.height = src.rows;

    cudaStream_t stream = StreamAccessor::getStream(s);

    NppStreamHandler h(stream);

    if (src.type() == CV_8UC1)
    {
        nppSafeCall( nppiMirror_8u_C1R(src.ptr<Npp8u>(), static_cast<int>(src.step),
            dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz,
            (flipCode == 0 ? NPP_HORIZONTAL_AXIS : (flipCode > 0 ? NPP_VERTICAL_AXIS : NPP_BOTH_AXIS))) );
    }
    else
    {
        nppSafeCall( nppiMirror_8u_C4R(src.ptr<Npp8u>(), static_cast<int>(src.step),
            dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz,
            (flipCode == 0 ? NPP_HORIZONTAL_AXIS : (flipCode > 0 ? NPP_VERTICAL_AXIS : NPP_BOTH_AXIS))) );
    }

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}

////////////////////////////////////////////////////////////////////////
// LUT

void cv::gpu::LUT(const GpuMat& src, const Mat& lut, GpuMat& dst, Stream& s)
{
    class LevelsInit
    {
    public:
        Npp32s pLevels[256];
        const Npp32s* pLevels3[3];
        int nValues3[3];

        LevelsInit()
        {
            nValues3[0] = nValues3[1] = nValues3[2] = 256;
            for (int i = 0; i < 256; ++i)
                pLevels[i] = i;
            pLevels3[0] = pLevels3[1] = pLevels3[2] = pLevels;
        }
    };
    static LevelsInit lvls;

    int cn = src.channels();

    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3);
    CV_Assert(lut.depth() == CV_8U && (lut.channels() == 1 || lut.channels() == cn) && lut.rows * lut.cols == 256 && lut.isContinuous());

    dst.create(src.size(), CV_MAKETYPE(lut.depth(), cn));

    NppiSize sz;
    sz.height = src.rows;
    sz.width = src.cols;

    Mat nppLut;
    lut.convertTo(nppLut, CV_32S);

    cudaStream_t stream = StreamAccessor::getStream(s);

    NppStreamHandler h(stream);

    if (src.type() == CV_8UC1)
    {
        nppSafeCall( nppiLUT_Linear_8u_C1R(src.ptr<Npp8u>(), static_cast<int>(src.step), 
            dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz, nppLut.ptr<Npp32s>(), lvls.pLevels, 256) );
    }
    else
    {
        Mat nppLut3[3];
        const Npp32s* pValues3[3];
        if (nppLut.channels() == 1)
            pValues3[0] = pValues3[1] = pValues3[2] = nppLut.ptr<Npp32s>();
        else
        {
            cv::split(nppLut, nppLut3);
            pValues3[0] = nppLut3[0].ptr<Npp32s>();
            pValues3[1] = nppLut3[1].ptr<Npp32s>();
            pValues3[2] = nppLut3[2].ptr<Npp32s>();
        }
        nppSafeCall( nppiLUT_Linear_8u_C3R(src.ptr<Npp8u>(), static_cast<int>(src.step), 
            dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz, pValues3, lvls.pLevels3, lvls.nValues3) );
    }

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}

////////////////////////////////////////////////////////////////////////
// exp

void cv::gpu::exp(const GpuMat& src, GpuMat& dst, Stream& s)
{
    CV_Assert(src.type() == CV_32FC1);

    dst.create(src.size(), src.type());

    NppiSize sz;
    sz.width = src.cols;
    sz.height = src.rows;

    cudaStream_t stream = StreamAccessor::getStream(s);

    NppStreamHandler h(stream);

    nppSafeCall( nppiExp_32f_C1R(src.ptr<Npp32f>(), static_cast<int>(src.step), dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz) );

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}

////////////////////////////////////////////////////////////////////////
// log

void cv::gpu::log(const GpuMat& src, GpuMat& dst, Stream& s)
{
    CV_Assert(src.type() == CV_32FC1);

    dst.create(src.size(), src.type());

    NppiSize sz;
    sz.width = src.cols;
    sz.height = src.rows;

    cudaStream_t stream = StreamAccessor::getStream(s);

    NppStreamHandler h(stream);

    nppSafeCall( nppiLn_32f_C1R(src.ptr<Npp32f>(), static_cast<int>(src.step), dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz) );

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}

////////////////////////////////////////////////////////////////////////
// NPP magnitide

namespace
{
    typedef NppStatus (*nppMagnitude_t)(const Npp32fc* pSrc, int nSrcStep, Npp32f* pDst, int nDstStep, NppiSize oSizeROI);

    inline void npp_magnitude(const GpuMat& src, GpuMat& dst, nppMagnitude_t func, cudaStream_t stream)
    {
        CV_Assert(src.type() == CV_32FC2);

        dst.create(src.size(), CV_32FC1);

        NppiSize sz;
        sz.width = src.cols;
        sz.height = src.rows;

        NppStreamHandler h(stream);

        nppSafeCall( func(src.ptr<Npp32fc>(), static_cast<int>(src.step), dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz) );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
}

void cv::gpu::magnitude(const GpuMat& src, GpuMat& dst, Stream& stream)
{
    ::npp_magnitude(src, dst, nppiMagnitude_32fc32f_C1R, StreamAccessor::getStream(stream));
}

void cv::gpu::magnitudeSqr(const GpuMat& src, GpuMat& dst, Stream& stream)
{
    ::npp_magnitude(src, dst, nppiMagnitudeSqr_32fc32f_C1R, StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// Polar <-> Cart

namespace cv { namespace gpu { namespace mathfunc
{
    void cartToPolar_gpu(const DevMem2Df& x, const DevMem2Df& y, const DevMem2Df& mag, bool magSqr, const DevMem2Df& angle, bool angleInDegrees, cudaStream_t stream);
    void polarToCart_gpu(const DevMem2Df& mag, const DevMem2Df& angle, const DevMem2Df& x, const DevMem2Df& y, bool angleInDegrees, cudaStream_t stream);
}}}

namespace
{
    inline void cartToPolar_caller(const GpuMat& x, const GpuMat& y, GpuMat* mag, bool magSqr, GpuMat* angle, bool angleInDegrees, cudaStream_t stream)
    {
        CV_DbgAssert(x.size() == y.size() && x.type() == y.type());
        CV_Assert(x.depth() == CV_32F);

        if (mag)
            mag->create(x.size(), x.type());
        if (angle)
            angle->create(x.size(), x.type());

        GpuMat x1cn = x.reshape(1);
        GpuMat y1cn = y.reshape(1);
        GpuMat mag1cn = mag ? mag->reshape(1) : GpuMat();
        GpuMat angle1cn = angle ? angle->reshape(1) : GpuMat();

        mathfunc::cartToPolar_gpu(x1cn, y1cn, mag1cn, magSqr, angle1cn, angleInDegrees, stream);
    }

    inline void polarToCart_caller(const GpuMat& mag, const GpuMat& angle, GpuMat& x, GpuMat& y, bool angleInDegrees, cudaStream_t stream)
    {
        CV_DbgAssert((mag.empty() || mag.size() == angle.size()) && mag.type() == angle.type());
        CV_Assert(mag.depth() == CV_32F);

        x.create(mag.size(), mag.type());
        y.create(mag.size(), mag.type());

        GpuMat mag1cn = mag.reshape(1);
        GpuMat angle1cn = angle.reshape(1);
        GpuMat x1cn = x.reshape(1);
        GpuMat y1cn = y.reshape(1);

        mathfunc::polarToCart_gpu(mag1cn, angle1cn, x1cn, y1cn, angleInDegrees, stream);
    }
}

void cv::gpu::magnitude(const GpuMat& x, const GpuMat& y, GpuMat& dst, Stream& stream)
{
    ::cartToPolar_caller(x, y, &dst, false, 0, false, StreamAccessor::getStream(stream));
}

void cv::gpu::magnitudeSqr(const GpuMat& x, const GpuMat& y, GpuMat& dst, Stream& stream)
{
    ::cartToPolar_caller(x, y, &dst, true, 0, false, StreamAccessor::getStream(stream));
}

void cv::gpu::phase(const GpuMat& x, const GpuMat& y, GpuMat& angle, bool angleInDegrees, Stream& stream)
{
    ::cartToPolar_caller(x, y, 0, false, &angle, angleInDegrees, StreamAccessor::getStream(stream));
}

void cv::gpu::cartToPolar(const GpuMat& x, const GpuMat& y, GpuMat& mag, GpuMat& angle, bool angleInDegrees, Stream& stream)
{
    ::cartToPolar_caller(x, y, &mag, false, &angle, angleInDegrees, StreamAccessor::getStream(stream));
}

void cv::gpu::polarToCart(const GpuMat& magnitude, const GpuMat& angle, GpuMat& x, GpuMat& y, bool angleInDegrees, Stream& stream)
{
    ::polarToCart_caller(magnitude, angle, x, y, angleInDegrees, StreamAccessor::getStream(stream));
}


#endif /* !defined (HAVE_CUDA) */
