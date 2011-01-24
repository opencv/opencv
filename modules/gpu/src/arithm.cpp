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

void cv::gpu::transpose(const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::flip(const GpuMat&, GpuMat&, int) { throw_nogpu(); }
void cv::gpu::LUT(const GpuMat&, const Mat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::exp(const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::log(const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::magnitude(const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::magnitudeSqr(const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::magnitude(const GpuMat&, const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::magnitude(const GpuMat&, const GpuMat&, GpuMat&, const Stream&) { throw_nogpu(); }
void cv::gpu::magnitudeSqr(const GpuMat&, const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::magnitudeSqr(const GpuMat&, const GpuMat&, GpuMat&, const Stream&) { throw_nogpu(); }
void cv::gpu::phase(const GpuMat&, const GpuMat&, GpuMat&, bool) { throw_nogpu(); }
void cv::gpu::phase(const GpuMat&, const GpuMat&, GpuMat&, bool, const Stream&) { throw_nogpu(); }
void cv::gpu::cartToPolar(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, bool) { throw_nogpu(); }
void cv::gpu::cartToPolar(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, bool, const Stream&) { throw_nogpu(); }
void cv::gpu::polarToCart(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, bool) { throw_nogpu(); }
void cv::gpu::polarToCart(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, bool, const Stream&) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

////////////////////////////////////////////////////////////////////////
// transpose

void cv::gpu::transpose(const GpuMat& src, GpuMat& dst)
{
    CV_Assert(src.elemSize() == 1 || src.elemSize() == 4 || src.elemSize() == 8);

    dst.create( src.cols, src.rows, src.type() );

    if (src.elemSize() == 1)
    {
        NppiSize sz;
        sz.width  = src.cols;
        sz.height = src.rows;

        nppSafeCall( nppiTranspose_8u_C1R(src.ptr<Npp8u>(), src.step, dst.ptr<Npp8u>(), dst.step, sz) );
    }
    else if (src.elemSize() == 4)
    {
        NppStSize32u sz;
        sz.width  = src.cols;
        sz.height = src.rows;

        nppSafeCall( nppiStTranspose_32u_C1R(const_cast<NppSt32u*>(src.ptr<NppSt32u>()), src.step, 
            dst.ptr<NppSt32u>(), dst.step, sz) );
    }
    else // if (src.elemSize() == 8)
    {
        NppStSize32u sz;
        sz.width  = src.cols;
        sz.height = src.rows;

        nppSafeCall( nppiStTranspose_64u_C1R(const_cast<NppSt64u*>(src.ptr<NppSt64u>()), src.step, 
            dst.ptr<NppSt64u>(), dst.step, sz) );
    }

    cudaSafeCall( cudaThreadSynchronize() );
}

////////////////////////////////////////////////////////////////////////
// flip

void cv::gpu::flip(const GpuMat& src, GpuMat& dst, int flipCode)
{
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC4);

    dst.create( src.size(), src.type() );

    NppiSize sz;
    sz.width  = src.cols;
    sz.height = src.rows;

    if (src.type() == CV_8UC1)
    {
        nppSafeCall( nppiMirror_8u_C1R(src.ptr<Npp8u>(), src.step,
            dst.ptr<Npp8u>(), dst.step, sz,
            (flipCode == 0 ? NPP_HORIZONTAL_AXIS : (flipCode > 0 ? NPP_VERTICAL_AXIS : NPP_BOTH_AXIS))) );
    }
    else
    {
        nppSafeCall( nppiMirror_8u_C4R(src.ptr<Npp8u>(), src.step,
            dst.ptr<Npp8u>(), dst.step, sz,
            (flipCode == 0 ? NPP_HORIZONTAL_AXIS : (flipCode > 0 ? NPP_VERTICAL_AXIS : NPP_BOTH_AXIS))) );
    }

    cudaSafeCall( cudaThreadSynchronize() );
}

////////////////////////////////////////////////////////////////////////
// LUT

void cv::gpu::LUT(const GpuMat& src, const Mat& lut, GpuMat& dst)
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

    if (src.type() == CV_8UC1)
    {
        nppSafeCall( nppiLUT_Linear_8u_C1R(src.ptr<Npp8u>(), src.step, dst.ptr<Npp8u>(), dst.step, sz, nppLut.ptr<Npp32s>(), lvls.pLevels, 256) );
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
        nppSafeCall( nppiLUT_Linear_8u_C3R(src.ptr<Npp8u>(), src.step, dst.ptr<Npp8u>(), dst.step, sz, pValues3, lvls.pLevels3, lvls.nValues3) );
    }

    cudaSafeCall( cudaThreadSynchronize() );
}

////////////////////////////////////////////////////////////////////////
// exp

void cv::gpu::exp(const GpuMat& src, GpuMat& dst)
{
    CV_Assert(src.type() == CV_32FC1);

    dst.create(src.size(), src.type());

    NppiSize sz;
    sz.width = src.cols;
    sz.height = src.rows;

    nppSafeCall( nppiExp_32f_C1R(src.ptr<Npp32f>(), src.step, dst.ptr<Npp32f>(), dst.step, sz) );

    cudaSafeCall( cudaThreadSynchronize() );
}

////////////////////////////////////////////////////////////////////////
// log

void cv::gpu::log(const GpuMat& src, GpuMat& dst)
{
    CV_Assert(src.type() == CV_32FC1);

    dst.create(src.size(), src.type());

    NppiSize sz;
    sz.width = src.cols;
    sz.height = src.rows;

    nppSafeCall( nppiLn_32f_C1R(src.ptr<Npp32f>(), src.step, dst.ptr<Npp32f>(), dst.step, sz) );

    cudaSafeCall( cudaThreadSynchronize() );
}

////////////////////////////////////////////////////////////////////////
// NPP magnitide

namespace
{
    typedef NppStatus (*nppMagnitude_t)(const Npp32fc* pSrc, int nSrcStep, Npp32f* pDst, int nDstStep, NppiSize oSizeROI);

    inline void npp_magnitude(const GpuMat& src, GpuMat& dst, nppMagnitude_t func)
    {
        CV_Assert(src.type() == CV_32FC2);

        dst.create(src.size(), CV_32FC1);

        NppiSize sz;
        sz.width = src.cols;
        sz.height = src.rows;

        nppSafeCall( func(src.ptr<Npp32fc>(), src.step, dst.ptr<Npp32f>(), dst.step, sz) );

        cudaSafeCall( cudaThreadSynchronize() );
    }
}

void cv::gpu::magnitude(const GpuMat& src, GpuMat& dst)
{
    ::npp_magnitude(src, dst, nppiMagnitude_32fc32f_C1R);
}

void cv::gpu::magnitudeSqr(const GpuMat& src, GpuMat& dst)
{
    ::npp_magnitude(src, dst, nppiMagnitudeSqr_32fc32f_C1R);
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

void cv::gpu::magnitude(const GpuMat& x, const GpuMat& y, GpuMat& dst)
{
    ::cartToPolar_caller(x, y, &dst, false, 0, false, 0);
}

void cv::gpu::magnitude(const GpuMat& x, const GpuMat& y, GpuMat& dst, const Stream& stream)
{
    ::cartToPolar_caller(x, y, &dst, false, 0, false, StreamAccessor::getStream(stream));
}

void cv::gpu::magnitudeSqr(const GpuMat& x, const GpuMat& y, GpuMat& dst)
{
    ::cartToPolar_caller(x, y, &dst, true, 0, false, 0);
}

void cv::gpu::magnitudeSqr(const GpuMat& x, const GpuMat& y, GpuMat& dst, const Stream& stream)
{
    ::cartToPolar_caller(x, y, &dst, true, 0, false, StreamAccessor::getStream(stream));
}

void cv::gpu::phase(const GpuMat& x, const GpuMat& y, GpuMat& angle, bool angleInDegrees)
{
    ::cartToPolar_caller(x, y, 0, false, &angle, angleInDegrees, 0);
}

void cv::gpu::phase(const GpuMat& x, const GpuMat& y, GpuMat& angle, bool angleInDegrees, const Stream& stream)
{
    ::cartToPolar_caller(x, y, 0, false, &angle, angleInDegrees, StreamAccessor::getStream(stream));
}

void cv::gpu::cartToPolar(const GpuMat& x, const GpuMat& y, GpuMat& mag, GpuMat& angle, bool angleInDegrees)
{
    ::cartToPolar_caller(x, y, &mag, false, &angle, angleInDegrees, 0);
}

void cv::gpu::cartToPolar(const GpuMat& x, const GpuMat& y, GpuMat& mag, GpuMat& angle, bool angleInDegrees, const Stream& stream)
{
    ::cartToPolar_caller(x, y, &mag, false, &angle, angleInDegrees, StreamAccessor::getStream(stream));
}

void cv::gpu::polarToCart(const GpuMat& magnitude, const GpuMat& angle, GpuMat& x, GpuMat& y, bool angleInDegrees)
{
    ::polarToCart_caller(magnitude, angle, x, y, angleInDegrees, 0);
}

void cv::gpu::polarToCart(const GpuMat& magnitude, const GpuMat& angle, GpuMat& x, GpuMat& y, bool angleInDegrees, const Stream& stream)
{
    ::polarToCart_caller(magnitude, angle, x, y, angleInDegrees, StreamAccessor::getStream(stream));
}


#endif /* !defined (HAVE_CUDA) */
