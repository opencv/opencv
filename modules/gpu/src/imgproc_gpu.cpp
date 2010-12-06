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

#include "precomp.hpp"

using namespace cv;
using namespace cv::gpu;

#if !defined (HAVE_CUDA)

void cv::gpu::remap(const GpuMat&, GpuMat&, const GpuMat&, const GpuMat&){ throw_nogpu(); }
void cv::gpu::meanShiftFiltering(const GpuMat&, GpuMat&, int, int, TermCriteria) { throw_nogpu(); }
void cv::gpu::meanShiftProc(const GpuMat&, GpuMat&, GpuMat&, int, int, TermCriteria) { throw_nogpu(); }
void cv::gpu::drawColorDisp(const GpuMat&, GpuMat&, int) { throw_nogpu(); }
void cv::gpu::drawColorDisp(const GpuMat&, GpuMat&, int, const Stream&) { throw_nogpu(); }
void cv::gpu::reprojectImageTo3D(const GpuMat&, GpuMat&, const Mat&) { throw_nogpu(); }
void cv::gpu::reprojectImageTo3D(const GpuMat&, GpuMat&, const Mat&, const Stream&) { throw_nogpu(); }
double cv::gpu::threshold(const GpuMat&, GpuMat&, double) { throw_nogpu(); return 0.0; }
void cv::gpu::resize(const GpuMat&, GpuMat&, Size, double, double, int) { throw_nogpu(); }
void cv::gpu::copyMakeBorder(const GpuMat&, GpuMat&, int, int, int, int, const Scalar&) { throw_nogpu(); }
void cv::gpu::warpAffine(const GpuMat&, GpuMat&, const Mat&, Size, int) { throw_nogpu(); }
void cv::gpu::warpPerspective(const GpuMat&, GpuMat&, const Mat&, Size, int) { throw_nogpu(); }
void cv::gpu::rotate(const GpuMat&, GpuMat&, Size, double, double, double, int) { throw_nogpu(); }
void cv::gpu::integral(GpuMat&, GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::rectStdDev(const GpuMat&, const GpuMat&, GpuMat&, const Rect&) { throw_nogpu(); }
void cv::gpu::Canny(const GpuMat&, GpuMat&, double, double, int) { throw_nogpu(); }
void cv::gpu::evenLevels(GpuMat&, int, int, int) { throw_nogpu(); }
void cv::gpu::histEven(const GpuMat&, GpuMat&, int, int, int) { throw_nogpu(); }
void cv::gpu::histEven(const GpuMat&, GpuMat*, int*, int*, int*) { throw_nogpu(); }
void cv::gpu::histRange(const GpuMat&, GpuMat&, const GpuMat&) { throw_nogpu(); }
void cv::gpu::histRange(const GpuMat&, GpuMat*, const GpuMat*) { throw_nogpu(); }
void cv::gpu::cornerHarris(const GpuMat&, GpuMat&, int, int, double, int) { throw_nogpu(); }
void cv::gpu::cornerMinEigenVal(const GpuMat&, GpuMat&, int, int, int) { throw_nogpu(); }


#else /* !defined (HAVE_CUDA) */

namespace cv { namespace gpu {  namespace imgproc
{
    void remap_gpu_1c(const DevMem2D& src, const DevMem2Df& xmap, const DevMem2Df& ymap, DevMem2D dst);
    void remap_gpu_3c(const DevMem2D& src, const DevMem2Df& xmap, const DevMem2Df& ymap, DevMem2D dst);

    extern "C" void meanShiftFiltering_gpu(const DevMem2D& src, DevMem2D dst, int sp, int sr, int maxIter, float eps);
    extern "C" void meanShiftProc_gpu(const DevMem2D& src, DevMem2D dstr, DevMem2D dstsp, int sp, int sr, int maxIter, float eps);

    void drawColorDisp_gpu(const DevMem2D& src, const DevMem2D& dst, int ndisp, const cudaStream_t& stream);
    void drawColorDisp_gpu(const DevMem2D_<short>& src, const DevMem2D& dst, int ndisp, const cudaStream_t& stream);

    void reprojectImageTo3D_gpu(const DevMem2D& disp, const DevMem2Df& xyzw, const float* q, const cudaStream_t& stream);
    void reprojectImageTo3D_gpu(const DevMem2D_<short>& disp, const DevMem2Df& xyzw, const float* q, const cudaStream_t& stream);
}}}

////////////////////////////////////////////////////////////////////////
// remap

void cv::gpu::remap(const GpuMat& src, GpuMat& dst, const GpuMat& xmap, const GpuMat& ymap)
{
    typedef void (*remap_gpu_t)(const DevMem2D& src, const DevMem2Df& xmap, const DevMem2Df& ymap, DevMem2D dst);
    static const remap_gpu_t callers[] = {imgproc::remap_gpu_1c, 0, imgproc::remap_gpu_3c};

    CV_Assert((src.type() == CV_8U || src.type() == CV_8UC3) && xmap.type() == CV_32F && ymap.type() == CV_32F);

    GpuMat out;
    if (dst.data != src.data)
        out = dst;

    out.create(xmap.size(), src.type());

    callers[src.channels() - 1](src, xmap, ymap, out);

    dst = out;
}

////////////////////////////////////////////////////////////////////////
// meanShiftFiltering_GPU

void cv::gpu::meanShiftFiltering(const GpuMat& src, GpuMat& dst, int sp, int sr, TermCriteria criteria)
{
    if( src.empty() )
        CV_Error( CV_StsBadArg, "The input image is empty" );

    if( src.depth() != CV_8U || src.channels() != 4 )
        CV_Error( CV_StsUnsupportedFormat, "Only 8-bit, 4-channel images are supported" );

    dst.create( src.size(), CV_8UC4 );

    if( !(criteria.type & TermCriteria::MAX_ITER) )
        criteria.maxCount = 5;

    int maxIter = std::min(std::max(criteria.maxCount, 1), 100);

    float eps;
    if( !(criteria.type & TermCriteria::EPS) )
        eps = 1.f;
    eps = (float)std::max(criteria.epsilon, 0.0);

    imgproc::meanShiftFiltering_gpu(src, dst, sp, sr, maxIter, eps);
}

////////////////////////////////////////////////////////////////////////
// meanShiftProc_GPU

void cv::gpu::meanShiftProc(const GpuMat& src, GpuMat& dstr, GpuMat& dstsp, int sp, int sr, TermCriteria criteria)
{
    if( src.empty() )
        CV_Error( CV_StsBadArg, "The input image is empty" );

    if( src.depth() != CV_8U || src.channels() != 4 )
        CV_Error( CV_StsUnsupportedFormat, "Only 8-bit, 4-channel images are supported" );

    dstr.create( src.size(), CV_8UC4 );
    dstsp.create( src.size(), CV_16SC2 );

    if( !(criteria.type & TermCriteria::MAX_ITER) )
        criteria.maxCount = 5;

    int maxIter = std::min(std::max(criteria.maxCount, 1), 100);

    float eps;
    if( !(criteria.type & TermCriteria::EPS) )
        eps = 1.f;
    eps = (float)std::max(criteria.epsilon, 0.0);

    imgproc::meanShiftProc_gpu(src, dstr, dstsp, sp, sr, maxIter, eps);
}

////////////////////////////////////////////////////////////////////////
// drawColorDisp

namespace
{
    template <typename T>
    void drawColorDisp_caller(const GpuMat& src, GpuMat& dst, int ndisp, const cudaStream_t& stream)
    {
        GpuMat out;
        if (dst.data != src.data)
            out = dst;
        out.create(src.size(), CV_8UC4);

        imgproc::drawColorDisp_gpu((DevMem2D_<T>)src, out, ndisp, stream);

        dst = out;
    }

    typedef void (*drawColorDisp_caller_t)(const GpuMat& src, GpuMat& dst, int ndisp, const cudaStream_t& stream);

    const drawColorDisp_caller_t drawColorDisp_callers[] = {drawColorDisp_caller<unsigned char>, 0, 0, drawColorDisp_caller<short>, 0, 0, 0, 0};
}

void cv::gpu::drawColorDisp(const GpuMat& src, GpuMat& dst, int ndisp)
{
    CV_Assert(src.type() == CV_8U || src.type() == CV_16S);

    drawColorDisp_callers[src.type()](src, dst, ndisp, 0);
}

void cv::gpu::drawColorDisp(const GpuMat& src, GpuMat& dst, int ndisp, const Stream& stream)
{
    CV_Assert(src.type() == CV_8U || src.type() == CV_16S);

    drawColorDisp_callers[src.type()](src, dst, ndisp, StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// reprojectImageTo3D

namespace
{
    template <typename T>
    void reprojectImageTo3D_caller(const GpuMat& disp, GpuMat& xyzw, const Mat& Q, const cudaStream_t& stream)
    {
        xyzw.create(disp.rows, disp.cols, CV_32FC4);
        imgproc::reprojectImageTo3D_gpu((DevMem2D_<T>)disp, xyzw, Q.ptr<float>(), stream);
    }

    typedef void (*reprojectImageTo3D_caller_t)(const GpuMat& disp, GpuMat& xyzw, const Mat& Q, const cudaStream_t& stream);

    const reprojectImageTo3D_caller_t reprojectImageTo3D_callers[] = {reprojectImageTo3D_caller<unsigned char>, 0, 0, reprojectImageTo3D_caller<short>, 0, 0, 0, 0};
}

void cv::gpu::reprojectImageTo3D(const GpuMat& disp, GpuMat& xyzw, const Mat& Q)
{
    CV_Assert((disp.type() == CV_8U || disp.type() == CV_16S) && Q.type() == CV_32F && Q.rows == 4 && Q.cols == 4);

    reprojectImageTo3D_callers[disp.type()](disp, xyzw, Q, 0);
}

void cv::gpu::reprojectImageTo3D(const GpuMat& disp, GpuMat& xyzw, const Mat& Q, const Stream& stream)
{
    CV_Assert((disp.type() == CV_8U || disp.type() == CV_16S) && Q.type() == CV_32F && Q.rows == 4 && Q.cols == 4);

    reprojectImageTo3D_callers[disp.type()](disp, xyzw, Q, StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// threshold

double cv::gpu::threshold(const GpuMat& src, GpuMat& dst, double thresh)
{
    CV_Assert(src.type() == CV_32FC1);

    dst.create( src.size(), src.type() );

    NppiSize sz;
    sz.width  = src.cols;
    sz.height = src.rows;

    nppSafeCall( nppiThreshold_32f_C1R(src.ptr<Npp32f>(), src.step,
        dst.ptr<Npp32f>(), dst.step, sz, static_cast<Npp32f>(thresh), NPP_CMP_GREATER) );

    return thresh;
}

////////////////////////////////////////////////////////////////////////
// resize

void cv::gpu::resize(const GpuMat& src, GpuMat& dst, Size dsize, double fx, double fy, int interpolation)
{
    static const int npp_inter[] = {NPPI_INTER_NN, NPPI_INTER_LINEAR/*, NPPI_INTER_CUBIC, 0, NPPI_INTER_LANCZOS*/};

    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC4);
    CV_Assert(interpolation == INTER_NEAREST || interpolation == INTER_LINEAR/* || interpolation == INTER_CUBIC || interpolation == INTER_LANCZOS4*/);

    CV_Assert( src.size().area() > 0 );
    CV_Assert( !(dsize == Size()) || (fx > 0 && fy > 0) );

    if( dsize == Size() )
    {
        dsize = Size(saturate_cast<int>(src.cols * fx), saturate_cast<int>(src.rows * fy));
    }
    else
    {
        fx = (double)dsize.width / src.cols;
        fy = (double)dsize.height / src.rows;
    }

    dst.create(dsize, src.type());

    NppiSize srcsz;
    srcsz.width  = src.cols;
    srcsz.height = src.rows;
    NppiRect srcrect;
    srcrect.x = srcrect.y = 0;
    srcrect.width  = src.cols;
    srcrect.height = src.rows;
    NppiSize dstsz;
    dstsz.width  = dst.cols;
    dstsz.height = dst.rows;

    if (src.type() == CV_8UC1)
    {
        nppSafeCall( nppiResize_8u_C1R(src.ptr<Npp8u>(), srcsz, src.step, srcrect,
            dst.ptr<Npp8u>(), dst.step, dstsz, fx, fy, npp_inter[interpolation]) );
    }
    else
    {
        nppSafeCall( nppiResize_8u_C4R(src.ptr<Npp8u>(), srcsz, src.step, srcrect,
            dst.ptr<Npp8u>(), dst.step, dstsz, fx, fy, npp_inter[interpolation]) );
    }
}

////////////////////////////////////////////////////////////////////////
// copyMakeBorder

void cv::gpu::copyMakeBorder(const GpuMat& src, GpuMat& dst, int top, int bottom, int left, int right, const Scalar& value)
{
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC4 || src.type() == CV_32SC1);

    dst.create(src.rows + top + bottom, src.cols + left + right, src.type());

	NppiSize srcsz;
	srcsz.width  = src.cols;
	srcsz.height = src.rows;
    NppiSize dstsz;
        dstsz.width  = dst.cols;
        dstsz.height = dst.rows;

    switch (src.type())
    {
    case CV_8UC1:
                {
            Npp8u nVal = static_cast<Npp8u>(value[0]);
            nppSafeCall( nppiCopyConstBorder_8u_C1R(src.ptr<Npp8u>(), src.step, srcsz,
                dst.ptr<Npp8u>(), dst.step, dstsz, top, left, nVal) );
            break;
                }
    case CV_8UC4:
                {
            Npp8u nVal[] = {static_cast<Npp8u>(value[0]), static_cast<Npp8u>(value[1]), static_cast<Npp8u>(value[2]), static_cast<Npp8u>(value[3])};
            nppSafeCall( nppiCopyConstBorder_8u_C4R(src.ptr<Npp8u>(), src.step, srcsz,
                dst.ptr<Npp8u>(), dst.step, dstsz, top, left, nVal) );
            break;
                }
    case CV_32SC1:
            {
            Npp32s nVal = static_cast<Npp32s>(value[0]);
            nppSafeCall( nppiCopyConstBorder_32s_C1R(src.ptr<Npp32s>(), src.step, srcsz,
                dst.ptr<Npp32s>(), dst.step, dstsz, top, left, nVal) );
            break;
            }
    default:
        CV_Assert(!"Unsupported source type");
    }
}

////////////////////////////////////////////////////////////////////////
// warp

namespace
{
    typedef NppStatus (*npp_warp_8u_t)(const Npp8u* pSrc, NppiSize srcSize, int srcStep, NppiRect srcRoi, Npp8u* pDst,
                                       int dstStep, NppiRect dstRoi, const double coeffs[][3],
                                       int interpolation);
    typedef NppStatus (*npp_warp_16u_t)(const Npp16u* pSrc, NppiSize srcSize, int srcStep, NppiRect srcRoi, Npp16u* pDst,
                                       int dstStep, NppiRect dstRoi, const double coeffs[][3],
                                       int interpolation);
    typedef NppStatus (*npp_warp_32s_t)(const Npp32s* pSrc, NppiSize srcSize, int srcStep, NppiRect srcRoi, Npp32s* pDst,
                                       int dstStep, NppiRect dstRoi, const double coeffs[][3],
                                       int interpolation);
    typedef NppStatus (*npp_warp_32f_t)(const Npp32f* pSrc, NppiSize srcSize, int srcStep, NppiRect srcRoi, Npp32f* pDst,
                                       int dstStep, NppiRect dstRoi, const double coeffs[][3],
                                       int interpolation);

    void nppWarpCaller(const GpuMat& src, GpuMat& dst, double coeffs[][3], const Size& dsize, int flags,
                       npp_warp_8u_t npp_warp_8u[][2], npp_warp_16u_t npp_warp_16u[][2],
                       npp_warp_32s_t npp_warp_32s[][2], npp_warp_32f_t npp_warp_32f[][2])
    {
        static const int npp_inter[] = {NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC};

        int interpolation = flags & INTER_MAX;

        CV_Assert((src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32S || src.depth() == CV_32F) && src.channels() != 2);
        CV_Assert(interpolation == INTER_NEAREST || interpolation == INTER_LINEAR || interpolation == INTER_CUBIC);

        dst.create(dsize, src.type());

        NppiSize srcsz;
        srcsz.height = src.rows;
        srcsz.width = src.cols;
        NppiRect srcroi;
        srcroi.x = srcroi.y = 0;
        srcroi.height = src.rows;
        srcroi.width = src.cols;
        NppiRect dstroi;
        dstroi.x = dstroi.y = 0;
        dstroi.height = dst.rows;
        dstroi.width = dst.cols;

        int warpInd = (flags & WARP_INVERSE_MAP) >> 4;

        switch (src.depth())
        {
        case CV_8U:
            nppSafeCall( npp_warp_8u[src.channels()][warpInd](src.ptr<Npp8u>(), srcsz, src.step, srcroi,
                dst.ptr<Npp8u>(), dst.step, dstroi, coeffs, npp_inter[interpolation]) );
            break;
        case CV_16U:
            nppSafeCall( npp_warp_16u[src.channels()][warpInd](src.ptr<Npp16u>(), srcsz, src.step, srcroi,
                dst.ptr<Npp16u>(), dst.step, dstroi, coeffs, npp_inter[interpolation]) );
            break;
        case CV_32S:
            nppSafeCall( npp_warp_32s[src.channels()][warpInd](src.ptr<Npp32s>(), srcsz, src.step, srcroi,
                dst.ptr<Npp32s>(), dst.step, dstroi, coeffs, npp_inter[interpolation]) );
            break;
        case CV_32F:
            nppSafeCall( npp_warp_32f[src.channels()][warpInd](src.ptr<Npp32f>(), srcsz, src.step, srcroi,
                dst.ptr<Npp32f>(), dst.step, dstroi, coeffs, npp_inter[interpolation]) );
            break;
        default:
            CV_Assert(!"Unsupported source type");
        }
    }
}

void cv::gpu::warpAffine(const GpuMat& src, GpuMat& dst, const Mat& M, Size dsize, int flags)
{
    static npp_warp_8u_t npp_warpAffine_8u[][2] =
        {
            {0, 0},
            {nppiWarpAffine_8u_C1R, nppiWarpAffineBack_8u_C1R},
            {0, 0},
            {nppiWarpAffine_8u_C3R, nppiWarpAffineBack_8u_C3R},
            {nppiWarpAffine_8u_C4R, nppiWarpAffineBack_8u_C4R}
        };
    static npp_warp_16u_t npp_warpAffine_16u[][2] =
        {
            {0, 0},
            {nppiWarpAffine_16u_C1R, nppiWarpAffineBack_16u_C1R},
            {0, 0},
            {nppiWarpAffine_16u_C3R, nppiWarpAffineBack_16u_C3R},
            {nppiWarpAffine_16u_C4R, nppiWarpAffineBack_16u_C4R}
        };
    static npp_warp_32s_t npp_warpAffine_32s[][2] =
        {
            {0, 0},
            {nppiWarpAffine_32s_C1R, nppiWarpAffineBack_32s_C1R},
            {0, 0},
            {nppiWarpAffine_32s_C3R, nppiWarpAffineBack_32s_C3R},
            {nppiWarpAffine_32s_C4R, nppiWarpAffineBack_32s_C4R}
        };
    static npp_warp_32f_t npp_warpAffine_32f[][2] =
        {
            {0, 0},
            {nppiWarpAffine_32f_C1R, nppiWarpAffineBack_32f_C1R},
            {0, 0},
            {nppiWarpAffine_32f_C3R, nppiWarpAffineBack_32f_C3R},
            {nppiWarpAffine_32f_C4R, nppiWarpAffineBack_32f_C4R}
        };

    CV_Assert(M.rows == 2 && M.cols == 3);

    double coeffs[2][3];
    Mat coeffsMat(2, 3, CV_64F, (void*)coeffs);
    M.convertTo(coeffsMat, coeffsMat.type());

    nppWarpCaller(src, dst, coeffs, dsize, flags, npp_warpAffine_8u, npp_warpAffine_16u, npp_warpAffine_32s, npp_warpAffine_32f);
}

void cv::gpu::warpPerspective(const GpuMat& src, GpuMat& dst, const Mat& M, Size dsize, int flags)
{
    static npp_warp_8u_t npp_warpPerspective_8u[][2] =
        {
            {0, 0},
            {nppiWarpPerspective_8u_C1R, nppiWarpPerspectiveBack_8u_C1R},
            {0, 0},
            {nppiWarpPerspective_8u_C3R, nppiWarpPerspectiveBack_8u_C3R},
            {nppiWarpPerspective_8u_C4R, nppiWarpPerspectiveBack_8u_C4R}
        };
    static npp_warp_16u_t npp_warpPerspective_16u[][2] =
        {
            {0, 0},
            {nppiWarpPerspective_16u_C1R, nppiWarpPerspectiveBack_16u_C1R},
            {0, 0},
            {nppiWarpPerspective_16u_C3R, nppiWarpPerspectiveBack_16u_C3R},
            {nppiWarpPerspective_16u_C4R, nppiWarpPerspectiveBack_16u_C4R}
        };
    static npp_warp_32s_t npp_warpPerspective_32s[][2] =
        {
            {0, 0},
            {nppiWarpPerspective_32s_C1R, nppiWarpPerspectiveBack_32s_C1R},
            {0, 0},
            {nppiWarpPerspective_32s_C3R, nppiWarpPerspectiveBack_32s_C3R},
            {nppiWarpPerspective_32s_C4R, nppiWarpPerspectiveBack_32s_C4R}
        };
    static npp_warp_32f_t npp_warpPerspective_32f[][2] =
        {
            {0, 0},
            {nppiWarpPerspective_32f_C1R, nppiWarpPerspectiveBack_32f_C1R},
            {0, 0},
            {nppiWarpPerspective_32f_C3R, nppiWarpPerspectiveBack_32f_C3R},
            {nppiWarpPerspective_32f_C4R, nppiWarpPerspectiveBack_32f_C4R}
        };

    CV_Assert(M.rows == 3 && M.cols == 3);

    double coeffs[3][3];
    Mat coeffsMat(3, 3, CV_64F, (void*)coeffs);
    M.convertTo(coeffsMat, coeffsMat.type());

    nppWarpCaller(src, dst, coeffs, dsize, flags, npp_warpPerspective_8u, npp_warpPerspective_16u, npp_warpPerspective_32s, npp_warpPerspective_32f);
}

////////////////////////////////////////////////////////////////////////
// rotate

void cv::gpu::rotate(const GpuMat& src, GpuMat& dst, Size dsize, double angle, double xShift, double yShift, int interpolation)
{
    static const int npp_inter[] = {NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC};

    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC4);
    CV_Assert(interpolation == INTER_NEAREST || interpolation == INTER_LINEAR || interpolation == INTER_CUBIC);

    dst.create(dsize, src.type());

    NppiSize srcsz;
    srcsz.height = src.rows;
    srcsz.width = src.cols;
    NppiRect srcroi;
    srcroi.x = srcroi.y = 0;
    srcroi.height = src.rows;
    srcroi.width = src.cols;
    NppiRect dstroi;
    dstroi.x = dstroi.y = 0;
    dstroi.height = dst.rows;
    dstroi.width = dst.cols;

    if (src.type() == CV_8UC1)
    {
        nppSafeCall( nppiRotate_8u_C1R(src.ptr<Npp8u>(), srcsz, src.step, srcroi,
            dst.ptr<Npp8u>(), dst.step, dstroi, angle, xShift, yShift, npp_inter[interpolation]) );
    }
    else
    {
        nppSafeCall( nppiRotate_8u_C4R(src.ptr<Npp8u>(), srcsz, src.step, srcroi,
            dst.ptr<Npp8u>(), dst.step, dstroi, angle, xShift, yShift, npp_inter[interpolation]) );
    }
}

////////////////////////////////////////////////////////////////////////
// integral

void cv::gpu::integral(GpuMat& src, GpuMat& sum, GpuMat& sqsum)
{
    CV_Assert(src.type() == CV_8UC1);

    int w = src.cols + 1, h = src.rows + 1;

    sum.create(h, w, CV_32S);
    sqsum.create(h, w, CV_32F);

    NppiSize sz;
    sz.width = src.cols;
    sz.height = src.rows;

    nppSafeCall( nppiSqrIntegral_8u32s32f_C1R(src.ptr<Npp8u>(), src.step, sum.ptr<Npp32s>(),
        sum.step, sqsum.ptr<Npp32f>(), sqsum.step, sz, 0, 0.0f, h) );
}

void cv::gpu::rectStdDev(const GpuMat& src, const GpuMat& sqr, GpuMat& dst, const Rect& rect)
{
    CV_Assert(src.type() == CV_32SC1 && sqr.type() == CV_32FC1);

    dst.create(src.size(), CV_32FC1);

    NppiSize sz;
    sz.width = src.cols;
    sz.height = src.rows;

    NppiRect nppRect;
    nppRect.height = rect.height;
    nppRect.width = rect.width;
    nppRect.x = rect.x;
    nppRect.y = rect.y;

    nppSafeCall( nppiRectStdDev_32s32f_C1R(src.ptr<Npp32s>(), src.step, sqr.ptr<Npp32f>(), sqr.step,
                dst.ptr<Npp32f>(), dst.step, sz, nppRect) );
}

////////////////////////////////////////////////////////////////////////
// Canny

void cv::gpu::Canny(const GpuMat& image, GpuMat& edges, double threshold1, double threshold2, int apertureSize)
{
    CV_Assert(!"disabled until fix crash");
    CV_Assert(image.type() == CV_8UC1);

    GpuMat srcDx, srcDy;

    Sobel(image, srcDx, -1, 1, 0, apertureSize);
    Sobel(image, srcDy, -1, 0, 1, apertureSize);

    srcDx.convertTo(srcDx, CV_32F);
    srcDy.convertTo(srcDy, CV_32F);

    edges.create(image.size(), CV_8UC1);

    NppiSize sz;
    sz.height = image.rows;
    sz.width = image.cols;

    int bufsz;
    nppSafeCall( nppiCannyGetBufferSize(sz, &bufsz) );
    GpuMat buf(1, bufsz, CV_8UC1);

    nppSafeCall( nppiCanny_32f8u_C1R(srcDx.ptr<Npp32f>(), srcDx.step, srcDy.ptr<Npp32f>(), srcDy.step,
        edges.ptr<Npp8u>(), edges.step, sz, (Npp32f)threshold1, (Npp32f)threshold2, buf.ptr<Npp8u>()) );
}

////////////////////////////////////////////////////////////////////////
// Histogram

namespace
{
    template<int n> struct NPPTypeTraits;
    template<> struct NPPTypeTraits<CV_8U>  { typedef Npp8u npp_type; };
    template<> struct NPPTypeTraits<CV_16U> { typedef Npp16u npp_type; };
    template<> struct NPPTypeTraits<CV_16S> { typedef Npp16s npp_type; };
    template<> struct NPPTypeTraits<CV_32F> { typedef Npp32f npp_type; };

    typedef NppStatus (*get_buf_size_c1_t)(NppiSize oSizeROI, int nLevels, int* hpBufferSize);
    typedef NppStatus (*get_buf_size_c4_t)(NppiSize oSizeROI, int nLevels[], int* hpBufferSize);

    template<int SDEPTH> struct NppHistogramEvenFuncC1
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

	typedef NppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s * pHist,
		    int nLevels, Npp32s nLowerLevel, Npp32s nUpperLevel, Npp8u * pBuffer);
    };
    template<int SDEPTH> struct NppHistogramEvenFuncC4
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        typedef NppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, NppiSize oSizeROI,
            Npp32s * pHist[4], int nLevels[4], Npp32s nLowerLevel[4], Npp32s nUpperLevel[4], Npp8u * pBuffer);
    };

    template<int SDEPTH, typename NppHistogramEvenFuncC1<SDEPTH>::func_ptr func, get_buf_size_c1_t get_buf_size>
    struct NppHistogramEvenC1
    {
        typedef typename NppHistogramEvenFuncC1<SDEPTH>::src_t src_t;

        static void hist(const GpuMat& src, GpuMat& hist, int histSize, int lowerLevel, int upperLevel)
        {
            int levels = histSize + 1;
            hist.create(1, histSize, CV_32S);

            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            GpuMat buffer;
            int buf_size;

            get_buf_size(sz, levels, &buf_size);
            buffer.create(1, buf_size, CV_8U);
            nppSafeCall( func(src.ptr<src_t>(), src.step, sz, hist.ptr<Npp32s>(), levels,
                lowerLevel, upperLevel, buffer.ptr<Npp8u>()) );
        }
    };
    template<int SDEPTH, typename NppHistogramEvenFuncC4<SDEPTH>::func_ptr func, get_buf_size_c4_t get_buf_size>
    struct NppHistogramEvenC4
    {
        typedef typename NppHistogramEvenFuncC4<SDEPTH>::src_t src_t;

        static void hist(const GpuMat& src, GpuMat hist[4], int histSize[4], int lowerLevel[4], int upperLevel[4])
        {
            int levels[] = {histSize[0] + 1, histSize[1] + 1, histSize[2] + 1, histSize[3] + 1};
            hist[0].create(1, histSize[0], CV_32S);
            hist[1].create(1, histSize[1], CV_32S);
            hist[2].create(1, histSize[2], CV_32S);
            hist[3].create(1, histSize[3], CV_32S);

            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            Npp32s* pHist[] = {hist[0].ptr<Npp32s>(), hist[1].ptr<Npp32s>(), hist[2].ptr<Npp32s>(), hist[3].ptr<Npp32s>()};

            GpuMat buffer;
            int buf_size;

            get_buf_size(sz, levels, &buf_size);
            buffer.create(1, buf_size, CV_8U);
            nppSafeCall( func(src.ptr<src_t>(), src.step, sz, pHist, levels, lowerLevel, upperLevel, buffer.ptr<Npp8u>()) );
        }
    };

    template<int SDEPTH> struct NppHistogramRangeFuncC1
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;
        typedef Npp32s level_t;
        enum {LEVEL_TYPE_CODE=CV_32SC1};

        typedef NppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s* pHist,
            const Npp32s* pLevels, int nLevels, Npp8u* pBuffer);
    };
    template<> struct NppHistogramRangeFuncC1<CV_32F>
    {
        typedef Npp32f src_t;
        typedef Npp32f level_t;
        enum {LEVEL_TYPE_CODE=CV_32FC1};

        typedef NppStatus (*func_ptr)(const Npp32f* pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s* pHist,
            const Npp32f* pLevels, int nLevels, Npp8u* pBuffer);
    };
    template<int SDEPTH> struct NppHistogramRangeFuncC4
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;
        typedef Npp32s level_t;
        enum {LEVEL_TYPE_CODE=CV_32SC1};

        typedef NppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s* pHist[4],
            const Npp32s* pLevels[4], int nLevels[4], Npp8u* pBuffer);
    };
    template<> struct NppHistogramRangeFuncC4<CV_32F>
    {
        typedef Npp32f src_t;
        typedef Npp32f level_t;
        enum {LEVEL_TYPE_CODE=CV_32FC1};

        typedef NppStatus (*func_ptr)(const Npp32f* pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s* pHist[4],
            const Npp32f* pLevels[4], int nLevels[4], Npp8u* pBuffer);
    };

    template<int SDEPTH, typename NppHistogramRangeFuncC1<SDEPTH>::func_ptr func, get_buf_size_c1_t get_buf_size>
    struct NppHistogramRangeC1
    {
        typedef typename NppHistogramRangeFuncC1<SDEPTH>::src_t src_t;
        typedef typename NppHistogramRangeFuncC1<SDEPTH>::level_t level_t;
        enum {LEVEL_TYPE_CODE=NppHistogramRangeFuncC1<SDEPTH>::LEVEL_TYPE_CODE};

        static void hist(const GpuMat& src, GpuMat& hist, const GpuMat& levels)
        {
            CV_Assert(levels.type() == LEVEL_TYPE_CODE && levels.rows == 1);

            hist.create(1, levels.cols - 1, CV_32S);

            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            GpuMat buffer;
            int buf_size;

            get_buf_size(sz, levels.cols, &buf_size);
            buffer.create(1, buf_size, CV_8U);
            nppSafeCall( func(src.ptr<src_t>(), src.step, sz, hist.ptr<Npp32s>(), levels.ptr<level_t>(), levels.cols, buffer.ptr<Npp8u>()) );
        }
    };
    template<int SDEPTH, typename NppHistogramRangeFuncC4<SDEPTH>::func_ptr func, get_buf_size_c4_t get_buf_size>
    struct NppHistogramRangeC4
    {
        typedef typename NppHistogramRangeFuncC4<SDEPTH>::src_t src_t;
        typedef typename NppHistogramRangeFuncC1<SDEPTH>::level_t level_t;
        enum {LEVEL_TYPE_CODE=NppHistogramRangeFuncC1<SDEPTH>::LEVEL_TYPE_CODE};

        static void hist(const GpuMat& src, GpuMat hist[4], const GpuMat levels[4])
        {
            CV_Assert(levels[0].type() == LEVEL_TYPE_CODE && levels[0].rows == 1);
            CV_Assert(levels[1].type() == LEVEL_TYPE_CODE && levels[1].rows == 1);
            CV_Assert(levels[2].type() == LEVEL_TYPE_CODE && levels[2].rows == 1);
            CV_Assert(levels[3].type() == LEVEL_TYPE_CODE && levels[3].rows == 1);

            hist[0].create(1, levels[0].cols - 1, CV_32S);
            hist[1].create(1, levels[1].cols - 1, CV_32S);
            hist[2].create(1, levels[2].cols - 1, CV_32S);
            hist[3].create(1, levels[3].cols - 1, CV_32S);

            Npp32s* pHist[] = {hist[0].ptr<Npp32s>(), hist[1].ptr<Npp32s>(), hist[2].ptr<Npp32s>(), hist[3].ptr<Npp32s>()};
            int nLevels[] = {levels[0].cols, levels[1].cols, levels[2].cols, levels[3].cols};
            const level_t* pLevels[] = {levels[0].ptr<level_t>(), levels[1].ptr<level_t>(), levels[2].ptr<level_t>(), levels[3].ptr<level_t>()};

            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            GpuMat buffer;
            int buf_size;

            get_buf_size(sz, nLevels, &buf_size);
            buffer.create(1, buf_size, CV_8U);
            nppSafeCall( func(src.ptr<src_t>(), src.step, sz, pHist, pLevels, nLevels, buffer.ptr<Npp8u>()) );
        }
    };
}

void cv::gpu::evenLevels(GpuMat& levels, int nLevels, int lowerLevel, int upperLevel)
{
    Mat host_levels(1, nLevels, CV_32SC1);
    nppSafeCall( nppiEvenLevelsHost_32s(host_levels.ptr<Npp32s>(), nLevels, lowerLevel, upperLevel) );
    levels.upload(host_levels);
}

void cv::gpu::histEven(const GpuMat& src, GpuMat& hist, int histSize, int lowerLevel, int upperLevel)
{
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_16UC1 || src.type() == CV_16SC1 );

    typedef void (*hist_t)(const GpuMat& src, GpuMat& hist, int levels, int lowerLevel, int upperLevel);
    static const hist_t hist_callers[] =
    {
        NppHistogramEvenC1<CV_8U , nppiHistogramEven_8u_C1R , nppiHistogramEvenGetBufferSize_8u_C1R >::hist,
        0,
        NppHistogramEvenC1<CV_16U, nppiHistogramEven_16u_C1R, nppiHistogramEvenGetBufferSize_16u_C1R>::hist,
        NppHistogramEvenC1<CV_16S, nppiHistogramEven_16s_C1R, nppiHistogramEvenGetBufferSize_16s_C1R>::hist
    };

    hist_callers[src.depth()](src, hist, histSize, lowerLevel, upperLevel);
}

void cv::gpu::histEven(const GpuMat& src, GpuMat hist[4], int histSize[4], int lowerLevel[4], int upperLevel[4])
{
    CV_Assert(src.type() == CV_8UC4 || src.type() == CV_16UC4 || src.type() == CV_16SC4 );

    typedef void (*hist_t)(const GpuMat& src, GpuMat hist[4], int levels[4], int lowerLevel[4], int upperLevel[4]);
    static const hist_t hist_callers[] =
    {
        NppHistogramEvenC4<CV_8U , nppiHistogramEven_8u_C4R , nppiHistogramEvenGetBufferSize_8u_C4R >::hist,
        0,
        NppHistogramEvenC4<CV_16U, nppiHistogramEven_16u_C4R, nppiHistogramEvenGetBufferSize_16u_C4R>::hist,
        NppHistogramEvenC4<CV_16S, nppiHistogramEven_16s_C4R, nppiHistogramEvenGetBufferSize_16s_C4R>::hist
    };

    hist_callers[src.depth()](src, hist, histSize, lowerLevel, upperLevel);
}

void cv::gpu::histRange(const GpuMat& src, GpuMat& hist, const GpuMat& levels)
{
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_16UC1 || src.type() == CV_16SC1 || src.type() == CV_32FC1);

    typedef void (*hist_t)(const GpuMat& src, GpuMat& hist, const GpuMat& levels);
    static const hist_t hist_callers[] =
    {
        NppHistogramRangeC1<CV_8U , nppiHistogramRange_8u_C1R , nppiHistogramRangeGetBufferSize_8u_C1R >::hist,
        0,
        NppHistogramRangeC1<CV_16U, nppiHistogramRange_16u_C1R, nppiHistogramRangeGetBufferSize_16u_C1R>::hist,
        NppHistogramRangeC1<CV_16S, nppiHistogramRange_16s_C1R, nppiHistogramRangeGetBufferSize_16s_C1R>::hist,
        0,
        NppHistogramRangeC1<CV_32F, nppiHistogramRange_32f_C1R, nppiHistogramRangeGetBufferSize_32f_C1R>::hist
    };

    hist_callers[src.depth()](src, hist, levels);
}

void cv::gpu::histRange(const GpuMat& src, GpuMat hist[4], const GpuMat levels[4])
{
    CV_Assert(src.type() == CV_8UC4 || src.type() == CV_16UC4 || src.type() == CV_16SC4 || src.type() == CV_32FC4);

    typedef void (*hist_t)(const GpuMat& src, GpuMat hist[4], const GpuMat levels[4]);
    static const hist_t hist_callers[] =
    {
        NppHistogramRangeC4<CV_8U , nppiHistogramRange_8u_C4R , nppiHistogramRangeGetBufferSize_8u_C4R >::hist,
        0,
        NppHistogramRangeC4<CV_16U, nppiHistogramRange_16u_C4R, nppiHistogramRangeGetBufferSize_16u_C4R>::hist,
        NppHistogramRangeC4<CV_16S, nppiHistogramRange_16s_C4R, nppiHistogramRangeGetBufferSize_16s_C4R>::hist,
        0,
        NppHistogramRangeC4<CV_32F, nppiHistogramRange_32f_C4R, nppiHistogramRangeGetBufferSize_32f_C4R>::hist
    };

    hist_callers[src.depth()](src, hist, levels);
}

namespace cv { namespace gpu { namespace imgproc {

    void extractCovData_caller(const DevMem2Df Dx, const DevMem2Df Dy, PtrStepf dst);
    void cornerHarris_caller(const int block_size, const float k, const DevMem2D Dx, const DevMem2D Dy, DevMem2D dst, int border_type);
    void cornerMinEigenVal_caller(const int block_size, const DevMem2D Dx, const DevMem2D Dy, DevMem2D dst, int border_type);

}}}

namespace cv { namespace gpu { namespace linear_filters {

    template <typename T>
    void rowFilterCaller(const DevMem2D_<T> src, PtrStepf dst, int anchor, const float* kernel, 
                         int ksize, int brd_interp);

    template <typename T>
    void colFilterCaller(const DevMem2D_<T> src, PtrStepf dst, int anchor, const float* kernel, 
                         int ksize, int brd_interp);

}}}

namespace 
{
    template <typename T>
    void extractCovData(const GpuMat& src, GpuMat& Dx, GpuMat& Dy, int blockSize, int ksize, int gpuBorderType)
    {   
        double scale = (double)(1 << ((ksize > 0 ? ksize : 3) - 1)) * blockSize;
        if (ksize < 0) 
            scale *= 2.;
        if (src.depth() == CV_8U)
            scale *= 255.;
        scale = 1./scale;

        GpuMat tmp_buf(src.size(), CV_32F);
        Dx.create(src.size(), CV_32F);
        Dy.create(src.size(), CV_32F);
        Mat kx, ky;

        getDerivKernels(kx, ky, 1, 0, ksize, false, CV_32F);
        kx = kx.reshape(1, 1) * scale;
        ky = ky.reshape(1, 1);

        linear_filters::rowFilterCaller<T>(
                src, tmp_buf, kx.cols >> 1, kx.ptr<float>(0), kx.cols,
                gpuBorderType);

        linear_filters::colFilterCaller<float>(
                tmp_buf, Dx, ky.cols >> 1, ky.ptr<float>(0), ky.cols, 
                gpuBorderType);

        getDerivKernels(kx, ky, 0, 1, ksize, false, CV_32F);
        kx = kx.reshape(1, 1);
        ky = ky.reshape(1, 1) * scale;

        linear_filters::rowFilterCaller<T>(
                src, tmp_buf, kx.cols >> 1, kx.ptr<float>(0), kx.cols, 
                gpuBorderType);

        linear_filters::colFilterCaller<float>(
                tmp_buf, Dy, ky.cols >> 1, ky.ptr<float>(0), ky.cols, 
                gpuBorderType);
    }

    void extractCovData(const GpuMat& src, GpuMat& Dx, GpuMat& Dy, int blockSize, int ksize, int gpuBorderType)
    {
        switch (src.type())
        {
        case CV_8U:
            extractCovData<unsigned char>(src, Dx, Dy, blockSize, ksize, gpuBorderType);
            break;
        case CV_32F:
            extractCovData<float>(src, Dx, Dy, blockSize, ksize, gpuBorderType);
            break;
        default:
            CV_Error(CV_StsBadArg, "extractCovData: unsupported type of the source matrix");
        }
    }

} // Anonymous namespace

void cv::gpu::cornerHarris(const GpuMat& src, GpuMat& dst, int blockSize, int ksize, double k, int borderType)
{
    int gpuBorderType;
    switch (borderType)
    {
    case cv::BORDER_REFLECT101:
        gpuBorderType = cv::gpu::BORDER_REFLECT101; 
        break;
    case cv::BORDER_REPLICATE:
        gpuBorderType = cv::gpu::BORDER_REPLICATE; 
        break;
    default:
        CV_Error(CV_StsBadArg, "cornerHarris: unsupported border extrapolation mode");
    }

    GpuMat Dx, Dy;
    extractCovData(src, Dx, Dy, blockSize, ksize, gpuBorderType);
    dst.create(src.size(), CV_32F);
    imgproc::cornerHarris_caller(blockSize, (float)k, Dx, Dy, dst, gpuBorderType);
}

void cv::gpu::cornerMinEigenVal(const GpuMat& src, GpuMat& dst, int blockSize, int ksize, int borderType)
{  
    int gpuBorderType;
    switch (borderType)
    {
    case cv::BORDER_REFLECT101:
        gpuBorderType = cv::gpu::BORDER_REFLECT101;
        break;
    case cv::BORDER_REPLICATE:
        gpuBorderType = cv::gpu::BORDER_REPLICATE; 
        break;
    default:
        CV_Error(CV_StsBadArg, "cornerMinEigenVal: unsupported border extrapolation mode");
    }

    GpuMat Dx, Dy;
    extractCovData(src, Dx, Dy, blockSize, ksize, gpuBorderType);    
    dst.create(src.size(), CV_32F);
    imgproc::cornerMinEigenVal_caller(blockSize, Dx, Dy, dst, gpuBorderType);
}


#endif /* !defined (HAVE_CUDA) */

