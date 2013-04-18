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

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

void cv::gpu::Canny(const GpuMat&, GpuMat&, double, double, int, bool) { throw_no_cuda(); }
void cv::gpu::Canny(const GpuMat&, CannyBuf&, GpuMat&, double, double, int, bool) { throw_no_cuda(); }
void cv::gpu::Canny(const GpuMat&, const GpuMat&, GpuMat&, double, double, bool) { throw_no_cuda(); }
void cv::gpu::Canny(const GpuMat&, const GpuMat&, CannyBuf&, GpuMat&, double, double, bool) { throw_no_cuda(); }
void cv::gpu::CannyBuf::create(const Size&, int) { throw_no_cuda(); }
void cv::gpu::CannyBuf::release() { throw_no_cuda(); }

#else /* !defined (HAVE_CUDA) */

void cv::gpu::CannyBuf::create(const Size& image_size, int apperture_size)
{
    if (apperture_size > 0)
    {
        ensureSizeIsEnough(image_size, CV_32SC1, dx);
        ensureSizeIsEnough(image_size, CV_32SC1, dy);

        if (apperture_size != 3)
        {
            filterDX = createDerivFilter_GPU(CV_8UC1, CV_32S, 1, 0, apperture_size, BORDER_REPLICATE);
            filterDY = createDerivFilter_GPU(CV_8UC1, CV_32S, 0, 1, apperture_size, BORDER_REPLICATE);
        }
    }

    ensureSizeIsEnough(image_size, CV_32FC1, mag);
    ensureSizeIsEnough(image_size, CV_32SC1, map);

    ensureSizeIsEnough(1, image_size.area(), CV_16UC2, st1);
    ensureSizeIsEnough(1, image_size.area(), CV_16UC2, st2);
}

void cv::gpu::CannyBuf::release()
{
    dx.release();
    dy.release();
    mag.release();
    map.release();
    st1.release();
    st2.release();
}

namespace canny
{
    void calcMagnitude(PtrStepSzb srcWhole, int xoff, int yoff, PtrStepSzi dx, PtrStepSzi dy, PtrStepSzf mag, bool L2Grad);
    void calcMagnitude(PtrStepSzi dx, PtrStepSzi dy, PtrStepSzf mag, bool L2Grad);

    void calcMap(PtrStepSzi dx, PtrStepSzi dy, PtrStepSzf mag, PtrStepSzi map, float low_thresh, float high_thresh);

    void edgesHysteresisLocal(PtrStepSzi map, ushort2* st1);

    void edgesHysteresisGlobal(PtrStepSzi map, ushort2* st1, ushort2* st2);

    void getEdges(PtrStepSzi map, PtrStepSzb dst);
}

namespace
{
    void CannyCaller(const GpuMat& dx, const GpuMat& dy, CannyBuf& buf, GpuMat& dst, float low_thresh, float high_thresh)
    {
        using namespace canny;

        buf.map.setTo(Scalar::all(0));
        calcMap(dx, dy, buf.mag, buf.map, low_thresh, high_thresh);

        edgesHysteresisLocal(buf.map, buf.st1.ptr<ushort2>());

        edgesHysteresisGlobal(buf.map, buf.st1.ptr<ushort2>(), buf.st2.ptr<ushort2>());

        getEdges(buf.map, dst);
    }
}

void cv::gpu::Canny(const GpuMat& src, GpuMat& dst, double low_thresh, double high_thresh, int apperture_size, bool L2gradient)
{
    CannyBuf buf;
    Canny(src, buf, dst, low_thresh, high_thresh, apperture_size, L2gradient);
}

void cv::gpu::Canny(const GpuMat& src, CannyBuf& buf, GpuMat& dst, double low_thresh, double high_thresh, int apperture_size, bool L2gradient)
{
    using namespace canny;

    CV_Assert(src.type() == CV_8UC1);

    if (!deviceSupports(SHARED_ATOMICS))
        CV_Error(cv::Error::StsNotImplemented, "The device doesn't support shared atomics");

    if( low_thresh > high_thresh )
        std::swap( low_thresh, high_thresh);

    dst.create(src.size(), CV_8U);
    buf.create(src.size(), apperture_size);

    if (apperture_size == 3)
    {
        Size wholeSize;
        Point ofs;
        src.locateROI(wholeSize, ofs);
        GpuMat srcWhole(wholeSize, src.type(), src.datastart, src.step);

        calcMagnitude(srcWhole, ofs.x, ofs.y, buf.dx, buf.dy, buf.mag, L2gradient);
    }
    else
    {
        buf.filterDX->apply(src, buf.dx, Rect(0, 0, src.cols, src.rows));
        buf.filterDY->apply(src, buf.dy, Rect(0, 0, src.cols, src.rows));

        calcMagnitude(buf.dx, buf.dy, buf.mag, L2gradient);
    }

    CannyCaller(buf.dx, buf.dy, buf, dst, static_cast<float>(low_thresh), static_cast<float>(high_thresh));
}

void cv::gpu::Canny(const GpuMat& dx, const GpuMat& dy, GpuMat& dst, double low_thresh, double high_thresh, bool L2gradient)
{
    CannyBuf buf;
    Canny(dx, dy, buf, dst, low_thresh, high_thresh, L2gradient);
}

void cv::gpu::Canny(const GpuMat& dx, const GpuMat& dy, CannyBuf& buf, GpuMat& dst, double low_thresh, double high_thresh, bool L2gradient)
{
    using namespace canny;

    CV_Assert(TargetArchs::builtWith(SHARED_ATOMICS) && DeviceInfo().supports(SHARED_ATOMICS));
    CV_Assert(dx.type() == CV_32SC1 && dy.type() == CV_32SC1 && dx.size() == dy.size());

    if( low_thresh > high_thresh )
        std::swap( low_thresh, high_thresh);

    dst.create(dx.size(), CV_8U);
    buf.create(dx.size(), -1);

    calcMagnitude(dx, dy, buf.mag, L2gradient);

    CannyCaller(dx, dy, buf, dst, static_cast<float>(low_thresh), static_cast<float>(high_thresh));
}

#endif /* !defined (HAVE_CUDA) */
