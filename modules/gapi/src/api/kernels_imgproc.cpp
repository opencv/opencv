// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include "opencv2/gapi/gscalar.hpp"
#include "opencv2/gapi/gcall.hpp"
#include "opencv2/gapi/gkernel.hpp"
#include "opencv2/gapi/imgproc.hpp"

namespace cv { namespace gapi {

GMat sepFilter(const GMat& src, int ddepth, const Mat& kernelX, const Mat& kernelY, const Point& anchor,
               const Scalar& delta, int borderType, const Scalar& borderVal)
{
    return imgproc::GSepFilter::on(src, ddepth, kernelX, kernelY, anchor, delta, borderType, borderVal);
}

GMat filter2D(const GMat& src, int ddepth, const Mat& kernel, const Point& anchor, const Scalar& delta, int borderType,
              const Scalar& bordVal)
{
    return imgproc::GFilter2D::on(src, ddepth, kernel, anchor, delta, borderType, bordVal);
}

GMat boxFilter(const GMat& src, int dtype, const Size& ksize, const Point& anchor,
               bool normalize, int borderType, const Scalar& bordVal)
{
    return imgproc::GBoxFilter::on(src, dtype, ksize, anchor, normalize, borderType, bordVal);
}

GMat blur(const GMat& src, const Size& ksize, const Point& anchor,
               int borderType, const Scalar& bordVal)
{
    return imgproc::GBlur::on(src, ksize, anchor, borderType, bordVal);
}

GMat gaussianBlur(const GMat& src, const Size& ksize, double sigmaX, double sigmaY,
                  int borderType, const Scalar& bordVal)
{
    return imgproc::GGaussBlur::on(src, ksize, sigmaX, sigmaY, borderType, bordVal);
}

GMat medianBlur(const GMat& src, int ksize)
{
    return imgproc::GMedianBlur::on(src, ksize);
}

GMat erode(const GMat& src, const Mat& kernel, const Point& anchor, int iterations,
           int borderType, const Scalar& borderValue )
{
    return imgproc::GErode::on(src, kernel, anchor, iterations, borderType, borderValue);
}

GMat erode3x3(const GMat& src, int iterations,
           int borderType, const Scalar& borderValue )
{
    return erode(src, cv::Mat(), cv::Point(-1, -1), iterations, borderType, borderValue);
}

GMat dilate(const GMat& src, const Mat& kernel, const Point& anchor, int iterations,
            int borderType, const Scalar& borderValue)
{
    return imgproc::GDilate::on(src, kernel, anchor, iterations, borderType, borderValue);
}

GMat dilate3x3(const GMat& src, int iterations,
            int borderType, const Scalar& borderValue)
{
    return dilate(src, cv::Mat(), cv::Point(-1,-1), iterations, borderType, borderValue);
}

GMat Sobel(const GMat& src, int ddepth, int dx, int dy, int ksize,
           double scale, double delta,
           int borderType, const Scalar& bordVal)
{
    return imgproc::GSobel::on(src, ddepth, dx, dy, ksize, scale, delta, borderType, bordVal);
}

GMat equalizeHist(const GMat& src)
{
    return imgproc::GEqHist::on(src);
}

GMat Canny(const GMat& src, double thr1, double thr2, int apertureSize, bool l2gradient)
{
    return imgproc::GCanny::on(src, thr1, thr2, apertureSize, l2gradient);
}

GMat RGB2Gray(const GMat& src)
{
    return imgproc::GRGB2Gray::on(src);
}

GMat RGB2Gray(const GMat& src, float rY, float gY, float bY)
{
    return imgproc::GRGB2GrayCustom::on(src, rY, gY, bY);
}

GMat BGR2Gray(const GMat& src)
{
    return imgproc::GBGR2Gray::on(src);
}

GMat RGB2YUV(const GMat& src)
{
    return imgproc::GRGB2YUV::on(src);
}

GMat BGR2LUV(const GMat& src)
{
    return imgproc::GBGR2LUV::on(src);
}

GMat LUV2BGR(const GMat& src)
{
    return imgproc::GLUV2BGR::on(src);
}

GMat BGR2YUV(const GMat& src)
{
    return imgproc::GBGR2YUV::on(src);
}

GMat YUV2BGR(const GMat& src)
{
    return imgproc::GYUV2BGR::on(src);
}

GMat YUV2RGB(const GMat& src)
{
    return imgproc::GYUV2RGB::on(src);
}

GMat RGB2Lab(const GMat& src)
{
    return imgproc::GRGB2Lab::on(src);
}

} //namespace gapi
} //namespace cv
