// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#include "precomp.hpp"

#include <opencv2/gapi/gscalar.hpp>
#include <opencv2/gapi/gcall.hpp>
#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/imgproc.hpp>

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

std::tuple<GMat, GMat> SobelXY(const GMat& src, int ddepth, int order, int ksize,
           double scale, double delta,
           int borderType, const Scalar& bordVal)
{
    return imgproc::GSobelXY::on(src, ddepth, order, ksize, scale, delta, borderType, bordVal);
}

GMat Laplacian(const GMat& src, int ddepth, int ksize, double scale, double delta, int borderType)
{
    return imgproc::GLaplacian::on(src, ddepth, ksize, scale, delta, borderType);
}

GMat bilateralFilter(const GMat& src, int d, double sigmaColor, double sigmaSpace, int borderType)
{
    return imgproc::GBilateralFilter::on(src, d, sigmaColor, sigmaSpace, borderType);
}

GMat equalizeHist(const GMat& src)
{
    return imgproc::GEqHist::on(src);
}

GMat Canny(const GMat& src, double thr1, double thr2, int apertureSize, bool l2gradient)
{
    return imgproc::GCanny::on(src, thr1, thr2, apertureSize, l2gradient);
}

cv::GArray<cv::Point2f> goodFeaturesToTrack(const GMat& image, int maxCorners, double qualityLevel,
                                            double minDistance, const Mat& mask, int blockSize,
                                            bool useHarrisDetector, double k)
{
    return imgproc::GGoodFeatures::on(image, maxCorners, qualityLevel, minDistance, mask, blockSize,
                                      useHarrisDetector, k);
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

GMat NV12toRGB(const GMat& src_y, const GMat& src_uv)
{
    return imgproc::GNV12toRGB::on(src_y, src_uv);
}

GMat NV12toBGR(const GMat& src_y, const GMat& src_uv)
{
    return imgproc::GNV12toBGR::on(src_y, src_uv);
}

GMat RGB2Lab(const GMat& src)
{
    return imgproc::GRGB2Lab::on(src);
}

GMat BayerGR2RGB(const GMat& src_gr)
{
    return imgproc::GBayerGR2RGB::on(src_gr);
}

GMat RGB2HSV(const GMat& src)
{
    return imgproc::GRGB2HSV::on(src);
}

GMat RGB2YUV422(const GMat& src)
{
    return imgproc::GRGB2YUV422::on(src);
}

GMat NV12toGray(const GMat &y, const GMat &uv)
{
    return imgproc::GNV12toGray::on(y, uv);
}

GMatP NV12toRGBp(const GMat &y, const GMat &uv)
{
    return imgproc::GNV12toRGBp::on(y, uv);
}

GMatP NV12toBGRp(const GMat &y, const GMat &uv)
{
    return imgproc::GNV12toBGRp::on(y, uv);
}

} //namespace gapi
} //namespace cv
