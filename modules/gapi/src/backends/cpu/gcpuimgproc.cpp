// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#include "precomp.hpp"

#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/cpu/imgproc.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/gcompoundkernel.hpp>

#include "backends/fluid/gfluidimgproc_func.hpp"


namespace {
    cv::Mat add_border(const cv::Mat& in, const int ksize, const int borderType, const cv::Scalar& bordVal){
        if( borderType == cv::BORDER_CONSTANT )
        {
            cv::Mat temp_in;
            int add = (ksize - 1) / 2;
            cv::copyMakeBorder(in, temp_in, add, add, add, add, borderType, bordVal);
            return temp_in(cv::Rect(add, add, in.cols, in.rows));
        }
        return in;
    }
}

GAPI_OCV_KERNEL(GCPUSepFilter, cv::gapi::imgproc::GSepFilter)
{
    static void run(const cv::Mat& in, int ddepth, const cv::Mat& kernX, const cv::Mat& kernY, const cv::Point& anchor, const cv::Scalar& delta,
                    int border, const cv::Scalar& bordVal, cv::Mat &out)
    {
        if( border == cv::BORDER_CONSTANT )
        {
            cv::Mat temp_in;
            int width_add = (kernY.cols - 1) / 2;
            int height_add =  (kernX.rows - 1) / 2;
            cv::copyMakeBorder(in, temp_in, height_add, height_add, width_add, width_add, border, bordVal);
            cv::Rect rect = cv::Rect(height_add, width_add, in.cols, in.rows);
            cv::sepFilter2D(temp_in(rect), out, ddepth, kernX, kernY, anchor, delta.val[0], border);
        }
        else
            cv::sepFilter2D(in, out, ddepth, kernX, kernY, anchor, delta.val[0], border);
    }
};

GAPI_OCV_KERNEL(GCPUBoxFilter, cv::gapi::imgproc::GBoxFilter)
{
    static void run(const cv::Mat& in, int ddepth, const cv::Size& ksize, const cv::Point& anchor, bool normalize, int borderType, const cv::Scalar& bordVal, cv::Mat &out)
    {
        if( borderType == cv::BORDER_CONSTANT )
        {
            cv::Mat temp_in;
            int width_add = (ksize.width - 1) / 2;
            int height_add =  (ksize.height - 1) / 2;
            cv::copyMakeBorder(in, temp_in, height_add, height_add, width_add, width_add, borderType, bordVal);
            cv::Rect rect = cv::Rect(height_add, width_add, in.cols, in.rows);
            cv::boxFilter(temp_in(rect), out, ddepth, ksize, anchor, normalize, borderType);
        }
        else
            cv::boxFilter(in, out, ddepth, ksize, anchor, normalize, borderType);
    }
};

GAPI_OCV_KERNEL(GCPUBlur, cv::gapi::imgproc::GBlur)
{
    static void run(const cv::Mat& in, const cv::Size& ksize, const cv::Point& anchor, int borderType, const cv::Scalar& bordVal, cv::Mat &out)
    {
        if( borderType == cv::BORDER_CONSTANT )
        {
            cv::Mat temp_in;
            int width_add = (ksize.width - 1) / 2;
            int height_add =  (ksize.height - 1) / 2;
            cv::copyMakeBorder(in, temp_in, height_add, height_add, width_add, width_add, borderType, bordVal);
            cv::Rect rect = cv::Rect(height_add, width_add, in.cols, in.rows);
            cv::blur(temp_in(rect), out, ksize, anchor, borderType);
        }
        else
            cv::blur(in, out, ksize, anchor, borderType);
    }
};


GAPI_OCV_KERNEL(GCPUFilter2D, cv::gapi::imgproc::GFilter2D)
{
    static void run(const cv::Mat& in, int ddepth, const cv::Mat& k, const cv::Point& anchor, const cv::Scalar& delta, int border,
                    const cv::Scalar& bordVal, cv::Mat &out)
    {
        if( border == cv::BORDER_CONSTANT )
        {
            cv::Mat temp_in;
            int width_add = (k.cols - 1) / 2;
            int height_add =  (k.rows - 1) / 2;
            cv::copyMakeBorder(in, temp_in, height_add, height_add, width_add, width_add, border, bordVal );
            cv::Rect rect = cv::Rect(height_add, width_add, in.cols, in.rows);
            cv::filter2D(temp_in(rect), out, ddepth, k, anchor, delta.val[0], border);
        }
        else
            cv::filter2D(in, out, ddepth, k, anchor, delta.val[0], border);
    }
};

GAPI_OCV_KERNEL(GCPUGaussBlur, cv::gapi::imgproc::GGaussBlur)
{
    static void run(const cv::Mat& in, const cv::Size& ksize, double sigmaX, double sigmaY, int borderType, const cv::Scalar& bordVal, cv::Mat &out)
    {
        if( borderType == cv::BORDER_CONSTANT )
        {
            cv::Mat temp_in;
            int width_add = (ksize.width - 1) / 2;
            int height_add =  (ksize.height - 1) / 2;
            cv::copyMakeBorder(in, temp_in, height_add, height_add, width_add, width_add, borderType, bordVal );
            cv::Rect rect = cv::Rect(height_add, width_add, in.cols, in.rows);
            cv::GaussianBlur(temp_in(rect), out, ksize, sigmaX, sigmaY, borderType);
        }
        else
            cv::GaussianBlur(in, out, ksize, sigmaX, sigmaY, borderType);
    }
};

GAPI_OCV_KERNEL(GCPUMedianBlur, cv::gapi::imgproc::GMedianBlur)
{
    static void run(const cv::Mat& in, int ksize, cv::Mat &out)
    {
        cv::medianBlur(in, out, ksize);
    }
};

GAPI_OCV_KERNEL(GCPUErode, cv::gapi::imgproc::GErode)
{
    static void run(const cv::Mat& in, const cv::Mat& kernel, const cv::Point& anchor, int iterations, int borderType, const cv::Scalar& borderValue, cv::Mat &out)
    {
        cv::erode(in, out, kernel, anchor, iterations, borderType, borderValue);
    }
};

GAPI_OCV_KERNEL(GCPUDilate, cv::gapi::imgproc::GDilate)
{
    static void run(const cv::Mat& in, const cv::Mat& kernel, const cv::Point& anchor, int iterations, int borderType, const cv::Scalar& borderValue, cv::Mat &out)
    {
        cv::dilate(in, out, kernel, anchor, iterations, borderType, borderValue);
    }
};

GAPI_OCV_KERNEL(GCPUMorphologyEx, cv::gapi::imgproc::GMorphologyEx)
{
    static void run(const cv::Mat &in, const cv::MorphTypes op, const cv::Mat &kernel,
                    const cv::Point &anchor, const int iterations,
                    const cv::BorderTypes borderType, const cv::Scalar &borderValue, cv::Mat &out)
    {
        cv::morphologyEx(in, out, op, kernel, anchor, iterations, borderType, borderValue);
    }
};

GAPI_OCV_KERNEL(GCPUSobel, cv::gapi::imgproc::GSobel)
{
    static void run(const cv::Mat& in, int ddepth, int dx, int dy, int ksize, double scale, double delta, int borderType,
                    const cv::Scalar& bordVal, cv::Mat &out)
    {
        cv::Mat temp_in = add_border(in, ksize, borderType, bordVal);
        cv::Sobel(temp_in, out, ddepth, dx, dy, ksize, scale, delta, borderType);
    }
};

GAPI_OCV_KERNEL(GCPUSobelXY, cv::gapi::imgproc::GSobelXY)
{
    static void run(const cv::Mat& in, int ddepth, int order, int ksize, double scale, double delta, int borderType,
                    const cv::Scalar& bordVal, cv::Mat &out_dx, cv::Mat &out_dy)
    {
        cv::Mat temp_in = add_border(in, ksize, borderType, bordVal);
        cv::Sobel(temp_in, out_dx, ddepth, order, 0, ksize, scale, delta, borderType);
        cv::Sobel(temp_in, out_dy, ddepth, 0, order, ksize, scale, delta, borderType);
    }
};

GAPI_OCV_KERNEL(GCPULaplacian, cv::gapi::imgproc::GLaplacian)
{
    static void run(const cv::Mat& in, int ddepth, int ksize, double scale,
                    double delta, int borderType, cv::Mat &out)
    {
        cv::Laplacian(in, out, ddepth, ksize, scale, delta, borderType);
    }
};

GAPI_OCV_KERNEL(GCPUBilateralFilter, cv::gapi::imgproc::GBilateralFilter)
{
    static void run(const cv::Mat& in, int d, double sigmaColor,
                    double sigmaSpace, int borderType, cv::Mat &out)
    {
        cv::bilateralFilter(in, out, d, sigmaColor, sigmaSpace, borderType);
    }
};

GAPI_OCV_KERNEL(GCPUEqualizeHist, cv::gapi::imgproc::GEqHist)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::equalizeHist(in, out);
    }
};

GAPI_OCV_KERNEL(GCPUCanny, cv::gapi::imgproc::GCanny)
{
    static void run(const cv::Mat& in, double thr1, double thr2, int apSize, bool l2gradient, cv::Mat &out)
    {
        cv::Canny(in, out, thr1, thr2, apSize, l2gradient);
    }
};

GAPI_OCV_KERNEL(GCPUGoodFeatures, cv::gapi::imgproc::GGoodFeatures)
{
    static void run(const cv::Mat& image, int maxCorners, double qualityLevel, double minDistance,
                    const cv::Mat& mask, int blockSize, bool useHarrisDetector, double k,
                    std::vector<cv::Point2f> &out)
    {
        cv::goodFeaturesToTrack(image, out, maxCorners, qualityLevel, minDistance,
                                mask, blockSize, useHarrisDetector, k);
    }
};

GAPI_OCV_KERNEL(GCPUFindContours, cv::gapi::imgproc::GFindContours)
{
    static void run(const cv::Mat& image, const cv::RetrievalModes mode,
                    const cv::ContourApproximationModes method, const cv::Point& offset,
                    std::vector<std::vector<cv::Point>> &outConts)
    {
        cv::findContours(image, outConts, mode, method, offset);
    }
};

GAPI_OCV_KERNEL(GCPUFindContoursNoOffset, cv::gapi::imgproc::GFindContoursNoOffset)
{
    static void run(const cv::Mat& image, const cv::RetrievalModes mode,
                    const cv::ContourApproximationModes method,
                    std::vector<std::vector<cv::Point>> &outConts)
    {
        cv::findContours(image, outConts, mode, method);
    }
};

GAPI_OCV_KERNEL(GCPUFindContoursH, cv::gapi::imgproc::GFindContoursH)
{
    static void run(const cv::Mat& image, const cv::RetrievalModes mode,
                    const cv::ContourApproximationModes method, const cv::Point& offset,
                    std::vector<std::vector<cv::Point>> &outConts, std::vector<cv::Vec4i> &outHier)
    {
        cv::findContours(image, outConts, outHier, mode, method, offset);
    }
};

GAPI_OCV_KERNEL(GCPUFindContoursHNoOffset, cv::gapi::imgproc::GFindContoursHNoOffset)
{
    static void run(const cv::Mat& image, const cv::RetrievalModes mode,
                    const cv::ContourApproximationModes method,
                    std::vector<std::vector<cv::Point>> &outConts, std::vector<cv::Vec4i> &outHier)
    {
        cv::findContours(image, outConts, outHier, mode, method);
    }
};

GAPI_OCV_KERNEL(GCPUBoundingRectMat, cv::gapi::imgproc::GBoundingRectMat)
{
    static void run(const cv::Mat& in, cv::Rect& out)
    {
        out = cv::boundingRect(in);
    }
};

GAPI_OCV_KERNEL(GCPUBoundingRectVector32S, cv::gapi::imgproc::GBoundingRectVector32S)
{
    static void run(const std::vector<cv::Point2i>& in, cv::Rect& out)
    {
        out = cv::boundingRect(in);
    }
};

GAPI_OCV_KERNEL(GCPUBoundingRectVector32F, cv::gapi::imgproc::GBoundingRectVector32F)
{
    static void run(const std::vector<cv::Point2f>& in, cv::Rect& out)
    {
        out = cv::boundingRect(in);
    }
};

GAPI_OCV_KERNEL(GCPUFitLine2DMat, cv::gapi::imgproc::GFitLine2DMat)
{
    static void run(const cv::Mat& in, const cv::DistanceTypes distType, const double param,
                    const double reps, const double aeps, cv::Vec4f& out)
    {
        cv::fitLine(in, out, distType, param, reps, aeps);
    }
};

GAPI_OCV_KERNEL(GCPUFitLine2DVector32S, cv::gapi::imgproc::GFitLine2DVector32S)
{
    static void run(const std::vector<cv::Point2i>& in, const cv::DistanceTypes distType,
                    const double param, const double reps, const double aeps, cv::Vec4f& out)
    {
        cv::fitLine(in, out, distType, param, reps, aeps);
    }
};

GAPI_OCV_KERNEL(GCPUFitLine2DVector32F, cv::gapi::imgproc::GFitLine2DVector32F)
{
    static void run(const std::vector<cv::Point2f>& in, const cv::DistanceTypes distType,
                    const double param, const double reps, const double aeps, cv::Vec4f& out)
    {
        cv::fitLine(in, out, distType, param, reps, aeps);
    }
};

GAPI_OCV_KERNEL(GCPUFitLine2DVector64F, cv::gapi::imgproc::GFitLine2DVector64F)
{
    static void run(const std::vector<cv::Point2d>& in, const cv::DistanceTypes distType,
                    const double param, const double reps, const double aeps, cv::Vec4f& out)
    {
        cv::fitLine(in, out, distType, param, reps, aeps);
    }
};

GAPI_OCV_KERNEL(GCPUFitLine3DMat, cv::gapi::imgproc::GFitLine3DMat)
{
    static void run(const cv::Mat& in, const cv::DistanceTypes distType, const double param,
                    const double reps, const double aeps, cv::Vec6f& out)
    {
        cv::fitLine(in, out, distType, param, reps, aeps);
    }
};

GAPI_OCV_KERNEL(GCPUFitLine3DVector32S, cv::gapi::imgproc::GFitLine3DVector32S)
{
    static void run(const std::vector<cv::Point3i>& in, const cv::DistanceTypes distType,
                    const double param, const double reps, const double aeps, cv::Vec6f& out)
    {
        cv::fitLine(in, out, distType, param, reps, aeps);
    }
};

GAPI_OCV_KERNEL(GCPUFitLine3DVector32F, cv::gapi::imgproc::GFitLine3DVector32F)
{
    static void run(const std::vector<cv::Point3f>& in, const cv::DistanceTypes distType,
                    const double param, const double reps, const double aeps, cv::Vec6f& out)
    {
        cv::fitLine(in, out, distType, param, reps, aeps);
    }
};

GAPI_OCV_KERNEL(GCPUFitLine3DVector64F, cv::gapi::imgproc::GFitLine3DVector64F)
{
    static void run(const std::vector<cv::Point3d>& in, const cv::DistanceTypes distType,
                    const double param, const double reps, const double aeps, cv::Vec6f& out)
    {
        cv::fitLine(in, out, distType, param, reps, aeps);
    }
};

GAPI_OCV_KERNEL(GCPUBGR2RGB, cv::gapi::imgproc::GBGR2RGB)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_BGR2RGB);
    }
};

GAPI_OCV_KERNEL(GCPUBGR2I420, cv::gapi::imgproc::GBGR2I420)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_BGR2YUV_I420);
    }
};

GAPI_OCV_KERNEL(GCPURGB2I420, cv::gapi::imgproc::GRGB2I420)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_RGB2YUV_I420);
    }
};

GAPI_OCV_KERNEL(GCPUI4202BGR, cv::gapi::imgproc::GI4202BGR)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_YUV2BGR_I420);
    }
};

GAPI_OCV_KERNEL(GCPUI4202RGB, cv::gapi::imgproc::GI4202RGB)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_YUV2RGB_I420);
    }
};

GAPI_OCV_KERNEL(GCPURGB2YUV, cv::gapi::imgproc::GRGB2YUV)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_RGB2YUV);
    }
};

GAPI_OCV_KERNEL(GCPUYUV2RGB, cv::gapi::imgproc::GYUV2RGB)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_YUV2RGB);
    }
};

GAPI_OCV_KERNEL(GCPUNV12toRGB, cv::gapi::imgproc::GNV12toRGB)
{
    static void run(const cv::Mat& in_y, const cv::Mat& in_uv, cv::Mat &out)
    {
        cv::cvtColorTwoPlane(in_y, in_uv, out, cv::COLOR_YUV2RGB_NV12);
    }
};

GAPI_OCV_KERNEL(GCPUNV12toBGR, cv::gapi::imgproc::GNV12toBGR)
{
    static void run(const cv::Mat& in_y, const cv::Mat& in_uv, cv::Mat &out)
    {
        cv::cvtColorTwoPlane(in_y, in_uv, out, cv::COLOR_YUV2BGR_NV12);
    }
};

GAPI_OCV_KERNEL(GCPURGB2Lab, cv::gapi::imgproc::GRGB2Lab)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_RGB2Lab);
    }
};

GAPI_OCV_KERNEL(GCPUBGR2LUV, cv::gapi::imgproc::GBGR2LUV)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_BGR2Luv);
    }
};

GAPI_OCV_KERNEL(GCPUBGR2YUV, cv::gapi::imgproc::GBGR2YUV)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_BGR2YUV);
    }
};

GAPI_OCV_KERNEL(GCPULUV2BGR, cv::gapi::imgproc::GLUV2BGR)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_Luv2BGR);
    }
};

GAPI_OCV_KERNEL(GCPUYUV2BGR, cv::gapi::imgproc::GYUV2BGR)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_YUV2BGR);
    }
};

GAPI_OCV_KERNEL(GCPURGB2Gray, cv::gapi::imgproc::GRGB2Gray)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_RGB2GRAY);
    }
};

GAPI_OCV_KERNEL(GCPUBGR2Gray, cv::gapi::imgproc::GBGR2Gray)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_BGR2GRAY);
    }
};

GAPI_OCV_KERNEL(GCPURGB2GrayCustom, cv::gapi::imgproc::GRGB2GrayCustom)
{
    static void run(const cv::Mat& in, float rY, float bY, float gY, cv::Mat &out)
    {
        cv::Mat planes[3];
        cv::split(in, planes);
        out = planes[0]*rY + planes[1]*bY + planes[2]*gY;
    }
};

GAPI_OCV_KERNEL(GCPUBayerGR2RGB, cv::gapi::imgproc::GBayerGR2RGB)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_BayerGR2RGB);
    }
};

GAPI_OCV_KERNEL(GCPURGB2HSV, cv::gapi::imgproc::GRGB2HSV)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        cv::cvtColor(in, out, cv::COLOR_RGB2HSV);
    }
};

GAPI_OCV_KERNEL(GCPURGB2YUV422, cv::gapi::imgproc::GRGB2YUV422)
{
    static void run(const cv::Mat& in, cv::Mat &out)
    {
        out.create(in.size(), CV_8UC2);

        for (int i = 0; i < in.rows; ++i)
        {
            const uchar* in_line_p  = in.ptr<uchar>(i);
            uchar* out_line_p = out.ptr<uchar>(i);
            cv::gapi::fluid::run_rgb2yuv422_impl(out_line_p, in_line_p, in.cols);
        }
    }
};

static void toPlanar(const cv::Mat& in, cv::Mat& out)
{
    GAPI_Assert(out.depth() == in.depth());
    GAPI_Assert(out.channels() == 1);
    GAPI_Assert(in.channels() == 3);
    GAPI_Assert(out.cols == in.cols);
    GAPI_Assert(out.rows == 3*in.rows);

    std::vector<cv::Mat> outs(3);
    for (int i = 0; i < 3; i++) {
        outs[i] = out(cv::Rect(0, i*in.rows, in.cols, in.rows));
    }
    cv::split(in, outs);
}


GAPI_OCV_KERNEL(GCPUNV12toRGBp, cv::gapi::imgproc::GNV12toRGBp)
{
    static void run(const cv::Mat& inY, const cv::Mat& inUV, cv::Mat& out)
    {
        cv::Mat rgb;
        cv::cvtColorTwoPlane(inY, inUV, rgb, cv::COLOR_YUV2RGB_NV12);
        toPlanar(rgb, out);
    }
};

G_TYPED_KERNEL(GYUV2Gray, <cv::GMat(cv::GMat)>, "yuvtogray") {
    static cv::GMatDesc outMeta(cv::GMatDesc in) {
        GAPI_Assert(in.depth  == CV_8U);
        GAPI_Assert(in.planar == false);
        GAPI_Assert(in.size.width  % 2 == 0);
        GAPI_Assert(in.size.height % 3 == 0);

        /* YUV format for this kernel:
         * Y Y Y Y Y Y Y Y
         * Y Y Y Y Y Y Y Y
         * Y Y Y Y Y Y Y Y
         * Y Y Y Y Y Y Y Y
         * U V U V U V U V
         * U V U V U V U V
         */

        return {CV_8U, 1, cv::Size{in.size.width, in.size.height - (in.size.height / 3)}, false};
    }
};

GAPI_OCV_KERNEL(GCPUYUV2Gray, GYUV2Gray)
{
    static void run(const cv::Mat& in, cv::Mat& out)
    {
        cv::cvtColor(in, out, cv::COLOR_YUV2GRAY_NV12);
    }
};

G_TYPED_KERNEL(GConcatYUVPlanes, <cv::GMat(cv::GMat, cv::GMat)>, "concatyuvplanes") {
    static cv::GMatDesc outMeta(cv::GMatDesc y, cv::GMatDesc uv) {
        return {CV_8U, 1, cv::Size{y.size.width, y.size.height + uv.size.height}, false};
    }
};

GAPI_OCV_KERNEL(GCPUConcatYUVPlanes, GConcatYUVPlanes)
{
    static void run(const cv::Mat& in_y, const cv::Mat& in_uv, cv::Mat& out)
    {
        cv::Mat uv_planar(in_uv.rows, in_uv.cols * 2, CV_8UC1, in_uv.data);
        cv::vconcat(in_y, uv_planar, out);
    }
};

GAPI_COMPOUND_KERNEL(GCPUNV12toGray, cv::gapi::imgproc::GNV12toGray)
{
    static cv::GMat expand(cv::GMat y, cv::GMat uv)
    {
        return GYUV2Gray::on(GConcatYUVPlanes::on(y, uv));
    }
};

GAPI_OCV_KERNEL(GCPUNV12toBGRp, cv::gapi::imgproc::GNV12toBGRp)
{
    static void run(const cv::Mat& inY, const cv::Mat& inUV, cv::Mat& out)
    {
        cv::Mat rgb;
        cv::cvtColorTwoPlane(inY, inUV, rgb, cv::COLOR_YUV2BGR_NV12);
        toPlanar(rgb, out);
    }
};

cv::gapi::GKernelPackage cv::gapi::imgproc::cpu::kernels()
{
    static auto pkg = cv::gapi::kernels
        < GCPUFilter2D
        , GCPUSepFilter
        , GCPUBoxFilter
        , GCPUBlur
        , GCPUGaussBlur
        , GCPUMedianBlur
        , GCPUErode
        , GCPUDilate
        , GCPUMorphologyEx
        , GCPUSobel
        , GCPUSobelXY
        , GCPULaplacian
        , GCPUBilateralFilter
        , GCPUCanny
        , GCPUGoodFeatures
        , GCPUEqualizeHist
        , GCPUFindContours
        , GCPUFindContoursNoOffset
        , GCPUFindContoursH
        , GCPUFindContoursHNoOffset
        , GCPUBGR2RGB
        , GCPURGB2YUV
        , GCPUBoundingRectMat
        , GCPUBoundingRectVector32S
        , GCPUBoundingRectVector32F
        , GCPUFitLine2DMat
        , GCPUFitLine2DVector32S
        , GCPUFitLine2DVector32F
        , GCPUFitLine2DVector64F
        , GCPUFitLine3DMat
        , GCPUFitLine3DVector32S
        , GCPUFitLine3DVector32F
        , GCPUFitLine3DVector64F
        , GCPUYUV2RGB
        , GCPUBGR2I420
        , GCPURGB2I420
        , GCPUI4202BGR
        , GCPUI4202RGB
        , GCPUNV12toRGB
        , GCPUNV12toBGR
        , GCPURGB2Lab
        , GCPUBGR2LUV
        , GCPUBGR2YUV
        , GCPUYUV2BGR
        , GCPULUV2BGR
        , GCPUBGR2Gray
        , GCPURGB2Gray
        , GCPURGB2GrayCustom
        , GCPUBayerGR2RGB
        , GCPURGB2HSV
        , GCPURGB2YUV422
        , GCPUYUV2Gray
        , GCPUNV12toRGBp
        , GCPUNV12toBGRp
        , GCPUNV12toGray
        , GCPUConcatYUVPlanes
        >();
    return pkg;
}
