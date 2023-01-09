// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#ifndef OPENCV_GAPI_IMGPROC_HPP
#define OPENCV_GAPI_IMGPROC_HPP

#include <opencv2/imgproc.hpp>

#include <utility> // std::tuple

#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/gmat.hpp>
#include <opencv2/gapi/gscalar.hpp>


/** \defgroup gapi_imgproc G-API Image processing functionality
@{
    @defgroup gapi_filters Graph API: Image filters
    @defgroup gapi_colorconvert Graph API: Converting image from one color space to another
    @defgroup gapi_feature Graph API: Image Feature Detection
    @defgroup gapi_shape Graph API: Image Structural Analysis and Shape Descriptors
    @defgroup gapi_transform Graph API: Image and channel composition functions
@}
 */

namespace {
void validateFindingContoursMeta(const int depth, const int chan, const int mode)
{
    GAPI_Assert(chan == 1);
    switch (mode)
    {
    case cv::RETR_CCOMP:
        GAPI_Assert(depth == CV_8U || depth == CV_32S);
        break;
    case cv::RETR_FLOODFILL:
        GAPI_Assert(depth == CV_32S);
        break;
    default:
        GAPI_Assert(depth == CV_8U);
        break;
    }
}
} // anonymous namespace

namespace cv { namespace gapi {

/**
 * @brief This namespace contains G-API Operation Types for OpenCV
 * ImgProc module functionality.
 */
namespace imgproc {
    using GMat2 = std::tuple<GMat,GMat>;
    using GMat3 = std::tuple<GMat,GMat,GMat>; // FIXME: how to avoid this?
    using GFindContoursOutput = std::tuple<GArray<GArray<Point>>,GArray<Vec4i>>;

    G_TYPED_KERNEL(GFilter2D, <GMat(GMat,int,Mat,Point,Scalar,int,Scalar)>, "org.opencv.imgproc.filters.filter2D") {
        static GMatDesc outMeta(GMatDesc in, int ddepth, Mat, Point, Scalar, int, Scalar) {
            return in.withDepth(ddepth);
        }
    };

    G_TYPED_KERNEL(GSepFilter, <GMat(GMat,int,Mat,Mat,Point,Scalar,int,Scalar)>, "org.opencv.imgproc.filters.sepfilter") {
        static GMatDesc outMeta(GMatDesc in, int ddepth, Mat, Mat, Point, Scalar, int, Scalar) {
            return in.withDepth(ddepth);
        }
    };

    G_TYPED_KERNEL(GBoxFilter, <GMat(GMat,int,Size,Point,bool,int,Scalar)>, "org.opencv.imgproc.filters.boxfilter") {
        static GMatDesc outMeta(GMatDesc in, int ddepth, Size, Point, bool, int, Scalar) {
            return in.withDepth(ddepth);
        }
    };

    G_TYPED_KERNEL(GBlur, <GMat(GMat,Size,Point,int,Scalar)>, "org.opencv.imgproc.filters.blur") {
        static GMatDesc outMeta(GMatDesc in, Size, Point, int, Scalar) {
            return in;
        }
    };

    G_TYPED_KERNEL(GGaussBlur, <GMat(GMat,Size,double,double,int,Scalar)>, "org.opencv.imgproc.filters.gaussianBlur") {
        static GMatDesc outMeta(GMatDesc in, Size, double, double, int, Scalar) {
            return in;
        }
    };

    G_TYPED_KERNEL(GMedianBlur, <GMat(GMat,int)>, "org.opencv.imgproc.filters.medianBlur") {
        static GMatDesc outMeta(GMatDesc in, int) {
            return in;
        }
    };

    G_TYPED_KERNEL(GErode, <GMat(GMat,Mat,Point,int,int,Scalar)>, "org.opencv.imgproc.filters.erode") {
        static GMatDesc outMeta(GMatDesc in, Mat, Point, int, int, Scalar) {
            return in;
        }
    };

    G_TYPED_KERNEL(GDilate, <GMat(GMat,Mat,Point,int,int,Scalar)>, "org.opencv.imgproc.filters.dilate") {
        static GMatDesc outMeta(GMatDesc in, Mat, Point, int, int, Scalar) {
            return in;
        }
    };

    G_TYPED_KERNEL(GMorphologyEx, <GMat(GMat,MorphTypes,Mat,Point,int,BorderTypes,Scalar)>,
                   "org.opencv.imgproc.filters.morphologyEx") {
        static GMatDesc outMeta(const GMatDesc &in, MorphTypes, Mat, Point, int,
                                BorderTypes, Scalar) {
            return in;
        }
    };

    G_TYPED_KERNEL(GSobel, <GMat(GMat,int,int,int,int,double,double,int,Scalar)>, "org.opencv.imgproc.filters.sobel") {
        static GMatDesc outMeta(GMatDesc in, int ddepth, int, int, int, double, double, int, Scalar) {
            return in.withDepth(ddepth);
        }
    };

    G_TYPED_KERNEL_M(GSobelXY, <GMat2(GMat,int,int,int,double,double,int,Scalar)>, "org.opencv.imgproc.filters.sobelxy") {
        static std::tuple<GMatDesc, GMatDesc> outMeta(GMatDesc in, int ddepth, int, int, double, double, int, Scalar) {
            return std::make_tuple(in.withDepth(ddepth), in.withDepth(ddepth));
        }
    };

    G_TYPED_KERNEL(GLaplacian, <GMat(GMat,int, int, double, double, int)>,
                   "org.opencv.imgproc.filters.laplacian") {
        static GMatDesc outMeta(GMatDesc in, int ddepth, int, double, double, int) {
            return in.withDepth(ddepth);
        }
    };

    G_TYPED_KERNEL(GBilateralFilter, <GMat(GMat,int, double, double, int)>,
                   "org.opencv.imgproc.filters.bilateralfilter") {
        static GMatDesc outMeta(GMatDesc in, int, double, double, int) {
            return in;
        }
    };

    G_TYPED_KERNEL(GEqHist, <GMat(GMat)>, "org.opencv.imgproc.equalizeHist") {
        static GMatDesc outMeta(GMatDesc in) {
            return in.withType(CV_8U, 1);
        }
    };

    G_TYPED_KERNEL(GCanny, <GMat(GMat,double,double,int,bool)>, "org.opencv.imgproc.feature.canny") {
        static GMatDesc outMeta(GMatDesc in, double, double, int, bool) {
            return in.withType(CV_8U, 1);
        }
    };

    G_TYPED_KERNEL(GGoodFeatures,
                   <cv::GArray<cv::Point2f>(GMat,int,double,double,Mat,int,bool,double)>,
                   "org.opencv.imgproc.feature.goodFeaturesToTrack") {
        static GArrayDesc outMeta(GMatDesc, int, double, double, const Mat&, int, bool, double) {
            return empty_array_desc();
        }
    };

    using RetrMode = RetrievalModes;
    using ContMethod = ContourApproximationModes;
    G_TYPED_KERNEL(GFindContours, <GArray<GArray<Point>>(GMat,RetrMode,ContMethod,GOpaque<Point>)>,
                   "org.opencv.imgproc.shape.findContours")
    {
        static GArrayDesc outMeta(GMatDesc in, RetrMode mode, ContMethod, GOpaqueDesc)
        {
            validateFindingContoursMeta(in.depth, in.chan, mode);
            return empty_array_desc();
        }
    };

    // FIXME oc: make default value offset = Point()
    G_TYPED_KERNEL(GFindContoursNoOffset, <GArray<GArray<Point>>(GMat,RetrMode,ContMethod)>,
                   "org.opencv.imgproc.shape.findContoursNoOffset")
    {
        static GArrayDesc outMeta(GMatDesc in, RetrMode mode, ContMethod)
        {
            validateFindingContoursMeta(in.depth, in.chan, mode);
            return empty_array_desc();
        }
    };

    G_TYPED_KERNEL(GFindContoursH,<GFindContoursOutput(GMat,RetrMode,ContMethod,GOpaque<Point>)>,
                   "org.opencv.imgproc.shape.findContoursH")
    {
        static std::tuple<GArrayDesc,GArrayDesc>
        outMeta(GMatDesc in, RetrMode mode, ContMethod, GOpaqueDesc)
        {
            validateFindingContoursMeta(in.depth, in.chan, mode);
            return std::make_tuple(empty_array_desc(), empty_array_desc());
        }
    };

    // FIXME oc: make default value offset = Point()
    G_TYPED_KERNEL(GFindContoursHNoOffset,<GFindContoursOutput(GMat,RetrMode,ContMethod)>,
                   "org.opencv.imgproc.shape.findContoursHNoOffset")
    {
        static std::tuple<GArrayDesc,GArrayDesc>
        outMeta(GMatDesc in, RetrMode mode, ContMethod)
        {
            validateFindingContoursMeta(in.depth, in.chan, mode);
            return std::make_tuple(empty_array_desc(), empty_array_desc());
        }
    };

    G_TYPED_KERNEL(GBoundingRectMat, <GOpaque<Rect>(GMat)>,
                   "org.opencv.imgproc.shape.boundingRectMat") {
        static GOpaqueDesc outMeta(GMatDesc in) {
            if (in.depth == CV_8U)
            {
                GAPI_Assert(in.chan == 1);
            }
            else
            {
                GAPI_Assert (in.depth == CV_32S || in.depth == CV_32F);
                int amount = detail::checkVector(in, 2u);
                GAPI_Assert(amount != -1 &&
                            "Input Mat can't be described as vector of 2-dimentional points");
            }
            return empty_gopaque_desc();
        }
    };

    G_TYPED_KERNEL(GBoundingRectVector32S, <GOpaque<Rect>(GArray<Point2i>)>,
                   "org.opencv.imgproc.shape.boundingRectVector32S") {
        static GOpaqueDesc outMeta(GArrayDesc) {
            return empty_gopaque_desc();
        }
    };

    G_TYPED_KERNEL(GBoundingRectVector32F, <GOpaque<Rect>(GArray<Point2f>)>,
                   "org.opencv.imgproc.shape.boundingRectVector32F") {
        static GOpaqueDesc outMeta(GArrayDesc) {
            return empty_gopaque_desc();
        }
    };

    G_TYPED_KERNEL(GFitLine2DMat, <GOpaque<Vec4f>(GMat,DistanceTypes,double,double,double)>,
                   "org.opencv.imgproc.shape.fitLine2DMat") {
        static GOpaqueDesc outMeta(GMatDesc in,DistanceTypes,double,double,double) {
            int amount = detail::checkVector(in, 2u);
            GAPI_Assert(amount != -1 &&
                        "Input Mat can't be described as vector of 2-dimentional points");
            return empty_gopaque_desc();
        }
    };

    G_TYPED_KERNEL(GFitLine2DVector32S,
                   <GOpaque<Vec4f>(GArray<Point2i>,DistanceTypes,double,double,double)>,
                   "org.opencv.imgproc.shape.fitLine2DVector32S") {
        static GOpaqueDesc outMeta(GArrayDesc,DistanceTypes,double,double,double) {
            return empty_gopaque_desc();
        }
    };

    G_TYPED_KERNEL(GFitLine2DVector32F,
                   <GOpaque<Vec4f>(GArray<Point2f>,DistanceTypes,double,double,double)>,
                   "org.opencv.imgproc.shape.fitLine2DVector32F") {
        static GOpaqueDesc outMeta(GArrayDesc,DistanceTypes,double,double,double) {
            return empty_gopaque_desc();
        }
    };

    G_TYPED_KERNEL(GFitLine2DVector64F,
                   <GOpaque<Vec4f>(GArray<Point2d>,DistanceTypes,double,double,double)>,
                   "org.opencv.imgproc.shape.fitLine2DVector64F") {
        static GOpaqueDesc outMeta(GArrayDesc,DistanceTypes,double,double,double) {
            return empty_gopaque_desc();
        }
    };

    G_TYPED_KERNEL(GFitLine3DMat, <GOpaque<Vec6f>(GMat,DistanceTypes,double,double,double)>,
                   "org.opencv.imgproc.shape.fitLine3DMat") {
        static GOpaqueDesc outMeta(GMatDesc in,int,double,double,double) {
            int amount = detail::checkVector(in, 3u);
            GAPI_Assert(amount != -1 &&
                        "Input Mat can't be described as vector of 3-dimentional points");
            return empty_gopaque_desc();
        }
    };

    G_TYPED_KERNEL(GFitLine3DVector32S,
                   <GOpaque<Vec6f>(GArray<Point3i>,DistanceTypes,double,double,double)>,
                   "org.opencv.imgproc.shape.fitLine3DVector32S") {
        static GOpaqueDesc outMeta(GArrayDesc,DistanceTypes,double,double,double) {
            return empty_gopaque_desc();
        }
    };

    G_TYPED_KERNEL(GFitLine3DVector32F,
                   <GOpaque<Vec6f>(GArray<Point3f>,DistanceTypes,double,double,double)>,
                   "org.opencv.imgproc.shape.fitLine3DVector32F") {
        static GOpaqueDesc outMeta(GArrayDesc,DistanceTypes,double,double,double) {
            return empty_gopaque_desc();
        }
    };

    G_TYPED_KERNEL(GFitLine3DVector64F,
                   <GOpaque<Vec6f>(GArray<Point3d>,DistanceTypes,double,double,double)>,
                   "org.opencv.imgproc.shape.fitLine3DVector64F") {
        static GOpaqueDesc outMeta(GArrayDesc,DistanceTypes,double,double,double) {
            return empty_gopaque_desc();
        }
    };

    G_TYPED_KERNEL(GBGR2RGB, <GMat(GMat)>, "org.opencv.imgproc.colorconvert.bgr2rgb") {
        static GMatDesc outMeta(GMatDesc in) {
            return in; // type still remains CV_8UC3;
        }
    };

    G_TYPED_KERNEL(GRGB2YUV, <GMat(GMat)>, "org.opencv.imgproc.colorconvert.rgb2yuv") {
        static GMatDesc outMeta(GMatDesc in) {
            return in; // type still remains CV_8UC3;
        }
    };

    G_TYPED_KERNEL(GYUV2RGB, <GMat(GMat)>, "org.opencv.imgproc.colorconvert.yuv2rgb") {
        static GMatDesc outMeta(GMatDesc in) {
            return in; // type still remains CV_8UC3;
        }
    };

    G_TYPED_KERNEL(GBGR2I420, <GMat(GMat)>, "org.opencv.imgproc.colorconvert.bgr2i420") {
        static GMatDesc outMeta(GMatDesc in) {
            GAPI_Assert(in.depth == CV_8U);
            GAPI_Assert(in.chan == 3);
            GAPI_Assert(in.size.height % 2 == 0);
            return in.withType(in.depth, 1).withSize(Size(in.size.width, in.size.height * 3 / 2));
        }
    };

    G_TYPED_KERNEL(GRGB2I420, <GMat(GMat)>, "org.opencv.imgproc.colorconvert.rgb2i420") {
        static GMatDesc outMeta(GMatDesc in) {
            GAPI_Assert(in.depth == CV_8U);
            GAPI_Assert(in.chan == 3);
            GAPI_Assert(in.size.height % 2 == 0);
            return in.withType(in.depth, 1).withSize(Size(in.size.width, in.size.height * 3 / 2));
        }
    };

    G_TYPED_KERNEL(GI4202BGR, <GMat(GMat)>, "org.opencv.imgproc.colorconvert.i4202bgr") {
        static GMatDesc outMeta(GMatDesc in) {
            GAPI_Assert(in.depth == CV_8U);
            GAPI_Assert(in.chan == 1);
            GAPI_Assert(in.size.height % 3 == 0);
            return in.withType(in.depth, 3).withSize(Size(in.size.width, in.size.height * 2 / 3));
        }
    };

    G_TYPED_KERNEL(GI4202RGB, <GMat(GMat)>, "org.opencv.imgproc.colorconvert.i4202rgb") {
        static GMatDesc outMeta(GMatDesc in) {
            GAPI_Assert(in.depth == CV_8U);
            GAPI_Assert(in.chan == 1);
            GAPI_Assert(in.size.height % 3 == 0);
            return in.withType(in.depth, 3).withSize(Size(in.size.width, in.size.height * 2 / 3));
        }
    };

    G_TYPED_KERNEL(GNV12toRGB, <GMat(GMat, GMat)>, "org.opencv.imgproc.colorconvert.nv12torgb") {
        static GMatDesc outMeta(GMatDesc in_y, GMatDesc in_uv) {
            GAPI_Assert(in_y.chan == 1);
            GAPI_Assert(in_uv.chan == 2);
            GAPI_Assert(in_y.depth == CV_8U);
            GAPI_Assert(in_uv.depth == CV_8U);
            // UV size should be aligned with Y
            GAPI_Assert(in_y.size.width == 2 * in_uv.size.width);
            GAPI_Assert(in_y.size.height == 2 * in_uv.size.height);
            return in_y.withType(CV_8U, 3); // type will be CV_8UC3;
        }
    };

    G_TYPED_KERNEL(GNV12toBGR, <GMat(GMat, GMat)>, "org.opencv.imgproc.colorconvert.nv12tobgr") {
        static GMatDesc outMeta(GMatDesc in_y, GMatDesc in_uv) {
            GAPI_Assert(in_y.chan == 1);
            GAPI_Assert(in_uv.chan == 2);
            GAPI_Assert(in_y.depth == CV_8U);
            GAPI_Assert(in_uv.depth == CV_8U);
            // UV size should be aligned with Y
            GAPI_Assert(in_y.size.width == 2 * in_uv.size.width);
            GAPI_Assert(in_y.size.height == 2 * in_uv.size.height);
            return in_y.withType(CV_8U, 3); // type will be CV_8UC3;
        }
    };

    G_TYPED_KERNEL(GRGB2Lab, <GMat(GMat)>, "org.opencv.imgproc.colorconvert.rgb2lab") {
        static GMatDesc outMeta(GMatDesc in) {
            return in; // type still remains CV_8UC3;
        }
    };

    G_TYPED_KERNEL(GBGR2LUV, <GMat(GMat)>, "org.opencv.imgproc.colorconvert.bgr2luv") {
        static GMatDesc outMeta(GMatDesc in) {
            return in; // type still remains CV_8UC3;
        }
    };

    G_TYPED_KERNEL(GLUV2BGR, <GMat(GMat)>, "org.opencv.imgproc.colorconvert.luv2bgr") {
        static GMatDesc outMeta(GMatDesc in) {
            return in; // type still remains CV_8UC3;
        }
    };

    G_TYPED_KERNEL(GYUV2BGR, <GMat(GMat)>, "org.opencv.imgproc.colorconvert.yuv2bgr") {
        static GMatDesc outMeta(GMatDesc in) {
            return in; // type still remains CV_8UC3;
        }
    };

    G_TYPED_KERNEL(GBGR2YUV, <GMat(GMat)>, "org.opencv.imgproc.colorconvert.bgr2yuv") {
        static GMatDesc outMeta(GMatDesc in) {
            return in; // type still remains CV_8UC3;
        }
    };

    G_TYPED_KERNEL(GRGB2Gray, <GMat(GMat)>, "org.opencv.imgproc.colorconvert.rgb2gray") {
        static GMatDesc outMeta(GMatDesc in) {
            return in.withType(CV_8U, 1);
        }
    };

    G_TYPED_KERNEL(GRGB2GrayCustom, <GMat(GMat,float,float,float)>, "org.opencv.imgproc.colorconvert.rgb2graycustom") {
        static GMatDesc outMeta(GMatDesc in, float, float, float) {
            return in.withType(CV_8U, 1);
        }
    };

    G_TYPED_KERNEL(GBGR2Gray, <GMat(GMat)>, "org.opencv.imgproc.colorconvert.bgr2gray") {
        static GMatDesc outMeta(GMatDesc in) {
            return in.withType(CV_8U, 1);
        }
    };

    G_TYPED_KERNEL(GBayerGR2RGB, <cv::GMat(cv::GMat)>, "org.opencv.imgproc.colorconvert.bayergr2rgb") {
        static cv::GMatDesc outMeta(cv::GMatDesc in) {
            return in.withType(CV_8U, 3);
        }
    };

    G_TYPED_KERNEL(GRGB2HSV, <cv::GMat(cv::GMat)>, "org.opencv.imgproc.colorconvert.rgb2hsv") {
        static cv::GMatDesc outMeta(cv::GMatDesc in) {
            return in;
        }
    };

    G_TYPED_KERNEL(GRGB2YUV422, <cv::GMat(cv::GMat)>, "org.opencv.imgproc.colorconvert.rgb2yuv422") {
        static cv::GMatDesc outMeta(cv::GMatDesc in) {
            GAPI_Assert(in.depth == CV_8U);
            GAPI_Assert(in.chan == 3);
            return in.withType(in.depth, 2);
        }
    };

    G_TYPED_KERNEL(GNV12toRGBp, <GMatP(GMat,GMat)>, "org.opencv.imgproc.colorconvert.nv12torgbp") {
        static GMatDesc outMeta(GMatDesc inY, GMatDesc inUV) {
            GAPI_Assert(inY.depth == CV_8U);
            GAPI_Assert(inUV.depth == CV_8U);
            GAPI_Assert(inY.chan == 1);
            GAPI_Assert(inY.planar == false);
            GAPI_Assert(inUV.chan == 2);
            GAPI_Assert(inUV.planar == false);
            GAPI_Assert(inY.size.width  == 2 * inUV.size.width);
            GAPI_Assert(inY.size.height == 2 * inUV.size.height);
            return inY.withType(CV_8U, 3).asPlanar();
        }
    };

    G_TYPED_KERNEL(GNV12toGray, <GMat(GMat,GMat)>, "org.opencv.imgproc.colorconvert.nv12togray") {
        static GMatDesc outMeta(GMatDesc inY, GMatDesc inUV) {
            GAPI_Assert(inY.depth   == CV_8U);
            GAPI_Assert(inUV.depth  == CV_8U);
            GAPI_Assert(inY.chan    == 1);
            GAPI_Assert(inY.planar  == false);
            GAPI_Assert(inUV.chan   == 2);
            GAPI_Assert(inUV.planar == false);

            GAPI_Assert(inY.size.width  == 2 * inUV.size.width);
            GAPI_Assert(inY.size.height == 2 * inUV.size.height);
            return inY.withType(CV_8U, 1);
        }
    };

    G_TYPED_KERNEL(GNV12toBGRp, <GMatP(GMat,GMat)>, "org.opencv.imgproc.colorconvert.nv12tobgrp") {
        static GMatDesc outMeta(GMatDesc inY, GMatDesc inUV) {
            GAPI_Assert(inY.depth == CV_8U);
            GAPI_Assert(inUV.depth == CV_8U);
            GAPI_Assert(inY.chan == 1);
            GAPI_Assert(inY.planar == false);
            GAPI_Assert(inUV.chan == 2);
            GAPI_Assert(inUV.planar == false);
            GAPI_Assert(inY.size.width  == 2 * inUV.size.width);
            GAPI_Assert(inY.size.height == 2 * inUV.size.height);
            return inY.withType(CV_8U, 3).asPlanar();
        }
    };

    G_TYPED_KERNEL(GResize, <GMat(GMat,Size,double,double,int)>, "org.opencv.imgproc.transform.resize") {
        static GMatDesc outMeta(GMatDesc in, Size sz, double fx, double fy, int /*interp*/) {
            if (sz.width != 0 && sz.height != 0)
            {
                return in.withSize(sz);
            }
            else
            {
                int outSz_w = saturate_cast<int>(in.size.width  * fx);
                int outSz_h = saturate_cast<int>(in.size.height * fy);
                GAPI_Assert(outSz_w > 0 && outSz_h > 0);
                return in.withSize(Size(outSz_w, outSz_h));
            }
        }
    };

    G_TYPED_KERNEL(GResizeP, <GMatP(GMatP,Size,int)>, "org.opencv.imgproc.transform.resizeP") {
        static GMatDesc outMeta(GMatDesc in, Size sz, int interp) {
            GAPI_Assert(in.depth == CV_8U);
            GAPI_Assert(in.chan == 3);
            GAPI_Assert(in.planar);
            GAPI_Assert(interp == cv::INTER_LINEAR);
            return in.withSize(sz);
        }
    };

} //namespace imgproc

//! @addtogroup gapi_filters
//! @{
/** @brief Applies a separable linear filter to a matrix(image).

The function applies a separable linear filter to the matrix. That is, first, every row of src is
filtered with the 1D kernel kernelX. Then, every column of the result is filtered with the 1D
kernel kernelY. The final result is returned.

Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
Output image must have the same type, size, and number of channels as the input image.
@note
 - In case of floating-point computation, rounding to nearest even is procedeed
if hardware supports it (if not - to nearest value).
 - Function textual ID is "org.opencv.imgproc.filters.sepfilter"
@param src Source image.
@param ddepth desired depth of the destination image (the following combinations of src.depth() and ddepth are supported:

        src.depth() = CV_8U, ddepth = -1/CV_16S/CV_32F/CV_64F
        src.depth() = CV_16U/CV_16S, ddepth = -1/CV_32F/CV_64F
        src.depth() = CV_32F, ddepth = -1/CV_32F/CV_64F
        src.depth() = CV_64F, ddepth = -1/CV_64F

when ddepth=-1, the output image will have the same depth as the source)
@param kernelX Coefficients for filtering each row.
@param kernelY Coefficients for filtering each column.
@param anchor Anchor position within the kernel. The default value \f$(-1,-1)\f$ means that the anchor
is at the kernel center.
@param delta Value added to the filtered results before storing them.
@param borderType Pixel extrapolation method, see cv::BorderTypes
@param borderValue border value in case of constant border type
@sa  boxFilter, gaussianBlur, medianBlur
 */
GAPI_EXPORTS_W GMat sepFilter(const GMat& src, int ddepth, const Mat& kernelX, const Mat& kernelY, const Point& anchor /*FIXME: = Point(-1,-1)*/,
                              const Scalar& delta /*FIXME = GScalar(0)*/, int borderType = BORDER_DEFAULT,
                              const Scalar& borderValue = Scalar(0));

/** @brief Convolves an image with the kernel.

The function applies an arbitrary linear filter to an image. When
the aperture is partially outside the image, the function interpolates outlier pixel values
according to the specified border mode.

The function does actually compute correlation, not the convolution:

\f[\texttt{dst} (x,y) =  \sum _{ \substack{0\leq x' < \texttt{kernel.cols}\\{0\leq y' < \texttt{kernel.rows}}}}  \texttt{kernel} (x',y')* \texttt{src} (x+x'- \texttt{anchor.x} ,y+y'- \texttt{anchor.y} )\f]

That is, the kernel is not mirrored around the anchor point. If you need a real convolution, flip
the kernel using flip and set the new anchor to `(kernel.cols - anchor.x - 1, kernel.rows -
anchor.y - 1)`.

Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
Output image must have the same size and number of channels an input image.
@note
 - Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.
 - Function textual ID is "org.opencv.imgproc.filters.filter2D"

@param src input image.
@param ddepth desired depth of the destination image
@param kernel convolution kernel (or rather a correlation kernel), a single-channel floating point
matrix; if you want to apply different kernels to different channels, split the image into
separate color planes using split and process them individually.
@param anchor anchor of the kernel that indicates the relative position of a filtered point within
the kernel; the anchor should lie within the kernel; default value (-1,-1) means that the anchor
is at the kernel center.
@param delta optional value added to the filtered pixels before storing them in dst.
@param borderType pixel extrapolation method, see cv::BorderTypes
@param borderValue border value in case of constant border type
@sa  sepFilter
 */
GAPI_EXPORTS_W GMat filter2D(const GMat& src, int ddepth, const Mat& kernel, const Point& anchor = Point(-1,-1), const Scalar& delta = Scalar(0),
                             int borderType = BORDER_DEFAULT, const Scalar& borderValue = Scalar(0));


/** @brief Blurs an image using the box filter.

The function smooths an image using the kernel:

\f[\texttt{K} =  \alpha \begin{bmatrix} 1 & 1 & 1 &  \cdots & 1 & 1  \\ 1 & 1 & 1 &  \cdots & 1 & 1  \\ \hdotsfor{6} \\ 1 & 1 & 1 &  \cdots & 1 & 1 \end{bmatrix}\f]

where

\f[\alpha = \begin{cases} \frac{1}{\texttt{ksize.width*ksize.height}} & \texttt{when } \texttt{normalize=true}  \\1 & \texttt{otherwise} \end{cases}\f]

Unnormalized box filter is useful for computing various integral characteristics over each pixel
neighborhood, such as covariance matrices of image derivatives (used in dense optical flow
algorithms, and so on). If you need to compute pixel sums over variable-size windows, use cv::integral.

Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
Output image must have the same type, size, and number of channels as the input image.
@note
 - Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.
 - Function textual ID is "org.opencv.imgproc.filters.boxfilter"

@param src Source image.
@param dtype the output image depth (-1 to set the input image data type).
@param ksize blurring kernel size.
@param anchor Anchor position within the kernel. The default value \f$(-1,-1)\f$ means that the anchor
is at the kernel center.
@param normalize flag, specifying whether the kernel is normalized by its area or not.
@param borderType Pixel extrapolation method, see cv::BorderTypes
@param borderValue border value in case of constant border type
@sa  sepFilter, gaussianBlur, medianBlur, integral
 */
GAPI_EXPORTS_W GMat boxFilter(const GMat& src, int dtype, const Size& ksize, const Point& anchor = Point(-1,-1),
                              bool normalize = true, int borderType = BORDER_DEFAULT,
                              const Scalar& borderValue = Scalar(0));

/** @brief Blurs an image using the normalized box filter.

The function smooths an image using the kernel:

\f[\texttt{K} =  \frac{1}{\texttt{ksize.width*ksize.height}} \begin{bmatrix} 1 & 1 & 1 &  \cdots & 1 & 1  \\ 1 & 1 & 1 &  \cdots & 1 & 1  \\ \hdotsfor{6} \\ 1 & 1 & 1 &  \cdots & 1 & 1  \\ \end{bmatrix}\f]

The call `blur(src, ksize, anchor, borderType)` is equivalent to `boxFilter(src, src.type(), ksize, anchor,
true, borderType)`.

Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
Output image must have the same type, size, and number of channels as the input image.
@note
 - Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.
 - Function textual ID is "org.opencv.imgproc.filters.blur"

@param src Source image.
@param ksize blurring kernel size.
@param anchor anchor point; default value Point(-1,-1) means that the anchor is at the kernel
center.
@param borderType border mode used to extrapolate pixels outside of the image, see cv::BorderTypes
@param borderValue border value in case of constant border type
@sa  boxFilter, bilateralFilter, GaussianBlur, medianBlur
 */
GAPI_EXPORTS_W GMat blur(const GMat& src, const Size& ksize, const Point& anchor = Point(-1,-1),
                         int borderType = BORDER_DEFAULT, const Scalar& borderValue = Scalar(0));


//GAPI_EXPORTS_W void blur( InputArray src, OutputArray dst,
 //                       Size ksize, Point anchor = Point(-1,-1),
 //                       int borderType = BORDER_DEFAULT );


/** @brief Blurs an image using a Gaussian filter.

The function filter2Ds the source image with the specified Gaussian kernel.
Output image must have the same type and number of channels an input image.

Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
Output image must have the same type, size, and number of channels as the input image.
@note
 - Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.
 - Function textual ID is "org.opencv.imgproc.filters.gaussianBlur"

@param src input image;
@param ksize Gaussian kernel size. ksize.width and ksize.height can differ but they both must be
positive and odd. Or, they can be zero's and then they are computed from sigma.
@param sigmaX Gaussian kernel standard deviation in X direction.
@param sigmaY Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be
equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height,
respectively (see cv::getGaussianKernel for details); to fully control the result regardless of
possible future modifications of all this semantics, it is recommended to specify all of ksize,
sigmaX, and sigmaY.
@param borderType pixel extrapolation method, see cv::BorderTypes
@param borderValue border value in case of constant border type
@sa  sepFilter, boxFilter, medianBlur
 */
GAPI_EXPORTS_W GMat gaussianBlur(const GMat& src, const Size& ksize, double sigmaX, double sigmaY = 0,
                                 int borderType = BORDER_DEFAULT, const Scalar& borderValue = Scalar(0));

/** @brief Blurs an image using the median filter.

The function smoothes an image using the median filter with the \f$\texttt{ksize} \times
\texttt{ksize}\f$ aperture. Each channel of a multi-channel image is processed independently.
Output image must have the same type, size, and number of channels as the input image.
@note
 - Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.
The median filter uses cv::BORDER_REPLICATE internally to cope with border pixels, see cv::BorderTypes
 - Function textual ID is "org.opencv.imgproc.filters.medianBlur"

@param src input matrix (image)
@param ksize aperture linear size; it must be odd and greater than 1, for example: 3, 5, 7 ...
@sa  boxFilter, gaussianBlur
 */
GAPI_EXPORTS_W GMat medianBlur(const GMat& src, int ksize);

/** @brief Erodes an image by using a specific structuring element.

The function erodes the source image using the specified structuring element that determines the
shape of a pixel neighborhood over which the minimum is taken:

\f[\texttt{dst} (x,y) =  \min _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')\f]

Erosion can be applied several (iterations) times. In case of multi-channel images, each channel is processed independently.
Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, and @ref CV_32FC1.
Output image must have the same type, size, and number of channels as the input image.
@note
 - Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.
 - Function textual ID is "org.opencv.imgproc.filters.erode"

@param src input image
@param kernel structuring element used for erosion; if `element=Mat()`, a `3 x 3` rectangular
structuring element is used. Kernel can be created using getStructuringElement.
@param anchor position of the anchor within the element; default value (-1, -1) means that the
anchor is at the element center.
@param iterations number of times erosion is applied.
@param borderType pixel extrapolation method, see cv::BorderTypes
@param borderValue border value in case of a constant border
@sa  dilate, morphologyEx
 */
GAPI_EXPORTS_W GMat erode(const GMat& src, const Mat& kernel, const Point& anchor = Point(-1,-1), int iterations = 1,
                          int borderType = BORDER_CONSTANT,
                          const  Scalar& borderValue = morphologyDefaultBorderValue());

/** @brief Erodes an image by using 3 by 3 rectangular structuring element.

The function erodes the source image using the rectangular structuring element with rectangle center as an anchor.
Erosion can be applied several (iterations) times. In case of multi-channel images, each channel is processed independently.
Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, and @ref CV_32FC1.
Output image must have the same type, size, and number of channels as the input image.
@note
 - Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.
 - Function textual ID is "org.opencv.imgproc.filters.erode"

@param src input image
@param iterations number of times erosion is applied.
@param borderType pixel extrapolation method, see cv::BorderTypes
@param borderValue border value in case of a constant border
@sa  erode, dilate3x3
 */
GAPI_EXPORTS_W GMat erode3x3(const GMat& src, int iterations = 1,
                             int borderType = BORDER_CONSTANT,
                             const  Scalar& borderValue = morphologyDefaultBorderValue());

/** @brief Dilates an image by using a specific structuring element.

The function dilates the source image using the specified structuring element that determines the
shape of a pixel neighborhood over which the maximum is taken:
\f[\texttt{dst} (x,y) =  \max _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')\f]

Dilation can be applied several (iterations) times. In case of multi-channel images, each channel is processed independently.
Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, and @ref CV_32FC1.
Output image must have the same type, size, and number of channels as the input image.
@note
 - Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.
 - Function textual ID is "org.opencv.imgproc.filters.dilate"

@param src input image.
@param kernel structuring element used for dilation; if elemenat=Mat(), a 3 x 3 rectangular
structuring element is used. Kernel can be created using getStructuringElement
@param anchor position of the anchor within the element; default value (-1, -1) means that the
anchor is at the element center.
@param iterations number of times dilation is applied.
@param borderType pixel extrapolation method, see cv::BorderTypes
@param borderValue border value in case of a constant border
@sa  erode, morphologyEx, getStructuringElement
 */
GAPI_EXPORTS_W GMat dilate(const GMat& src, const Mat& kernel, const Point& anchor = Point(-1,-1), int iterations = 1,
                           int borderType = BORDER_CONSTANT,
                           const  Scalar& borderValue = morphologyDefaultBorderValue());

/** @brief Dilates an image by using 3 by 3 rectangular structuring element.

The function dilates the source image using the specified structuring element that determines the
shape of a pixel neighborhood over which the maximum is taken:
\f[\texttt{dst} (x,y) =  \max _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')\f]

Dilation can be applied several (iterations) times. In case of multi-channel images, each channel is processed independently.
Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, and @ref CV_32FC1.
Output image must have the same type, size, and number of channels as the input image.
@note
 - Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.
 - Function textual ID is "org.opencv.imgproc.filters.dilate"

@param src input image.
@param iterations number of times dilation is applied.
@param borderType pixel extrapolation method, see cv::BorderTypes
@param borderValue border value in case of a constant border
@sa  dilate, erode3x3
 */

GAPI_EXPORTS_W GMat dilate3x3(const GMat& src, int iterations = 1,
                              int borderType = BORDER_CONSTANT,
                              const  Scalar& borderValue = morphologyDefaultBorderValue());

/** @brief Performs advanced morphological transformations.

The function can perform advanced morphological transformations using an erosion and dilation as
basic operations.

Any of the operations can be done in-place. In case of multi-channel images, each channel is
processed independently.

@note
 - Function textual ID is "org.opencv.imgproc.filters.morphologyEx"
 - The number of iterations is the number of times erosion or dilatation operation will be
applied. For instance, an opening operation (#MORPH_OPEN) with two iterations is equivalent to
apply successively: erode -> erode -> dilate -> dilate
(and not erode -> dilate -> erode -> dilate).

@param src Input image.
@param op Type of a morphological operation, see #MorphTypes
@param kernel Structuring element. It can be created using #getStructuringElement.
@param anchor Anchor position within the element. Both negative values mean that the anchor is at
the kernel center.
@param iterations Number of times erosion and dilation are applied.
@param borderType Pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
@param borderValue Border value in case of a constant border. The default value has a special
meaning.
@sa  dilate, erode, getStructuringElement
 */
GAPI_EXPORTS_W GMat morphologyEx(const GMat &src, const MorphTypes op, const Mat &kernel,
                                 const Point       &anchor      = Point(-1,-1),
                                 const int          iterations  = 1,
                                 const BorderTypes  borderType  = BORDER_CONSTANT,
                                 const Scalar      &borderValue = morphologyDefaultBorderValue());

/** @brief Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.

In all cases except one, the \f$\texttt{ksize} \times \texttt{ksize}\f$ separable kernel is used to
calculate the derivative. When \f$\texttt{ksize = 1}\f$, the \f$3 \times 1\f$ or \f$1 \times 3\f$
kernel is used (that is, no Gaussian smoothing is done). `ksize = 1` can only be used for the first
or the second x- or y- derivatives.

There is also the special value `ksize = FILTER_SCHARR (-1)` that corresponds to the \f$3\times3\f$ Scharr
filter that may give more accurate results than the \f$3\times3\f$ Sobel. The Scharr aperture is

\f[\vecthreethree{-3}{0}{3}{-10}{0}{10}{-3}{0}{3}\f]

for the x-derivative, or transposed for the y-derivative.

The function calculates an image derivative by convolving the image with the appropriate kernel:

\f[\texttt{dst} =  \frac{\partial^{xorder+yorder} \texttt{src}}{\partial x^{xorder} \partial y^{yorder}}\f]

The Sobel operators combine Gaussian smoothing and differentiation, so the result is more or less
resistant to the noise. Most often, the function is called with ( xorder = 1, yorder = 0, ksize = 3)
or ( xorder = 0, yorder = 1, ksize = 3) to calculate the first x- or y- image derivative. The first
case corresponds to a kernel of:

\f[\vecthreethree{-1}{0}{1}{-2}{0}{2}{-1}{0}{1}\f]

The second case corresponds to a kernel of:

\f[\vecthreethree{-1}{-2}{-1}{0}{0}{0}{1}{2}{1}\f]

@note
 - Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.
 - Function textual ID is "org.opencv.imgproc.filters.sobel"

@param src input image.
@param ddepth output image depth, see @ref filter_depths "combinations"; in the case of
    8-bit input images it will result in truncated derivatives.
@param dx order of the derivative x.
@param dy order of the derivative y.
@param ksize size of the extended Sobel kernel; it must be odd.
@param scale optional scale factor for the computed derivative values; by default, no scaling is
applied (see cv::getDerivKernels for details).
@param delta optional delta value that is added to the results prior to storing them in dst.
@param borderType pixel extrapolation method, see cv::BorderTypes
@param borderValue border value in case of constant border type
@sa filter2D, gaussianBlur, cartToPolar
 */
GAPI_EXPORTS_W GMat Sobel(const GMat& src, int ddepth, int dx, int dy, int ksize = 3,
                          double scale = 1, double delta = 0,
                          int borderType = BORDER_DEFAULT,
                          const Scalar& borderValue = Scalar(0));

/** @brief Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.

In all cases except one, the \f$\texttt{ksize} \times \texttt{ksize}\f$ separable kernel is used to
calculate the derivative. When \f$\texttt{ksize = 1}\f$, the \f$3 \times 1\f$ or \f$1 \times 3\f$
kernel is used (that is, no Gaussian smoothing is done). `ksize = 1` can only be used for the first
or the second x- or y- derivatives.

There is also the special value `ksize = FILTER_SCHARR (-1)` that corresponds to the \f$3\times3\f$ Scharr
filter that may give more accurate results than the \f$3\times3\f$ Sobel. The Scharr aperture is

\f[\vecthreethree{-3}{0}{3}{-10}{0}{10}{-3}{0}{3}\f]

for the x-derivative, or transposed for the y-derivative.

The function calculates an image derivative by convolving the image with the appropriate kernel:

\f[\texttt{dst} =  \frac{\partial^{xorder+yorder} \texttt{src}}{\partial x^{xorder} \partial y^{yorder}}\f]

The Sobel operators combine Gaussian smoothing and differentiation, so the result is more or less
resistant to the noise. Most often, the function is called with ( xorder = 1, yorder = 0, ksize = 3)
or ( xorder = 0, yorder = 1, ksize = 3) to calculate the first x- or y- image derivative. The first
case corresponds to a kernel of:

\f[\vecthreethree{-1}{0}{1}{-2}{0}{2}{-1}{0}{1}\f]

The second case corresponds to a kernel of:

\f[\vecthreethree{-1}{-2}{-1}{0}{0}{0}{1}{2}{1}\f]

@note
 - First returned matrix correspons to dx derivative while the second one to dy.
 - Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.
 - Function textual ID is "org.opencv.imgproc.filters.sobelxy"

@param src input image.
@param ddepth output image depth, see @ref filter_depths "combinations"; in the case of
    8-bit input images it will result in truncated derivatives.
@param order order of the derivatives.
@param ksize size of the extended Sobel kernel; it must be odd.
@param scale optional scale factor for the computed derivative values; by default, no scaling is
applied (see cv::getDerivKernels for details).
@param delta optional delta value that is added to the results prior to storing them in dst.
@param borderType pixel extrapolation method, see cv::BorderTypes
@param borderValue border value in case of constant border type
@sa filter2D, gaussianBlur, cartToPolar
 */
GAPI_EXPORTS_W std::tuple<GMat, GMat> SobelXY(const GMat& src, int ddepth, int order, int ksize = 3,
                                              double scale = 1, double delta = 0,
                                              int borderType = BORDER_DEFAULT,
                                              const Scalar& borderValue = Scalar(0));

/** @brief Calculates the Laplacian of an image.

The function calculates the Laplacian of the source image by adding up the second x and y
derivatives calculated using the Sobel operator:

\f[\texttt{dst} =  \Delta \texttt{src} =  \frac{\partial^2 \texttt{src}}{\partial x^2} +  \frac{\partial^2 \texttt{src}}{\partial y^2}\f]

This is done when `ksize > 1`. When `ksize == 1`, the Laplacian is computed by filtering the image
with the following \f$3 \times 3\f$ aperture:

\f[\vecthreethree {0}{1}{0}{1}{-4}{1}{0}{1}{0}\f]

@note Function textual ID is "org.opencv.imgproc.filters.laplacian"

@param src Source image.
@param ddepth Desired depth of the destination image.
@param ksize Aperture size used to compute the second-derivative filters. See #getDerivKernels for
details. The size must be positive and odd.
@param scale Optional scale factor for the computed Laplacian values. By default, no scaling is
applied. See #getDerivKernels for details.
@param delta Optional delta value that is added to the results prior to storing them in dst .
@param borderType Pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
@return Destination image of the same size and the same number of channels as src.
@sa  Sobel, Scharr
 */
GAPI_EXPORTS_W GMat Laplacian(const GMat& src, int ddepth, int ksize = 1,
                              double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT);

/** @brief Applies the bilateral filter to an image.

The function applies bilateral filtering to the input image, as described in
http://www.dai.ed.ac.uk/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html
bilateralFilter can reduce unwanted noise very well while keeping edges fairly sharp. However, it is
very slow compared to most filters.

_Sigma values_: For simplicity, you can set the 2 sigma values to be the same. If they are small (\<
10), the filter will not have much effect, whereas if they are large (\> 150), they will have a very
strong effect, making the image look "cartoonish".

_Filter size_: Large filters (d \> 5) are very slow, so it is recommended to use d=5 for real-time
applications, and perhaps d=9 for offline applications that need heavy noise filtering.

This filter does not work inplace.

@note Function textual ID is "org.opencv.imgproc.filters.bilateralfilter"

@param src Source 8-bit or floating-point, 1-channel or 3-channel image.
@param d Diameter of each pixel neighborhood that is used during filtering. If it is non-positive,
it is computed from sigmaSpace.
@param sigmaColor Filter sigma in the color space. A larger value of the parameter means that
farther colors within the pixel neighborhood (see sigmaSpace) will be mixed together, resulting
in larger areas of semi-equal color.
@param sigmaSpace Filter sigma in the coordinate space. A larger value of the parameter means that
farther pixels will influence each other as long as their colors are close enough (see sigmaColor
). When d\>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is
proportional to sigmaSpace.
@param borderType border mode used to extrapolate pixels outside of the image, see #BorderTypes
@return Destination image of the same size and type as src.
 */
GAPI_EXPORTS_W GMat bilateralFilter(const GMat& src, int d, double sigmaColor, double sigmaSpace,
                                    int borderType = BORDER_DEFAULT);

//! @} gapi_filters

//! @addtogroup gapi_feature
//! @{
/** @brief Finds edges in an image using the Canny algorithm.

The function finds edges in the input image and marks them in the output map edges using the
Canny algorithm. The smallest value between threshold1 and threshold2 is used for edge linking. The
largest value is used to find initial segments of strong edges. See
<http://en.wikipedia.org/wiki/Canny_edge_detector>

@note Function textual ID is "org.opencv.imgproc.feature.canny"

@param image 8-bit input image.
@param threshold1 first threshold for the hysteresis procedure.
@param threshold2 second threshold for the hysteresis procedure.
@param apertureSize aperture size for the Sobel operator.
@param L2gradient a flag, indicating whether a more accurate \f$L_2\f$ norm
\f$=\sqrt{(dI/dx)^2 + (dI/dy)^2}\f$ should be used to calculate the image gradient magnitude (
L2gradient=true ), or whether the default \f$L_1\f$ norm \f$=|dI/dx|+|dI/dy|\f$ is enough (
L2gradient=false ).
 */
GAPI_EXPORTS_W GMat Canny(const GMat& image, double threshold1, double threshold2,
                          int apertureSize = 3, bool L2gradient = false);

/** @brief Determines strong corners on an image.

The function finds the most prominent corners in the image or in the specified image region, as
described in @cite Shi94

-   Function calculates the corner quality measure at every source image pixel using the
    #cornerMinEigenVal or #cornerHarris .
-   Function performs a non-maximum suppression (the local maximums in *3 x 3* neighborhood are
    retained).
-   The corners with the minimal eigenvalue less than
    \f$\texttt{qualityLevel} \cdot \max_{x,y} qualityMeasureMap(x,y)\f$ are rejected.
-   The remaining corners are sorted by the quality measure in the descending order.
-   Function throws away each corner for which there is a stronger corner at a distance less than
    maxDistance.

The function can be used to initialize a point-based tracker of an object.

@note
 - If the function is called with different values A and B of the parameter qualityLevel , and
A \> B, the vector of returned corners with qualityLevel=A will be the prefix of the output vector
with qualityLevel=B .
 - Function textual ID is "org.opencv.imgproc.feature.goodFeaturesToTrack"

@param image Input 8-bit or floating-point 32-bit, single-channel image.
@param maxCorners Maximum number of corners to return. If there are more corners than are found,
the strongest of them is returned. `maxCorners <= 0` implies that no limit on the maximum is set
and all detected corners are returned.
@param qualityLevel Parameter characterizing the minimal accepted quality of image corners. The
parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue
(see #cornerMinEigenVal ) or the Harris function response (see #cornerHarris ). The corners with the
quality measure less than the product are rejected. For example, if the best corner has the
quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure
less than 15 are rejected.
@param minDistance Minimum possible Euclidean distance between the returned corners.
@param mask Optional region of interest. If the image is not empty (it needs to have the type
CV_8UC1 and the same size as image ), it specifies the region in which the corners are detected.
@param blockSize Size of an average block for computing a derivative covariation matrix over each
pixel neighborhood. See cornerEigenValsAndVecs .
@param useHarrisDetector Parameter indicating whether to use a Harris detector (see #cornerHarris)
or #cornerMinEigenVal.
@param k Free parameter of the Harris detector.

@return vector of detected corners.
 */
GAPI_EXPORTS_W GArray<Point2f> goodFeaturesToTrack(const GMat   &image,
                                                         int    maxCorners,
                                                         double qualityLevel,
                                                         double minDistance,
                                                   const Mat    &mask = Mat(),
                                                         int    blockSize = 3,
                                                         bool   useHarrisDetector = false,
                                                         double k = 0.04);

/** @brief Equalizes the histogram of a grayscale image.

//! @} gapi_feature

The function equalizes the histogram of the input image using the following algorithm:

- Calculate the histogram \f$H\f$ for src .
- Normalize the histogram so that the sum of histogram bins is 255.
- Compute the integral of the histogram:
\f[H'_i =  \sum _{0  \le j < i} H(j)\f]
- Transform the image using \f$H'\f$ as a look-up table: \f$\texttt{dst}(x,y) = H'(\texttt{src}(x,y))\f$

The algorithm normalizes the brightness and increases the contrast of the image.
@note
 - The returned image is of the same size and type as input.
 - Function textual ID is "org.opencv.imgproc.equalizeHist"

@param src Source 8-bit single channel image.
 */
GAPI_EXPORTS_W GMat equalizeHist(const GMat& src);

//! @addtogroup gapi_shape
//! @{
/** @brief Finds contours in a binary image.

The function retrieves contours from the binary image using the algorithm @cite Suzuki85 .
The contours are a useful tool for shape analysis and object detection and recognition.
See squares.cpp in the OpenCV sample directory.

@note Function textual ID is "org.opencv.imgproc.shape.findContours"

@param src Input gray-scale image @ref CV_8UC1. Non-zero pixels are treated as 1's. Zero
pixels remain 0's, so the image is treated as binary . You can use #compare, #inRange, #threshold ,
#adaptiveThreshold, #Canny, and others to create a binary image out of a grayscale or color one.
If mode equals to #RETR_CCOMP, the input can also be a 32-bit integer
image of labels ( @ref CV_32SC1 ). If #RETR_FLOODFILL then @ref CV_32SC1 is supported only.
@param mode Contour retrieval mode, see #RetrievalModes
@param method Contour approximation method, see #ContourApproximationModes
@param offset Optional offset by which every contour point is shifted. This is useful if the
contours are extracted from the image ROI and then they should be analyzed in the whole image
context.

@return GArray of detected contours. Each contour is stored as a GArray of points.
 */
GAPI_EXPORTS GArray<GArray<Point>>
findContours(const GMat &src, const RetrievalModes mode, const ContourApproximationModes method,
             const GOpaque<Point> &offset);

// FIXME oc: make default value offset = Point()
/** @overload
@note Function textual ID is "org.opencv.imgproc.shape.findContoursNoOffset"
 */
GAPI_EXPORTS GArray<GArray<Point>>
findContours(const GMat &src, const RetrievalModes mode, const ContourApproximationModes method);

/** @brief Finds contours and their hierarchy in a binary image.

The function retrieves contours from the binary image using the algorithm @cite Suzuki85
and calculates their hierarchy.
The contours are a useful tool for shape analysis and object detection and recognition.
See squares.cpp in the OpenCV sample directory.

@note Function textual ID is "org.opencv.imgproc.shape.findContoursH"

@param src Input gray-scale image @ref CV_8UC1. Non-zero pixels are treated as 1's. Zero
pixels remain 0's, so the image is treated as binary . You can use #compare, #inRange, #threshold ,
#adaptiveThreshold, #Canny, and others to create a binary image out of a grayscale or color one.
If mode equals to #RETR_CCOMP, the input can also be a 32-bit integer
image of labels ( @ref CV_32SC1 ). If #RETR_FLOODFILL -- @ref CV_32SC1 supports only.
@param mode Contour retrieval mode, see #RetrievalModes
@param method Contour approximation method, see #ContourApproximationModes
@param offset Optional offset by which every contour point is shifted. This is useful if the
contours are extracted from the image ROI and then they should be analyzed in the whole image
context.

@return
 - GArray of detected contours. Each contour is stored as a GArray of points.
 - Optional output GArray of cv::Vec4i, containing information about the image topology.
It has as many elements as the number of contours. For each i-th contour contours[i], the elements
hierarchy[i][0] , hierarchy[i][1] , hierarchy[i][2] , and hierarchy[i][3] are set to 0-based
indices in contours of the next and previous contours at the same hierarchical level, the first
child contour and the parent contour, respectively. If for the contour i there are no next,
previous, parent, or nested contours, the corresponding elements of hierarchy[i] will be negative.
 */
GAPI_EXPORTS std::tuple<GArray<GArray<Point>>,GArray<Vec4i>>
findContoursH(const GMat &src, const RetrievalModes mode, const ContourApproximationModes method,
              const GOpaque<Point> &offset);

// FIXME oc: make default value offset = Point()
/** @overload
@note Function textual ID is "org.opencv.imgproc.shape.findContoursHNoOffset"
 */
GAPI_EXPORTS std::tuple<GArray<GArray<Point>>,GArray<Vec4i>>
findContoursH(const GMat &src, const RetrievalModes mode, const ContourApproximationModes method);

/** @brief Calculates the up-right bounding rectangle of a point set or non-zero pixels
of gray-scale image.

The function calculates and returns the minimal up-right bounding rectangle for the specified
point set or non-zero pixels of gray-scale image.

@note
 - Function textual ID is "org.opencv.imgproc.shape.boundingRectMat"
 - In case of a 2D points' set given, Mat should be 2-dimensional, have a single row or column
if there are 2 channels, or have 2 columns if there is a single channel. Mat should have either
@ref CV_32S or @ref CV_32F depth

@param src Input gray-scale image @ref CV_8UC1; or input set of @ref CV_32S or @ref CV_32F
2D points stored in Mat.
 */
GAPI_EXPORTS_W GOpaque<Rect> boundingRect(const GMat& src);

/** @overload

Calculates the up-right bounding rectangle of a point set.

@note Function textual ID is "org.opencv.imgproc.shape.boundingRectVector32S"

@param src Input 2D point set, stored in std::vector<cv::Point2i>.
 */
GAPI_EXPORTS_W GOpaque<Rect> boundingRect(const GArray<Point2i>& src);

/** @overload

Calculates the up-right bounding rectangle of a point set.

@note Function textual ID is "org.opencv.imgproc.shape.boundingRectVector32F"

@param src Input 2D point set, stored in std::vector<cv::Point2f>.
 */
GAPI_EXPORTS_W GOpaque<Rect> boundingRect(const GArray<Point2f>& src);

/** @brief Fits a line to a 2D point set.

The function fits a line to a 2D point set by minimizing \f$\sum_i \rho(r_i)\f$ where
\f$r_i\f$ is a distance between the \f$i^{th}\f$ point, the line and \f$\rho(r)\f$ is a distance
function, one of the following:
-  DIST_L2
\f[\rho (r) = r^2/2  \quad \text{(the simplest and the fastest least-squares method)}\f]
- DIST_L1
\f[\rho (r) = r\f]
- DIST_L12
\f[\rho (r) = 2  \cdot ( \sqrt{1 + \frac{r^2}{2}} - 1)\f]
- DIST_FAIR
\f[\rho \left (r \right ) = C^2  \cdot \left (  \frac{r}{C} -  \log{\left(1 + \frac{r}{C}\right)} \right )  \quad \text{where} \quad C=1.3998\f]
- DIST_WELSCH
\f[\rho \left (r \right ) =  \frac{C^2}{2} \cdot \left ( 1 -  \exp{\left(-\left(\frac{r}{C}\right)^2\right)} \right )  \quad \text{where} \quad C=2.9846\f]
- DIST_HUBER
\f[\rho (r) =  \fork{r^2/2}{if \(r < C\)}{C \cdot (r-C/2)}{otherwise} \quad \text{where} \quad C=1.345\f]

The algorithm is based on the M-estimator ( <http://en.wikipedia.org/wiki/M-estimator> ) technique
that iteratively fits the line using the weighted least-squares algorithm. After each iteration the
weights \f$w_i\f$ are adjusted to be inversely proportional to \f$\rho(r_i)\f$ .

@note
 - Function textual ID is "org.opencv.imgproc.shape.fitLine2DMat"
 - In case of an N-dimentional points' set given, Mat should be 2-dimensional, have a single row
or column if there are N channels, or have N columns if there is a single channel.

@param src Input set of 2D points stored in one of possible containers: Mat,
std::vector<cv::Point2i>, std::vector<cv::Point2f>, std::vector<cv::Point2d>.
@param distType Distance used by the M-estimator, see #DistanceTypes. @ref DIST_USER
and @ref DIST_C are not supported.
@param param Numerical parameter ( C ) for some types of distances. If it is 0, an optimal value
is chosen.
@param reps Sufficient accuracy for the radius (distance between the coordinate origin and the
line). 1.0 would be a good default value for reps. If it is 0, a default value is chosen.
@param aeps Sufficient accuracy for the angle. 0.01 would be a good default value for aeps.
If it is 0, a default value is chosen.

@return Output line parameters: a vector of 4 elements (like Vec4f) - (vx, vy, x0, y0),
where (vx, vy) is a normalized vector collinear to the line and (x0, y0) is a point on the line.
 */
GAPI_EXPORTS GOpaque<Vec4f> fitLine2D(const GMat& src, const DistanceTypes distType,
                                      const double param = 0., const double reps = 0.,
                                      const double aeps = 0.);

/** @overload

@note Function textual ID is "org.opencv.imgproc.shape.fitLine2DVector32S"

 */
GAPI_EXPORTS GOpaque<Vec4f> fitLine2D(const GArray<Point2i>& src, const DistanceTypes distType,
                                      const double param = 0., const double reps = 0.,
                                      const double aeps = 0.);

/** @overload

@note Function textual ID is "org.opencv.imgproc.shape.fitLine2DVector32F"

 */
GAPI_EXPORTS GOpaque<Vec4f> fitLine2D(const GArray<Point2f>& src, const DistanceTypes distType,
                                      const double param = 0., const double reps = 0.,
                                      const double aeps = 0.);

/** @overload

@note Function textual ID is "org.opencv.imgproc.shape.fitLine2DVector64F"

 */
GAPI_EXPORTS GOpaque<Vec4f> fitLine2D(const GArray<Point2d>& src, const DistanceTypes distType,
                                      const double param = 0., const double reps = 0.,
                                      const double aeps = 0.);

/** @brief Fits a line to a 3D point set.

The function fits a line to a 3D point set by minimizing \f$\sum_i \rho(r_i)\f$ where
\f$r_i\f$ is a distance between the \f$i^{th}\f$ point, the line and \f$\rho(r)\f$ is a distance
function, one of the following:
-  DIST_L2
\f[\rho (r) = r^2/2  \quad \text{(the simplest and the fastest least-squares method)}\f]
- DIST_L1
\f[\rho (r) = r\f]
- DIST_L12
\f[\rho (r) = 2  \cdot ( \sqrt{1 + \frac{r^2}{2}} - 1)\f]
- DIST_FAIR
\f[\rho \left (r \right ) = C^2  \cdot \left (  \frac{r}{C} -  \log{\left(1 + \frac{r}{C}\right)} \right )  \quad \text{where} \quad C=1.3998\f]
- DIST_WELSCH
\f[\rho \left (r \right ) =  \frac{C^2}{2} \cdot \left ( 1 -  \exp{\left(-\left(\frac{r}{C}\right)^2\right)} \right )  \quad \text{where} \quad C=2.9846\f]
- DIST_HUBER
\f[\rho (r) =  \fork{r^2/2}{if \(r < C\)}{C \cdot (r-C/2)}{otherwise} \quad \text{where} \quad C=1.345\f]

The algorithm is based on the M-estimator ( <http://en.wikipedia.org/wiki/M-estimator> ) technique
that iteratively fits the line using the weighted least-squares algorithm. After each iteration the
weights \f$w_i\f$ are adjusted to be inversely proportional to \f$\rho(r_i)\f$ .

@note
 - Function textual ID is "org.opencv.imgproc.shape.fitLine3DMat"
 - In case of an N-dimentional points' set given, Mat should be 2-dimensional, have a single row
or column if there are N channels, or have N columns if there is a single channel.

@param src Input set of 3D points stored in one of possible containers: Mat,
std::vector<cv::Point3i>, std::vector<cv::Point3f>, std::vector<cv::Point3d>.
@param distType Distance used by the M-estimator, see #DistanceTypes. @ref DIST_USER
and @ref DIST_C are not supported.
@param param Numerical parameter ( C ) for some types of distances. If it is 0, an optimal value
is chosen.
@param reps Sufficient accuracy for the radius (distance between the coordinate origin and the
line). 1.0 would be a good default value for reps. If it is 0, a default value is chosen.
@param aeps Sufficient accuracy for the angle. 0.01 would be a good default value for aeps.
If it is 0, a default value is chosen.

@return Output line parameters: a vector of 6 elements (like Vec6f) - (vx, vy, vz, x0, y0, z0),
where (vx, vy, vz) is a normalized vector collinear to the line and (x0, y0, z0) is a point on
the line.
 */
GAPI_EXPORTS GOpaque<Vec6f> fitLine3D(const GMat& src, const DistanceTypes distType,
                                      const double param = 0., const double reps = 0.,
                                      const double aeps = 0.);

/** @overload

@note Function textual ID is "org.opencv.imgproc.shape.fitLine3DVector32S"

 */
GAPI_EXPORTS GOpaque<Vec6f> fitLine3D(const GArray<Point3i>& src, const DistanceTypes distType,
                                      const double param = 0., const double reps = 0.,
                                      const double aeps = 0.);

/** @overload

@note Function textual ID is "org.opencv.imgproc.shape.fitLine3DVector32F"

 */
GAPI_EXPORTS GOpaque<Vec6f> fitLine3D(const GArray<Point3f>& src, const DistanceTypes distType,
                                      const double param = 0., const double reps = 0.,
                                      const double aeps = 0.);

/** @overload

@note Function textual ID is "org.opencv.imgproc.shape.fitLine3DVector64F"

 */
GAPI_EXPORTS GOpaque<Vec6f> fitLine3D(const GArray<Point3d>& src, const DistanceTypes distType,
                                      const double param = 0., const double reps = 0.,
                                      const double aeps = 0.);

//! @} gapi_shape

//! @addtogroup gapi_colorconvert
//! @{
/** @brief Converts an image from BGR color space to RGB color space.

The function converts an input image from BGR color space to RGB.
The conventional ranges for B, G, and R channel values are 0 to 255.

Output image is 8-bit unsigned 3-channel image @ref CV_8UC3.

@note Function textual ID is "org.opencv.imgproc.colorconvert.bgr2rgb"

@param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.
@sa RGB2BGR
*/
GAPI_EXPORTS_W GMat BGR2RGB(const GMat& src);

/** @brief Converts an image from RGB color space to gray-scaled.

The conventional ranges for R, G, and B channel values are 0 to 255.
Resulting gray color value computed as
\f[\texttt{dst} (I)= \texttt{0.299} * \texttt{src}(I).R + \texttt{0.587} * \texttt{src}(I).G  + \texttt{0.114} * \texttt{src}(I).B \f]

@note Function textual ID is "org.opencv.imgproc.colorconvert.rgb2gray"

@param src input image: 8-bit unsigned 3-channel image @ref CV_8UC1.
@sa RGB2YUV
 */
GAPI_EXPORTS_W GMat RGB2Gray(const GMat& src);

/** @overload
Resulting gray color value computed as
\f[\texttt{dst} (I)= \texttt{rY} * \texttt{src}(I).R + \texttt{gY} * \texttt{src}(I).G  + \texttt{bY} * \texttt{src}(I).B \f]

@note Function textual ID is "org.opencv.imgproc.colorconvert.rgb2graycustom"

@param src input image: 8-bit unsigned 3-channel image @ref CV_8UC1.
@param rY float multiplier for R channel.
@param gY float multiplier for G channel.
@param bY float multiplier for B channel.
@sa RGB2YUV
 */
GAPI_EXPORTS_W GMat RGB2Gray(const GMat& src, float rY, float gY, float bY);

/** @brief Converts an image from BGR color space to gray-scaled.

The conventional ranges for B, G, and R channel values are 0 to 255.
Resulting gray color value computed as
\f[\texttt{dst} (I)= \texttt{0.114} * \texttt{src}(I).B + \texttt{0.587} * \texttt{src}(I).G  + \texttt{0.299} * \texttt{src}(I).R \f]

@note Function textual ID is "org.opencv.imgproc.colorconvert.bgr2gray"

@param src input image: 8-bit unsigned 3-channel image @ref CV_8UC1.
@sa BGR2LUV
 */
GAPI_EXPORTS_W GMat BGR2Gray(const GMat& src);

/** @brief Converts an image from RGB color space to YUV color space.

The function converts an input image from RGB color space to YUV.
The conventional ranges for R, G, and B channel values are 0 to 255.

In case of linear transformations, the range does not matter. But in case of a non-linear
transformation, an input RGB image should be normalized to the proper value range to get the correct
results, like here, at RGB \f$\rightarrow\f$ Y\*u\*v\* transformation.
Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.

@note Function textual ID is "org.opencv.imgproc.colorconvert.rgb2yuv"

@param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.
@sa YUV2RGB, RGB2Lab
*/
GAPI_EXPORTS_W GMat RGB2YUV(const GMat& src);

/** @brief Converts an image from BGR color space to I420 color space.

The function converts an input image from BGR color space to I420.
The conventional ranges for R, G, and B channel values are 0 to 255.

Output image must be 8-bit unsigned 1-channel image. @ref CV_8UC1.
Width of I420 output image must be the same as width of input image.
Height of I420 output image must be equal 3/2 from height of input image.

@note Function textual ID is "org.opencv.imgproc.colorconvert.bgr2i420"

@param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.
@sa I4202BGR
*/
GAPI_EXPORTS_W GMat BGR2I420(const GMat& src);

/** @brief Converts an image from RGB color space to I420 color space.

The function converts an input image from RGB color space to I420.
The conventional ranges for R, G, and B channel values are 0 to 255.

Output image must be 8-bit unsigned 1-channel image. @ref CV_8UC1.
Width of I420 output image must be the same as width of input image.
Height of I420 output image must be equal 3/2 from height of input image.

@note Function textual ID is "org.opencv.imgproc.colorconvert.rgb2i420"

@param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.
@sa I4202RGB
*/
GAPI_EXPORTS_W GMat RGB2I420(const GMat& src);

/** @brief Converts an image from I420 color space to BGR color space.

The function converts an input image from I420 color space to BGR.
The conventional ranges for B, G, and R channel values are 0 to 255.

Output image must be 8-bit unsigned 3-channel image. @ref CV_8UC3.
Width of BGR output image must be the same as width of input image.
Height of BGR output image must be equal 2/3 from height of input image.

@note Function textual ID is "org.opencv.imgproc.colorconvert.i4202bgr"

@param src input image: 8-bit unsigned 1-channel image @ref CV_8UC1.
@sa BGR2I420
*/
GAPI_EXPORTS_W GMat I4202BGR(const GMat& src);

/** @brief Converts an image from I420 color space to BGR color space.

The function converts an input image from I420 color space to BGR.
The conventional ranges for B, G, and R channel values are 0 to 255.

Output image must be 8-bit unsigned 3-channel image. @ref CV_8UC3.
Width of RGB output image must be the same as width of input image.
Height of RGB output image must be equal 2/3 from height of input image.

@note Function textual ID is "org.opencv.imgproc.colorconvert.i4202rgb"

@param src input image: 8-bit unsigned 1-channel image @ref CV_8UC1.
@sa RGB2I420
*/
GAPI_EXPORTS_W GMat I4202RGB(const GMat& src);

/** @brief Converts an image from BGR color space to LUV color space.

The function converts an input image from BGR color space to LUV.
The conventional ranges for B, G, and R channel values are 0 to 255.

Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.

@note Function textual ID is "org.opencv.imgproc.colorconvert.bgr2luv"

@param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.
@sa RGB2Lab, RGB2LUV
*/
GAPI_EXPORTS_W GMat BGR2LUV(const GMat& src);

/** @brief Converts an image from LUV color space to BGR color space.

The function converts an input image from LUV color space to BGR.
The conventional ranges for B, G, and R channel values are 0 to 255.

Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.

@note Function textual ID is "org.opencv.imgproc.colorconvert.luv2bgr"

@param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.
@sa BGR2LUV
*/
GAPI_EXPORTS_W GMat LUV2BGR(const GMat& src);

/** @brief Converts an image from YUV color space to BGR color space.

The function converts an input image from YUV color space to BGR.
The conventional ranges for B, G, and R channel values are 0 to 255.

Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.

@note Function textual ID is "org.opencv.imgproc.colorconvert.yuv2bgr"

@param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.
@sa BGR2YUV
*/
GAPI_EXPORTS_W GMat YUV2BGR(const GMat& src);

/** @brief Converts an image from BGR color space to YUV color space.

The function converts an input image from BGR color space to YUV.
The conventional ranges for B, G, and R channel values are 0 to 255.

Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.

@note Function textual ID is "org.opencv.imgproc.colorconvert.bgr2yuv"

@param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.
@sa YUV2BGR
*/
GAPI_EXPORTS_W GMat BGR2YUV(const GMat& src);

/** @brief Converts an image from RGB color space to Lab color space.

The function converts an input image from BGR color space to Lab.
The conventional ranges for R, G, and B channel values are 0 to 255.

Output image must be 8-bit unsigned 3-channel image @ref CV_8UC1.

@note Function textual ID is "org.opencv.imgproc.colorconvert.rgb2lab"

@param src input image: 8-bit unsigned 3-channel image @ref CV_8UC1.
@sa RGB2YUV, RGB2LUV
*/
GAPI_EXPORTS_W GMat RGB2Lab(const GMat& src);

/** @brief Converts an image from YUV color space to RGB.
The function converts an input image from YUV color space to RGB.
The conventional ranges for Y, U, and V channel values are 0 to 255.

Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.

@note Function textual ID is "org.opencv.imgproc.colorconvert.yuv2rgb"

@param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.

@sa RGB2Lab, RGB2YUV
*/
GAPI_EXPORTS_W GMat YUV2RGB(const GMat& src);

/** @brief Converts an image from NV12 (YUV420p) color space to RGB.
The function converts an input image from NV12 color space to RGB.
The conventional ranges for Y, U, and V channel values are 0 to 255.

Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.

@note Function textual ID is "org.opencv.imgproc.colorconvert.nv12torgb"

@param src_y input image: 8-bit unsigned 1-channel image @ref CV_8UC1.
@param src_uv input image: 8-bit unsigned 2-channel image @ref CV_8UC2.

@sa YUV2RGB, NV12toBGR
*/
GAPI_EXPORTS_W GMat NV12toRGB(const GMat& src_y, const GMat& src_uv);

/** @brief Converts an image from NV12 (YUV420p) color space to gray-scaled.
The function converts an input image from NV12 color space to gray-scaled.
The conventional ranges for Y, U, and V channel values are 0 to 255.

Output image must be 8-bit unsigned 1-channel image @ref CV_8UC1.

@note Function textual ID is "org.opencv.imgproc.colorconvert.nv12togray"

@param src_y input image: 8-bit unsigned 1-channel image @ref CV_8UC1.
@param src_uv input image: 8-bit unsigned 2-channel image @ref CV_8UC2.

@sa YUV2RGB, NV12toBGR
*/
GAPI_EXPORTS_W GMat NV12toGray(const GMat& src_y, const GMat& src_uv);

/** @brief Converts an image from NV12 (YUV420p) color space to BGR.
The function converts an input image from NV12 color space to RGB.
The conventional ranges for Y, U, and V channel values are 0 to 255.

Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.

@note Function textual ID is "org.opencv.imgproc.colorconvert.nv12tobgr"

@param src_y input image: 8-bit unsigned 1-channel image @ref CV_8UC1.
@param src_uv input image: 8-bit unsigned 2-channel image @ref CV_8UC2.

@sa YUV2BGR, NV12toRGB
*/
GAPI_EXPORTS_W GMat NV12toBGR(const GMat& src_y, const GMat& src_uv);

/** @brief Converts an image from BayerGR color space to RGB.
The function converts an input image from BayerGR color space to RGB.
The conventional ranges for G, R, and B channel values are 0 to 255.

Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.

@note Function textual ID is "org.opencv.imgproc.colorconvert.bayergr2rgb"

@param src_gr input image: 8-bit unsigned 1-channel image @ref CV_8UC1.

@sa YUV2BGR, NV12toRGB
*/
GAPI_EXPORTS_W GMat BayerGR2RGB(const GMat& src_gr);

/** @brief Converts an image from RGB color space to HSV.
The function converts an input image from RGB color space to HSV.
The conventional ranges for R, G, and B channel values are 0 to 255.

Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.

@note Function textual ID is "org.opencv.imgproc.colorconvert.rgb2hsv"

@param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.

@sa YUV2BGR, NV12toRGB
*/
GAPI_EXPORTS_W GMat RGB2HSV(const GMat& src);

/** @brief Converts an image from RGB color space to YUV422.
The function converts an input image from RGB color space to YUV422.
The conventional ranges for R, G, and B channel values are 0 to 255.

Output image must be 8-bit unsigned 2-channel image @ref CV_8UC2.

@note Function textual ID is "org.opencv.imgproc.colorconvert.rgb2yuv422"

@param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.

@sa YUV2BGR, NV12toRGB
*/
GAPI_EXPORTS_W GMat RGB2YUV422(const GMat& src);

/** @brief Converts an image from NV12 (YUV420p) color space to RGB.
The function converts an input image from NV12 color space to RGB.
The conventional ranges for Y, U, and V channel values are 0 to 255.

Output image must be 8-bit unsigned planar 3-channel image @ref CV_8UC1.
Planar image memory layout is three planes laying in the memory contiguously,
so the image height should be plane_height*plane_number,
image type is @ref CV_8UC1.

@note Function textual ID is "org.opencv.imgproc.colorconvert.nv12torgbp"

@param src_y input image: 8-bit unsigned 1-channel image @ref CV_8UC1.
@param src_uv input image: 8-bit unsigned 2-channel image @ref CV_8UC2.

@sa YUV2RGB, NV12toBGRp, NV12toRGB
*/
GAPI_EXPORTS GMatP NV12toRGBp(const GMat &src_y, const GMat &src_uv);

/** @brief Converts an image from NV12 (YUV420p) color space to BGR.
The function converts an input image from NV12 color space to BGR.
The conventional ranges for Y, U, and V channel values are 0 to 255.

Output image must be 8-bit unsigned planar 3-channel image @ref CV_8UC1.
Planar image memory layout is three planes laying in the memory contiguously,
so the image height should be plane_height*plane_number,
image type is @ref CV_8UC1.

@note Function textual ID is "org.opencv.imgproc.colorconvert.nv12torgbp"

@param src_y input image: 8-bit unsigned 1-channel image @ref CV_8UC1.
@param src_uv input image: 8-bit unsigned 2-channel image @ref CV_8UC2.

@sa YUV2RGB, NV12toRGBp, NV12toBGR
*/
GAPI_EXPORTS GMatP NV12toBGRp(const GMat &src_y, const GMat &src_uv);

//! @} gapi_colorconvert
//! @addtogroup gapi_transform
//! @{
/** @brief Resizes an image.

The function resizes the image src down to or up to the specified size.

Output image size will have the size dsize (when dsize is non-zero) or the size computed from
src.size(), fx, and fy; the depth of output is the same as of src.

If you want to resize src so that it fits the pre-created dst,
you may call the function as follows:
@code
    // explicitly specify dsize=dst.size(); fx and fy will be computed from that.
    resize(src, dst, dst.size(), 0, 0, interpolation);
@endcode
If you want to decimate the image by factor of 2 in each direction, you can call the function this
way:
@code
    // specify fx and fy and let the function compute the destination image size.
    resize(src, dst, Size(), 0.5, 0.5, interpolation);
@endcode
To shrink an image, it will generally look best with cv::INTER_AREA interpolation, whereas to
enlarge an image, it will generally look best with cv::INTER_CUBIC (slow) or cv::INTER_LINEAR
(faster but still looks OK).

@note Function textual ID is "org.opencv.imgproc.transform.resize"

@param src input image.
@param dsize output image size; if it equals zero, it is computed as:
 \f[\texttt{dsize = Size(round(fx*src.cols), round(fy*src.rows))}\f]
 Either dsize or both fx and fy must be non-zero.
@param fx scale factor along the horizontal axis; when it equals 0, it is computed as
\f[\texttt{(double)dsize.width/src.cols}\f]
@param fy scale factor along the vertical axis; when it equals 0, it is computed as
\f[\texttt{(double)dsize.height/src.rows}\f]
@param interpolation interpolation method, see cv::InterpolationFlags

@sa  warpAffine, warpPerspective, remap, resizeP
 */
GAPI_EXPORTS_W GMat resize(const GMat& src, const Size& dsize, double fx = 0, double fy = 0, int interpolation = INTER_LINEAR);

/** @brief Resizes a planar image.

The function resizes the image src down to or up to the specified size.
Planar image memory layout is three planes laying in the memory contiguously,
so the image height should be plane_height*plane_number, image type is @ref CV_8UC1.

Output image size will have the size dsize, the depth of output is the same as of src.

@note Function textual ID is "org.opencv.imgproc.transform.resizeP"

@param src input image, must be of @ref CV_8UC1 type;
@param dsize output image size;
@param interpolation interpolation method, only cv::INTER_LINEAR is supported at the moment

@sa  warpAffine, warpPerspective, remap, resize
 */
GAPI_EXPORTS GMatP resizeP(const GMatP& src, const Size& dsize, int interpolation = cv::INTER_LINEAR);

//! @} gapi_transform
} //namespace gapi
} //namespace cv

#endif // OPENCV_GAPI_IMGPROC_HPP
