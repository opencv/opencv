// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


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
@}
 */

namespace cv { namespace gapi {

namespace imgproc {
    using GMat2 = std::tuple<GMat,GMat>;
    using GMat3 = std::tuple<GMat,GMat,GMat>; // FIXME: how to avoid this?

    G_TYPED_KERNEL(GFilter2D, <GMat(GMat,int,Mat,Point,Scalar,int,Scalar)>,"org.opencv.imgproc.filters.filter2D") {
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

    G_TYPED_KERNEL(GBlur, <GMat(GMat,Size,Point,int,Scalar)>,         "org.opencv.imgproc.filters.blur"){
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

    G_TYPED_KERNEL(GEqHist, <GMat(GMat)>, "org.opencv.imgproc.equalizeHist"){
        static GMatDesc outMeta(GMatDesc in) {
            return in.withType(CV_8U, 1);
        }
    };

    G_TYPED_KERNEL(GCanny, <GMat(GMat,double,double,int,bool)>, "org.opencv.imgproc.canny"){
        static GMatDesc outMeta(GMatDesc in, double, double, int, bool) {
            return in.withType(CV_8U, 1);
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

    G_TYPED_KERNEL(GNV12toRGBp, <GMatP(GMat,GMat)>, "org.opencv.colorconvert.imgproc.nv12torgbp") {
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

    G_TYPED_KERNEL(GNV12toBGRp, <GMatP(GMat,GMat)>, "org.opencv.colorconvert.imgproc.nv12tobgrp") {
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

}


//! @addtogroup gapi_filters
//! @{
/** @brief Applies a separable linear filter to a matrix(image).

The function applies a separable linear filter to the matrix. That is, first, every row of src is
filtered with the 1D kernel kernelX. Then, every column of the result is filtered with the 1D
kernel kernelY. The final result is returned.

Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
Output image must have the same type, size, and number of channels as the input image.
@note In case of floating-point computation, rounding to nearest even is procedeed
if hardware supports it (if not - to nearest value).

@note Function textual ID is "org.opencv.imgproc.filters.sepfilter"
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
GAPI_EXPORTS GMat sepFilter(const GMat& src, int ddepth, const Mat& kernelX, const Mat& kernelY, const Point& anchor /*FIXME: = Point(-1,-1)*/,
                            const Scalar& delta /*FIXME = GScalar(0)*/, int borderType = BORDER_DEFAULT,
                            const Scalar& borderValue = Scalar(0));

/** @brief Convolves an image with the kernel.

The function applies an arbitrary linear filter to an image. When
the aperture is partially outside the image, the function interpolates outlier pixel values
according to the specified border mode.

The function does actually compute correlation, not the convolution:

\f[\texttt{dst} (x,y) =  \sum _{ \stackrel{0\leq x' < \texttt{kernel.cols},}{0\leq y' < \texttt{kernel.rows}} }  \texttt{kernel} (x',y')* \texttt{src} (x+x'- \texttt{anchor.x} ,y+y'- \texttt{anchor.y} )\f]

That is, the kernel is not mirrored around the anchor point. If you need a real convolution, flip
the kernel using flip and set the new anchor to `(kernel.cols - anchor.x - 1, kernel.rows -
anchor.y - 1)`.

Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
Output image must have the same size and number of channels an input image.
@note Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.

@note Function textual ID is "org.opencv.imgproc.filters.filter2D"

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
GAPI_EXPORTS GMat filter2D(const GMat& src, int ddepth, const Mat& kernel, const Point& anchor = Point(-1,-1), const Scalar& delta = Scalar(0),
                           int borderType = BORDER_DEFAULT, const Scalar& borderValue = Scalar(0));


/** @brief Blurs an image using the box filter.

The function smooths an image using the kernel:

\f[\texttt{K} =  \alpha \begin{bmatrix} 1 & 1 & 1 &  \cdots & 1 & 1  \\ 1 & 1 & 1 &  \cdots & 1 & 1  \\ \hdotsfor{6} \\ 1 & 1 & 1 &  \cdots & 1 & 1 \end{bmatrix}\f]

where

\f[\alpha = \fork{\frac{1}{\texttt{ksize.width*ksize.height}}}{when \texttt{normalize=true}}{1}{otherwise}\f]

Unnormalized box filter is useful for computing various integral characteristics over each pixel
neighborhood, such as covariance matrices of image derivatives (used in dense optical flow
algorithms, and so on). If you need to compute pixel sums over variable-size windows, use cv::integral.

Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
Output image must have the same type, size, and number of channels as the input image.
@note Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.

@note Function textual ID is "org.opencv.imgproc.filters.boxfilter"

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
GAPI_EXPORTS GMat boxFilter(const GMat& src, int dtype, const Size& ksize, const Point& anchor = Point(-1,-1),
                            bool normalize = true, int borderType = BORDER_DEFAULT,
                            const Scalar& borderValue = Scalar(0));

/** @brief Blurs an image using the normalized box filter.

The function smooths an image using the kernel:

\f[\texttt{K} =  \frac{1}{\texttt{ksize.width*ksize.height}} \begin{bmatrix} 1 & 1 & 1 &  \cdots & 1 & 1  \\ 1 & 1 & 1 &  \cdots & 1 & 1  \\ \hdotsfor{6} \\ 1 & 1 & 1 &  \cdots & 1 & 1  \\ \end{bmatrix}\f]

The call `blur(src, dst, ksize, anchor, borderType)` is equivalent to `boxFilter(src, dst, src.type(),
anchor, true, borderType)`.

Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
Output image must have the same type, size, and number of channels as the input image.
@note Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.

@note Function textual ID is "org.opencv.imgproc.filters.blur"

@param src Source image.
@param ksize blurring kernel size.
@param anchor anchor point; default value Point(-1,-1) means that the anchor is at the kernel
center.
@param borderType border mode used to extrapolate pixels outside of the image, see cv::BorderTypes
@param borderValue border value in case of constant border type
@sa  boxFilter, bilateralFilter, GaussianBlur, medianBlur
 */
GAPI_EXPORTS GMat blur(const GMat& src, const Size& ksize, const Point& anchor = Point(-1,-1),
                       int borderType = BORDER_DEFAULT, const Scalar& borderValue = Scalar(0));


//GAPI_EXPORTS_W void blur( InputArray src, OutputArray dst,
 //                       Size ksize, Point anchor = Point(-1,-1),
 //                       int borderType = BORDER_DEFAULT );


/** @brief Blurs an image using a Gaussian filter.

The function filter2Ds the source image with the specified Gaussian kernel.
Output image must have the same type and number of channels an input image.

Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
Output image must have the same type, size, and number of channels as the input image.
@note Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.

@note Function textual ID is "org.opencv.imgproc.filters.gaussianBlur"

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
GAPI_EXPORTS GMat gaussianBlur(const GMat& src, const Size& ksize, double sigmaX, double sigmaY = 0,
                               int borderType = BORDER_DEFAULT, const Scalar& borderValue = Scalar(0));

/** @brief Blurs an image using the median filter.

The function smoothes an image using the median filter with the \f$\texttt{ksize} \times
\texttt{ksize}\f$ aperture. Each channel of a multi-channel image is processed independently.
Output image must have the same type, size, and number of channels as the input image.
@note Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.
The median filter uses cv::BORDER_REPLICATE internally to cope with border pixels, see cv::BorderTypes

@note Function textual ID is "org.opencv.imgproc.filters.medianBlur"

@param src input matrix (image)
@param ksize aperture linear size; it must be odd and greater than 1, for example: 3, 5, 7 ...
@sa  boxFilter, gaussianBlur
 */
GAPI_EXPORTS GMat medianBlur(const GMat& src, int ksize);

/** @brief Erodes an image by using a specific structuring element.

The function erodes the source image using the specified structuring element that determines the
shape of a pixel neighborhood over which the minimum is taken:

\f[\texttt{dst} (x,y) =  \min _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')\f]

Erosion can be applied several (iterations) times. In case of multi-channel images, each channel is processed independently.
Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, and @ref CV_32FC1.
Output image must have the same type, size, and number of channels as the input image.
@note Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.

@note Function textual ID is "org.opencv.imgproc.filters.erode"

@param src input image
@param kernel structuring element used for erosion; if `element=Mat()`, a `3 x 3` rectangular
structuring element is used. Kernel can be created using getStructuringElement.
@param anchor position of the anchor within the element; default value (-1, -1) means that the
anchor is at the element center.
@param iterations number of times erosion is applied.
@param borderType pixel extrapolation method, see cv::BorderTypes
@param borderValue border value in case of a constant border
@sa  dilate
 */
GAPI_EXPORTS GMat erode(const GMat& src, const Mat& kernel, const Point& anchor = Point(-1,-1), int iterations = 1,
                        int borderType = BORDER_CONSTANT,
                        const  Scalar& borderValue = morphologyDefaultBorderValue());

/** @brief Erodes an image by using 3 by 3 rectangular structuring element.

The function erodes the source image using the rectangular structuring element with rectangle center as an anchor.
Erosion can be applied several (iterations) times. In case of multi-channel images, each channel is processed independently.
Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, and @ref CV_32FC1.
Output image must have the same type, size, and number of channels as the input image.
@note Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.

@param src input image
@param iterations number of times erosion is applied.
@param borderType pixel extrapolation method, see cv::BorderTypes
@param borderValue border value in case of a constant border
@sa  erode, dilate3x3
 */
GAPI_EXPORTS GMat erode3x3(const GMat& src, int iterations = 1,
                           int borderType = BORDER_CONSTANT,
                           const  Scalar& borderValue = morphologyDefaultBorderValue());

/** @brief Dilates an image by using a specific structuring element.

The function dilates the source image using the specified structuring element that determines the
shape of a pixel neighborhood over which the maximum is taken:
\f[\texttt{dst} (x,y) =  \max _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')\f]

Dilation can be applied several (iterations) times. In case of multi-channel images, each channel is processed independently.
Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, and @ref CV_32FC1.
Output image must have the same type, size, and number of channels as the input image.
@note Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.

@note Function textual ID is "org.opencv.imgproc.filters.dilate"

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
GAPI_EXPORTS GMat dilate(const GMat& src, const Mat& kernel, const Point& anchor = Point(-1,-1), int iterations = 1,
                         int borderType = BORDER_CONSTANT,
                         const  Scalar& borderValue = morphologyDefaultBorderValue());

/** @brief Dilates an image by using 3 by 3 rectangular structuring element.

The function dilates the source image using the specified structuring element that determines the
shape of a pixel neighborhood over which the maximum is taken:
\f[\texttt{dst} (x,y) =  \max _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')\f]

Dilation can be applied several (iterations) times. In case of multi-channel images, each channel is processed independently.
Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, and @ref CV_32FC1.
Output image must have the same type, size, and number of channels as the input image.
@note Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.

@note Function textual ID is "org.opencv.imgproc.filters.dilate"

@param src input image.
@param iterations number of times dilation is applied.
@param borderType pixel extrapolation method, see cv::BorderTypes
@param borderValue border value in case of a constant border
@sa  dilate, erode3x3
 */

GAPI_EXPORTS GMat dilate3x3(const GMat& src, int iterations = 1,
                            int borderType = BORDER_CONSTANT,
                            const  Scalar& borderValue = morphologyDefaultBorderValue());

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

@note Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.

@note Function textual ID is "org.opencv.imgproc.filters.sobel"

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
GAPI_EXPORTS GMat Sobel(const GMat& src, int ddepth, int dx, int dy, int ksize = 3,
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

@note First returned matrix correspons to dx derivative while the second one to dy.

@note Rounding to nearest even is procedeed if hardware supports it, if not - to nearest.

@note Function textual ID is "org.opencv.imgproc.filters.sobelxy"

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
GAPI_EXPORTS std::tuple<GMat, GMat> SobelXY(const GMat& src, int ddepth, int order, int ksize = 3,
                        double scale = 1, double delta = 0,
                        int borderType = BORDER_DEFAULT,
                        const Scalar& borderValue = Scalar(0));

/** @brief Finds edges in an image using the Canny algorithm.

The function finds edges in the input image and marks them in the output map edges using the
Canny algorithm. The smallest value between threshold1 and threshold2 is used for edge linking. The
largest value is used to find initial segments of strong edges. See
<http://en.wikipedia.org/wiki/Canny_edge_detector>

@note Function textual ID is "org.opencv.imgproc.filters.canny"

@param image 8-bit input image.
@param threshold1 first threshold for the hysteresis procedure.
@param threshold2 second threshold for the hysteresis procedure.
@param apertureSize aperture size for the Sobel operator.
@param L2gradient a flag, indicating whether a more accurate \f$L_2\f$ norm
\f$=\sqrt{(dI/dx)^2 + (dI/dy)^2}\f$ should be used to calculate the image gradient magnitude (
L2gradient=true ), or whether the default \f$L_1\f$ norm \f$=|dI/dx|+|dI/dy|\f$ is enough (
L2gradient=false ).
 */
GAPI_EXPORTS GMat Canny(const GMat& image, double threshold1, double threshold2,
                        int apertureSize = 3, bool L2gradient = false);

/** @brief Equalizes the histogram of a grayscale image.

The function equalizes the histogram of the input image using the following algorithm:

- Calculate the histogram \f$H\f$ for src .
- Normalize the histogram so that the sum of histogram bins is 255.
- Compute the integral of the histogram:
\f[H'_i =  \sum _{0  \le j < i} H(j)\f]
- Transform the image using \f$H'\f$ as a look-up table: \f$\texttt{dst}(x,y) = H'(\texttt{src}(x,y))\f$

The algorithm normalizes the brightness and increases the contrast of the image.
@note The returned image is of the same size and type as input.

@note Function textual ID is "org.opencv.imgproc.equalizeHist"

@param src Source 8-bit single channel image.
 */
GAPI_EXPORTS GMat equalizeHist(const GMat& src);

//! @} gapi_filters

//! @addtogroup gapi_colorconvert
//! @{
/** @brief Converts an image from RGB color space to gray-scaled.
The conventional ranges for R, G, and B channel values are 0 to 255.
Resulting gray color value computed as
\f[\texttt{dst} (I)= \texttt{0.299} * \texttt{src}(I).R + \texttt{0.587} * \texttt{src}(I).G  + \texttt{0.114} * \texttt{src}(I).B \f]

@note Function textual ID is "org.opencv.imgproc.colorconvert.rgb2gray"

@param src input image: 8-bit unsigned 3-channel image @ref CV_8UC1.
@sa RGB2YUV
 */
GAPI_EXPORTS GMat RGB2Gray(const GMat& src);

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
GAPI_EXPORTS GMat RGB2Gray(const GMat& src, float rY, float gY, float bY);

/** @brief Converts an image from BGR color space to gray-scaled.
The conventional ranges for B, G, and R channel values are 0 to 255.
Resulting gray color value computed as
\f[\texttt{dst} (I)= \texttt{0.114} * \texttt{src}(I).B + \texttt{0.587} * \texttt{src}(I).G  + \texttt{0.299} * \texttt{src}(I).R \f]

@note Function textual ID is "org.opencv.imgproc.colorconvert.bgr2gray"

@param src input image: 8-bit unsigned 3-channel image @ref CV_8UC1.
@sa BGR2LUV
 */
GAPI_EXPORTS GMat BGR2Gray(const GMat& src);

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
GAPI_EXPORTS GMat RGB2YUV(const GMat& src);

/** @brief Converts an image from BGR color space to LUV color space.

The function converts an input image from BGR color space to LUV.
The conventional ranges for B, G, and R channel values are 0 to 255.

Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.

@note Function textual ID is "org.opencv.imgproc.colorconvert.bgr2luv"

@param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.
@sa RGB2Lab, RGB2LUV
*/
GAPI_EXPORTS GMat BGR2LUV(const GMat& src);

/** @brief Converts an image from LUV color space to BGR color space.

The function converts an input image from LUV color space to BGR.
The conventional ranges for B, G, and R channel values are 0 to 255.

Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.

@note Function textual ID is "org.opencv.imgproc.colorconvert.luv2bgr"

@param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.
@sa BGR2LUV
*/
GAPI_EXPORTS GMat LUV2BGR(const GMat& src);

/** @brief Converts an image from YUV color space to BGR color space.

The function converts an input image from YUV color space to BGR.
The conventional ranges for B, G, and R channel values are 0 to 255.

Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.

@note Function textual ID is "org.opencv.imgproc.colorconvert.yuv2bgr"

@param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.
@sa BGR2YUV
*/
GAPI_EXPORTS GMat YUV2BGR(const GMat& src);

/** @brief Converts an image from BGR color space to YUV color space.

The function converts an input image from BGR color space to YUV.
The conventional ranges for B, G, and R channel values are 0 to 255.

Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.

@note Function textual ID is "org.opencv.imgproc.colorconvert.bgr2yuv"

@param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.
@sa YUV2BGR
*/
GAPI_EXPORTS GMat BGR2YUV(const GMat& src);

/** @brief Converts an image from RGB color space to Lab color space.

The function converts an input image from BGR color space to Lab.
The conventional ranges for R, G, and B channel values are 0 to 255.

Output image must be 8-bit unsigned 3-channel image @ref CV_8UC1.

@note Function textual ID is "org.opencv.imgproc.colorconvert.rgb2lab"

@param src input image: 8-bit unsigned 3-channel image @ref CV_8UC1.
@sa RGB2YUV, RGB2LUV
*/
GAPI_EXPORTS GMat RGB2Lab(const GMat& src);

/** @brief Converts an image from YUV color space to RGB.
The function converts an input image from YUV color space to RGB.
The conventional ranges for Y, U, and V channel values are 0 to 255.

Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.

@note Function textual ID is "org.opencv.imgproc.colorconvert.yuv2rgb"

@param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.

@sa RGB2Lab, RGB2YUV
*/
GAPI_EXPORTS GMat YUV2RGB(const GMat& src);

/** @brief Converts an image from NV12 (YUV420p) color space to RGB.
The function converts an input image from NV12 color space to RGB.
The conventional ranges for Y, U, and V channel values are 0 to 255.

Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.

@note Function textual ID is "org.opencv.imgproc.colorconvert.nv12torgb"

@param src_y input image: 8-bit unsigned 1-channel image @ref CV_8UC1.
@param src_uv input image: 8-bit unsigned 2-channel image @ref CV_8UC2.

@sa YUV2RGB, NV12toBGR
*/
GAPI_EXPORTS GMat NV12toRGB(const GMat& src_y, const GMat& src_uv);

/** @brief Converts an image from NV12 (YUV420p) color space to BGR.
The function converts an input image from NV12 color space to RGB.
The conventional ranges for Y, U, and V channel values are 0 to 255.

Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.

@note Function textual ID is "org.opencv.imgproc.colorconvert.nv12tobgr"

@param src_y input image: 8-bit unsigned 1-channel image @ref CV_8UC1.
@param src_uv input image: 8-bit unsigned 2-channel image @ref CV_8UC2.

@sa YUV2BGR, NV12toRGB
*/
GAPI_EXPORTS GMat NV12toBGR(const GMat& src_y, const GMat& src_uv);

/** @brief Converts an image from BayerGR color space to RGB.
The function converts an input image from BayerGR color space to RGB.
The conventional ranges for G, R, and B channel values are 0 to 255.

Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.

@note Function textual ID is "org.opencv.imgproc.colorconvert.bayergr2rgb"

@param src_gr input image: 8-bit unsigned 1-channel image @ref CV_8UC1.

@sa YUV2BGR, NV12toRGB
*/
GAPI_EXPORTS GMat BayerGR2RGB(const GMat& src_gr);

/** @brief Converts an image from RGB color space to HSV.
The function converts an input image from RGB color space to HSV.
The conventional ranges for R, G, and B channel values are 0 to 255.

Output image must be 8-bit unsigned 3-channel image @ref CV_8UC3.

@note Function textual ID is "org.opencv.imgproc.colorconvert.rgb2hsv"

@param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.

@sa YUV2BGR, NV12toRGB
*/
GAPI_EXPORTS GMat RGB2HSV(const GMat& src);

/** @brief Converts an image from RGB color space to YUV422.
The function converts an input image from RGB color space to YUV422.
The conventional ranges for R, G, and B channel values are 0 to 255.

Output image must be 8-bit unsigned 2-channel image @ref CV_8UC2.

@note Function textual ID is "org.opencv.imgproc.colorconvert.rgb2yuv422"

@param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3.

@sa YUV2BGR, NV12toRGB
*/
GAPI_EXPORTS GMat RGB2YUV422(const GMat& src);

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
} //namespace gapi
} //namespace cv

#endif // OPENCV_GAPI_IMGPROC_HPP
