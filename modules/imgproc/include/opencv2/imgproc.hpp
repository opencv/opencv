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

#ifndef __OPENCV_IMGPROC_HPP__
#define __OPENCV_IMGPROC_HPP__

#include "opencv2/core.hpp"

/*! \namespace cv
 Namespace where all the C++ OpenCV functionality resides
 */
namespace cv
{

//! type of the kernel
enum { KERNEL_GENERAL      = 0, // the kernel is generic. No any type of symmetry or other properties.
       KERNEL_SYMMETRICAL  = 1, // kernel[i] == kernel[ksize-i-1] , and the anchor is at the center
       KERNEL_ASYMMETRICAL = 2, // kernel[i] == -kernel[ksize-i-1] , and the anchor is at the center
       KERNEL_SMOOTH       = 4, // all the kernel elements are non-negative and summed to 1
       KERNEL_INTEGER      = 8  // all the kernel coefficients are integer numbers
     };

//! type of morphological operation
enum { MORPH_ERODE    = 0,
       MORPH_DILATE   = 1,
       MORPH_OPEN     = 2,
       MORPH_CLOSE    = 3,
       MORPH_GRADIENT = 4,
       MORPH_TOPHAT   = 5,
       MORPH_BLACKHAT = 6
     };

//! shape of the structuring element
enum { MORPH_RECT    = 0,
       MORPH_CROSS   = 1,
       MORPH_ELLIPSE = 2
     };

//! interpolation algorithm
enum { INTER_NEAREST        = 0, //!< nearest neighbor interpolation
       INTER_LINEAR         = 1, //!< bilinear interpolation
       INTER_CUBIC          = 2, //!< bicubic interpolation
       INTER_AREA           = 3, //!< area-based (or super) interpolation
       INTER_LANCZOS4       = 4, //!< Lanczos interpolation over 8x8 neighborhood

       INTER_MAX            = 7, //!< mask for interpolation codes
       WARP_FILL_OUTLIERS   = 8,
       WARP_INVERSE_MAP     = 16
     };

enum { INTER_BITS      = 5,
       INTER_BITS2     = INTER_BITS * 2,
       INTER_TAB_SIZE  = 1 << INTER_BITS,
       INTER_TAB_SIZE2 = INTER_TAB_SIZE * INTER_TAB_SIZE
     };

//! Distance types for Distance Transform and M-estimators
enum { DIST_USER    = -1,  // User defined distance
       DIST_L1      = 1,   // distance = |x1-x2| + |y1-y2|
       DIST_L2      = 2,   // the simple euclidean distance
       DIST_C       = 3,   // distance = max(|x1-x2|,|y1-y2|)
       DIST_L12     = 4,   // L1-L2 metric: distance = 2(sqrt(1+x*x/2) - 1))
       DIST_FAIR    = 5,   // distance = c^2(|x|/c-log(1+|x|/c)), c = 1.3998
       DIST_WELSCH  = 6,   // distance = c^2/2(1-exp(-(x/c)^2)), c = 2.9846
       DIST_HUBER   = 7    // distance = |x|<c ? x^2/2 : c(|x|-c/2), c=1.345
};

//! Mask size for distance transform
enum { DIST_MASK_3       = 3,
       DIST_MASK_5       = 5,
       DIST_MASK_PRECISE = 0
     };

//! type of the threshold operation
enum { THRESH_BINARY     = 0, // value = value > threshold ? max_value : 0
       THRESH_BINARY_INV = 1, // value = value > threshold ? 0 : max_value
       THRESH_TRUNC      = 2, // value = value > threshold ? threshold : value
       THRESH_TOZERO     = 3, // value = value > threshold ? value : 0
       THRESH_TOZERO_INV = 4, // value = value > threshold ? 0 : value
       THRESH_MASK       = 7,
       THRESH_OTSU       = 8  // use Otsu algorithm to choose the optimal threshold value
     };

//! adaptive threshold algorithm
enum { ADAPTIVE_THRESH_MEAN_C     = 0,
       ADAPTIVE_THRESH_GAUSSIAN_C = 1
     };

enum { PROJ_SPHERICAL_ORTHO  = 0,
       PROJ_SPHERICAL_EQRECT = 1
     };

//! class of the pixel in GrabCut algorithm
enum { GC_BGD    = 0,  //!< background
       GC_FGD    = 1,  //!< foreground
       GC_PR_BGD = 2,  //!< most probably background
       GC_PR_FGD = 3   //!< most probably foreground
     };

//! GrabCut algorithm flags
enum { GC_INIT_WITH_RECT  = 0,
       GC_INIT_WITH_MASK  = 1,
       GC_EVAL            = 2
};

//! distanceTransform algorithm flags
enum { DIST_LABEL_CCOMP = 0,
       DIST_LABEL_PIXEL = 1
     };

//! floodfill algorithm flags
enum { FLOODFILL_FIXED_RANGE = 1 << 16,
       FLOODFILL_MASK_ONLY   = 1 << 17
     };

//! type of the template matching operation
enum { TM_SQDIFF        = 0,
       TM_SQDIFF_NORMED = 1,
       TM_CCORR         = 2,
       TM_CCORR_NORMED  = 3,
       TM_CCOEFF        = 4,
       TM_CCOEFF_NORMED = 5
     };

//! connected components algorithm output formats
enum { CC_STAT_LEFT   = 0,
       CC_STAT_TOP    = 1,
       CC_STAT_WIDTH  = 2,
       CC_STAT_HEIGHT = 3,
       CC_STAT_AREA   = 4,
       CC_STAT_MAX    = 5
     };

//! mode of the contour retrieval algorithm
enum { RETR_EXTERNAL  = 0, //!< retrieve only the most external (top-level) contours
       RETR_LIST      = 1, //!< retrieve all the contours without any hierarchical information
       RETR_CCOMP     = 2, //!< retrieve the connected components (that can possibly be nested)
       RETR_TREE      = 3, //!< retrieve all the contours and the whole hierarchy
       RETR_FLOODFILL = 4
     };

//! the contour approximation algorithm
enum { CHAIN_APPROX_NONE      = 1,
       CHAIN_APPROX_SIMPLE    = 2,
       CHAIN_APPROX_TC89_L1   = 3,
       CHAIN_APPROX_TC89_KCOS = 4
     };

//! Variants of a Hough transform
enum { HOUGH_STANDARD      = 0,
       HOUGH_PROBABILISTIC = 1,
       HOUGH_MULTI_SCALE   = 2,
       HOUGH_GRADIENT      = 3
     };

//! Variants of Line Segment Detector
enum { LSD_REFINE_NONE = 0,
       LSD_REFINE_STD  = 1,
       LSD_REFINE_ADV  = 2
     };

//! Histogram comparison methods
enum { HISTCMP_CORREL        = 0,
       HISTCMP_CHISQR        = 1,
       HISTCMP_INTERSECT     = 2,
       HISTCMP_BHATTACHARYYA = 3,
       HISTCMP_HELLINGER     = HISTCMP_BHATTACHARYYA,
       HISTCMP_CHISQR_ALT    = 4
     };

//! the color conversion code
enum { COLOR_BGR2BGRA     = 0,
       COLOR_RGB2RGBA     = COLOR_BGR2BGRA,

       COLOR_BGRA2BGR     = 1,
       COLOR_RGBA2RGB     = COLOR_BGRA2BGR,

       COLOR_BGR2RGBA     = 2,
       COLOR_RGB2BGRA     = COLOR_BGR2RGBA,

       COLOR_RGBA2BGR     = 3,
       COLOR_BGRA2RGB     = COLOR_RGBA2BGR,

       COLOR_BGR2RGB      = 4,
       COLOR_RGB2BGR      = COLOR_BGR2RGB,

       COLOR_BGRA2RGBA    = 5,
       COLOR_RGBA2BGRA    = COLOR_BGRA2RGBA,

       COLOR_BGR2GRAY     = 6,
       COLOR_RGB2GRAY     = 7,
       COLOR_GRAY2BGR     = 8,
       COLOR_GRAY2RGB     = COLOR_GRAY2BGR,
       COLOR_GRAY2BGRA    = 9,
       COLOR_GRAY2RGBA    = COLOR_GRAY2BGRA,
       COLOR_BGRA2GRAY    = 10,
       COLOR_RGBA2GRAY    = 11,

       COLOR_BGR2BGR565   = 12,
       COLOR_RGB2BGR565   = 13,
       COLOR_BGR5652BGR   = 14,
       COLOR_BGR5652RGB   = 15,
       COLOR_BGRA2BGR565  = 16,
       COLOR_RGBA2BGR565  = 17,
       COLOR_BGR5652BGRA  = 18,
       COLOR_BGR5652RGBA  = 19,

       COLOR_GRAY2BGR565  = 20,
       COLOR_BGR5652GRAY  = 21,

       COLOR_BGR2BGR555   = 22,
       COLOR_RGB2BGR555   = 23,
       COLOR_BGR5552BGR   = 24,
       COLOR_BGR5552RGB   = 25,
       COLOR_BGRA2BGR555  = 26,
       COLOR_RGBA2BGR555  = 27,
       COLOR_BGR5552BGRA  = 28,
       COLOR_BGR5552RGBA  = 29,

       COLOR_GRAY2BGR555  = 30,
       COLOR_BGR5552GRAY  = 31,

       COLOR_BGR2XYZ      = 32,
       COLOR_RGB2XYZ      = 33,
       COLOR_XYZ2BGR      = 34,
       COLOR_XYZ2RGB      = 35,

       COLOR_BGR2YCrCb    = 36,
       COLOR_RGB2YCrCb    = 37,
       COLOR_YCrCb2BGR    = 38,
       COLOR_YCrCb2RGB    = 39,

       COLOR_BGR2HSV      = 40,
       COLOR_RGB2HSV      = 41,

       COLOR_BGR2Lab      = 44,
       COLOR_RGB2Lab      = 45,

       COLOR_BGR2Luv      = 50,
       COLOR_RGB2Luv      = 51,
       COLOR_BGR2HLS      = 52,
       COLOR_RGB2HLS      = 53,

       COLOR_HSV2BGR      = 54,
       COLOR_HSV2RGB      = 55,

       COLOR_Lab2BGR      = 56,
       COLOR_Lab2RGB      = 57,
       COLOR_Luv2BGR      = 58,
       COLOR_Luv2RGB      = 59,
       COLOR_HLS2BGR      = 60,
       COLOR_HLS2RGB      = 61,

       COLOR_BGR2HSV_FULL = 66,
       COLOR_RGB2HSV_FULL = 67,
       COLOR_BGR2HLS_FULL = 68,
       COLOR_RGB2HLS_FULL = 69,

       COLOR_HSV2BGR_FULL = 70,
       COLOR_HSV2RGB_FULL = 71,
       COLOR_HLS2BGR_FULL = 72,
       COLOR_HLS2RGB_FULL = 73,

       COLOR_LBGR2Lab     = 74,
       COLOR_LRGB2Lab     = 75,
       COLOR_LBGR2Luv     = 76,
       COLOR_LRGB2Luv     = 77,

       COLOR_Lab2LBGR     = 78,
       COLOR_Lab2LRGB     = 79,
       COLOR_Luv2LBGR     = 80,
       COLOR_Luv2LRGB     = 81,

       COLOR_BGR2YUV      = 82,
       COLOR_RGB2YUV      = 83,
       COLOR_YUV2BGR      = 84,
       COLOR_YUV2RGB      = 85,

       // YUV 4:2:0 family to RGB
       COLOR_YUV2RGB_NV12  = 90,
       COLOR_YUV2BGR_NV12  = 91,
       COLOR_YUV2RGB_NV21  = 92,
       COLOR_YUV2BGR_NV21  = 93,
       COLOR_YUV420sp2RGB  = COLOR_YUV2RGB_NV21,
       COLOR_YUV420sp2BGR  = COLOR_YUV2BGR_NV21,

       COLOR_YUV2RGBA_NV12 = 94,
       COLOR_YUV2BGRA_NV12 = 95,
       COLOR_YUV2RGBA_NV21 = 96,
       COLOR_YUV2BGRA_NV21 = 97,
       COLOR_YUV420sp2RGBA = COLOR_YUV2RGBA_NV21,
       COLOR_YUV420sp2BGRA = COLOR_YUV2BGRA_NV21,

       COLOR_YUV2RGB_YV12  = 98,
       COLOR_YUV2BGR_YV12  = 99,
       COLOR_YUV2RGB_IYUV  = 100,
       COLOR_YUV2BGR_IYUV  = 101,
       COLOR_YUV2RGB_I420  = COLOR_YUV2RGB_IYUV,
       COLOR_YUV2BGR_I420  = COLOR_YUV2BGR_IYUV,
       COLOR_YUV420p2RGB   = COLOR_YUV2RGB_YV12,
       COLOR_YUV420p2BGR   = COLOR_YUV2BGR_YV12,

       COLOR_YUV2RGBA_YV12 = 102,
       COLOR_YUV2BGRA_YV12 = 103,
       COLOR_YUV2RGBA_IYUV = 104,
       COLOR_YUV2BGRA_IYUV = 105,
       COLOR_YUV2RGBA_I420 = COLOR_YUV2RGBA_IYUV,
       COLOR_YUV2BGRA_I420 = COLOR_YUV2BGRA_IYUV,
       COLOR_YUV420p2RGBA  = COLOR_YUV2RGBA_YV12,
       COLOR_YUV420p2BGRA  = COLOR_YUV2BGRA_YV12,

       COLOR_YUV2GRAY_420  = 106,
       COLOR_YUV2GRAY_NV21 = COLOR_YUV2GRAY_420,
       COLOR_YUV2GRAY_NV12 = COLOR_YUV2GRAY_420,
       COLOR_YUV2GRAY_YV12 = COLOR_YUV2GRAY_420,
       COLOR_YUV2GRAY_IYUV = COLOR_YUV2GRAY_420,
       COLOR_YUV2GRAY_I420 = COLOR_YUV2GRAY_420,
       COLOR_YUV420sp2GRAY = COLOR_YUV2GRAY_420,
       COLOR_YUV420p2GRAY  = COLOR_YUV2GRAY_420,

       // YUV 4:2:2 family to RGB
       COLOR_YUV2RGB_UYVY = 107,
       COLOR_YUV2BGR_UYVY = 108,
     //COLOR_YUV2RGB_VYUY = 109,
     //COLOR_YUV2BGR_VYUY = 110,
       COLOR_YUV2RGB_Y422 = COLOR_YUV2RGB_UYVY,
       COLOR_YUV2BGR_Y422 = COLOR_YUV2BGR_UYVY,
       COLOR_YUV2RGB_UYNV = COLOR_YUV2RGB_UYVY,
       COLOR_YUV2BGR_UYNV = COLOR_YUV2BGR_UYVY,

       COLOR_YUV2RGBA_UYVY = 111,
       COLOR_YUV2BGRA_UYVY = 112,
     //COLOR_YUV2RGBA_VYUY = 113,
     //COLOR_YUV2BGRA_VYUY = 114,
       COLOR_YUV2RGBA_Y422 = COLOR_YUV2RGBA_UYVY,
       COLOR_YUV2BGRA_Y422 = COLOR_YUV2BGRA_UYVY,
       COLOR_YUV2RGBA_UYNV = COLOR_YUV2RGBA_UYVY,
       COLOR_YUV2BGRA_UYNV = COLOR_YUV2BGRA_UYVY,

       COLOR_YUV2RGB_YUY2 = 115,
       COLOR_YUV2BGR_YUY2 = 116,
       COLOR_YUV2RGB_YVYU = 117,
       COLOR_YUV2BGR_YVYU = 118,
       COLOR_YUV2RGB_YUYV = COLOR_YUV2RGB_YUY2,
       COLOR_YUV2BGR_YUYV = COLOR_YUV2BGR_YUY2,
       COLOR_YUV2RGB_YUNV = COLOR_YUV2RGB_YUY2,
       COLOR_YUV2BGR_YUNV = COLOR_YUV2BGR_YUY2,

       COLOR_YUV2RGBA_YUY2 = 119,
       COLOR_YUV2BGRA_YUY2 = 120,
       COLOR_YUV2RGBA_YVYU = 121,
       COLOR_YUV2BGRA_YVYU = 122,
       COLOR_YUV2RGBA_YUYV = COLOR_YUV2RGBA_YUY2,
       COLOR_YUV2BGRA_YUYV = COLOR_YUV2BGRA_YUY2,
       COLOR_YUV2RGBA_YUNV = COLOR_YUV2RGBA_YUY2,
       COLOR_YUV2BGRA_YUNV = COLOR_YUV2BGRA_YUY2,

       COLOR_YUV2GRAY_UYVY = 123,
       COLOR_YUV2GRAY_YUY2 = 124,
     //CV_YUV2GRAY_VYUY    = CV_YUV2GRAY_UYVY,
       COLOR_YUV2GRAY_Y422 = COLOR_YUV2GRAY_UYVY,
       COLOR_YUV2GRAY_UYNV = COLOR_YUV2GRAY_UYVY,
       COLOR_YUV2GRAY_YVYU = COLOR_YUV2GRAY_YUY2,
       COLOR_YUV2GRAY_YUYV = COLOR_YUV2GRAY_YUY2,
       COLOR_YUV2GRAY_YUNV = COLOR_YUV2GRAY_YUY2,

       // alpha premultiplication
       COLOR_RGBA2mRGBA    = 125,
       COLOR_mRGBA2RGBA    = 126,

       // RGB to YUV 4:2:0 family
       COLOR_RGB2YUV_I420  = 127,
       COLOR_BGR2YUV_I420  = 128,
       COLOR_RGB2YUV_IYUV  = COLOR_RGB2YUV_I420,
       COLOR_BGR2YUV_IYUV  = COLOR_BGR2YUV_I420,

       COLOR_RGBA2YUV_I420 = 129,
       COLOR_BGRA2YUV_I420 = 130,
       COLOR_RGBA2YUV_IYUV = COLOR_RGBA2YUV_I420,
       COLOR_BGRA2YUV_IYUV = COLOR_BGRA2YUV_I420,
       COLOR_RGB2YUV_YV12  = 131,
       COLOR_BGR2YUV_YV12  = 132,
       COLOR_RGBA2YUV_YV12 = 133,
       COLOR_BGRA2YUV_YV12 = 134,

       // Demosaicing
       COLOR_BayerBG2BGR = 46,
       COLOR_BayerGB2BGR = 47,
       COLOR_BayerRG2BGR = 48,
       COLOR_BayerGR2BGR = 49,

       COLOR_BayerBG2RGB = COLOR_BayerRG2BGR,
       COLOR_BayerGB2RGB = COLOR_BayerGR2BGR,
       COLOR_BayerRG2RGB = COLOR_BayerBG2BGR,
       COLOR_BayerGR2RGB = COLOR_BayerGB2BGR,

       COLOR_BayerBG2GRAY = 86,
       COLOR_BayerGB2GRAY = 87,
       COLOR_BayerRG2GRAY = 88,
       COLOR_BayerGR2GRAY = 89,

       // Demosaicing using Variable Number of Gradients
       COLOR_BayerBG2BGR_VNG = 62,
       COLOR_BayerGB2BGR_VNG = 63,
       COLOR_BayerRG2BGR_VNG = 64,
       COLOR_BayerGR2BGR_VNG = 65,

       COLOR_BayerBG2RGB_VNG = COLOR_BayerRG2BGR_VNG,
       COLOR_BayerGB2RGB_VNG = COLOR_BayerGR2BGR_VNG,
       COLOR_BayerRG2RGB_VNG = COLOR_BayerBG2BGR_VNG,
       COLOR_BayerGR2RGB_VNG = COLOR_BayerGB2BGR_VNG,

       // Edge-Aware Demosaicing
       COLOR_BayerBG2BGR_EA  = 135,
       COLOR_BayerGB2BGR_EA  = 136,
       COLOR_BayerRG2BGR_EA  = 137,
       COLOR_BayerGR2BGR_EA  = 138,

       COLOR_BayerBG2RGB_EA  = COLOR_BayerRG2BGR_EA,
       COLOR_BayerGB2RGB_EA  = COLOR_BayerGR2BGR_EA,
       COLOR_BayerRG2RGB_EA  = COLOR_BayerBG2BGR_EA,
       COLOR_BayerGR2RGB_EA  = COLOR_BayerGB2BGR_EA,


       COLOR_COLORCVT_MAX  = 139
};

//! types of intersection between rectangles
enum { INTERSECT_NONE = 0,
       INTERSECT_PARTIAL  = 1,
       INTERSECT_FULL  = 2
     };

/*!
 The Base Class for 1D or Row-wise Filters

 This is the base class for linear or non-linear filters that process 1D data.
 In particular, such filters are used for the "horizontal" filtering parts in separable filters.

 Several functions in OpenCV return Ptr<BaseRowFilter> for the specific types of filters,
 and those pointers can be used directly or within cv::FilterEngine.
*/
class CV_EXPORTS BaseRowFilter
{
public:
    //! the default constructor
    BaseRowFilter();
    //! the destructor
    virtual ~BaseRowFilter();
    //! the filtering operator. Must be overrided in the derived classes. The horizontal border interpolation is done outside of the class.
    virtual void operator()(const uchar* src, uchar* dst, int width, int cn) = 0;

    int ksize;
    int anchor;
};


/*!
 The Base Class for Column-wise Filters

 This is the base class for linear or non-linear filters that process columns of 2D arrays.
 Such filters are used for the "vertical" filtering parts in separable filters.

 Several functions in OpenCV return Ptr<BaseColumnFilter> for the specific types of filters,
 and those pointers can be used directly or within cv::FilterEngine.

 Unlike cv::BaseRowFilter, cv::BaseColumnFilter may have some context information,
 i.e. box filter keeps the sliding sum of elements. To reset the state BaseColumnFilter::reset()
 must be called (e.g. the method is called by cv::FilterEngine)
 */
class CV_EXPORTS BaseColumnFilter
{
public:
    //! the default constructor
    BaseColumnFilter();
    //! the destructor
    virtual ~BaseColumnFilter();
    //! the filtering operator. Must be overrided in the derived classes. The vertical border interpolation is done outside of the class.
    virtual void operator()(const uchar** src, uchar* dst, int dststep, int dstcount, int width) = 0;
    //! resets the internal buffers, if any
    virtual void reset();

    int ksize;
    int anchor;
};


/*!
 The Base Class for Non-Separable 2D Filters.

 This is the base class for linear or non-linear 2D filters.

 Several functions in OpenCV return Ptr<BaseFilter> for the specific types of filters,
 and those pointers can be used directly or within cv::FilterEngine.

 Similar to cv::BaseColumnFilter, the class may have some context information,
 that should be reset using BaseFilter::reset() method before processing the new array.
*/
class CV_EXPORTS BaseFilter
{
public:
    //! the default constructor
    BaseFilter();
    //! the destructor
    virtual ~BaseFilter();
    //! the filtering operator. The horizontal and the vertical border interpolation is done outside of the class.
    virtual void operator()(const uchar** src, uchar* dst, int dststep, int dstcount, int width, int cn) = 0;
    //! resets the internal buffers, if any
    virtual void reset();

    Size ksize;
    Point anchor;
};


/*!
 The Main Class for Image Filtering.

 The class can be used to apply an arbitrary filtering operation to an image.
 It contains all the necessary intermediate buffers, it computes extrapolated values
 of the "virtual" pixels outside of the image etc.
 Pointers to the initialized cv::FilterEngine instances
 are returned by various OpenCV functions, such as cv::createSeparableLinearFilter(),
 cv::createLinearFilter(), cv::createGaussianFilter(), cv::createDerivFilter(),
 cv::createBoxFilter() and cv::createMorphologyFilter().

 Using the class you can process large images by parts and build complex pipelines
 that include filtering as some of the stages. If all you need is to apply some pre-defined
 filtering operation, you may use cv::filter2D(), cv::erode(), cv::dilate() etc.
 functions that create FilterEngine internally.

 Here is the example on how to use the class to implement Laplacian operator, which is the sum of
 second-order derivatives. More complex variant for different types is implemented in cv::Laplacian().

 \code
 void laplace_f(const Mat& src, Mat& dst)
 {
     CV_Assert( src.type() == CV_32F );
     // make sure the destination array has the proper size and type
     dst.create(src.size(), src.type());

     // get the derivative and smooth kernels for d2I/dx2.
     // for d2I/dy2 we could use the same kernels, just swapped
     Mat kd, ks;
     getSobelKernels( kd, ks, 2, 0, ksize, false, ktype );

     // let's process 10 source rows at once
     int DELTA = std::min(10, src.rows);
     Ptr<FilterEngine> Fxx = createSeparableLinearFilter(src.type(),
     dst.type(), kd, ks, Point(-1,-1), 0, borderType, borderType, Scalar() );
     Ptr<FilterEngine> Fyy = createSeparableLinearFilter(src.type(),
     dst.type(), ks, kd, Point(-1,-1), 0, borderType, borderType, Scalar() );

     int y = Fxx->start(src), dsty = 0, dy = 0;
     Fyy->start(src);
     const uchar* sptr = src.data + y*src.step;

     // allocate the buffers for the spatial image derivatives;
     // the buffers need to have more than DELTA rows, because at the
     // last iteration the output may take max(kd.rows-1,ks.rows-1)
     // rows more than the input.
     Mat Ixx( DELTA + kd.rows - 1, src.cols, dst.type() );
     Mat Iyy( DELTA + kd.rows - 1, src.cols, dst.type() );

     // inside the loop we always pass DELTA rows to the filter
     // (note that the "proceed" method takes care of possibe overflow, since
     // it was given the actual image height in the "start" method)
     // on output we can get:
     //  * < DELTA rows (the initial buffer accumulation stage)
     //  * = DELTA rows (settled state in the middle)
     //  * > DELTA rows (then the input image is over, but we generate
     //                  "virtual" rows using the border mode and filter them)
     // this variable number of output rows is dy.
     // dsty is the current output row.
     // sptr is the pointer to the first input row in the portion to process
     for( ; dsty < dst.rows; sptr += DELTA*src.step, dsty += dy )
     {
         Fxx->proceed( sptr, (int)src.step, DELTA, Ixx.data, (int)Ixx.step );
         dy = Fyy->proceed( sptr, (int)src.step, DELTA, d2y.data, (int)Iyy.step );
         if( dy > 0 )
         {
             Mat dstripe = dst.rowRange(dsty, dsty + dy);
             add(Ixx.rowRange(0, dy), Iyy.rowRange(0, dy), dstripe);
         }
     }
 }
 \endcode
*/
class CV_EXPORTS FilterEngine
{
public:
    //! the default constructor
    FilterEngine();
    //! the full constructor. Either _filter2D or both _rowFilter and _columnFilter must be non-empty.
    FilterEngine(const Ptr<BaseFilter>& _filter2D,
                 const Ptr<BaseRowFilter>& _rowFilter,
                 const Ptr<BaseColumnFilter>& _columnFilter,
                 int srcType, int dstType, int bufType,
                 int _rowBorderType = BORDER_REPLICATE,
                 int _columnBorderType = -1,
                 const Scalar& _borderValue = Scalar());
    //! the destructor
    virtual ~FilterEngine();
    //! reinitializes the engine. The previously assigned filters are released.
    void init(const Ptr<BaseFilter>& _filter2D,
              const Ptr<BaseRowFilter>& _rowFilter,
              const Ptr<BaseColumnFilter>& _columnFilter,
              int srcType, int dstType, int bufType,
              int _rowBorderType = BORDER_REPLICATE,
              int _columnBorderType = -1,
              const Scalar& _borderValue = Scalar());
    //! starts filtering of the specified ROI of an image of size wholeSize.
    virtual int start(Size wholeSize, Rect roi, int maxBufRows = -1);
    //! starts filtering of the specified ROI of the specified image.
    virtual int start(const Mat& src, const Rect& srcRoi = Rect(0,0,-1,-1),
                      bool isolated = false, int maxBufRows = -1);
    //! processes the next srcCount rows of the image.
    virtual int proceed(const uchar* src, int srcStep, int srcCount,
                        uchar* dst, int dstStep);
    //! applies filter to the specified ROI of the image. if srcRoi=(0,0,-1,-1), the whole image is filtered.
    virtual void apply( const Mat& src, Mat& dst,
                        const Rect& srcRoi = Rect(0,0,-1,-1),
                        Point dstOfs = Point(0,0),
                        bool isolated = false);
    //! returns true if the filter is separable
    bool isSeparable() const { return !filter2D; }
    //! returns the number
    int remainingInputRows() const;
    int remainingOutputRows() const;

    int srcType;
    int dstType;
    int bufType;
    Size ksize;
    Point anchor;
    int maxWidth;
    Size wholeSize;
    Rect roi;
    int dx1;
    int dx2;
    int rowBorderType;
    int columnBorderType;
    std::vector<int> borderTab;
    int borderElemSize;
    std::vector<uchar> ringBuf;
    std::vector<uchar> srcRow;
    std::vector<uchar> constBorderValue;
    std::vector<uchar> constBorderRow;
    int bufStep;
    int startY;
    int startY0;
    int endY;
    int rowCount;
    int dstY;
    std::vector<uchar*> rows;

    Ptr<BaseFilter> filter2D;
    Ptr<BaseRowFilter> rowFilter;
    Ptr<BaseColumnFilter> columnFilter;
};


//! finds arbitrary template in the grayscale image using Generalized Hough Transform
class CV_EXPORTS GeneralizedHough : public Algorithm
{
public:
    //! set template to search
    virtual void setTemplate(InputArray templ, Point templCenter = Point(-1, -1)) = 0;
    virtual void setTemplate(InputArray edges, InputArray dx, InputArray dy, Point templCenter = Point(-1, -1)) = 0;

    //! find template on image
    virtual void detect(InputArray image, OutputArray positions, OutputArray votes = noArray()) = 0;
    virtual void detect(InputArray edges, InputArray dx, InputArray dy, OutputArray positions, OutputArray votes = noArray()) = 0;

    //! Canny low threshold.
    virtual void setCannyLowThresh(int cannyLowThresh) = 0;
    virtual int getCannyLowThresh() const = 0;

    //! Canny high threshold.
    virtual void setCannyHighThresh(int cannyHighThresh) = 0;
    virtual int getCannyHighThresh() const = 0;

    //! Minimum distance between the centers of the detected objects.
    virtual void setMinDist(double minDist) = 0;
    virtual double getMinDist() const = 0;

    //! Inverse ratio of the accumulator resolution to the image resolution.
    virtual void setDp(double dp) = 0;
    virtual double getDp() const = 0;

    //! Maximal size of inner buffers.
    virtual void setMaxBufferSize(int maxBufferSize) = 0;
    virtual int getMaxBufferSize() const = 0;
};

//! Ballard, D.H. (1981). Generalizing the Hough transform to detect arbitrary shapes. Pattern Recognition 13 (2): 111-122.
//! Detects position only without traslation and rotation
class CV_EXPORTS GeneralizedHoughBallard : public GeneralizedHough
{
public:
    //! R-Table levels.
    virtual void setLevels(int levels) = 0;
    virtual int getLevels() const = 0;

    //! The accumulator threshold for the template centers at the detection stage. The smaller it is, the more false positions may be detected.
    virtual void setVotesThreshold(int votesThreshold) = 0;
    virtual int getVotesThreshold() const = 0;
};

//! Guil, N., González-Linares, J.M. and Zapata, E.L. (1999). Bidimensional shape detection using an invariant approach. Pattern Recognition 32 (6): 1025-1038.
//! Detects position, traslation and rotation
class CV_EXPORTS GeneralizedHoughGuil : public GeneralizedHough
{
public:
    //! Angle difference in degrees between two points in feature.
    virtual void setXi(double xi) = 0;
    virtual double getXi() const = 0;

    //! Feature table levels.
    virtual void setLevels(int levels) = 0;
    virtual int getLevels() const = 0;

    //! Maximal difference between angles that treated as equal.
    virtual void setAngleEpsilon(double angleEpsilon) = 0;
    virtual double getAngleEpsilon() const = 0;

    //! Minimal rotation angle to detect in degrees.
    virtual void setMinAngle(double minAngle) = 0;
    virtual double getMinAngle() const = 0;

    //! Maximal rotation angle to detect in degrees.
    virtual void setMaxAngle(double maxAngle) = 0;
    virtual double getMaxAngle() const = 0;

    //! Angle step in degrees.
    virtual void setAngleStep(double angleStep) = 0;
    virtual double getAngleStep() const = 0;

    //! Angle votes threshold.
    virtual void setAngleThresh(int angleThresh) = 0;
    virtual int getAngleThresh() const = 0;

    //! Minimal scale to detect.
    virtual void setMinScale(double minScale) = 0;
    virtual double getMinScale() const = 0;

    //! Maximal scale to detect.
    virtual void setMaxScale(double maxScale) = 0;
    virtual double getMaxScale() const = 0;

    //! Scale step.
    virtual void setScaleStep(double scaleStep) = 0;
    virtual double getScaleStep() const = 0;

    //! Scale votes threshold.
    virtual void setScaleThresh(int scaleThresh) = 0;
    virtual int getScaleThresh() const = 0;

    //! Position votes threshold.
    virtual void setPosThresh(int posThresh) = 0;
    virtual int getPosThresh() const = 0;
};


class CV_EXPORTS_W CLAHE : public Algorithm
{
public:
    CV_WRAP virtual void apply(InputArray src, OutputArray dst) = 0;

    CV_WRAP virtual void setClipLimit(double clipLimit) = 0;
    CV_WRAP virtual double getClipLimit() const = 0;

    CV_WRAP virtual void setTilesGridSize(Size tileGridSize) = 0;
    CV_WRAP virtual Size getTilesGridSize() const = 0;

    CV_WRAP virtual void collectGarbage() = 0;
};


class CV_EXPORTS_W Subdiv2D
{
public:
    enum { PTLOC_ERROR        = -2,
           PTLOC_OUTSIDE_RECT = -1,
           PTLOC_INSIDE       = 0,
           PTLOC_VERTEX       = 1,
           PTLOC_ON_EDGE      = 2
         };

    enum { NEXT_AROUND_ORG   = 0x00,
           NEXT_AROUND_DST   = 0x22,
           PREV_AROUND_ORG   = 0x11,
           PREV_AROUND_DST   = 0x33,
           NEXT_AROUND_LEFT  = 0x13,
           NEXT_AROUND_RIGHT = 0x31,
           PREV_AROUND_LEFT  = 0x20,
           PREV_AROUND_RIGHT = 0x02
         };

    CV_WRAP Subdiv2D();
    CV_WRAP Subdiv2D(Rect rect);
    CV_WRAP void initDelaunay(Rect rect);

    CV_WRAP int insert(Point2f pt);
    CV_WRAP void insert(const std::vector<Point2f>& ptvec);
    CV_WRAP int locate(Point2f pt, CV_OUT int& edge, CV_OUT int& vertex);

    CV_WRAP int findNearest(Point2f pt, CV_OUT Point2f* nearestPt = 0);
    CV_WRAP void getEdgeList(CV_OUT std::vector<Vec4f>& edgeList) const;
    CV_WRAP void getTriangleList(CV_OUT std::vector<Vec6f>& triangleList) const;
    CV_WRAP void getVoronoiFacetList(const std::vector<int>& idx, CV_OUT std::vector<std::vector<Point2f> >& facetList,
                                     CV_OUT std::vector<Point2f>& facetCenters);

    CV_WRAP Point2f getVertex(int vertex, CV_OUT int* firstEdge = 0) const;

    CV_WRAP int getEdge( int edge, int nextEdgeType ) const;
    CV_WRAP int nextEdge(int edge) const;
    CV_WRAP int rotateEdge(int edge, int rotate) const;
    CV_WRAP int symEdge(int edge) const;
    CV_WRAP int edgeOrg(int edge, CV_OUT Point2f* orgpt = 0) const;
    CV_WRAP int edgeDst(int edge, CV_OUT Point2f* dstpt = 0) const;

protected:
    int newEdge();
    void deleteEdge(int edge);
    int newPoint(Point2f pt, bool isvirtual, int firstEdge = 0);
    void deletePoint(int vtx);
    void setEdgePoints( int edge, int orgPt, int dstPt );
    void splice( int edgeA, int edgeB );
    int connectEdges( int edgeA, int edgeB );
    void swapEdges( int edge );
    int isRightOf(Point2f pt, int edge) const;
    void calcVoronoi();
    void clearVoronoi();
    void checkSubdiv() const;

    struct CV_EXPORTS Vertex
    {
        Vertex();
        Vertex(Point2f pt, bool _isvirtual, int _firstEdge=0);
        bool isvirtual() const;
        bool isfree() const;

        int firstEdge;
        int type;
        Point2f pt;
    };

    struct CV_EXPORTS QuadEdge
    {
        QuadEdge();
        QuadEdge(int edgeidx);
        bool isfree() const;

        int next[4];
        int pt[4];
    };

    std::vector<Vertex> vtx;
    std::vector<QuadEdge> qedges;
    int freeQEdge;
    int freePoint;
    bool validGeometry;

    int recentEdge;
    Point2f topLeft;
    Point2f bottomRight;
};

class CV_EXPORTS_W LineSegmentDetector : public Algorithm
{
public:
/**
 * Detect lines in the input image.
 *
 * @param _image    A grayscale(CV_8UC1) input image.
 *                  If only a roi needs to be selected, use
 *                  lsd_ptr->detect(image(roi), ..., lines);
 *                  lines += Scalar(roi.x, roi.y, roi.x, roi.y);
 * @param _lines    Return: A vector of Vec4i elements specifying the beginning and ending point of a line.
 *                          Where Vec4i is (x1, y1, x2, y2), point 1 is the start, point 2 - end.
 *                          Returned lines are strictly oriented depending on the gradient.
 * @param width     Return: Vector of widths of the regions, where the lines are found. E.g. Width of line.
 * @param prec      Return: Vector of precisions with which the lines are found.
 * @param nfa       Return: Vector containing number of false alarms in the line region, with precision of 10%.
 *                          The bigger the value, logarithmically better the detection.
 *                              * -1 corresponds to 10 mean false alarms
 *                              * 0 corresponds to 1 mean false alarm
 *                              * 1 corresponds to 0.1 mean false alarms
 *                          This vector will be calculated _only_ when the objects type is REFINE_ADV
 */
    CV_WRAP virtual void detect(InputArray _image, OutputArray _lines,
                        OutputArray width = noArray(), OutputArray prec = noArray(),
                        OutputArray nfa = noArray()) = 0;

/**
 * Draw lines on the given canvas.
 *
 * @param image     The image, where lines will be drawn.
 *                  Should have the size of the image, where the lines were found
 * @param lines     The lines that need to be drawn
 */
    CV_WRAP virtual void drawSegments(InputOutputArray _image, InputArray lines) = 0;

/**
 * Draw both vectors on the image canvas. Uses blue for lines 1 and red for lines 2.
 *
 * @param size      The size of the image, where lines were found.
 * @param lines1    The first lines that need to be drawn. Color - Blue.
 * @param lines2    The second lines that need to be drawn. Color - Red.
 * @param image     Optional image, where lines will be drawn.
 *                  Should have the size of the image, where the lines were found
 * @return          The number of mismatching pixels between lines1 and lines2.
 */
    CV_WRAP virtual int compareSegments(const Size& size, InputArray lines1, InputArray lines2, InputOutputArray _image = noArray()) = 0;

    virtual ~LineSegmentDetector() { }
};

//! Returns a pointer to a LineSegmentDetector class.
CV_EXPORTS_W Ptr<LineSegmentDetector> createLineSegmentDetector(
    int _refine = LSD_REFINE_STD, double _scale = 0.8,
    double _sigma_scale = 0.6, double _quant = 2.0, double _ang_th = 22.5,
    double _log_eps = 0, double _density_th = 0.7, int _n_bins = 1024);

//! returns type (one of KERNEL_*) of 1D or 2D kernel specified by its coefficients.
CV_EXPORTS int getKernelType(InputArray kernel, Point anchor);

//! returns the primitive row filter with the specified kernel
CV_EXPORTS Ptr<BaseRowFilter> getLinearRowFilter(int srcType, int bufType,
                                            InputArray kernel, int anchor,
                                            int symmetryType);

//! returns the primitive column filter with the specified kernel
CV_EXPORTS Ptr<BaseColumnFilter> getLinearColumnFilter(int bufType, int dstType,
                                            InputArray kernel, int anchor,
                                            int symmetryType, double delta = 0,
                                            int bits = 0);

//! returns 2D filter with the specified kernel
CV_EXPORTS Ptr<BaseFilter> getLinearFilter(int srcType, int dstType,
                                           InputArray kernel,
                                           Point anchor = Point(-1,-1),
                                           double delta = 0, int bits = 0);

//! returns the separable linear filter engine
CV_EXPORTS Ptr<FilterEngine> createSeparableLinearFilter(int srcType, int dstType,
                          InputArray rowKernel, InputArray columnKernel,
                          Point anchor = Point(-1,-1), double delta = 0,
                          int rowBorderType = BORDER_DEFAULT,
                          int columnBorderType = -1,
                          const Scalar& borderValue = Scalar());

//! returns the non-separable linear filter engine
CV_EXPORTS Ptr<FilterEngine> createLinearFilter(int srcType, int dstType,
                 InputArray kernel, Point _anchor = Point(-1,-1),
                 double delta = 0, int rowBorderType = BORDER_DEFAULT,
                 int columnBorderType = -1, const Scalar& borderValue = Scalar());

//! returns the Gaussian kernel with the specified parameters
CV_EXPORTS_W Mat getGaussianKernel( int ksize, double sigma, int ktype = CV_64F );

//! returns the Gaussian filter engine
CV_EXPORTS Ptr<FilterEngine> createGaussianFilter( int type, Size ksize,
                                    double sigma1, double sigma2 = 0,
                                    int borderType = BORDER_DEFAULT);

//! initializes kernels of the generalized Sobel operator
CV_EXPORTS_W void getDerivKernels( OutputArray kx, OutputArray ky,
                                   int dx, int dy, int ksize,
                                   bool normalize = false, int ktype = CV_32F );

//! returns filter engine for the generalized Sobel operator
CV_EXPORTS Ptr<FilterEngine> createDerivFilter( int srcType, int dstType,
                                        int dx, int dy, int ksize,
                                        int borderType = BORDER_DEFAULT );

//! returns horizontal 1D box filter
CV_EXPORTS Ptr<BaseRowFilter> getRowSumFilter(int srcType, int sumType,
                                              int ksize, int anchor = -1);

//! returns vertical 1D box filter
CV_EXPORTS Ptr<BaseColumnFilter> getColumnSumFilter( int sumType, int dstType,
                                                     int ksize, int anchor = -1,
                                                     double scale = 1);
//! returns box filter engine
CV_EXPORTS Ptr<FilterEngine> createBoxFilter( int srcType, int dstType, Size ksize,
                                              Point anchor = Point(-1,-1),
                                              bool normalize = true,
                                              int borderType = BORDER_DEFAULT);

//! returns the Gabor kernel with the specified parameters
CV_EXPORTS_W Mat getGaborKernel( Size ksize, double sigma, double theta, double lambd,
                                 double gamma, double psi = CV_PI*0.5, int ktype = CV_64F );

//! returns horizontal 1D morphological filter
CV_EXPORTS Ptr<BaseRowFilter> getMorphologyRowFilter(int op, int type, int ksize, int anchor = -1);

//! returns vertical 1D morphological filter
CV_EXPORTS Ptr<BaseColumnFilter> getMorphologyColumnFilter(int op, int type, int ksize, int anchor = -1);

//! returns 2D morphological filter
CV_EXPORTS Ptr<BaseFilter> getMorphologyFilter(int op, int type, InputArray kernel,
                                               Point anchor = Point(-1,-1));

//! returns "magic" border value for erosion and dilation. It is automatically transformed to Scalar::all(-DBL_MAX) for dilation.
static inline Scalar morphologyDefaultBorderValue() { return Scalar::all(DBL_MAX); }

//! returns morphological filter engine. Only MORPH_ERODE and MORPH_DILATE are supported.
CV_EXPORTS Ptr<FilterEngine> createMorphologyFilter(int op, int type, InputArray kernel,
                    Point anchor = Point(-1,-1), int rowBorderType = BORDER_CONSTANT,
                    int columnBorderType = -1, const Scalar& borderValue = morphologyDefaultBorderValue());

//! returns structuring element of the specified shape and size
CV_EXPORTS_W Mat getStructuringElement(int shape, Size ksize, Point anchor = Point(-1,-1));

//! smooths the image using median filter.
CV_EXPORTS_W void medianBlur( InputArray src, OutputArray dst, int ksize );

//! smooths the image using Gaussian filter.
CV_EXPORTS_W void GaussianBlur( InputArray src, OutputArray dst, Size ksize,
                                double sigmaX, double sigmaY = 0,
                                int borderType = BORDER_DEFAULT );

//! smooths the image using bilateral filter
CV_EXPORTS_W void bilateralFilter( InputArray src, OutputArray dst, int d,
                                   double sigmaColor, double sigmaSpace,
                                   int borderType = BORDER_DEFAULT );

//! smooths the image using the box filter. Each pixel is processed in O(1) time
CV_EXPORTS_W void boxFilter( InputArray src, OutputArray dst, int ddepth,
                             Size ksize, Point anchor = Point(-1,-1),
                             bool normalize = true,
                             int borderType = BORDER_DEFAULT );

CV_EXPORTS_W void sqrBoxFilter( InputArray _src, OutputArray _dst, int ddepth,
                                Size ksize, Point anchor = Point(-1, -1),
                                bool normalize = true,
                                int borderType = BORDER_DEFAULT );

//! a synonym for normalized box filter
CV_EXPORTS_W void blur( InputArray src, OutputArray dst,
                        Size ksize, Point anchor = Point(-1,-1),
                        int borderType = BORDER_DEFAULT );

//! applies non-separable 2D linear filter to the image
CV_EXPORTS_W void filter2D( InputArray src, OutputArray dst, int ddepth,
                            InputArray kernel, Point anchor = Point(-1,-1),
                            double delta = 0, int borderType = BORDER_DEFAULT );

//! applies separable 2D linear filter to the image
CV_EXPORTS_W void sepFilter2D( InputArray src, OutputArray dst, int ddepth,
                               InputArray kernelX, InputArray kernelY,
                               Point anchor = Point(-1,-1),
                               double delta = 0, int borderType = BORDER_DEFAULT );

//! applies generalized Sobel operator to the image
CV_EXPORTS_W void Sobel( InputArray src, OutputArray dst, int ddepth,
                         int dx, int dy, int ksize = 3,
                         double scale = 1, double delta = 0,
                         int borderType = BORDER_DEFAULT );

//! applies the vertical or horizontal Scharr operator to the image
CV_EXPORTS_W void Scharr( InputArray src, OutputArray dst, int ddepth,
                          int dx, int dy, double scale = 1, double delta = 0,
                          int borderType = BORDER_DEFAULT );

//! applies Laplacian operator to the image
CV_EXPORTS_W void Laplacian( InputArray src, OutputArray dst, int ddepth,
                             int ksize = 1, double scale = 1, double delta = 0,
                             int borderType = BORDER_DEFAULT );

//! applies Canny edge detector and produces the edge map.
CV_EXPORTS_W void Canny( InputArray image, OutputArray edges,
                         double threshold1, double threshold2,
                         int apertureSize = 3, bool L2gradient = false );

//! computes minimum eigen value of 2x2 derivative covariation matrix at each pixel - the cornerness criteria
CV_EXPORTS_W void cornerMinEigenVal( InputArray src, OutputArray dst,
                                     int blockSize, int ksize = 3,
                                     int borderType = BORDER_DEFAULT );

//! computes Harris cornerness criteria at each image pixel
CV_EXPORTS_W void cornerHarris( InputArray src, OutputArray dst, int blockSize,
                                int ksize, double k,
                                int borderType = BORDER_DEFAULT );

//! computes both eigenvalues and the eigenvectors of 2x2 derivative covariation matrix  at each pixel. The output is stored as 6-channel matrix.
CV_EXPORTS_W void cornerEigenValsAndVecs( InputArray src, OutputArray dst,
                                          int blockSize, int ksize,
                                          int borderType = BORDER_DEFAULT );

//! computes another complex cornerness criteria at each pixel
CV_EXPORTS_W void preCornerDetect( InputArray src, OutputArray dst, int ksize,
                                   int borderType = BORDER_DEFAULT );

//! adjusts the corner locations with sub-pixel accuracy to maximize the certain cornerness criteria
CV_EXPORTS_W void cornerSubPix( InputArray image, InputOutputArray corners,
                                Size winSize, Size zeroZone,
                                TermCriteria criteria );

//! finds the strong enough corners where the cornerMinEigenVal() or cornerHarris() report the local maxima
CV_EXPORTS_W void goodFeaturesToTrack( InputArray image, OutputArray corners,
                                     int maxCorners, double qualityLevel, double minDistance,
                                     InputArray mask = noArray(), int blockSize = 3,
                                     bool useHarrisDetector = false, double k = 0.04 );

//! finds lines in the black-n-white image using the standard or pyramid Hough transform
CV_EXPORTS_W void HoughLines( InputArray image, OutputArray lines,
                              double rho, double theta, int threshold,
                              double srn = 0, double stn = 0,
                              double min_theta = 0, double max_theta = CV_PI );

//! finds line segments in the black-n-white image using probabilistic Hough transform
CV_EXPORTS_W void HoughLinesP( InputArray image, OutputArray lines,
                               double rho, double theta, int threshold,
                               double minLineLength = 0, double maxLineGap = 0 );

//! finds circles in the grayscale image using 2+1 gradient Hough transform
CV_EXPORTS_W void HoughCircles( InputArray image, OutputArray circles,
                               int method, double dp, double minDist,
                               double param1 = 100, double param2 = 100,
                               int minRadius = 0, int maxRadius = 0 );

//! erodes the image (applies the local minimum operator)
CV_EXPORTS_W void erode( InputArray src, OutputArray dst, InputArray kernel,
                         Point anchor = Point(-1,-1), int iterations = 1,
                         int borderType = BORDER_CONSTANT,
                         const Scalar& borderValue = morphologyDefaultBorderValue() );

//! dilates the image (applies the local maximum operator)
CV_EXPORTS_W void dilate( InputArray src, OutputArray dst, InputArray kernel,
                          Point anchor = Point(-1,-1), int iterations = 1,
                          int borderType = BORDER_CONSTANT,
                          const Scalar& borderValue = morphologyDefaultBorderValue() );

//! applies an advanced morphological operation to the image
CV_EXPORTS_W void morphologyEx( InputArray src, OutputArray dst,
                                int op, InputArray kernel,
                                Point anchor = Point(-1,-1), int iterations = 1,
                                int borderType = BORDER_CONSTANT,
                                const Scalar& borderValue = morphologyDefaultBorderValue() );

//! resizes the image
CV_EXPORTS_W void resize( InputArray src, OutputArray dst,
                          Size dsize, double fx = 0, double fy = 0,
                          int interpolation = INTER_LINEAR );

//! warps the image using affine transformation
CV_EXPORTS_W void warpAffine( InputArray src, OutputArray dst,
                              InputArray M, Size dsize,
                              int flags = INTER_LINEAR,
                              int borderMode = BORDER_CONSTANT,
                              const Scalar& borderValue = Scalar());

//! warps the image using perspective transformation
CV_EXPORTS_W void warpPerspective( InputArray src, OutputArray dst,
                                   InputArray M, Size dsize,
                                   int flags = INTER_LINEAR,
                                   int borderMode = BORDER_CONSTANT,
                                   const Scalar& borderValue = Scalar());

//! warps the image using the precomputed maps. The maps are stored in either floating-point or integer fixed-point format
CV_EXPORTS_W void remap( InputArray src, OutputArray dst,
                         InputArray map1, InputArray map2,
                         int interpolation, int borderMode = BORDER_CONSTANT,
                         const Scalar& borderValue = Scalar());

//! converts maps for remap from floating-point to fixed-point format or backwards
CV_EXPORTS_W void convertMaps( InputArray map1, InputArray map2,
                               OutputArray dstmap1, OutputArray dstmap2,
                               int dstmap1type, bool nninterpolation = false );

//! returns 2x3 affine transformation matrix for the planar rotation.
CV_EXPORTS_W Mat getRotationMatrix2D( Point2f center, double angle, double scale );

//! returns 3x3 perspective transformation for the corresponding 4 point pairs.
CV_EXPORTS Mat getPerspectiveTransform( const Point2f src[], const Point2f dst[] );

//! returns 2x3 affine transformation for the corresponding 3 point pairs.
CV_EXPORTS Mat getAffineTransform( const Point2f src[], const Point2f dst[] );

//! computes 2x3 affine transformation matrix that is inverse to the specified 2x3 affine transformation.
CV_EXPORTS_W void invertAffineTransform( InputArray M, OutputArray iM );

CV_EXPORTS_W Mat getPerspectiveTransform( InputArray src, InputArray dst );

CV_EXPORTS_W Mat getAffineTransform( InputArray src, InputArray dst );

//! extracts rectangle from the image at sub-pixel location
CV_EXPORTS_W void getRectSubPix( InputArray image, Size patchSize,
                                 Point2f center, OutputArray patch, int patchType = -1 );

//! computes the log polar transform
CV_EXPORTS_W void logPolar( InputArray src, OutputArray dst,
                            Point2f center, double M, int flags );

//! computes the linear polar transform
CV_EXPORTS_W void linearPolar( InputArray src, OutputArray dst,
                               Point2f center, double maxRadius, int flags );

//! computes the integral image
CV_EXPORTS_W void integral( InputArray src, OutputArray sum, int sdepth = -1 );

//! computes the integral image and integral for the squared image
CV_EXPORTS_AS(integral2) void integral( InputArray src, OutputArray sum,
                                        OutputArray sqsum, int sdepth = -1, int sqdepth = -1 );

//! computes the integral image, integral for the squared image and the tilted integral image
CV_EXPORTS_AS(integral3) void integral( InputArray src, OutputArray sum,
                                        OutputArray sqsum, OutputArray tilted,
                                        int sdepth = -1, int sqdepth = -1 );

//! adds image to the accumulator (dst += src). Unlike cv::add, dst and src can have different types.
CV_EXPORTS_W void accumulate( InputArray src, InputOutputArray dst,
                              InputArray mask = noArray() );

//! adds squared src image to the accumulator (dst += src*src).
CV_EXPORTS_W void accumulateSquare( InputArray src, InputOutputArray dst,
                                    InputArray mask = noArray() );
//! adds product of the 2 images to the accumulator (dst += src1*src2).
CV_EXPORTS_W void accumulateProduct( InputArray src1, InputArray src2,
                                     InputOutputArray dst, InputArray mask=noArray() );

//! updates the running average (dst = dst*(1-alpha) + src*alpha)
CV_EXPORTS_W void accumulateWeighted( InputArray src, InputOutputArray dst,
                                      double alpha, InputArray mask = noArray() );

CV_EXPORTS_W Point2d phaseCorrelate(InputArray src1, InputArray src2,
                                    InputArray window = noArray(), CV_OUT double* response = 0);

CV_EXPORTS_W void createHanningWindow(OutputArray dst, Size winSize, int type);

//! applies fixed threshold to the image
CV_EXPORTS_W double threshold( InputArray src, OutputArray dst,
                               double thresh, double maxval, int type );


//! applies variable (adaptive) threshold to the image
CV_EXPORTS_W void adaptiveThreshold( InputArray src, OutputArray dst,
                                     double maxValue, int adaptiveMethod,
                                     int thresholdType, int blockSize, double C );

//! smooths and downsamples the image
CV_EXPORTS_W void pyrDown( InputArray src, OutputArray dst,
                           const Size& dstsize = Size(), int borderType = BORDER_DEFAULT );

//! upsamples and smoothes the image
CV_EXPORTS_W void pyrUp( InputArray src, OutputArray dst,
                         const Size& dstsize = Size(), int borderType = BORDER_DEFAULT );

//! builds the gaussian pyramid using pyrDown() as a basic operation
CV_EXPORTS void buildPyramid( InputArray src, OutputArrayOfArrays dst,
                              int maxlevel, int borderType = BORDER_DEFAULT );

//! corrects lens distortion for the given camera matrix and distortion coefficients
CV_EXPORTS_W void undistort( InputArray src, OutputArray dst,
                             InputArray cameraMatrix,
                             InputArray distCoeffs,
                             InputArray newCameraMatrix = noArray() );

//! initializes maps for cv::remap() to correct lens distortion and optionally rectify the image
CV_EXPORTS_W void initUndistortRectifyMap( InputArray cameraMatrix, InputArray distCoeffs,
                           InputArray R, InputArray newCameraMatrix,
                           Size size, int m1type, OutputArray map1, OutputArray map2 );

//! initializes maps for cv::remap() for wide-angle
CV_EXPORTS_W float initWideAngleProjMap( InputArray cameraMatrix, InputArray distCoeffs,
                                         Size imageSize, int destImageWidth,
                                         int m1type, OutputArray map1, OutputArray map2,
                                         int projType = PROJ_SPHERICAL_EQRECT, double alpha = 0);

//! returns the default new camera matrix (by default it is the same as cameraMatrix unless centerPricipalPoint=true)
CV_EXPORTS_W Mat getDefaultNewCameraMatrix( InputArray cameraMatrix, Size imgsize = Size(),
                                            bool centerPrincipalPoint = false );

//! returns points' coordinates after lens distortion correction
CV_EXPORTS_W void undistortPoints( InputArray src, OutputArray dst,
                                   InputArray cameraMatrix, InputArray distCoeffs,
                                   InputArray R = noArray(), InputArray P = noArray());

//! computes the joint dense histogram for a set of images.
CV_EXPORTS void calcHist( const Mat* images, int nimages,
                          const int* channels, InputArray mask,
                          OutputArray hist, int dims, const int* histSize,
                          const float** ranges, bool uniform = true, bool accumulate = false );

//! computes the joint sparse histogram for a set of images.
CV_EXPORTS void calcHist( const Mat* images, int nimages,
                          const int* channels, InputArray mask,
                          SparseMat& hist, int dims,
                          const int* histSize, const float** ranges,
                          bool uniform = true, bool accumulate = false );

CV_EXPORTS_W void calcHist( InputArrayOfArrays images,
                            const std::vector<int>& channels,
                            InputArray mask, OutputArray hist,
                            const std::vector<int>& histSize,
                            const std::vector<float>& ranges,
                            bool accumulate = false );

//! computes back projection for the set of images
CV_EXPORTS void calcBackProject( const Mat* images, int nimages,
                                 const int* channels, InputArray hist,
                                 OutputArray backProject, const float** ranges,
                                 double scale = 1, bool uniform = true );

//! computes back projection for the set of images
CV_EXPORTS void calcBackProject( const Mat* images, int nimages,
                                 const int* channels, const SparseMat& hist,
                                 OutputArray backProject, const float** ranges,
                                 double scale = 1, bool uniform = true );

CV_EXPORTS_W void calcBackProject( InputArrayOfArrays images, const std::vector<int>& channels,
                                   InputArray hist, OutputArray dst,
                                   const std::vector<float>& ranges,
                                   double scale );

//! compares two histograms stored in dense arrays
CV_EXPORTS_W double compareHist( InputArray H1, InputArray H2, int method );

//! compares two histograms stored in sparse arrays
CV_EXPORTS double compareHist( const SparseMat& H1, const SparseMat& H2, int method );

//! normalizes the grayscale image brightness and contrast by normalizing its histogram
CV_EXPORTS_W void equalizeHist( InputArray src, OutputArray dst );

CV_EXPORTS float EMD( InputArray signature1, InputArray signature2,
                      int distType, InputArray cost=noArray(),
                      float* lowerBound = 0, OutputArray flow = noArray() );

//! segments the image using watershed algorithm
CV_EXPORTS_W void watershed( InputArray image, InputOutputArray markers );

//! filters image using meanshift algorithm
CV_EXPORTS_W void pyrMeanShiftFiltering( InputArray src, OutputArray dst,
                                         double sp, double sr, int maxLevel = 1,
                                         TermCriteria termcrit=TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS,5,1) );

//! segments the image using GrabCut algorithm
CV_EXPORTS_W void grabCut( InputArray img, InputOutputArray mask, Rect rect,
                           InputOutputArray bgdModel, InputOutputArray fgdModel,
                           int iterCount, int mode = GC_EVAL );


//! builds the discrete Voronoi diagram
CV_EXPORTS_AS(distanceTransformWithLabels) void distanceTransform( InputArray src, OutputArray dst,
                                     OutputArray labels, int distanceType, int maskSize,
                                     int labelType = DIST_LABEL_CCOMP );

//! computes the distance transform map
CV_EXPORTS_W void distanceTransform( InputArray src, OutputArray dst,
                                     int distanceType, int maskSize, int dstType=CV_32F);


//! fills the semi-uniform image region starting from the specified seed point
CV_EXPORTS int floodFill( InputOutputArray image,
                          Point seedPoint, Scalar newVal, CV_OUT Rect* rect = 0,
                          Scalar loDiff = Scalar(), Scalar upDiff = Scalar(),
                          int flags = 4 );

//! fills the semi-uniform image region and/or the mask starting from the specified seed point
CV_EXPORTS_W int floodFill( InputOutputArray image, InputOutputArray mask,
                            Point seedPoint, Scalar newVal, CV_OUT Rect* rect=0,
                            Scalar loDiff = Scalar(), Scalar upDiff = Scalar(),
                            int flags = 4 );

//! converts image from one color space to another
CV_EXPORTS_W void cvtColor( InputArray src, OutputArray dst, int code, int dstCn = 0 );

// main function for all demosaicing procceses
CV_EXPORTS_W void demosaicing(InputArray _src, OutputArray _dst, int code, int dcn = 0);

//! computes moments of the rasterized shape or a vector of points
CV_EXPORTS_W Moments moments( InputArray array, bool binaryImage = false );

//! computes 7 Hu invariants from the moments
CV_EXPORTS void HuMoments( const Moments& moments, double hu[7] );

CV_EXPORTS_W void HuMoments( const Moments& m, OutputArray hu );

//! computes the proximity map for the raster template and the image where the template is searched for
CV_EXPORTS_W void matchTemplate( InputArray image, InputArray templ,
                                 OutputArray result, int method );


// computes the connected components labeled image of boolean image ``image``
// with 4 or 8 way connectivity - returns N, the total
// number of labels [0, N-1] where 0 represents the background label.
// ltype specifies the output label image type, an important
// consideration based on the total number of labels or
// alternatively the total number of pixels in the source image.
CV_EXPORTS_W int connectedComponents(InputArray image, OutputArray labels,
                                     int connectivity = 8, int ltype = CV_32S);

CV_EXPORTS_W int connectedComponentsWithStats(InputArray image, OutputArray labels,
                                              OutputArray stats, OutputArray centroids,
                                              int connectivity = 8, int ltype = CV_32S);


//! retrieves contours and the hierarchical information from black-n-white image.
CV_EXPORTS_W void findContours( InputOutputArray image, OutputArrayOfArrays contours,
                              OutputArray hierarchy, int mode,
                              int method, Point offset = Point());

//! retrieves contours from black-n-white image.
CV_EXPORTS void findContours( InputOutputArray image, OutputArrayOfArrays contours,
                              int mode, int method, Point offset = Point());

//! approximates contour or a curve using Douglas-Peucker algorithm
CV_EXPORTS_W void approxPolyDP( InputArray curve,
                                OutputArray approxCurve,
                                double epsilon, bool closed );

//! computes the contour perimeter (closed=true) or a curve length
CV_EXPORTS_W double arcLength( InputArray curve, bool closed );

//! computes the bounding rectangle for a contour
CV_EXPORTS_W Rect boundingRect( InputArray points );

//! computes the contour area
CV_EXPORTS_W double contourArea( InputArray contour, bool oriented = false );

//! computes the minimal rotated rectangle for a set of points
CV_EXPORTS_W RotatedRect minAreaRect( InputArray points );

//! computes boxpoints
CV_EXPORTS_W void boxPoints(RotatedRect box, OutputArray points);

//! computes the minimal enclosing circle for a set of points
CV_EXPORTS_W void minEnclosingCircle( InputArray points,
                                      CV_OUT Point2f& center, CV_OUT float& radius );

//! computes the minimal enclosing triangle for a set of points and returns its area
CV_EXPORTS_W double minEnclosingTriangle( InputArray points, CV_OUT OutputArray triangle );

//! matches two contours using one of the available algorithms
CV_EXPORTS_W double matchShapes( InputArray contour1, InputArray contour2,
                                 int method, double parameter );

//! computes convex hull for a set of 2D points.
CV_EXPORTS_W void convexHull( InputArray points, OutputArray hull,
                              bool clockwise = false, bool returnPoints = true );

//! computes the contour convexity defects
CV_EXPORTS_W void convexityDefects( InputArray contour, InputArray convexhull, OutputArray convexityDefects );

//! returns true if the contour is convex. Does not support contours with self-intersection
CV_EXPORTS_W bool isContourConvex( InputArray contour );

//! finds intersection of two convex polygons
CV_EXPORTS_W float intersectConvexConvex( InputArray _p1, InputArray _p2,
                                          OutputArray _p12, bool handleNested = true );

//! fits ellipse to the set of 2D points
CV_EXPORTS_W RotatedRect fitEllipse( InputArray points );

//! fits line to the set of 2D points using M-estimator algorithm
CV_EXPORTS_W void fitLine( InputArray points, OutputArray line, int distType,
                           double param, double reps, double aeps );

//! checks if the point is inside the contour. Optionally computes the signed distance from the point to the contour boundary
CV_EXPORTS_W double pointPolygonTest( InputArray contour, Point2f pt, bool measureDist );

//! computes whether two rotated rectangles intersect and returns the vertices of the intersecting region
CV_EXPORTS_W int rotatedRectangleIntersection( const RotatedRect& rect1, const RotatedRect& rect2, OutputArray intersectingRegion  );

CV_EXPORTS_W Ptr<CLAHE> createCLAHE(double clipLimit = 40.0, Size tileGridSize = Size(8, 8));

//! Ballard, D.H. (1981). Generalizing the Hough transform to detect arbitrary shapes. Pattern Recognition 13 (2): 111-122.
//! Detects position only without traslation and rotation
CV_EXPORTS Ptr<GeneralizedHoughBallard> createGeneralizedHoughBallard();

//! Guil, N., González-Linares, J.M. and Zapata, E.L. (1999). Bidimensional shape detection using an invariant approach. Pattern Recognition 32 (6): 1025-1038.
//! Detects position, traslation and rotation
CV_EXPORTS Ptr<GeneralizedHoughGuil> createGeneralizedHoughGuil();

//! Performs linear blending of two images
CV_EXPORTS void blendLinear(InputArray src1, InputArray src2, InputArray weights1, InputArray weights2, OutputArray dst);

} // cv

#endif
