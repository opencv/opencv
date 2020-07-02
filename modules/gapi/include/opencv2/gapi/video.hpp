// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_VIDEO_HPP
#define OPENCV_GAPI_VIDEO_HPP

#include <utility> // std::tuple

#include <opencv2/gapi/gkernel.hpp>


/** \defgroup gapi_video G-API Video processing functionality
 */

namespace cv { namespace gapi {
namespace  video
{
using GBuildPyrOutput  = std::tuple<GArray<GMat>, GScalar>;

using GOptFlowLKOutput = std::tuple<cv::GArray<cv::Point2f>,
                                    cv::GArray<uchar>,
                                    cv::GArray<float>>;

G_TYPED_KERNEL(GBuildOptFlowPyramid, <GBuildPyrOutput(GMat,Size,GScalar,bool,int,int,bool)>,
               "org.opencv.video.buildOpticalFlowPyramid")
{
    static std::tuple<GArrayDesc,GScalarDesc>
            outMeta(GMatDesc,const Size&,GScalarDesc,bool,int,int,bool)
    {
        return std::make_tuple(empty_array_desc(), empty_scalar_desc());
    }
};

G_TYPED_KERNEL(GCalcOptFlowLK,
               <GOptFlowLKOutput(GMat,GMat,cv::GArray<cv::Point2f>,cv::GArray<cv::Point2f>,Size,
                                 GScalar,TermCriteria,int,double)>,
               "org.opencv.video.calcOpticalFlowPyrLK")
{
    static std::tuple<GArrayDesc,GArrayDesc,GArrayDesc> outMeta(GMatDesc,GMatDesc,GArrayDesc,
                                                                GArrayDesc,const Size&,GScalarDesc,
                                                                const TermCriteria&,int,double)
    {
        return std::make_tuple(empty_array_desc(), empty_array_desc(), empty_array_desc());
    }

};

G_TYPED_KERNEL(GCalcOptFlowLKForPyr,
               <GOptFlowLKOutput(cv::GArray<cv::GMat>,cv::GArray<cv::GMat>,
                                 cv::GArray<cv::Point2f>,cv::GArray<cv::Point2f>,Size,GScalar,
                                 TermCriteria,int,double)>,
               "org.opencv.video.calcOpticalFlowPyrLKForPyr")
{
    static std::tuple<GArrayDesc,GArrayDesc,GArrayDesc> outMeta(GArrayDesc,GArrayDesc,
                                                                GArrayDesc,GArrayDesc,
                                                                const Size&,GScalarDesc,
                                                                const TermCriteria&,int,double)
    {
        return std::make_tuple(empty_array_desc(), empty_array_desc(), empty_array_desc());
    }
};
} //namespace video

//! @addtogroup gapi_video
//! @{
/** @brief Constructs the image pyramid which can be passed to calcOpticalFlowPyrLK.

@note Function textual ID is "org.opencv.video.buildOpticalFlowPyramid"

@param img                8-bit input image.
@param winSize            window size of optical flow algorithm. Must be not less than winSize
                          argument of calcOpticalFlowPyrLK. It is needed to calculate required
                          padding for pyramid levels.
@param maxLevel           0-based maximal pyramid level number.
@param withDerivatives    set to precompute gradients for the every pyramid level. If pyramid is
                          constructed without the gradients then calcOpticalFlowPyrLK will calculate
                          them internally.
@param pyrBorder          the border mode for pyramid layers.
@param derivBorder        the border mode for gradients.
@param tryReuseInputImage put ROI of input image into the pyramid if possible. You can pass false
                          to force data copying.

@return output pyramid.
@return number of levels in constructed pyramid. Can be less than maxLevel.
 */
GAPI_EXPORTS std::tuple<GArray<GMat>, GScalar>
buildOpticalFlowPyramid(const GMat     &img,
                        const Size     &winSize,
                        const GScalar  &maxLevel,
                              bool      withDerivatives    = true,
                              int       pyrBorder          = BORDER_REFLECT_101,
                              int       derivBorder        = BORDER_CONSTANT,
                              bool      tryReuseInputImage = true);

/** @brief Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade
method with pyramids.

See @cite Bouguet00 .

@note Function textual ID is "org.opencv.video.calcOpticalFlowPyrLK"

@param prevImg first 8-bit input image (GMat) or pyramid (GArray<GMat>) constructed by
buildOpticalFlowPyramid.
@param nextImg second input image (GMat) or pyramid (GArray<GMat>) of the same size and the same
type as prevImg.
@param prevPts GArray of 2D points for which the flow needs to be found; point coordinates must be
single-precision floating-point numbers.
@param predPts GArray of 2D points initial for the flow search; make sense only when
OPTFLOW_USE_INITIAL_FLOW flag is passed; in that case the vector must have the same size as in
the input.
@param winSize size of the search window at each pyramid level.
@param maxLevel 0-based maximal pyramid level number; if set to 0, pyramids are not used (single
level), if set to 1, two levels are used, and so on; if pyramids are passed to input then
algorithm will use as many levels as pyramids have but no more than maxLevel.
@param criteria parameter, specifying the termination criteria of the iterative search algorithm
(after the specified maximum number of iterations criteria.maxCount or when the search window
moves by less than criteria.epsilon).
@param flags operation flags:
 -   **OPTFLOW_USE_INITIAL_FLOW** uses initial estimations, stored in nextPts; if the flag is
     not set, then prevPts is copied to nextPts and is considered the initial estimate.
 -   **OPTFLOW_LK_GET_MIN_EIGENVALS** use minimum eigen values as an error measure (see
     minEigThreshold description); if the flag is not set, then L1 distance between patches
     around the original and a moved point, divided by number of pixels in a window, is used as a
     error measure.
@param minEigThresh the algorithm calculates the minimum eigen value of a 2x2 normal matrix of
optical flow equations (this matrix is called a spatial gradient matrix in @cite Bouguet00), divided
by number of pixels in a window; if this value is less than minEigThreshold, then a corresponding
feature is filtered out and its flow is not processed, so it allows to remove bad points and get a
performance boost.

@return GArray of 2D points (with single-precision floating-point coordinates)
containing the calculated new positions of input features in the second image.
@return status GArray (of unsigned chars); each element of the vector is set to 1 if
the flow for the corresponding features has been found, otherwise, it is set to 0.
@return GArray of errors (doubles); each element of the vector is set to an error for the
corresponding feature, type of the error measure can be set in flags parameter; if the flow wasn't
found then the error is not defined (use the status parameter to find such cases).
 */
GAPI_EXPORTS std::tuple<GArray<Point2f>, GArray<uchar>, GArray<float>>
calcOpticalFlowPyrLK(const GMat            &prevImg,
                     const GMat            &nextImg,
                     const GArray<Point2f> &prevPts,
                     const GArray<Point2f> &predPts,
                     const Size            &winSize      = Size(21, 21),
                     const GScalar         &maxLevel     = 3,
                     const TermCriteria    &criteria     = TermCriteria(TermCriteria::COUNT |
                                                                        TermCriteria::EPS,
                                                                        30, 0.01),
                           int              flags        = 0,
                           double           minEigThresh = 1e-4);

/**
@overload
@note Function textual ID is "org.opencv.video.calcOpticalFlowPyrLKForPyr"
*/
GAPI_EXPORTS std::tuple<GArray<Point2f>, GArray<uchar>, GArray<float>>
calcOpticalFlowPyrLK(const GArray<GMat>    &prevPyr,
                     const GArray<GMat>    &nextPyr,
                     const GArray<Point2f> &prevPts,
                     const GArray<Point2f> &predPts,
                     const Size            &winSize      = Size(21, 21),
                     const GScalar         &maxLevel     = 3,
                     const TermCriteria    &criteria     = TermCriteria(TermCriteria::COUNT |
                                                                        TermCriteria::EPS,
                                                                        30, 0.01),
                           int              flags        = 0,
                           double           minEigThresh = 1e-4);

//! @} gapi_video
} //namespace gapi
} //namespace cv

#endif // OPENCV_GAPI_VIDEO_HPP
