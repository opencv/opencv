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

/** @brief Structure for the Kalman filter's initialization parameters.*/

struct GAPI_EXPORTS KalmanParams
{
    // initial state

    //! corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))
    Mat state;
    //! posteriori error estimate covariance matrix (P(k)): P(k)=(I-K(k)*H)*P'(k)
    Mat errorCov;

    // dynamic system description

    //! state transition matrix (A)
    Mat transitionMatrix;
    //! measurement matrix (H)
    Mat measurementMatrix;
    //! process noise covariance matrix (Q)
    Mat processNoiseCov;
    //! measurement noise covariance matrix (R)
    Mat measurementNoiseCov;
    //! control matrix (B) (Optional: not used if there's no control)
    Mat controlMatrix;
};

/**
 * @brief This namespace contains G-API Operations and functions for
 * video-oriented algorithms, like optical flow and background subtraction.
 */
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

enum BackgroundSubtractorType
{
    TYPE_BS_MOG2,
    TYPE_BS_KNN
};

/** @brief Structure for the Background Subtractor operation's initialization parameters.*/

struct BackgroundSubtractorParams
{
    //! Type of the Background Subtractor operation.
    BackgroundSubtractorType operation = TYPE_BS_MOG2;

    //! Length of the history.
    int history = 500;

    //! For MOG2: Threshold on the squared Mahalanobis distance between the pixel
    //! and the model to decide whether a pixel is well described by
    //! the background model.
    //! For KNN: Threshold on the squared distance between the pixel and the sample
    //! to decide whether a pixel is close to that sample.
    double threshold = 16;

    //! If true, the algorithm will detect shadows and mark them.
    bool detectShadows = true;

    //! The value between 0 and 1 that indicates how fast
    //! the background model is learnt.
    //! Negative parameter value makes the algorithm use some automatically
    //! chosen learning rate.
    double learningRate = -1;

    //! default constructor
    BackgroundSubtractorParams() {}

    /** Full constructor
    @param op MOG2/KNN Background Subtractor type.
    @param histLength Length of the history.
    @param thrshld For MOG2: Threshold on the squared Mahalanobis distance between
    the pixel and the model to decide whether a pixel is well described by the background model.
    For KNN: Threshold on the squared distance between the pixel and the sample to decide
    whether a pixel is close to that sample.
    @param detect If true, the algorithm will detect shadows and mark them. It decreases the
    speed a bit, so if you do not need this feature, set the parameter to false.
    @param lRate The value between 0 and 1 that indicates how fast the background model is learnt.
    Negative parameter value makes the algorithm to use some automatically chosen learning rate.
    */
    BackgroundSubtractorParams(BackgroundSubtractorType op, int histLength,
                               double thrshld, bool detect, double lRate) : operation(op),
                                                                            history(histLength),
                                                                            threshold(thrshld),
                                                                            detectShadows(detect),
                                                                            learningRate(lRate){}
};

G_TYPED_KERNEL(GBackgroundSubtractor, <GMat(GMat, BackgroundSubtractorParams)>,
               "org.opencv.video.BackgroundSubtractor")
{
    static GMatDesc outMeta(const GMatDesc& in, const BackgroundSubtractorParams& bsParams)
    {
        GAPI_Assert(bsParams.history >= 0);
        GAPI_Assert(bsParams.learningRate <= 1);
        return in.withType(CV_8U, 1);
    }
};

void checkParams(const cv::gapi::KalmanParams& kfParams,
                 const cv::GMatDesc& measurement, const cv::GMatDesc& control = {});

G_TYPED_KERNEL(GKalmanFilter, <GMat(GMat, GOpaque<bool>, GMat, KalmanParams)>,
               "org.opencv.video.KalmanFilter")
{
    static GMatDesc outMeta(const GMatDesc& measurement, const GOpaqueDesc&,
                            const GMatDesc& control, const KalmanParams& kfParams)
    {
        checkParams(kfParams, measurement, control);
        return measurement.withSize(Size(1, kfParams.transitionMatrix.rows));
    }
};

G_TYPED_KERNEL(GKalmanFilterNoControl, <GMat(GMat, GOpaque<bool>, KalmanParams)>, "org.opencv.video.KalmanFilterNoControl")
{
    static GMatDesc outMeta(const GMatDesc& measurement, const GOpaqueDesc&, const KalmanParams& kfParams)
    {
        checkParams(kfParams, measurement);
        return measurement.withSize(Size(1, kfParams.transitionMatrix.rows));
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

@return
 - output pyramid.
 - number of levels in constructed pyramid. Can be less than maxLevel.
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

@return
 - GArray of 2D points (with single-precision floating-point coordinates)
containing the calculated new positions of input features in the second image.
 - status GArray (of unsigned chars); each element of the vector is set to 1 if
the flow for the corresponding features has been found, otherwise, it is set to 0.
 - GArray of errors (doubles); each element of the vector is set to an error for the
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

/** @brief Gaussian Mixture-based or K-nearest neighbours-based Background/Foreground Segmentation Algorithm.
The operation generates a foreground mask.

@return Output image is foreground mask, i.e. 8-bit unsigned 1-channel (binary) matrix @ref CV_8UC1.

@note Functional textual ID is "org.opencv.video.BackgroundSubtractor"

@param src input image: Floating point frame is used without scaling and should be in range [0,255].
@param bsParams Set of initialization parameters for Background Subtractor kernel.
*/
GAPI_EXPORTS GMat BackgroundSubtractor(const GMat& src, const cv::gapi::video::BackgroundSubtractorParams& bsParams);

/** @brief Standard Kalman filter algorithm <http://en.wikipedia.org/wiki/Kalman_filter>.

@note Functional textual ID is "org.opencv.video.KalmanFilter"

@param measurement input matrix: 32-bit or 64-bit float 1-channel matrix containing measurements.
@param haveMeasurement dynamic input flag that indicates whether we get measurements
at a particular iteration .
@param control input matrix: 32-bit or 64-bit float 1-channel matrix contains control data
for changing dynamic system.
@param kfParams Set of initialization parameters for Kalman filter kernel.

@return Output matrix is predicted or corrected state. They can be 32-bit or 64-bit float
1-channel matrix @ref CV_32FC1 or @ref CV_64FC1.

@details If measurement matrix is given (haveMeasurements == true), corrected state will
be returned which corresponds to the pipeline
cv::KalmanFilter::predict(control) -> cv::KalmanFilter::correct(measurement).
Otherwise, predicted state will be returned which corresponds to the call of
cv::KalmanFilter::predict(control).
@sa cv::KalmanFilter
*/
GAPI_EXPORTS GMat KalmanFilter(const GMat& measurement, const GOpaque<bool>& haveMeasurement,
                               const GMat& control, const cv::gapi::KalmanParams& kfParams);

/** @overload
The case of Standard Kalman filter algorithm when there is no control in a dynamic system.
In this case the controlMatrix is empty and control vector is absent.

@note Function textual ID is "org.opencv.video.KalmanFilterNoControl"

@param measurement input matrix: 32-bit or 64-bit float 1-channel matrix containing measurements.
@param haveMeasurement dynamic input flag that indicates whether we get measurements
at a particular iteration.
@param kfParams Set of initialization parameters for Kalman filter kernel.

@return Output matrix is predicted or corrected state. They can be 32-bit or 64-bit float
1-channel matrix @ref CV_32FC1 or @ref CV_64FC1.

@sa cv::KalmanFilter
 */
GAPI_EXPORTS GMat KalmanFilter(const GMat& measurement, const GOpaque<bool>& haveMeasurement,
                               const cv::gapi::KalmanParams& kfParams);

//! @} gapi_video
} //namespace gapi
} //namespace cv


namespace cv { namespace detail {
template<> struct CompileArgTag<cv::gapi::video::BackgroundSubtractorParams>
{
    static const char* tag()
    {
        return "org.opencv.video.background_substractor_params";
    }
};
}  // namespace detail
}  // namespace cv

#endif // OPENCV_GAPI_VIDEO_HPP
