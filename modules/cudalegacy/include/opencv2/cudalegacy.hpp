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

#ifndef OPENCV_CUDALEGACY_HPP
#define OPENCV_CUDALEGACY_HPP

#include "opencv2/core/cuda.hpp"
#include "opencv2/cudalegacy/NCV.hpp"
#include "opencv2/cudalegacy/NPP_staging.hpp"
#include "opencv2/cudalegacy/NCVPyramid.hpp"
#include "opencv2/cudalegacy/NCVHaarObjectDetection.hpp"
#include "opencv2/cudalegacy/NCVBroxOpticalFlow.hpp"
#include "opencv2/video/background_segm.hpp"

/**
  @addtogroup cuda
  @{
    @defgroup cudalegacy Legacy support
  @}
*/

namespace cv { namespace cuda {

//! @addtogroup cudalegacy
//! @{

//
// ImagePyramid
//

class CV_EXPORTS ImagePyramid : public Algorithm
{
public:
    virtual void getLayer(OutputArray outImg, Size outRoi, Stream& stream = Stream::Null()) const = 0;
};

CV_EXPORTS Ptr<ImagePyramid> createImagePyramid(InputArray img, int nLayers = -1, Stream& stream = Stream::Null());

//
// GMG
//

/** @brief Background/Foreground Segmentation Algorithm.

The class discriminates between foreground and background pixels by building and maintaining a model
of the background. Any pixel which does not fit this model is then deemed to be foreground. The
class implements algorithm described in @cite Gold2012 .
 */
class CV_EXPORTS BackgroundSubtractorGMG : public cv::BackgroundSubtractor
{
public:
    using cv::BackgroundSubtractor::apply;
    virtual void apply(InputArray image, OutputArray fgmask, double learningRate, Stream& stream) = 0;

    virtual int getMaxFeatures() const = 0;
    virtual void setMaxFeatures(int maxFeatures) = 0;

    virtual double getDefaultLearningRate() const = 0;
    virtual void setDefaultLearningRate(double lr) = 0;

    virtual int getNumFrames() const = 0;
    virtual void setNumFrames(int nframes) = 0;

    virtual int getQuantizationLevels() const = 0;
    virtual void setQuantizationLevels(int nlevels) = 0;

    virtual double getBackgroundPrior() const = 0;
    virtual void setBackgroundPrior(double bgprior) = 0;

    virtual int getSmoothingRadius() const = 0;
    virtual void setSmoothingRadius(int radius) = 0;

    virtual double getDecisionThreshold() const = 0;
    virtual void setDecisionThreshold(double thresh) = 0;

    virtual bool getUpdateBackgroundModel() const = 0;
    virtual void setUpdateBackgroundModel(bool update) = 0;

    virtual double getMinVal() const = 0;
    virtual void setMinVal(double val) = 0;

    virtual double getMaxVal() const = 0;
    virtual void setMaxVal(double val) = 0;
};

/** @brief Creates GMG Background Subtractor

@param initializationFrames Number of frames of video to use to initialize histograms.
@param decisionThreshold Value above which pixel is determined to be FG.
 */
CV_EXPORTS Ptr<cuda::BackgroundSubtractorGMG>
    createBackgroundSubtractorGMG(int initializationFrames = 120, double decisionThreshold = 0.8);

//
// FGD
//

/** @brief The class discriminates between foreground and background pixels by building and maintaining a model
of the background.

Any pixel which does not fit this model is then deemed to be foreground. The class implements
algorithm described in @cite FGD2003 .
@sa BackgroundSubtractor
 */
class CV_EXPORTS BackgroundSubtractorFGD : public cv::BackgroundSubtractor
{
public:
    /** @brief Returns the output foreground regions calculated by findContours.

    @param foreground_regions Output array (CPU memory).
     */
    virtual void getForegroundRegions(OutputArrayOfArrays foreground_regions) = 0;
};

struct CV_EXPORTS FGDParams
{
    int Lc;  //!< Quantized levels per 'color' component. Power of two, typically 32, 64 or 128.
    int N1c; //!< Number of color vectors used to model normal background color variation at a given pixel.
    int N2c; //!< Number of color vectors retained at given pixel.  Must be > N1c, typically ~ 5/3 of N1c.
    //!< Used to allow the first N1c vectors to adapt over time to changing background.

    int Lcc;  //!< Quantized levels per 'color co-occurrence' component.  Power of two, typically 16, 32 or 64.
    int N1cc; //!< Number of color co-occurrence vectors used to model normal background color variation at a given pixel.
    int N2cc; //!< Number of color co-occurrence vectors retained at given pixel.  Must be > N1cc, typically ~ 5/3 of N1cc.
    //!< Used to allow the first N1cc vectors to adapt over time to changing background.

    bool is_obj_without_holes; //!< If TRUE we ignore holes within foreground blobs. Defaults to TRUE.
    int perform_morphing;     //!< Number of erode-dilate-erode foreground-blob cleanup iterations.
    //!< These erase one-pixel junk blobs and merge almost-touching blobs. Default value is 1.

    float alpha1; //!< How quickly we forget old background pixel values seen. Typically set to 0.1.
    float alpha2; //!< "Controls speed of feature learning". Depends on T. Typical value circa 0.005.
    float alpha3; //!< Alternate to alpha2, used (e.g.) for quicker initial convergence. Typical value 0.1.

    float delta;   //!< Affects color and color co-occurrence quantization, typically set to 2.
    float T;       //!< A percentage value which determines when new features can be recognized as new background. (Typically 0.9).
    float minArea; //!< Discard foreground blobs whose bounding box is smaller than this threshold.

    //! default Params
    FGDParams();
};

/** @brief Creates FGD Background Subtractor

@param params Algorithm's parameters. See @cite FGD2003 for explanation.
 */
CV_EXPORTS Ptr<cuda::BackgroundSubtractorFGD>
    createBackgroundSubtractorFGD(const FGDParams& params = FGDParams());

//
// Optical flow
//

//! Calculates optical flow for 2 images using block matching algorithm */
CV_EXPORTS void calcOpticalFlowBM(const GpuMat& prev, const GpuMat& curr,
                                  Size block_size, Size shift_size, Size max_range, bool use_previous,
                                  GpuMat& velx, GpuMat& vely, GpuMat& buf,
                                  Stream& stream = Stream::Null());

class CV_EXPORTS FastOpticalFlowBM
{
public:
    void operator ()(const GpuMat& I0, const GpuMat& I1, GpuMat& flowx, GpuMat& flowy, int search_window = 21, int block_window = 7, Stream& s = Stream::Null());

private:
    GpuMat buffer;
    GpuMat extended_I0;
    GpuMat extended_I1;
};

/** @brief Interpolates frames (images) using provided optical flow (displacement field).

@param frame0 First frame (32-bit floating point images, single channel).
@param frame1 Second frame. Must have the same type and size as frame0 .
@param fu Forward horizontal displacement.
@param fv Forward vertical displacement.
@param bu Backward horizontal displacement.
@param bv Backward vertical displacement.
@param pos New frame position.
@param newFrame Output image.
@param buf Temporary buffer, will have width x 6\*height size, CV_32FC1 type and contain 6
GpuMat: occlusion masks for first frame, occlusion masks for second, interpolated forward
horizontal flow, interpolated forward vertical flow, interpolated backward horizontal flow,
interpolated backward vertical flow.
@param stream Stream for the asynchronous version.
 */
CV_EXPORTS void interpolateFrames(const GpuMat& frame0, const GpuMat& frame1,
                                  const GpuMat& fu, const GpuMat& fv,
                                  const GpuMat& bu, const GpuMat& bv,
                                  float pos, GpuMat& newFrame, GpuMat& buf,
                                  Stream& stream = Stream::Null());

CV_EXPORTS void createOpticalFlowNeedleMap(const GpuMat& u, const GpuMat& v, GpuMat& vertex, GpuMat& colors);

//
// Labeling
//

//!performs labeling via graph cuts of a 2D regular 4-connected graph.
CV_EXPORTS void graphcut(GpuMat& terminals, GpuMat& leftTransp, GpuMat& rightTransp, GpuMat& top, GpuMat& bottom, GpuMat& labels,
                         GpuMat& buf, Stream& stream = Stream::Null());

//!performs labeling via graph cuts of a 2D regular 8-connected graph.
CV_EXPORTS void graphcut(GpuMat& terminals, GpuMat& leftTransp, GpuMat& rightTransp, GpuMat& top, GpuMat& topLeft, GpuMat& topRight,
                         GpuMat& bottom, GpuMat& bottomLeft, GpuMat& bottomRight,
                         GpuMat& labels,
                         GpuMat& buf, Stream& stream = Stream::Null());

//! compute mask for Generalized Flood fill componetns labeling.
CV_EXPORTS void connectivityMask(const GpuMat& image, GpuMat& mask, const cv::Scalar& lo, const cv::Scalar& hi, Stream& stream = Stream::Null());

//! performs connected componnents labeling.
CV_EXPORTS void labelComponents(const GpuMat& mask, GpuMat& components, int flags = 0, Stream& stream = Stream::Null());

//
// Calib3d
//

CV_EXPORTS void transformPoints(const GpuMat& src, const Mat& rvec, const Mat& tvec,
                                GpuMat& dst, Stream& stream = Stream::Null());

CV_EXPORTS void projectPoints(const GpuMat& src, const Mat& rvec, const Mat& tvec,
                              const Mat& camera_mat, const Mat& dist_coef, GpuMat& dst,
                              Stream& stream = Stream::Null());

/** @brief Finds the object pose from 3D-2D point correspondences.

@param object Single-row matrix of object points.
@param image Single-row matrix of image points.
@param camera_mat 3x3 matrix of intrinsic camera parameters.
@param dist_coef Distortion coefficients. See undistortPoints for details.
@param rvec Output 3D rotation vector.
@param tvec Output 3D translation vector.
@param use_extrinsic_guess Flag to indicate that the function must use rvec and tvec as an
initial transformation guess. It is not supported for now.
@param num_iters Maximum number of RANSAC iterations.
@param max_dist Euclidean distance threshold to detect whether point is inlier or not.
@param min_inlier_count Flag to indicate that the function must stop if greater or equal number
of inliers is achieved. It is not supported for now.
@param inliers Output vector of inlier indices.
 */
CV_EXPORTS void solvePnPRansac(const Mat& object, const Mat& image, const Mat& camera_mat,
                               const Mat& dist_coef, Mat& rvec, Mat& tvec, bool use_extrinsic_guess=false,
                               int num_iters=100, float max_dist=8.0, int min_inlier_count=100,
                               std::vector<int>* inliers=NULL);

//! @}

}}

#endif /* OPENCV_CUDALEGACY_HPP */
