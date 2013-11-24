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

#ifndef __OPENCV_CUDASTEREO_HPP__
#define __OPENCV_CUDASTEREO_HPP__

#ifndef __cplusplus
#  error cudastereo.hpp header must be compiled as C++
#endif

#include "opencv2/core/cuda.hpp"
#include "opencv2/calib3d.hpp"

namespace cv { namespace cuda {

/////////////////////////////////////////
// StereoBM

class CV_EXPORTS StereoBM : public cv::StereoBM
{
public:
    using cv::StereoBM::compute;

    virtual void compute(InputArray left, InputArray right, OutputArray disparity, Stream& stream) = 0;
};

CV_EXPORTS Ptr<cuda::StereoBM> createStereoBM(int numDisparities = 64, int blockSize = 19);

/////////////////////////////////////////
// StereoBeliefPropagation

//! "Efficient Belief Propagation for Early Vision" P.Felzenszwalb
class CV_EXPORTS StereoBeliefPropagation : public cv::StereoMatcher
{
public:
    using cv::StereoMatcher::compute;

    virtual void compute(InputArray left, InputArray right, OutputArray disparity, Stream& stream) = 0;

    //! version for user specified data term
    virtual void compute(InputArray data, OutputArray disparity, Stream& stream = Stream::Null()) = 0;

    //! number of BP iterations on each level
    virtual int getNumIters() const = 0;
    virtual void setNumIters(int iters) = 0;

    //! number of levels
    virtual int getNumLevels() const = 0;
    virtual void setNumLevels(int levels) = 0;

    //! truncation of data cost
    virtual double getMaxDataTerm() const = 0;
    virtual void setMaxDataTerm(double max_data_term) = 0;

    //! data weight
    virtual double getDataWeight() const = 0;
    virtual void setDataWeight(double data_weight) = 0;

    //! truncation of discontinuity cost
    virtual double getMaxDiscTerm() const = 0;
    virtual void setMaxDiscTerm(double max_disc_term) = 0;

    //! discontinuity single jump
    virtual double getDiscSingleJump() const = 0;
    virtual void setDiscSingleJump(double disc_single_jump) = 0;

    //! type for messages (CV_16SC1 or CV_32FC1)
    virtual int getMsgType() const = 0;
    virtual void setMsgType(int msg_type) = 0;

    static void estimateRecommendedParams(int width, int height, int& ndisp, int& iters, int& levels);
};

CV_EXPORTS Ptr<cuda::StereoBeliefPropagation>
    createStereoBeliefPropagation(int ndisp = 64, int iters = 5, int levels = 5, int msg_type = CV_32F);

/////////////////////////////////////////
// StereoConstantSpaceBP

//! "A Constant-Space Belief Propagation Algorithm for Stereo Matching"
//! Qingxiong Yang, Liang Wang, Narendra Ahuja
//! http://vision.ai.uiuc.edu/~qyang6/
class CV_EXPORTS StereoConstantSpaceBP : public cuda::StereoBeliefPropagation
{
public:
    //! number of active disparity on the first level
    virtual int getNrPlane() const = 0;
    virtual void setNrPlane(int nr_plane) = 0;

    virtual bool getUseLocalInitDataCost() const = 0;
    virtual void setUseLocalInitDataCost(bool use_local_init_data_cost) = 0;

    static void estimateRecommendedParams(int width, int height, int& ndisp, int& iters, int& levels, int& nr_plane);
};

CV_EXPORTS Ptr<cuda::StereoConstantSpaceBP>
    createStereoConstantSpaceBP(int ndisp = 128, int iters = 8, int levels = 4, int nr_plane = 4, int msg_type = CV_32F);

/////////////////////////////////////////
// DisparityBilateralFilter

//! Disparity map refinement using joint bilateral filtering given a single color image.
//! Qingxiong Yang, Liang Wang, Narendra Ahuja
//! http://vision.ai.uiuc.edu/~qyang6/
class CV_EXPORTS DisparityBilateralFilter : public cv::Algorithm
{
public:
    //! the disparity map refinement operator. Refine disparity map using joint bilateral filtering given a single color image.
    //! disparity must have CV_8U or CV_16S type, image must have CV_8UC1 or CV_8UC3 type.
    virtual void apply(InputArray disparity, InputArray image, OutputArray dst, Stream& stream = Stream::Null()) = 0;

    virtual int getNumDisparities() const = 0;
    virtual void setNumDisparities(int numDisparities) = 0;

    virtual int getRadius() const = 0;
    virtual void setRadius(int radius) = 0;

    virtual int getNumIters() const = 0;
    virtual void setNumIters(int iters) = 0;

    //! truncation of data continuity
    virtual double getEdgeThreshold() const = 0;
    virtual void setEdgeThreshold(double edge_threshold) = 0;

    //! truncation of disparity continuity
    virtual double getMaxDiscThreshold() const = 0;
    virtual void setMaxDiscThreshold(double max_disc_threshold) = 0;

    //! filter range sigma
    virtual double getSigmaRange() const = 0;
    virtual void setSigmaRange(double sigma_range) = 0;
};

CV_EXPORTS Ptr<cuda::DisparityBilateralFilter>
    createDisparityBilateralFilter(int ndisp = 64, int radius = 3, int iters = 1);

/////////////////////////////////////////
// Utility

//! Reprojects disparity image to 3D space.
//! Supports CV_8U and CV_16S types of input disparity.
//! The output is a 3- or 4-channel floating-point matrix.
//! Each element of this matrix will contain the 3D coordinates of the point (x,y,z,1), computed from the disparity map.
//! Q is the 4x4 perspective transformation matrix that can be obtained with cvStereoRectify.
CV_EXPORTS void reprojectImageTo3D(InputArray disp, OutputArray xyzw, InputArray Q, int dst_cn = 4, Stream& stream = Stream::Null());

//! Does coloring of disparity image: [0..ndisp) -> [0..240, 1, 1] in HSV.
//! Supported types of input disparity: CV_8U, CV_16S.
//! Output disparity has CV_8UC4 type in BGRA format (alpha = 255).
CV_EXPORTS void drawColorDisp(InputArray src_disp, OutputArray dst_disp, int ndisp, Stream& stream = Stream::Null());

}} // namespace cv { namespace cuda {

#endif /* __OPENCV_CUDASTEREO_HPP__ */
