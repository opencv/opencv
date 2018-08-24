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

#ifndef OPENCV_CUDABGSEGM_HPP
#define OPENCV_CUDABGSEGM_HPP

#ifndef __cplusplus
#  error cudabgsegm.hpp header must be compiled as C++
#endif

#include "opencv2/core/cuda.hpp"
#include "opencv2/video/background_segm.hpp"

/**
  @addtogroup cuda
  @{
    @defgroup cudabgsegm Background Segmentation
  @}
 */

namespace cv { namespace cuda {

//! @addtogroup cudabgsegm
//! @{

////////////////////////////////////////////////////
// MOG

/** @brief Gaussian Mixture-based Background/Foreground Segmentation Algorithm.

The class discriminates between foreground and background pixels by building and maintaining a model
of the background. Any pixel which does not fit this model is then deemed to be foreground. The
class implements algorithm described in @cite MOG2001 .

@sa BackgroundSubtractorMOG

@note
   -   An example on gaussian mixture based background/foreground segmantation can be found at
        opencv_source_code/samples/gpu/bgfg_segm.cpp
 */
class CV_EXPORTS_W BackgroundSubtractorMOG : public cv::BackgroundSubtractor
{
public:

    using cv::BackgroundSubtractor::apply;
    CV_WRAP virtual void apply(InputArray image, OutputArray fgmask, double learningRate, Stream& stream) = 0;

    using cv::BackgroundSubtractor::getBackgroundImage;
    CV_WRAP virtual void getBackgroundImage(OutputArray backgroundImage, Stream& stream) const = 0;

    CV_WRAP virtual int getHistory() const = 0;
    CV_WRAP virtual void setHistory(int nframes) = 0;

    CV_WRAP virtual int getNMixtures() const = 0;
    CV_WRAP virtual void setNMixtures(int nmix) = 0;

    CV_WRAP virtual double getBackgroundRatio() const = 0;
    CV_WRAP virtual void setBackgroundRatio(double backgroundRatio) = 0;

    CV_WRAP virtual double getNoiseSigma() const = 0;
    CV_WRAP virtual void setNoiseSigma(double noiseSigma) = 0;
};

/** @brief Creates mixture-of-gaussian background subtractor

@param history Length of the history.
@param nmixtures Number of Gaussian mixtures.
@param backgroundRatio Background ratio.
@param noiseSigma Noise strength (standard deviation of the brightness or each color channel). 0
means some automatic value.
 */
CV_EXPORTS_W Ptr<cuda::BackgroundSubtractorMOG>
    createBackgroundSubtractorMOG(int history = 200, int nmixtures = 5,
                                  double backgroundRatio = 0.7, double noiseSigma = 0);

////////////////////////////////////////////////////
// MOG2

/** @brief Gaussian Mixture-based Background/Foreground Segmentation Algorithm.

The class discriminates between foreground and background pixels by building and maintaining a model
of the background. Any pixel which does not fit this model is then deemed to be foreground. The
class implements algorithm described in @cite Zivkovic2004 .

@sa BackgroundSubtractorMOG2
 */
class CV_EXPORTS_W BackgroundSubtractorMOG2 : public cv::BackgroundSubtractorMOG2
{
public:
    using cv::BackgroundSubtractorMOG2::apply;
    using cv::BackgroundSubtractorMOG2::getBackgroundImage;

    CV_WRAP virtual void apply(InputArray image, OutputArray fgmask, double learningRate, Stream& stream) = 0;

    CV_WRAP virtual void getBackgroundImage(OutputArray backgroundImage, Stream& stream) const = 0;
};

/** @brief Creates MOG2 Background Subtractor

@param history Length of the history.
@param varThreshold Threshold on the squared Mahalanobis distance between the pixel and the model
to decide whether a pixel is well described by the background model. This parameter does not
affect the background update.
@param detectShadows If true, the algorithm will detect shadows and mark them. It decreases the
speed a bit, so if you do not need this feature, set the parameter to false.
 */
CV_EXPORTS_W Ptr<cuda::BackgroundSubtractorMOG2>
    createBackgroundSubtractorMOG2(int history = 500, double varThreshold = 16,
                                   bool detectShadows = true);

//! @}

}} // namespace cv { namespace cuda {

#endif /* OPENCV_CUDABGSEGM_HPP */
