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

#ifndef __OPENCV_GPUBGSEGM_HPP__
#define __OPENCV_GPUBGSEGM_HPP__

#ifndef __cplusplus
#  error gpubgsegm.hpp header must be compiled as C++
#endif

#include "opencv2/core/gpu.hpp"
#include "opencv2/video/background_segm.hpp"

namespace cv { namespace gpu {

////////////////////////////////////////////////////
// MOG

class CV_EXPORTS BackgroundSubtractorMOG : public cv::BackgroundSubtractorMOG
{
public:
    using cv::BackgroundSubtractorMOG::apply;
    using cv::BackgroundSubtractorMOG::getBackgroundImage;

    virtual void apply(InputArray image, OutputArray fgmask, double learningRate, Stream& stream) = 0;

    virtual void getBackgroundImage(OutputArray backgroundImage, Stream& stream) const = 0;
};

CV_EXPORTS Ptr<gpu::BackgroundSubtractorMOG>
    createBackgroundSubtractorMOG(int history = 200, int nmixtures = 5,
                                  double backgroundRatio = 0.7, double noiseSigma = 0);

////////////////////////////////////////////////////
// MOG2

class CV_EXPORTS BackgroundSubtractorMOG2 : public cv::BackgroundSubtractorMOG2
{
public:
    using cv::BackgroundSubtractorMOG2::apply;
    using cv::BackgroundSubtractorMOG2::getBackgroundImage;

    virtual void apply(InputArray image, OutputArray fgmask, double learningRate, Stream& stream) = 0;

    virtual void getBackgroundImage(OutputArray backgroundImage, Stream& stream) const = 0;
};

CV_EXPORTS Ptr<gpu::BackgroundSubtractorMOG2>
    createBackgroundSubtractorMOG2(int history = 500, double varThreshold = 16,
                                   bool detectShadows = true);

////////////////////////////////////////////////////
// GMG

class CV_EXPORTS BackgroundSubtractorGMG : public cv::BackgroundSubtractorGMG
{
public:
    using cv::BackgroundSubtractorGMG::apply;

    virtual void apply(InputArray image, OutputArray fgmask, double learningRate, Stream& stream) = 0;
};

CV_EXPORTS Ptr<gpu::BackgroundSubtractorGMG>
    createBackgroundSubtractorGMG(int initializationFrames = 120, double decisionThreshold = 0.8);

////////////////////////////////////////////////////
// FGD

/**
 * Foreground Object Detection from Videos Containing Complex Background.
 * Liyuan Li, Weimin Huang, Irene Y.H. Gu, and Qi Tian.
 * ACM MM2003 9p
 */
class CV_EXPORTS BackgroundSubtractorFGD : public cv::BackgroundSubtractor
{
public:
    virtual void getForegroundRegions(OutputArrayOfArrays foreground_regions) = 0;
};

struct CV_EXPORTS FGDParams
{
    int Lc;  // Quantized levels per 'color' component. Power of two, typically 32, 64 or 128.
    int N1c; // Number of color vectors used to model normal background color variation at a given pixel.
    int N2c; // Number of color vectors retained at given pixel.  Must be > N1c, typically ~ 5/3 of N1c.
    // Used to allow the first N1c vectors to adapt over time to changing background.

    int Lcc;  // Quantized levels per 'color co-occurrence' component.  Power of two, typically 16, 32 or 64.
    int N1cc; // Number of color co-occurrence vectors used to model normal background color variation at a given pixel.
    int N2cc; // Number of color co-occurrence vectors retained at given pixel.  Must be > N1cc, typically ~ 5/3 of N1cc.
    // Used to allow the first N1cc vectors to adapt over time to changing background.

    bool is_obj_without_holes; // If TRUE we ignore holes within foreground blobs. Defaults to TRUE.
    int perform_morphing;     // Number of erode-dilate-erode foreground-blob cleanup iterations.
    // These erase one-pixel junk blobs and merge almost-touching blobs. Default value is 1.

    float alpha1; // How quickly we forget old background pixel values seen. Typically set to 0.1.
    float alpha2; // "Controls speed of feature learning". Depends on T. Typical value circa 0.005.
    float alpha3; // Alternate to alpha2, used (e.g.) for quicker initial convergence. Typical value 0.1.

    float delta;   // Affects color and color co-occurrence quantization, typically set to 2.
    float T;       // A percentage value which determines when new features can be recognized as new background. (Typically 0.9).
    float minArea; // Discard foreground blobs whose bounding box is smaller than this threshold.

    // default Params
    FGDParams();
};

CV_EXPORTS Ptr<gpu::BackgroundSubtractorFGD>
    createBackgroundSubtractorFGD(const FGDParams& params = FGDParams());

}} // namespace cv { namespace gpu {

#endif /* __OPENCV_GPUBGSEGM_HPP__ */
