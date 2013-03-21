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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#ifndef __OPENCV_BACKGROUND_SEGM_HPP__
#define __OPENCV_BACKGROUND_SEGM_HPP__

#include "opencv2/core.hpp"
#include <list>
namespace cv
{

/*!
 The Base Class for Background/Foreground Segmentation

 The class is only used to define the common interface for
 the whole family of background/foreground segmentation algorithms.
*/
class BackgroundSubtractor : public Algorithm
{
public:
    //! the update operator that takes the next video frame and returns the current foreground mask as 8-bit binary image.
    virtual void apply(InputArray image, OutputArray fgmask, double learningRate=-1) = 0;

    //! computes a background image
    virtual void getBackgroundImage(OutputArray backgroundImage) const = 0;
};


/*!
 Gaussian Mixture-based Backbround/Foreground Segmentation Algorithm

 The class implements the following algorithm:
 "An improved adaptive background mixture model for real-time tracking with shadow detection"
 P. KadewTraKuPong and R. Bowden,
 Proc. 2nd European Workshp on Advanced Video-Based Surveillance Systems, 2001."
 http://personal.ee.surrey.ac.uk/Personal/R.Bowden/publications/avbs01/avbs01.pdf

*/
class BackgroundSubtractorMOG : public BackgroundSubtractor
{
public:
    virtual int getHistory() const = 0;
    virtual void setHistory(int nframes) = 0;

    virtual int getNMixtures() const = 0;
    virtual void setNMixtures(int nmix) = 0;

    virtual double getBackgroundRatio() const = 0;
    virtual void setBackgroundRatio(double backgroundRatio) = 0;

    virtual double getNoiseSigma() const = 0;
    virtual void setNoiseSigma(double noiseSigma) = 0;
};

CV_EXPORTS Ptr<BackgroundSubtractorMOG>
    createBackgroundSubtractorMOG(int history=200, int nmixtures=5,
                                  double backgroundRatio=0.7, double noiseSigma=0);


/*!
 The class implements the following algorithm:
 "Improved adaptive Gausian mixture model for background subtraction"
 Z.Zivkovic
 International Conference Pattern Recognition, UK, August, 2004.
 http://www.zoranz.net/Publications/zivkovic2004ICPR.pdf
 */
class BackgroundSubtractorMOG2 : public BackgroundSubtractor
{
public:
    virtual int getHistory() const = 0;
    virtual void setHistory(int nframes) = 0;

    virtual int getNMixtures() const = 0;
    virtual void setNMixtures(int nmix) = 0;

    virtual double getBackgroundRatio() const = 0;
    virtual void setBackgroundRatio(double backgroundRatio) = 0;

    virtual double getVarThreshold() const = 0;
    virtual void setVarThreshold(double varThreshold) = 0;

    virtual double getVarThresholdGen() const = 0;
    virtual void setVarThresholdGen(double varThresholdGen) = 0;

    virtual double getVarInit() const = 0;
    virtual void setVarInit(double varInit) = 0;

    virtual double getVarMin() const = 0;
    virtual void setVarMin(double varMin) = 0;

    virtual double getVarMax() const = 0;
    virtual void setVarMax(double varMax) = 0;

    virtual double getComplexityReductionThreshold() const = 0;
    virtual void setComplexityReductionThreshold(double ct) = 0;

    virtual bool getDetectShadows() const = 0;
    virtual void setDetectShadows(bool detectshadows) = 0;

    virtual int getShadowValue() const = 0;
    virtual void setShadowValue(int value) = 0;

    virtual double getShadowThreshold() const = 0;
    virtual void setShadowThreshold(double value) = 0;
};

CV_EXPORTS Ptr<BackgroundSubtractorMOG2>
    createBackgroundSubtractorMOG2(int history=500, double varThreshold=16,
                                   bool detectShadows=true);

/**
 * Background Subtractor module. Takes a series of images and returns a sequence of mask (8UC1)
 * images of the same size, where 255 indicates Foreground and 0 represents Background.
 * This class implements an algorithm described in "Visual Tracking of Human Visitors under
 * Variable-Lighting Conditions for a Responsive Audio Art Installation," A. Godbehere,
 * A. Matsukawa, K. Goldberg, American Control Conference, Montreal, June 2012.
 */
class BackgroundSubtractorGMG : public BackgroundSubtractor
{
public:
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


CV_EXPORTS Ptr<BackgroundSubtractorGMG> createBackgroundSubtractorGMG(int initializationFrames=120,
                                                                        double decisionThreshold=0.8);

}

#endif
