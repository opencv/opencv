/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
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

namespace cv
{

/*!
 The Base Class for Background/Foreground Segmentation

 The class is only used to define the common interface for
 the whole family of background/foreground segmentation algorithms.
*/
class CV_EXPORTS_W BackgroundSubtractor : public Algorithm
{
public:
    //! the update operator that takes the next video frame and returns the current foreground mask as 8-bit binary image.
    CV_WRAP virtual void apply(InputArray image, OutputArray fgmask, double learningRate=-1) = 0;

    //! computes a background image
    CV_WRAP virtual void getBackgroundImage(OutputArray backgroundImage) const = 0;
};


/*!
 The class implements the following algorithm:
 "Improved adaptive Gausian mixture model for background subtraction"
 Z.Zivkovic
 International Conference Pattern Recognition, UK, August, 2004.
 http://www.zoranz.net/Publications/zivkovic2004ICPR.pdf
 */
class CV_EXPORTS_W BackgroundSubtractorMOG2 : public BackgroundSubtractor
{
public:
    CV_WRAP virtual int getHistory() const = 0;
    CV_WRAP virtual void setHistory(int history) = 0;

    CV_WRAP virtual int getNMixtures() const = 0;
    CV_WRAP virtual void setNMixtures(int nmixtures) = 0;//needs reinitialization!

    CV_WRAP virtual double getBackgroundRatio() const = 0;
    CV_WRAP virtual void setBackgroundRatio(double ratio) = 0;

    CV_WRAP virtual double getVarThreshold() const = 0;
    CV_WRAP virtual void setVarThreshold(double varThreshold) = 0;

    CV_WRAP virtual double getVarThresholdGen() const = 0;
    CV_WRAP virtual void setVarThresholdGen(double varThresholdGen) = 0;

    CV_WRAP virtual double getVarInit() const = 0;
    CV_WRAP virtual void setVarInit(double varInit) = 0;

    CV_WRAP virtual double getVarMin() const = 0;
    CV_WRAP virtual void setVarMin(double varMin) = 0;

    CV_WRAP virtual double getVarMax() const = 0;
    CV_WRAP virtual void setVarMax(double varMax) = 0;

    CV_WRAP virtual double getComplexityReductionThreshold() const = 0;
    CV_WRAP virtual void setComplexityReductionThreshold(double ct) = 0;

    CV_WRAP virtual bool getDetectShadows() const = 0;
    CV_WRAP virtual void setDetectShadows(bool detectShadows) = 0;

    CV_WRAP virtual int getShadowValue() const = 0;
    CV_WRAP virtual void setShadowValue(int value) = 0;

    CV_WRAP virtual double getShadowThreshold() const = 0;
    CV_WRAP virtual void setShadowThreshold(double threshold) = 0;
};

CV_EXPORTS_W Ptr<BackgroundSubtractorMOG2>
    createBackgroundSubtractorMOG2(int history=500, double varThreshold=16,
                                   bool detectShadows=true);

/*!
 The class implements the K nearest neigbours algorithm from:
 "Efficient Adaptive Density Estimation per Image Pixel for the Task of Background Subtraction"
 Z.Zivkovic, F. van der Heijden
 Pattern Recognition Letters, vol. 27, no. 7, pages 773-780, 2006
 http://www.zoranz.net/Publications/zivkovicPRL2006.pdf

 Fast for small foreground object. Results on the benchmark data is at http://www.changedetection.net.
*/

class CV_EXPORTS_W BackgroundSubtractorKNN : public BackgroundSubtractor
{
public:
    CV_WRAP virtual int getHistory() const = 0;
    CV_WRAP virtual void setHistory(int history) = 0;

    CV_WRAP virtual int getNSamples() const = 0;
    CV_WRAP virtual void setNSamples(int _nN) = 0;//needs reinitialization!

    CV_WRAP virtual double getDist2Threshold() const = 0;
    CV_WRAP virtual void setDist2Threshold(double _dist2Threshold) = 0;

    CV_WRAP virtual int getkNNSamples() const = 0;
    CV_WRAP virtual void setkNNSamples(int _nkNN) = 0;

    CV_WRAP virtual bool getDetectShadows() const = 0;
    CV_WRAP virtual void setDetectShadows(bool detectShadows) = 0;

    CV_WRAP virtual int getShadowValue() const = 0;
    CV_WRAP virtual void setShadowValue(int value) = 0;

    CV_WRAP virtual double getShadowThreshold() const = 0;
    CV_WRAP virtual void setShadowThreshold(double threshold) = 0;
};

CV_EXPORTS_W Ptr<BackgroundSubtractorKNN>
    createBackgroundSubtractorKNN(int history=500, double dist2Threshold=400.0,
                                   bool detectShadows=true);

} // cv

#endif
