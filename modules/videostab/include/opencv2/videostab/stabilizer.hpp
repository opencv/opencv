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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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

#ifndef __OPENCV_VIDEOSTAB_STABILIZER_HPP__
#define __OPENCV_VIDEOSTAB_STABILIZER_HPP__

#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videostab/global_motion.hpp"
#include "opencv2/videostab/motion_stabilizing.hpp"
#include "opencv2/videostab/frame_source.hpp"
#include "opencv2/videostab/log.hpp"
#include "opencv2/videostab/inpainting.hpp"
#include "opencv2/videostab/deblurring.hpp"

namespace cv
{
namespace videostab
{

class CV_EXPORTS StabilizerBase
{
public:
    virtual ~StabilizerBase() {}

    void setLog(Ptr<ILog> log) { log_ = log; }
    Ptr<ILog> log() const { return log_; }

    void setRadius(int val) { radius_ = val; }
    int radius() const { return radius_; }

    void setFrameSource(Ptr<IFrameSource> val) { frameSource_ = val; }
    Ptr<IFrameSource> frameSource() const { return frameSource_; }

    void setMotionEstimator(Ptr<IGlobalMotionEstimator> val) { motionEstimator_ = val; }
    Ptr<IGlobalMotionEstimator> motionEstimator() const { return motionEstimator_; }

    void setDeblurer(Ptr<DeblurerBase> val) { deblurer_ = val; }
    Ptr<DeblurerBase> deblurrer() const { return deblurer_; }

    void setTrimRatio(float val) { trimRatio_ = val; }
    float trimRatio() const { return trimRatio_; }

    void setCorrectionForInclusion(bool val) { doCorrectionForInclusion_ = val; }
    bool doCorrectionForInclusion() const { return doCorrectionForInclusion_; }

    void setBorderMode(int val) { borderMode_ = val; }
    int borderMode() const { return borderMode_; }

    void setInpainter(Ptr<InpainterBase> val) { inpainter_ = val; }
    Ptr<InpainterBase> inpainter() const { return inpainter_; }

protected:
    StabilizerBase();

    void setUp(int cacheSize, const Mat &frame);
    Mat nextStabilizedFrame();
    bool doOneIteration();
    void stabilizeFrame(const Mat &stabilizationMotion);

    virtual void setUp(Mat &firstFrame) = 0;
    virtual void stabilizeFrame() = 0;
    virtual void estimateMotion() = 0;

    Ptr<ILog> log_;
    Ptr<IFrameSource> frameSource_;
    Ptr<IGlobalMotionEstimator> motionEstimator_;
    Ptr<DeblurerBase> deblurer_;
    Ptr<InpainterBase> inpainter_;
    int radius_;
    float trimRatio_;
    bool doCorrectionForInclusion_;
    int borderMode_;

    Size frameSize_;
    Mat frameMask_;
    int curPos_;
    int curStabilizedPos_;
    bool doDeblurring_;
    Mat preProcessedFrame_;
    bool doInpainting_;
    Mat inpaintingMask_;
    std::vector<Mat> frames_;
    std::vector<Mat> motions_; // motions_[i] is the motion from i-th to i+1-th frame
    std::vector<float> blurrinessRates_;
    std::vector<Mat> stabilizedFrames_;
    std::vector<Mat> stabilizedMasks_;
    std::vector<Mat> stabilizationMotions_;
};

class CV_EXPORTS OnePassStabilizer : public StabilizerBase, public IFrameSource
{
public:
    OnePassStabilizer();

    void setMotionFilter(Ptr<MotionFilterBase> val) { motionFilter_ = val; }
    Ptr<MotionFilterBase> motionFilter() const { return motionFilter_; }

    virtual void reset() { resetImpl(); }
    virtual Mat nextFrame() { return nextStabilizedFrame(); }

private:
    void resetImpl();

    virtual void setUp(Mat &firstFrame);
    virtual void estimateMotion();
    virtual void stabilizeFrame();

    Ptr<MotionFilterBase> motionFilter_;
};

class CV_EXPORTS TwoPassStabilizer : public StabilizerBase, public IFrameSource
{
public:
    TwoPassStabilizer();

    void setMotionStabilizer(Ptr<IMotionStabilizer> val) { motionStabilizer_ = val; }
    Ptr<IMotionStabilizer> motionStabilizer() const { return motionStabilizer_; }

    void setEstimateTrimRatio(bool val) { mustEstTrimRatio_ = val; }
    bool mustEstimateTrimaRatio() const { return mustEstTrimRatio_; }

    virtual void reset() { resetImpl(); }
    virtual Mat nextFrame();

private:
    void resetImpl();
    void runPrePassIfNecessary();

    virtual void setUp(Mat &firstFrame);
    virtual void estimateMotion() { /* do nothing as motion was estimation in pre-pass */ }
    virtual void stabilizeFrame();

    Ptr<IMotionStabilizer> motionStabilizer_;
    bool mustEstTrimRatio_;

    int frameCount_;
    bool isPrePassDone_;
};

} // namespace videostab
} // namespace cv

#endif
