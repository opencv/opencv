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
#include "opencv2/videostab/motion_filtering.hpp"
#include "opencv2/videostab/frame_source.hpp"
#include "opencv2/videostab/log.hpp"
#include "opencv2/videostab/inpainting.hpp"
#include "opencv2/videostab/deblurring.hpp"

namespace cv
{
namespace videostab
{

class CV_EXPORTS Stabilizer : public IFrameSource
{
public:
    Stabilizer();

    void setLog(Ptr<ILog> log) { log_ = log; }
    Ptr<ILog> log() const { return log_; }

    void setFrameSource(Ptr<IFrameSource> val) { frameSource_ = val; reset(); }
    Ptr<IFrameSource> frameSource() const { return frameSource_; }

    void setMotionEstimator(Ptr<IGlobalMotionEstimator> val) { motionEstimator_ = val; }
    Ptr<IGlobalMotionEstimator> motionEstimator() const { return motionEstimator_; }

    void setMotionFilter(Ptr<IMotionFilter> val) { motionFilter_ = val; reset(); }
    Ptr<IMotionFilter> motionFilter() const { return motionFilter_; }

    void setDeblurer(Ptr<IDeblurer> val) { deblurer_ = val; reset(); }
    Ptr<IDeblurer> deblurrer() const { return deblurer_; }

    void setEstimateTrimRatio(bool val) { mustEstimateTrimRatio_ = val; reset(); }
    bool mustEstimateTrimRatio() const { return mustEstimateTrimRatio_; }

    void setTrimRatio(float val) { trimRatio_ = val; reset(); }
    int trimRatio() const { return trimRatio_; }

    void setInclusionConstraint(bool val) { inclusionConstraint_ = val; }
    bool inclusionConstraint() const { return inclusionConstraint_; }

    void setBorderMode(int val) { borderMode_ = val; }
    int borderMode() const { return borderMode_; }

    void setInpainter(Ptr<IInpainter> val) { inpainter_ = val; reset(); }
    Ptr<IInpainter> inpainter() const { return inpainter_; }

    virtual void reset();
    virtual Mat nextFrame();

private:
    void estimateMotionsAndTrimRatio();
    void processFirstFrame(Mat &frame);
    bool processNextFrame();
    void stabilizeFrame(int idx);

    Ptr<IFrameSource> frameSource_;
    Ptr<IGlobalMotionEstimator> motionEstimator_;
    Ptr<IMotionFilter> motionFilter_;
    Ptr<IDeblurer> deblurer_;
    Ptr<IInpainter> inpainter_;
    bool mustEstimateTrimRatio_;
    float trimRatio_;
    bool inclusionConstraint_;
    int borderMode_;    
    Ptr<ILog> log_;

    Size frameSize_;
    Mat frameMask_;
    int radius_;
    int curPos_;
    int curStabilizedPos_;
    bool auxPassWasDone_;
    bool doDeblurring_;
    Mat preProcessedFrame_;
    bool doInpainting_;
    Mat inpaintingMask_;
    std::vector<Mat> frames_;
    std::vector<Mat> motions_; // motions_[i] is the motion from i to i+1 frame
    std::vector<float> blurrinessRates_;
    std::vector<Mat> stabilizedFrames_;
    std::vector<Mat> stabilizedMasks_;
    std::vector<Mat> stabilizationMotions_;
};

} // namespace videostab
} // namespace cv

#endif
