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

#ifndef __OPENCV_VIDEOSTAB_INPAINTINT_HPP__
#define __OPENCV_VIDEOSTAB_INPAINTINT_HPP__

#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/videostab/optical_flow.hpp"
#include "opencv2/videostab/fast_marching.hpp"
#include "opencv2/videostab/global_motion.hpp"
#include "opencv2/photo/photo.hpp"

namespace cv
{
namespace videostab
{

class CV_EXPORTS InpainterBase
{
public:
    InpainterBase()
        : radius_(0), motionModel_(UNKNOWN), frames_(0), motions_(0),
          stabilizedFrames_(0), stabilizationMotions_(0) {}

    virtual ~InpainterBase() {}

    virtual void setRadius(int val) { radius_ = val; }
    virtual int radius() const { return radius_; }

    virtual void setMotionModel(MotionModel val) { motionModel_ = val; }
    virtual MotionModel motionModel() const { return motionModel_; }

    virtual void inpaint(int idx, Mat &frame, Mat &mask) = 0;


    // data from stabilizer

    virtual void setFrames(const std::vector<Mat> &val) { frames_ = &val; }
    virtual const std::vector<Mat>& frames() const { return *frames_; }

    virtual void setMotions(const std::vector<Mat> &val) { motions_ = &val; }
    virtual const std::vector<Mat>& motions() const { return *motions_; }

    virtual void setStabilizedFrames(const std::vector<Mat> &val) { stabilizedFrames_ = &val; }
    virtual const std::vector<Mat>& stabilizedFrames() const { return *stabilizedFrames_; }

    virtual void setStabilizationMotions(const std::vector<Mat> &val) { stabilizationMotions_ = &val; }
    virtual const std::vector<Mat>& stabilizationMotions() const { return *stabilizationMotions_; }

protected:
    int radius_;
    MotionModel motionModel_;
    const std::vector<Mat> *frames_;
    const std::vector<Mat> *motions_;
    const std::vector<Mat> *stabilizedFrames_;
    const std::vector<Mat> *stabilizationMotions_;
};

class CV_EXPORTS NullInpainter : public InpainterBase
{
public:
    virtual void inpaint(int /*idx*/, Mat &/*frame*/, Mat &/*mask*/) {}
};

class CV_EXPORTS InpaintingPipeline : public InpainterBase
{
public:
    void pushBack(Ptr<InpainterBase> inpainter) { inpainters_.push_back(inpainter); }
    bool empty() const { return inpainters_.empty(); }

    virtual void setRadius(int val);
    virtual void setMotionModel(MotionModel val);
    virtual void setFrames(const std::vector<Mat> &val);
    virtual void setMotions(const std::vector<Mat> &val);
    virtual void setStabilizedFrames(const std::vector<Mat> &val);
    virtual void setStabilizationMotions(const std::vector<Mat> &val);

    virtual void inpaint(int idx, Mat &frame, Mat &mask);

private:
    std::vector<Ptr<InpainterBase> > inpainters_;
};

class CV_EXPORTS ConsistentMosaicInpainter : public InpainterBase
{
public:
    ConsistentMosaicInpainter();

    void setStdevThresh(float val) { stdevThresh_ = val; }
    float stdevThresh() const { return stdevThresh_; }

    virtual void inpaint(int idx, Mat &frame, Mat &mask);

private:
    float stdevThresh_;
};

class CV_EXPORTS MotionInpainter : public InpainterBase
{
public:
    MotionInpainter();

    void setOptFlowEstimator(Ptr<IDenseOptFlowEstimator> val) { optFlowEstimator_ = val; }
    Ptr<IDenseOptFlowEstimator> optFlowEstimator() const { return optFlowEstimator_; }

    void setFlowErrorThreshold(float val) { flowErrorThreshold_ = val; }
    float flowErrorThreshold() const { return flowErrorThreshold_; }

    void setDistThreshold(float val) { distThresh_ = val; }
    float distThresh() const { return distThresh_; }

    void setBorderMode(int val) { borderMode_ = val; }
    int borderMode() const { return borderMode_; }

    virtual void inpaint(int idx, Mat &frame, Mat &mask);

private:
    FastMarchingMethod fmm_;
    Ptr<IDenseOptFlowEstimator> optFlowEstimator_;
    float flowErrorThreshold_;
    float distThresh_;
    int borderMode_;

    Mat frame1_, transformedFrame1_;
    Mat_<uchar> grayFrame_, transformedGrayFrame1_;
    Mat_<uchar> mask1_, transformedMask1_;
    Mat_<float> flowX_, flowY_, flowErrors_;
    Mat_<uchar> flowMask_;
};

class CV_EXPORTS ColorAverageInpainter : public InpainterBase
{
public:
    virtual void inpaint(int idx, Mat &frame, Mat &mask);

private:
    FastMarchingMethod fmm_;
};

class CV_EXPORTS ColorInpainter : public InpainterBase
{
public:
    ColorInpainter(int method = INPAINT_TELEA, double radius = 2.)
        : method_(method), radius_(radius) {}

    virtual void inpaint(int idx, Mat &frame, Mat &mask);

private:
    int method_;
    double radius_;
    Mat invMask_;
};

CV_EXPORTS void calcFlowMask(
        const Mat &flowX, const Mat &flowY, const Mat &errors, float maxError,
        const Mat &mask0, const Mat &mask1, Mat &flowMask);

CV_EXPORTS void completeFrameAccordingToFlow(
        const Mat &flowMask, const Mat &flowX, const Mat &flowY, const Mat &frame1, const Mat &mask1,
        float distThresh, Mat& frame0, Mat &mask0);

} // namespace videostab
} // namespace cv

#endif
