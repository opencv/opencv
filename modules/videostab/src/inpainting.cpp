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

#include "precomp.hpp"
#include <queue>
#include "opencv2/videostab/inpainting.hpp"
#include "opencv2/videostab/global_motion.hpp"
#include "opencv2/videostab/fast_marching.hpp"
#include "opencv2/videostab/ring_buffer.hpp"
#include "opencv2/opencv_modules.hpp"

namespace cv
{
namespace videostab
{

void InpaintingPipeline::setRadius(int val)
{
    for (size_t i = 0; i < inpainters_.size(); ++i)
        inpainters_[i]->setRadius(val);
    InpainterBase::setRadius(val);
}


void InpaintingPipeline::setFrames(const std::vector<Mat> &val)
{
    for (size_t i = 0; i < inpainters_.size(); ++i)
        inpainters_[i]->setFrames(val);
    InpainterBase::setFrames(val);
}


void InpaintingPipeline::setMotionModel(MotionModel val)
{
    for (size_t i = 0; i < inpainters_.size(); ++i)
        inpainters_[i]->setMotionModel(val);
    InpainterBase::setMotionModel(val);
}


void InpaintingPipeline::setMotions(const std::vector<Mat> &val)
{
    for (size_t i = 0; i < inpainters_.size(); ++i)
        inpainters_[i]->setMotions(val);
    InpainterBase::setMotions(val);
}


void InpaintingPipeline::setStabilizedFrames(const std::vector<Mat> &val)
{
    for (size_t i = 0; i < inpainters_.size(); ++i)
        inpainters_[i]->setStabilizedFrames(val);
    InpainterBase::setStabilizedFrames(val);
}


void InpaintingPipeline::setStabilizationMotions(const std::vector<Mat> &val)
{
    for (size_t i = 0; i < inpainters_.size(); ++i)
        inpainters_[i]->setStabilizationMotions(val);
    InpainterBase::setStabilizationMotions(val);
}


void InpaintingPipeline::inpaint(int idx, Mat &frame, Mat &mask)
{
    CV_INSTRUMENT_REGION()

    for (size_t i = 0; i < inpainters_.size(); ++i)
        inpainters_[i]->inpaint(idx, frame, mask);
}


struct Pixel3
{
    float intens;
    Point3_<uchar> color;
    bool operator <(const Pixel3 &other) const { return intens < other.intens; }
};


ConsistentMosaicInpainter::ConsistentMosaicInpainter()
{
    setStdevThresh(20.f);
}


void ConsistentMosaicInpainter::inpaint(int idx, Mat &frame, Mat &mask)
{
    CV_INSTRUMENT_REGION()

    CV_Assert(frame.type() == CV_8UC3);
    CV_Assert(mask.size() == frame.size() && mask.type() == CV_8U);

    Mat invS = at(idx, *stabilizationMotions_).inv();
    std::vector<Mat_<float> > vmotions(2*radius_ + 1);
    for (int i = -radius_; i <= radius_; ++i)
        vmotions[radius_ + i] = getMotion(idx, idx + i, *motions_) * invS;

    int n;
    float mean, var;
    std::vector<Pixel3> pixels(2*radius_ + 1);

    Mat_<Point3_<uchar> > frame_(frame);
    Mat_<uchar> mask_(mask);

    for (int y = 0; y < mask.rows; ++y)
    {
        for (int x = 0; x < mask.cols; ++x)
        {
            if (!mask_(y, x))
            {
                n = 0;
                mean = 0;
                var = 0;

                for (int i = -radius_; i <= radius_; ++i)
                {
                    const Mat_<Point3_<uchar> > &framei = at(idx + i, *frames_);
                    const Mat_<float> &Mi = vmotions[radius_ + i];
                    int xi = cvRound(Mi(0,0)*x + Mi(0,1)*y + Mi(0,2));
                    int yi = cvRound(Mi(1,0)*x + Mi(1,1)*y + Mi(1,2));
                    if (xi >= 0 && xi < framei.cols && yi >= 0 && yi < framei.rows)
                    {
                        pixels[n].color = framei(yi, xi);
                        mean += pixels[n].intens = intensity(pixels[n].color);
                        n++;
                    }
                }

                if (n > 0)
                {
                    mean /= n;
                    for (int i = 0; i < n; ++i)
                        var += sqr(pixels[i].intens - mean);
                    var /= std::max(n - 1, 1);

                    if (var < stdevThresh_ * stdevThresh_)
                    {
                        std::sort(pixels.begin(), pixels.begin() + n);
                        int nh = (n-1)/2;
                        int c1 = pixels[nh].color.x;
                        int c2 = pixels[nh].color.y;
                        int c3 = pixels[nh].color.z;
                        if (n-2*nh)
                        {
                            c1 = (c1 + pixels[nh].color.x) / 2;
                            c2 = (c2 + pixels[nh].color.y) / 2;
                            c3 = (c3 + pixels[nh].color.z) / 2;
                        }
                        frame_(y, x) = Point3_<uchar>(
                                static_cast<uchar>(c1),
                                static_cast<uchar>(c2),
                                static_cast<uchar>(c3));
                        mask_(y, x) = 255;
                    }
                }
            }
        }
    }
}


static float alignementError(
        const Mat &M, const Mat &frame0, const Mat &mask0, const Mat &frame1)
{
    CV_Assert(frame0.type() == CV_8UC3 && frame1.type() == CV_8UC3);
    CV_Assert(mask0.type() == CV_8U && mask0.size() == frame0.size());
    CV_Assert(frame0.size() == frame1.size());
    CV_Assert(M.size() == Size(3,3) && M.type() == CV_32F);

    Mat_<uchar> mask0_(mask0);
    Mat_<float> M_(M);
    float err = 0;

    for (int y0 = 0; y0 < frame0.rows; ++y0)
    {
        for (int x0 = 0; x0 < frame0.cols; ++x0)
        {
            if (mask0_(y0,x0))
            {
                int x1 = cvRound(M_(0,0)*x0 + M_(0,1)*y0 + M_(0,2));
                int y1 = cvRound(M_(1,0)*x0 + M_(1,1)*y0 + M_(1,2));
                if (y1 >= 0 && y1 < frame1.rows && x1 >= 0 && x1 < frame1.cols)
                    err += std::abs(intensity(frame1.at<Point3_<uchar> >(y1,x1)) -
                                    intensity(frame0.at<Point3_<uchar> >(y0,x0)));
            }
        }
    }

    return err;
}


class MotionInpaintBody
{
public:
    void operator ()(int x, int y)
    {
        float uEst = 0.f, vEst = 0.f, wSum = 0.f;

        for (int dy = -rad; dy <= rad; ++dy)
        {
            for (int dx = -rad; dx <= rad; ++dx)
            {
                int qx0 = x + dx;
                int qy0 = y + dy;

                if (qy0 >= 0 && qy0 < mask0.rows && qx0 >= 0 && qx0 < mask0.cols && mask0(qy0,qx0))
                {
                    int qx1 = cvRound(qx0 + flowX(qy0,qx0));
                    int qy1 = cvRound(qy0 + flowY(qy0,qx0));
                    int px1 = qx1 - dx;
                    int py1 = qy1 - dy;

                    if (qx1 >= 0 && qx1 < mask1.cols && qy1 >= 0 && qy1 < mask1.rows && mask1(qy1,qx1) &&
                        px1 >= 0 && px1 < mask1.cols && py1 >= 0 && py1 < mask1.rows && mask1(py1,px1))
                    {
                        float dudx = 0.f, dvdx = 0.f, dudy = 0.f, dvdy = 0.f;

                        if (qx0 > 0 && mask0(qy0,qx0-1))
                        {
                            if (qx0+1 < mask0.cols && mask0(qy0,qx0+1))
                            {
                                dudx = (flowX(qy0,qx0+1) - flowX(qy0,qx0-1)) * 0.5f;
                                dvdx = (flowY(qy0,qx0+1) - flowY(qy0,qx0-1)) * 0.5f;
                            }
                            else
                            {
                                dudx = flowX(qy0,qx0) - flowX(qy0,qx0-1);
                                dvdx = flowY(qy0,qx0) - flowY(qy0,qx0-1);
                            }
                        }
                        else if (qx0+1 < mask0.cols && mask0(qy0,qx0+1))
                        {
                            dudx = flowX(qy0,qx0+1) - flowX(qy0,qx0);
                            dvdx = flowY(qy0,qx0+1) - flowY(qy0,qx0);
                        }

                        if (qy0 > 0 && mask0(qy0-1,qx0))
                        {
                            if (qy0+1 < mask0.rows && mask0(qy0+1,qx0))
                            {
                                dudy = (flowX(qy0+1,qx0) - flowX(qy0-1,qx0)) * 0.5f;
                                dvdy = (flowY(qy0+1,qx0) - flowY(qy0-1,qx0)) * 0.5f;
                            }
                            else
                            {
                                dudy = flowX(qy0,qx0) - flowX(qy0-1,qx0);
                                dvdy = flowY(qy0,qx0) - flowY(qy0-1,qx0);
                            }
                        }
                        else if (qy0+1 < mask0.rows && mask0(qy0+1,qx0))
                        {
                            dudy = flowX(qy0+1,qx0) - flowX(qy0,qx0);
                            dvdy = flowY(qy0+1,qx0) - flowY(qy0,qx0);
                        }

                        Point3_<uchar> cp = frame1(py1,px1), cq = frame1(qy1,qx1);
                        float distColor = sqr(static_cast<float>(cp.x-cq.x))
                                        + sqr(static_cast<float>(cp.y-cq.y))
                                        + sqr(static_cast<float>(cp.z-cq.z));
                        float w = 1.f / (std::sqrt(distColor * (dx*dx + dy*dy)) + eps);

                        uEst += w * (flowX(qy0,qx0) - dudx*dx - dudy*dy);
                        vEst += w * (flowY(qy0,qx0) - dvdx*dx - dvdy*dy);
                        wSum += w;
                    }
                }
            }
        }

        if (wSum > 0.f)
        {
            flowX(y,x) = uEst / wSum;
            flowY(y,x) = vEst / wSum;
            mask0(y,x) = 255;
        }
    }

    Mat_<Point3_<uchar> > frame1;
    Mat_<uchar> mask0, mask1;
    Mat_<float> flowX, flowY;
    float eps;
    int rad;
};


MotionInpainter::MotionInpainter()
{
#ifdef HAVE_OPENCV_CUDAOPTFLOW
    setOptFlowEstimator(makePtr<DensePyrLkOptFlowEstimatorGpu>());
#else
    CV_Error(Error::StsNotImplemented, "Current implementation of MotionInpainter requires CUDA");
#endif
    setFlowErrorThreshold(1e-4f);
    setDistThreshold(5.f);
    setBorderMode(BORDER_REPLICATE);
}


void MotionInpainter::inpaint(int idx, Mat &frame, Mat &mask)
{
    CV_INSTRUMENT_REGION()

    std::priority_queue<std::pair<float,int> > neighbors;
    std::vector<Mat> vmotions(2*radius_ + 1);

    for (int i = -radius_; i <= radius_; ++i)
    {
        Mat motion0to1 = getMotion(idx, idx + i, *motions_) * at(idx, *stabilizationMotions_).inv();
        vmotions[radius_ + i] = motion0to1;

        if (i != 0)
        {
            float err = alignementError(motion0to1, frame, mask, at(idx + i, *frames_));
            neighbors.push(std::make_pair(-err, idx + i));
        }
    }

    if (mask1_.size() != mask.size())
    {
        mask1_.create(mask.size());
        mask1_.setTo(255);
    }

    cvtColor(frame, grayFrame_, COLOR_BGR2GRAY);

    MotionInpaintBody body;
    body.rad = 2;
    body.eps = 1e-4f;

    while (!neighbors.empty())
    {
        int neighbor = neighbors.top().second;
        neighbors.pop();

        Mat motion1to0 = vmotions[radius_ + neighbor - idx].inv();

        // warp frame

        frame1_ = at(neighbor, *frames_);

        if (motionModel_ != MM_HOMOGRAPHY)
            warpAffine(
                    frame1_, transformedFrame1_, motion1to0(Rect(0,0,3,2)), frame1_.size(),
                    INTER_LINEAR, borderMode_);
        else
            warpPerspective(
                    frame1_, transformedFrame1_, motion1to0, frame1_.size(), INTER_LINEAR,
                    borderMode_);

        cvtColor(transformedFrame1_, transformedGrayFrame1_, COLOR_BGR2GRAY);

        // warp mask

        if (motionModel_ != MM_HOMOGRAPHY)
            warpAffine(
                    mask1_, transformedMask1_, motion1to0(Rect(0,0,3,2)), mask1_.size(),
                    INTER_NEAREST);
        else
            warpPerspective(mask1_, transformedMask1_, motion1to0, mask1_.size(), INTER_NEAREST);

        erode(transformedMask1_, transformedMask1_, Mat());

        // update flow

        optFlowEstimator_->run(grayFrame_, transformedGrayFrame1_, flowX_, flowY_, flowErrors_);

        calcFlowMask(
                flowX_, flowY_, flowErrors_, flowErrorThreshold_, mask, transformedMask1_,
                flowMask_);

        body.flowX = flowX_;
        body.flowY = flowY_;
        body.mask0 = flowMask_;
        body.mask1 = transformedMask1_;
        body.frame1 = transformedFrame1_;
        fmm_.run(flowMask_, body);

        completeFrameAccordingToFlow(
                flowMask_, flowX_, flowY_, transformedFrame1_, transformedMask1_, distThresh_,
                frame, mask);
    }
}


class ColorAverageInpaintBody
{
public:
    void operator ()(int x, int y)
    {
        float c1 = 0, c2 = 0, c3 = 0;
        float wSum = 0;

        static const int lut[8][2] = {{-1,-1}, {-1,0}, {-1,1}, {0,-1}, {0,1}, {1,-1}, {1,0}, {1,1}};

        for (int i = 0; i < 8; ++i)
        {
            int qx = x + lut[i][0];
            int qy = y + lut[i][1];
            if (qy >= 0 && qy < mask.rows && qx >= 0 && qx < mask.cols && mask(qy,qx))
            {
                c1 += frame.at<uchar>(qy,3*qx);
                c2 += frame.at<uchar>(qy,3*qx+1);
                c3 += frame.at<uchar>(qy,3*qx+2);
                wSum += 1;
            }
        }

        float wSumInv = 1.f / wSum;
        frame(y,x) = Point3_<uchar>(
                static_cast<uchar>(c1*wSumInv),
                static_cast<uchar>(c2*wSumInv),
                static_cast<uchar>(c3*wSumInv));
        mask(y,x) = 255;
    }

    cv::Mat_<uchar> mask;
    cv::Mat_<cv::Point3_<uchar> > frame;
};


void ColorAverageInpainter::inpaint(int /*idx*/, Mat &frame, Mat &mask)
{
    CV_INSTRUMENT_REGION()

    ColorAverageInpaintBody body;
    body.mask = mask;
    body.frame = frame;
    fmm_.run(mask, body);
}


void ColorInpainter::inpaint(int /*idx*/, Mat &frame, Mat &mask)
{
    CV_INSTRUMENT_REGION()

    bitwise_not(mask, invMask_);
    cv::inpaint(frame, invMask_, frame, radius_, method_);
}


void calcFlowMask(
        const Mat &flowX, const Mat &flowY, const Mat &errors, float maxError,
        const Mat &mask0, const Mat &mask1, Mat &flowMask)
{
    CV_INSTRUMENT_REGION()

    CV_Assert(flowX.type() == CV_32F && flowX.size() == mask0.size());
    CV_Assert(flowY.type() == CV_32F && flowY.size() == mask0.size());
    CV_Assert(errors.type() == CV_32F && errors.size() == mask0.size());
    CV_Assert(mask0.type() == CV_8U);
    CV_Assert(mask1.type() == CV_8U && mask1.size() == mask0.size());

    Mat_<float> flowX_(flowX), flowY_(flowY), errors_(errors);
    Mat_<uchar> mask0_(mask0), mask1_(mask1);

    flowMask.create(mask0.size(), CV_8U);
    flowMask.setTo(0);
    Mat_<uchar> flowMask_(flowMask);

    for (int y0 = 0; y0 < flowMask_.rows; ++y0)
    {
        for (int x0 = 0; x0 < flowMask_.cols; ++x0)
        {
            if (mask0_(y0,x0) && errors_(y0,x0) < maxError)
            {
                int x1 = cvRound(x0 + flowX_(y0,x0));
                int y1 = cvRound(y0 + flowY_(y0,x0));

                if (x1 >= 0 && x1 < mask1_.cols && y1 >= 0 && y1 < mask1_.rows && mask1_(y1,x1))
                    flowMask_(y0,x0) = 255;
            }
        }
    }
}


void completeFrameAccordingToFlow(
        const Mat &flowMask, const Mat &flowX, const Mat &flowY, const Mat &frame1, const Mat &mask1,
        float distThresh, Mat &frame0, Mat &mask0)
{
    CV_INSTRUMENT_REGION()

    CV_Assert(flowMask.type() == CV_8U);
    CV_Assert(flowX.type() == CV_32F && flowX.size() == flowMask.size());
    CV_Assert(flowY.type() == CV_32F && flowY.size() == flowMask.size());
    CV_Assert(frame1.type() == CV_8UC3 && frame1.size() == flowMask.size());
    CV_Assert(mask1.type() == CV_8U && mask1.size() == flowMask.size());
    CV_Assert(frame0.type() == CV_8UC3 && frame0.size() == flowMask.size());
    CV_Assert(mask0.type() == CV_8U && mask0.size() == flowMask.size());

    Mat_<uchar> flowMask_(flowMask), mask1_(mask1), mask0_(mask0);
    Mat_<float> flowX_(flowX), flowY_(flowY);

    for (int y0 = 0; y0 < frame0.rows; ++y0)
    {
        for (int x0 = 0; x0 < frame0.cols; ++x0)
        {
            if (!mask0_(y0,x0) && flowMask_(y0,x0))
            {
                int x1 = cvRound(x0 + flowX_(y0,x0));
                int y1 = cvRound(y0 + flowY_(y0,x0));

                if (x1 >= 0 && x1 < frame1.cols && y1 >= 0 && y1 < frame1.rows && mask1_(y1,x1)
                    && sqr(flowX_(y0,x0)) + sqr(flowY_(y0,x0)) < sqr(distThresh))
                {
                    frame0.at<Point3_<uchar> >(y0,x0) = frame1.at<Point3_<uchar> >(y1,x1);
                    mask0_(y0,x0) = 255;
                }
            }
        }
    }
}

} // namespace videostab
} // namespace cv
