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
//                       (3-clause BSD License)
//                     For BackgroundSubtractorCNT
//               (Background Subtraction based on Counting)
//
// Copyright (C) 2016, Sagi Zeevi (www.theimpossiblecode.com), all rights reserved.
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
#include <functional>

namespace cv
{

/** @brief Implementation of background subtraction based on counting.
 *  About as fast as MOG2 on a high end system (benchmarked on )
 *  More than twice faster than MOG2 on cheap hardware (benchmarked on Raspberry Pi3).
 *  Algorithm by Sagi Zeevi
 */
class BackgroundSubtractorCNTImpl: public BackgroundSubtractorCNT
{
public:
    /**
     * @brief BackgroundSubtractorCNTImpl
     * @param stability number of frames with same pixel color to consider stable
     * @param useHistory determines if we're giving a pixel credit for being stable for a long time
     * @param maxStability maximum allowed credit for a pixel in history
     * @param isParallel determines if we're parallelizing the algorithm
     */
    BackgroundSubtractorCNTImpl(int minStability,
                                bool useHistory,
                                int maxStability,
                                bool isParallel);

    // BackgroundSubtractor interface
    virtual void apply(InputArray image, OutputArray fgmask, double learningRate);
    virtual void getBackgroundImage(OutputArray backgroundImage) const;

    int getMinPixelStability() const;
    void setMinPixelStability(int value);

    int getMaxPixelStability() const;
    void setMaxPixelStability(int value);

    bool getUseHistory() const;
    void setUseHistory(bool value);

    bool getIsParallel() const;
    void setIsParallel(bool value);

    //! the destructor
    virtual ~BackgroundSubtractorCNTImpl() {}

private:
    int minPixelStability;
    int maxPixelStability;
    int threshold;
    bool useHistory;
    bool isParallel;
    // These 3 commented expressed in 1 'data' for faster single access
    //    Mat_<int> stability;        // data[0]  => Candidate for historyStability if pixel is ~same as in prevFrame
    //    Mat_<int> history;          // data[1]  => Color which got most hits for the past maxPixelStability frames
    //    Mat_<int> historyStability; // data[2]  => How many hits this pixel got for the color in history
    //    Mat_<int> background;       // data[3]  => Current background as detected by algorithm
    Mat_<Vec4i> data;
    Mat prevFrame;
    Mat fgMaskPrev;
};

BackgroundSubtractorCNTImpl::BackgroundSubtractorCNTImpl(int minStability,
                                                         bool _useHistory,
                                                         int maxStability,
                                                         bool _isParallel)
    : minPixelStability(minStability),
      maxPixelStability(maxStability),
      threshold(5),
      useHistory(_useHistory),
      isParallel(_isParallel)
{
}

void BackgroundSubtractorCNTImpl::getBackgroundImage(OutputArray _backgroundImage) const
{
    CV_Assert(! data.empty());

    _backgroundImage.create(prevFrame.size(), CV_8U); // OutputArray usage requires this step
    Mat backgroundImage = _backgroundImage.getMat();

    // mixChannels requires same types to mix,
    //  so imixing with tmp Mat and conerting
    Mat_<int> tmp(prevFrame.rows, prevFrame.cols);
    int from_bg_model_to_user[] = {3, 0};
    mixChannels(&data, 1, &tmp, 1, from_bg_model_to_user, 1);
    tmp.convertTo(backgroundImage, CV_8U);
}

int BackgroundSubtractorCNTImpl::getMinPixelStability() const
{
    return minPixelStability;
}

void BackgroundSubtractorCNTImpl::setMinPixelStability(int value)
{
    CV_Assert(value > 0 && value < maxPixelStability);
    minPixelStability = value;
}

int BackgroundSubtractorCNTImpl::getMaxPixelStability() const
{
    return maxPixelStability;
}

void BackgroundSubtractorCNTImpl::setMaxPixelStability(int value)
{
    CV_Assert(value > minPixelStability);
    maxPixelStability = value;
}

bool BackgroundSubtractorCNTImpl::getUseHistory() const
{
    return useHistory;
}

void BackgroundSubtractorCNTImpl::setUseHistory(bool value)
{
    useHistory = value;
}

bool BackgroundSubtractorCNTImpl::getIsParallel() const
{
    return isParallel;
}

void BackgroundSubtractorCNTImpl::setIsParallel(bool value)
{
    isParallel = value;
}

class CNTFunctor
{
public:
    virtual void operator()(Vec4i &vec, uchar currColor, uchar prevColor, uchar &fgMaskPixelRef) = 0;
    //! the destructor
    virtual ~CNTFunctor() {}
};

struct BGSubtractPixel : public CNTFunctor
{
    BGSubtractPixel(int _minPixelStability, int _threshold,
                    const Mat &_frame, const Mat &_prevFrame, Mat &_fgMask)
        : minPixelStability(_minPixelStability),
          threshold(_threshold),
          frame(_frame),
          prevFrame(_prevFrame),
          fgMask(_fgMask)
    {}

    //! the destructor
    virtual ~BGSubtractPixel() {}

    void operator()(Vec4i &vec, uchar currColor, uchar prevColor, uchar &fgMaskPixelRef)
    {
        int &stabilityRef = vec[0];
        int &bgImgRef = vec[3];
        if (abs(currColor - prevColor) < threshold)
        {
            ++stabilityRef;
            if (stabilityRef == minPixelStability)
            {   // bg
                --stabilityRef;
                bgImgRef = prevColor;
            }
            else
            {   // fg
                fgMaskPixelRef = 255;
            }
        }
        else
        {   // fg
            stabilityRef = 0;
            fgMaskPixelRef = 255;
        }
    }

    int minPixelStability;
    int threshold;
    const Mat &frame;
    const Mat &prevFrame;
    Mat &fgMask;
};

struct BGSubtractPixelWithHistory : public CNTFunctor
{
    BGSubtractPixelWithHistory(int _minPixelStability, int _maxPixelStability, int _threshold,
                               const Mat &_frame, const Mat &_prevFrame, Mat &_fgMask)
        : minPixelStability(_minPixelStability),
          maxPixelStability(_maxPixelStability),
          threshold(_threshold),
          thresholdHistory(30),
          frame(_frame),
          prevFrame(_prevFrame),
          fgMask(_fgMask)
    {}

    //! the destructor
    virtual ~BGSubtractPixelWithHistory() {}

    void incrStability(int &histStabilityRef)
    {
        if (histStabilityRef < maxPixelStability)
        {
            ++histStabilityRef;
        }
    }

    void decrStability(int &histStabilityRef)
    {
        if (histStabilityRef > 0)
        {
            --histStabilityRef;
        }
    }

    void operator()(Vec4i &vec, uchar currColor, uchar prevColor, uchar &fgMaskPixelRef)
    {
        int &stabilityRef = vec[0];
        int &historyColorRef = vec[1];
        int &histStabilityRef = vec[2];
        int &bgImgRef = vec[3];
        if (abs(currColor - historyColorRef) < thresholdHistory)
        {   // No change compared to history - this is maybe a background
            stabilityRef = 0;
            incrStability(histStabilityRef);
            if (histStabilityRef <= minPixelStability)
            {
                fgMaskPixelRef = 255;
            }
            else
            {
                bgImgRef = historyColorRef;
            }
        }
        else if (abs(currColor - prevColor) < threshold)
        {   // No change compared to prev - this is maybe a background
            incrStability(stabilityRef);
            if (stabilityRef > minPixelStability)
            {   // Stable color - this is maybe a background
                if (stabilityRef >= histStabilityRef)
                {
                    historyColorRef = currColor;
                    histStabilityRef = stabilityRef;
                    bgImgRef = historyColorRef;
                }
                else
                {   // Stable but different from stable history - this is a foreground
                    decrStability(histStabilityRef);
                    fgMaskPixelRef = 255;
                }
            }
            else
            {   // This is FG.
                fgMaskPixelRef = 255;
            }
        }
        else
        {   // Color changed - this is defently a foreground
            stabilityRef = 0;
            decrStability(histStabilityRef);
            fgMaskPixelRef = 255;
        }

    }

    int minPixelStability;
    int maxPixelStability;
    int threshold;
    int thresholdHistory;
    const Mat &frame;
    const Mat &prevFrame;
    Mat &fgMask;
};

class CNTInvoker : public ParallelLoopBody
{
public:
    CNTInvoker(Mat_<Vec4i> &_data, Mat &_img, Mat &_prevFrame, Mat &_fgMask, CNTFunctor &_functor)
        : data(_data), img(_img), prevFrame(_prevFrame), fgMask(_fgMask), functor(_functor)
    {
    }

    // Iterate rows
    void operator()(const Range& range) const
    {
        for (int r = range.start; r < range.end; ++r)
        {
            Vec4i* row = data.ptr<Vec4i>(r);
            uchar* frameRow = img.ptr<uchar>(r);
            uchar* prevFrameRow = prevFrame.ptr<uchar>(r);
            uchar* fgMaskRow = fgMask.ptr<uchar>(r);
            for (int c = 0; c < data.cols; ++c)
            {
                functor(row[c], frameRow[c], prevFrameRow[c], fgMaskRow[c]);
            }
        }
    }

private:
    Mat_<Vec4i> &data;
    Mat &img;
    Mat &prevFrame;
    Mat &fgMask;
    CNTFunctor &functor;
};

void BackgroundSubtractorCNTImpl::apply(InputArray image, OutputArray _fgmask, double learningRate)
{
    CV_Assert(image.type() == CV_8UC1);

    Mat frameIn = image.getMat();
    _fgmask.create(image.size(), CV_8U); // OutputArray usage requires this step
    Mat fgMask = _fgmask.getMat();

    bool needToInitialize = data.empty() || learningRate >= 1 || frameIn.size() != prevFrame.size();

    Mat frame = frameIn.clone();

    if (needToInitialize)
    {   // Usually done only once
        data = Mat_<Vec4i>::zeros(frame.rows, frame.cols);
        prevFrame = frame;

        // mixChannels requires same types to mix,
        //  so imixing with tmp Mat and conerting
        Mat tmp;
        prevFrame.convertTo(tmp, CV_32S);
        int from_gray_to_history_color[] = {0,1};
        mixChannels(&tmp, 1, &data, 1, from_gray_to_history_color, 1);
    }

    fgMask = Scalar(0);
    CNTFunctor *functor;
    if (useHistory && learningRate)
    {
        double scaleMaxStability = 1.0;
        if (learningRate > 0 && learningRate < 1.0)
        {
            scaleMaxStability = learningRate;
        }
        functor = new BGSubtractPixelWithHistory(minPixelStability, int(maxPixelStability * scaleMaxStability),
                                                 threshold, frame, prevFrame, fgMask);
    }
    else
    {
        functor = new BGSubtractPixel(minPixelStability, threshold*3, frame, prevFrame, fgMask);
    }

    if (isParallel)
    {
        parallel_for_(Range(0, frame.rows),
                      CNTInvoker(data, frame, prevFrame, fgMask, *functor));
    }
    else
    {
        for (int r = 0; r < data.rows; ++r)
        {
            Vec4i* row = data.ptr<Vec4i>(r);
            uchar* frameRow = frame.ptr<uchar>(r);
            uchar* prevFrameRow = prevFrame.ptr<uchar>(r);
            uchar* fgMaskRow = fgMask.ptr<uchar>(r);
            for (int c = 0; c < data.cols; ++c)
            {
                (*functor)(row[c], frameRow[c], prevFrameRow[c], fgMaskRow[c]);
            }
        }
    }

    delete functor;

    prevFrame = frame;
}


Ptr<BackgroundSubtractorCNT> createBackgroundSubtractorCNT(int minPixelStability, bool useHistory, int maxStability, bool isParallel)
{
    return makePtr<BackgroundSubtractorCNTImpl>(minPixelStability, useHistory, maxStability, isParallel);
}

}

/* End of file. */
