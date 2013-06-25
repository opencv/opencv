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
#include "precomp.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;
using namespace cv::superres;
using namespace cv::superres::detail;

cv::superres::FrameSource::~FrameSource()
{
}

//////////////////////////////////////////////////////
// EmptyFrameSource

namespace
{
    class EmptyFrameSource : public FrameSource
    {
    public:
        void nextFrame(OutputArray frame);
        void reset();
    };

    void EmptyFrameSource::nextFrame(OutputArray frame)
    {
        frame.release();
    }

    void EmptyFrameSource::reset()
    {
    }
}

Ptr<FrameSource> cv::superres::createFrameSource_Empty()
{
    return new EmptyFrameSource;
}

//////////////////////////////////////////////////////
// VideoFrameSource & CameraFrameSource

#ifndef HAVE_OPENCV_HIGHGUI

Ptr<FrameSource> cv::superres::createFrameSource_Video(const string& fileName)
{
    (void) fileName;
    CV_Error(CV_StsNotImplemented, "The called functionality is disabled for current build or platform");
    return Ptr<FrameSource>();
}

Ptr<FrameSource> cv::superres::createFrameSource_Camera(int deviceId)
{
    (void) deviceId;
    CV_Error(CV_StsNotImplemented, "The called functionality is disabled for current build or platform");
    return Ptr<FrameSource>();
}

#else // HAVE_OPENCV_HIGHGUI

namespace
{
    class CaptureFrameSource : public FrameSource
    {
    public:
        void nextFrame(OutputArray frame);

    protected:
        VideoCapture vc_;

    private:
        Mat frame_;
    };

    void CaptureFrameSource::nextFrame(OutputArray _frame)
    {
        if (_frame.kind() == _InputArray::MAT)
        {
            vc_ >> _frame.getMatRef();
        }
        else if(_frame.kind() == _InputArray::GPU_MAT)
        {
            vc_ >> frame_;
            arrCopy(frame_, _frame);
        }
        else if(_frame.kind() == _InputArray::OCL_MAT)
        {
            vc_ >> frame_;
            if(!frame_.empty())
            {
                arrCopy(frame_, _frame);
            }
        }
        else
        {
            //should never get here
        }
    }

    class VideoFrameSource : public CaptureFrameSource
    {
    public:
        VideoFrameSource(const string& fileName);

        void reset();

    private:
        string fileName_;
    };

    VideoFrameSource::VideoFrameSource(const string& fileName) : fileName_(fileName)
    {
        reset();
    }

    void VideoFrameSource::reset()
    {
        vc_.release();
        vc_.open(fileName_);
        CV_Assert( vc_.isOpened() );
    }

    class CameraFrameSource : public CaptureFrameSource
    {
    public:
        CameraFrameSource(int deviceId);

        void reset();

    private:
        int deviceId_;
    };

    CameraFrameSource::CameraFrameSource(int deviceId) : deviceId_(deviceId)
    {
        reset();
    }

    void CameraFrameSource::reset()
    {
        vc_.release();
        vc_.open(deviceId_);
        CV_Assert( vc_.isOpened() );
    }
}

Ptr<FrameSource> cv::superres::createFrameSource_Video(const string& fileName)
{
    return new VideoFrameSource(fileName);
}

Ptr<FrameSource> cv::superres::createFrameSource_Camera(int deviceId)
{
    return new CameraFrameSource(deviceId);
}

#endif // HAVE_OPENCV_HIGHGUI

//////////////////////////////////////////////////////
// VideoFrameSource_GPU

#ifndef HAVE_OPENCV_GPU

Ptr<FrameSource> cv::superres::createFrameSource_Video_GPU(const string& fileName)
{
    (void) fileName;
    CV_Error(CV_StsNotImplemented, "The called functionality is disabled for current build or platform");
    return Ptr<FrameSource>();
}

#else // HAVE_OPENCV_GPU

namespace
{
    class VideoFrameSource_GPU : public FrameSource
    {
    public:
        VideoFrameSource_GPU(const string& fileName);

        void nextFrame(OutputArray frame);
        void reset();

    private:
        string fileName_;
        VideoReader_GPU reader_;
        GpuMat frame_;
    };

    VideoFrameSource_GPU::VideoFrameSource_GPU(const string& fileName) : fileName_(fileName)
    {
        reset();
    }

    void VideoFrameSource_GPU::nextFrame(OutputArray _frame)
    {
        if (_frame.kind() == _InputArray::GPU_MAT)
        {
            bool res = reader_.read(_frame.getGpuMatRef());
            if (!res)
                _frame.release();
        }
        else
        {
            bool res = reader_.read(frame_);
            if (!res)
                _frame.release();
            else
                arrCopy(frame_, _frame);
        }
    }

    void VideoFrameSource_GPU::reset()
    {
        reader_.close();
        reader_.open(fileName_);
        CV_Assert( reader_.isOpened() );
    }
}

Ptr<FrameSource> cv::superres::createFrameSource_Video_GPU(const string& fileName)
{
    return new VideoFrameSource(fileName);
}

#endif // HAVE_OPENCV_GPU
