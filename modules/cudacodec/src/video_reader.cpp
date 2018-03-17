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

using namespace cv;
using namespace cv::cuda;
using namespace cv::cudacodec;

#ifndef HAVE_NVCUVID

Ptr<VideoReader> cv::cudacodec::createVideoReader(const String&) { throw_no_cuda(); return Ptr<VideoReader>(); }
Ptr<VideoReader> cv::cudacodec::createVideoReader(const Ptr<RawVideoSource>&) { throw_no_cuda(); return Ptr<VideoReader>(); }

#else // HAVE_NVCUVID

void videoDecPostProcessFrame(const GpuMat& decodedFrame, OutputArray _outFrame, int width, int height);

using namespace cv::cudacodec::detail;

namespace
{
    class VideoReaderImpl : public VideoReader
    {
    public:
        explicit VideoReaderImpl(const Ptr<VideoSource>& source);
        ~VideoReaderImpl();

        bool nextFrame(OutputArray frame);

        FormatInfo format() const;

    private:
        Ptr<VideoSource> videoSource_;

        Ptr<FrameQueue> frameQueue_;
        Ptr<VideoDecoder> videoDecoder_;
        Ptr<VideoParser> videoParser_;

        CUvideoctxlock lock_;

        std::deque< std::pair<CUVIDPARSERDISPINFO, CUVIDPROCPARAMS> > frames_;
    };

    FormatInfo VideoReaderImpl::format() const
    {
        return videoSource_->format();
    }

    VideoReaderImpl::VideoReaderImpl(const Ptr<VideoSource>& source) :
        videoSource_(source),
        lock_(0)
    {
        // init context
        GpuMat temp(1, 1, CV_8UC1);
        temp.release();

        CUcontext ctx;
        cuSafeCall( cuCtxGetCurrent(&ctx) );
        cuSafeCall( cuvidCtxLockCreate(&lock_, ctx) );

        frameQueue_.reset(new FrameQueue);
        videoDecoder_.reset(new VideoDecoder(videoSource_->format(), lock_));
        videoParser_.reset(new VideoParser(videoDecoder_, frameQueue_));

        videoSource_->setVideoParser(videoParser_);
        videoSource_->start();
    }

    VideoReaderImpl::~VideoReaderImpl()
    {
        frameQueue_->endDecode();
        videoSource_->stop();
    }

    class VideoCtxAutoLock
    {
    public:
        VideoCtxAutoLock(CUvideoctxlock lock) : m_lock(lock) { cuSafeCall( cuvidCtxLock(m_lock, 0) ); }
        ~VideoCtxAutoLock() { cuvidCtxUnlock(m_lock, 0); }

    private:
        CUvideoctxlock m_lock;
    };

    bool VideoReaderImpl::nextFrame(OutputArray frame)
    {
        if (videoSource_->hasError() || videoParser_->hasError())
            CV_Error(Error::StsUnsupportedFormat, "Unsupported video source");

        if (!videoSource_->isStarted() || frameQueue_->isEndOfDecode())
            return false;

        if (frames_.empty())
        {
            CUVIDPARSERDISPINFO displayInfo;

            for (;;)
            {
                if (frameQueue_->dequeue(displayInfo))
                    break;

                if (videoSource_->hasError() || videoParser_->hasError())
                    CV_Error(Error::StsUnsupportedFormat, "Unsupported video source");

                if (frameQueue_->isEndOfDecode())
                    return false;

                // Wait a bit
                Thread::sleep(1);
            }

            bool isProgressive = displayInfo.progressive_frame != 0;
            const int num_fields = isProgressive ? 1 : 2 + displayInfo.repeat_first_field;

            for (int active_field = 0; active_field < num_fields; ++active_field)
            {
                CUVIDPROCPARAMS videoProcParams;
                std::memset(&videoProcParams, 0, sizeof(CUVIDPROCPARAMS));

                videoProcParams.progressive_frame = displayInfo.progressive_frame;
                videoProcParams.second_field      = active_field;
                videoProcParams.top_field_first   = displayInfo.top_field_first;
                videoProcParams.unpaired_field    = (num_fields == 1);

                frames_.push_back(std::make_pair(displayInfo, videoProcParams));
            }
        }

        if (frames_.empty())
            return false;

        std::pair<CUVIDPARSERDISPINFO, CUVIDPROCPARAMS> frameInfo = frames_.front();
        frames_.pop_front();

        {
            VideoCtxAutoLock autoLock(lock_);

            // map decoded video frame to CUDA surface
            GpuMat decodedFrame = videoDecoder_->mapFrame(frameInfo.first.picture_index, frameInfo.second);

            // perform post processing on the CUDA surface (performs colors space conversion and post processing)
            // comment this out if we include the line of code seen above
            videoDecPostProcessFrame(decodedFrame, frame, videoDecoder_->targetWidth(), videoDecoder_->targetHeight());

            // unmap video frame
            // unmapFrame() synchronizes with the VideoDecode API (ensures the frame has finished decoding)
            videoDecoder_->unmapFrame(decodedFrame);
        }

        // release the frame, so it can be re-used in decoder
        if (frames_.empty())
            frameQueue_->releaseFrame(frameInfo.first);

        return true;
    }
}

Ptr<VideoReader> cv::cudacodec::createVideoReader(const String& filename)
{
    CV_Assert( !filename.empty() );

    Ptr<VideoSource> videoSource;

    try
    {
        videoSource.reset(new CuvidVideoSource(filename));
    }
    catch (...)
    {
        Ptr<RawVideoSource> source(new FFmpegVideoSource(filename));
        videoSource.reset(new RawVideoSourceWrapper(source));
    }

    return makePtr<VideoReaderImpl>(videoSource);
}

Ptr<VideoReader> cv::cudacodec::createVideoReader(const Ptr<RawVideoSource>& source)
{
    Ptr<VideoSource> videoSource(new RawVideoSourceWrapper(source));
    return makePtr<VideoReaderImpl>(videoSource);
}

#endif // HAVE_NVCUVID
