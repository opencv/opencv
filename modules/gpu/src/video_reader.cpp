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

#if !defined(HAVE_CUDA) || defined(CUDA_DISABLER) || !defined(HAVE_NVCUVID)

class cv::gpu::VideoReader_GPU::Impl
{
};

cv::gpu::VideoReader_GPU::VideoReader_GPU() { throw_nogpu(); }
cv::gpu::VideoReader_GPU::VideoReader_GPU(const std::string&) { throw_nogpu(); }
cv::gpu::VideoReader_GPU::VideoReader_GPU(const cv::Ptr<VideoSource>&) { throw_nogpu(); }
cv::gpu::VideoReader_GPU::~VideoReader_GPU() { }
void cv::gpu::VideoReader_GPU::open(const std::string&) { throw_nogpu(); }
void cv::gpu::VideoReader_GPU::open(const cv::Ptr<VideoSource>&) { throw_nogpu(); }
bool cv::gpu::VideoReader_GPU::isOpened() const { return false; }
void cv::gpu::VideoReader_GPU::close() { }
bool cv::gpu::VideoReader_GPU::read(GpuMat&) { throw_nogpu(); return false; }
cv::gpu::VideoReader_GPU::FormatInfo cv::gpu::VideoReader_GPU::format() const { throw_nogpu(); FormatInfo format_ = {MPEG1,Monochrome,0,0}; return format_; }
bool cv::gpu::VideoReader_GPU::VideoSource::parseVideoData(const unsigned char*, size_t, bool) { throw_nogpu(); return false; }
void cv::gpu::VideoReader_GPU::dumpFormat(std::ostream&) { throw_nogpu(); }

#else // HAVE_CUDA

#include "frame_queue.h"
#include "video_decoder.h"
#include "video_parser.h"

#include "cuvid_video_source.h"
#include "ffmpeg_video_source.h"

#include "cu_safe_call.h"

class cv::gpu::VideoReader_GPU::Impl
{
public:
    explicit Impl(const cv::Ptr<cv::gpu::VideoReader_GPU::VideoSource>& source);
    ~Impl();

    bool grab(cv::gpu::GpuMat& frame);

    cv::gpu::VideoReader_GPU::FormatInfo format() const { return videoSource_->format(); }

private:
    Impl(const Impl&);
    Impl& operator =(const Impl&);

    cv::Ptr<cv::gpu::VideoReader_GPU::VideoSource> videoSource_;

    std::auto_ptr<cv::gpu::detail::FrameQueue> frameQueue_;
    std::auto_ptr<cv::gpu::detail::VideoDecoder> videoDecoder_;
    std::auto_ptr<cv::gpu::detail::VideoParser> videoParser_;

    CUvideoctxlock lock_;

    std::deque< std::pair<CUVIDPARSERDISPINFO, CUVIDPROCPARAMS> > frames_;
};

cv::gpu::VideoReader_GPU::Impl::Impl(const cv::Ptr<VideoSource>& source) :
    videoSource_(source),
    lock_(0)
{
    // init context
    GpuMat temp(1, 1, CV_8UC1);
    temp.release();

    DeviceInfo devInfo;
    CV_Assert( devInfo.supports(FEATURE_SET_COMPUTE_11) );

    CUcontext ctx;
    cuSafeCall( cuCtxGetCurrent(&ctx) );
    cuSafeCall( cuvidCtxLockCreate(&lock_, ctx) );

    frameQueue_.reset(new detail::FrameQueue);
    videoDecoder_.reset(new detail::VideoDecoder(videoSource_->format(), lock_));
    videoParser_.reset(new detail::VideoParser(videoDecoder_.get(), frameQueue_.get()));

    videoSource_->setFrameQueue(frameQueue_.get());
    videoSource_->setVideoParser(videoParser_.get());

    videoSource_->start();
}

cv::gpu::VideoReader_GPU::Impl::~Impl()
{
    frameQueue_->endDecode();
    videoSource_->stop();
}

namespace cv { namespace gpu { namespace device {
    namespace video_decoding
    {
        void loadHueCSC(float hueCSC[9]);
        void NV12ToARGB_gpu(const PtrStepb decodedFrame, PtrStepSz<unsigned int> interopFrame, cudaStream_t stream = 0);
    }
}}}

namespace
{
    class VideoCtxAutoLock
    {
    public:
        VideoCtxAutoLock(CUvideoctxlock lock) : m_lock(lock) { cuSafeCall( cuvidCtxLock(m_lock, 0) ); }
        ~VideoCtxAutoLock() { cuvidCtxUnlock(m_lock, 0); }

    private:
        CUvideoctxlock m_lock;
    };

    enum ColorSpace
    {
        ITU601 = 1,
        ITU709 = 2
    };

    void setColorSpaceMatrix(ColorSpace CSC, float hueCSC[9], float hue)
    {
        float hueSin = std::sin(hue);
        float hueCos = std::cos(hue);

        if (CSC == ITU601)
        {
            //CCIR 601
            hueCSC[0] = 1.1644f;
            hueCSC[1] = hueSin * 1.5960f;
            hueCSC[2] = hueCos * 1.5960f;
            hueCSC[3] = 1.1644f;
            hueCSC[4] = (hueCos * -0.3918f) - (hueSin * 0.8130f);
            hueCSC[5] = (hueSin *  0.3918f) - (hueCos * 0.8130f);
            hueCSC[6] = 1.1644f;
            hueCSC[7] = hueCos *  2.0172f;
            hueCSC[8] = hueSin * -2.0172f;
        }
        else if (CSC == ITU709)
        {
            //CCIR 709
            hueCSC[0] = 1.0f;
            hueCSC[1] = hueSin * 1.57480f;
            hueCSC[2] = hueCos * 1.57480f;
            hueCSC[3] = 1.0;
            hueCSC[4] = (hueCos * -0.18732f) - (hueSin * 0.46812f);
            hueCSC[5] = (hueSin *  0.18732f) - (hueCos * 0.46812f);
            hueCSC[6] = 1.0f;
            hueCSC[7] = hueCos *  1.85560f;
            hueCSC[8] = hueSin * -1.85560f;
        }
    }

    void cudaPostProcessFrame(const cv::gpu::GpuMat& decodedFrame, cv::gpu::GpuMat& interopFrame, int width, int height)
    {
        using namespace cv::gpu::device::video_decoding;

        static bool updateCSC = true;
        static float hueColorSpaceMat[9];

        // Upload the Color Space Conversion Matrices
        if (updateCSC)
        {
            const ColorSpace colorSpace = ITU601;
            const float hue = 0.0f;

            // CCIR 601/709
            setColorSpaceMatrix(colorSpace, hueColorSpaceMat, hue);

            updateCSC = false;
        }

        // Final Stage: NV12toARGB color space conversion

        interopFrame.create(height, width, CV_8UC4);

        loadHueCSC(hueColorSpaceMat);

        NV12ToARGB_gpu(decodedFrame, interopFrame);
    }
}

bool cv::gpu::VideoReader_GPU::Impl::grab(GpuMat& frame)
{
    if (videoSource_->hasError() || videoParser_->hasError())
        CV_Error(CV_StsUnsupportedFormat, "Unsupported video source");

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
                CV_Error(CV_StsUnsupportedFormat, "Unsupported video source");

            if (frameQueue_->isEndOfDecode())
                return false;

            // Wait a bit
            detail::Thread::sleep(1);
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
        cv::gpu::GpuMat decodedFrame = videoDecoder_->mapFrame(frameInfo.first.picture_index, frameInfo.second);

        // perform post processing on the CUDA surface (performs colors space conversion and post processing)
        // comment this out if we inclue the line of code seen above
        cudaPostProcessFrame(decodedFrame, frame, videoDecoder_->targetWidth(), videoDecoder_->targetHeight());

        // unmap video frame
        // unmapFrame() synchronizes with the VideoDecode API (ensures the frame has finished decoding)
        videoDecoder_->unmapFrame(decodedFrame);
    }

    // release the frame, so it can be re-used in decoder
    if (frames_.empty())
        frameQueue_->releaseFrame(frameInfo.first);

    return true;
}

////////////////////////////////////////////////////////////////////////////

cv::gpu::VideoReader_GPU::VideoReader_GPU()
{
}

cv::gpu::VideoReader_GPU::VideoReader_GPU(const std::string& filename)
{
    open(filename);
}

cv::gpu::VideoReader_GPU::VideoReader_GPU(const cv::Ptr<VideoSource>& source)
{
    open(source);
}

cv::gpu::VideoReader_GPU::~VideoReader_GPU()
{
    close();
}

void cv::gpu::VideoReader_GPU::open(const std::string& filename)
{
    CV_Assert( !filename.empty() );

#ifndef __APPLE__
    try
    {
        cv::Ptr<VideoSource> source(new detail::CuvidVideoSource(filename));
        open(source);
    }
    catch (const std::runtime_error&)
#endif
    {
        cv::Ptr<VideoSource> source(new cv::gpu::detail::FFmpegVideoSource(filename));
        open(source);
    }
}

void cv::gpu::VideoReader_GPU::open(const cv::Ptr<VideoSource>& source)
{
    CV_Assert( !source.empty() );
    close();
    impl_.reset(new Impl(source));
}

bool cv::gpu::VideoReader_GPU::isOpened() const
{
    return impl_.get() != 0;
}

void cv::gpu::VideoReader_GPU::close()
{
    impl_.reset();
}

bool cv::gpu::VideoReader_GPU::read(GpuMat& image)
{
    if (!isOpened())
        return false;

    if (!impl_->grab(image))
    {
        close();
        return false;
    }

    return true;
}

cv::gpu::VideoReader_GPU::FormatInfo cv::gpu::VideoReader_GPU::format() const
{
    CV_Assert( isOpened() );
    return impl_->format();
}

bool cv::gpu::VideoReader_GPU::VideoSource::parseVideoData(const unsigned char* data, size_t size, bool endOfStream)
{
    return videoParser_->parseVideoData(data, size, endOfStream);
}

void cv::gpu::VideoReader_GPU::dumpFormat(std::ostream& st)
{
    static const char* codecs[] =
    {
        "MPEG1",
        "MPEG2",
        "MPEG4",
        "VC1",
        "H264",
        "JPEG",
        "H264_SVC",
        "H264_MVC"
    };

    static const char* chromas[] =
    {
        "Monochrome",
        "YUV420",
        "YUV422",
        "YUV444"
    };

    FormatInfo _format = this->format();

    st << "Frame Size    : " << _format.width << "x" << _format.height << std::endl;
    st << "Codec         : " << (_format.codec <= H264_MVC ? codecs[_format.codec] : "Uncompressed YUV") << std::endl;
    st << "Chroma Format : " << chromas[_format.chromaFormat] << std::endl;
}

#endif // HAVE_CUDA
