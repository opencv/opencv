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

#ifndef __OPENCV_GPUCODEC_HPP__
#define __OPENCV_GPUCODEC_HPP__

#ifndef __cplusplus
#  error gpucodec.hpp header must be compiled as C++
#endif

#include <iosfwd>

#include "opencv2/core/gpumat.hpp"

namespace cv { namespace gpu {

////////////////////////////////// Video Encoding //////////////////////////////////

// Works only under Windows
// Supports olny H264 video codec and AVI files
class CV_EXPORTS VideoWriter_GPU
{
public:
    struct EncoderParams;

    // Callbacks for video encoder, use it if you want to work with raw video stream
    class EncoderCallBack;

    enum SurfaceFormat
    {
        SF_UYVY = 0,
        SF_YUY2,
        SF_YV12,
        SF_NV12,
        SF_IYUV,
        SF_BGR,
        SF_GRAY = SF_BGR
    };

    VideoWriter_GPU();
    VideoWriter_GPU(const String& fileName, cv::Size frameSize, double fps, SurfaceFormat format = SF_BGR);
    VideoWriter_GPU(const String& fileName, cv::Size frameSize, double fps, const EncoderParams& params, SurfaceFormat format = SF_BGR);
    VideoWriter_GPU(const cv::Ptr<EncoderCallBack>& encoderCallback, cv::Size frameSize, double fps, SurfaceFormat format = SF_BGR);
    VideoWriter_GPU(const cv::Ptr<EncoderCallBack>& encoderCallback, cv::Size frameSize, double fps, const EncoderParams& params, SurfaceFormat format = SF_BGR);
    ~VideoWriter_GPU();

    // all methods throws cv::Exception if error occurs
    void open(const String& fileName, cv::Size frameSize, double fps, SurfaceFormat format = SF_BGR);
    void open(const String& fileName, cv::Size frameSize, double fps, const EncoderParams& params, SurfaceFormat format = SF_BGR);
    void open(const cv::Ptr<EncoderCallBack>& encoderCallback, cv::Size frameSize, double fps, SurfaceFormat format = SF_BGR);
    void open(const cv::Ptr<EncoderCallBack>& encoderCallback, cv::Size frameSize, double fps, const EncoderParams& params, SurfaceFormat format = SF_BGR);

    bool isOpened() const;
    void close();

    void write(const cv::gpu::GpuMat& image, bool lastFrame = false);

    struct CV_EXPORTS EncoderParams
    {
        int       P_Interval;      //    NVVE_P_INTERVAL,
        int       IDR_Period;      //    NVVE_IDR_PERIOD,
        int       DynamicGOP;      //    NVVE_DYNAMIC_GOP,
        int       RCType;          //    NVVE_RC_TYPE,
        int       AvgBitrate;      //    NVVE_AVG_BITRATE,
        int       PeakBitrate;     //    NVVE_PEAK_BITRATE,
        int       QP_Level_Intra;  //    NVVE_QP_LEVEL_INTRA,
        int       QP_Level_InterP; //    NVVE_QP_LEVEL_INTER_P,
        int       QP_Level_InterB; //    NVVE_QP_LEVEL_INTER_B,
        int       DeblockMode;     //    NVVE_DEBLOCK_MODE,
        int       ProfileLevel;    //    NVVE_PROFILE_LEVEL,
        int       ForceIntra;      //    NVVE_FORCE_INTRA,
        int       ForceIDR;        //    NVVE_FORCE_IDR,
        int       ClearStat;       //    NVVE_CLEAR_STAT,
        int       DIMode;          //    NVVE_SET_DEINTERLACE,
        int       Presets;         //    NVVE_PRESETS,
        int       DisableCabac;    //    NVVE_DISABLE_CABAC,
        int       NaluFramingType; //    NVVE_CONFIGURE_NALU_FRAMING_TYPE
        int       DisableSPSPPS;   //    NVVE_DISABLE_SPS_PPS

        EncoderParams();
        explicit EncoderParams(const String& configFile);

        void load(const String& configFile);
        void save(const String& configFile) const;
    };

    EncoderParams getParams() const;

    class CV_EXPORTS EncoderCallBack
    {
    public:
        enum PicType
        {
            IFRAME = 1,
            PFRAME = 2,
            BFRAME = 3
        };

        virtual ~EncoderCallBack() {}

        // callback function to signal the start of bitstream that is to be encoded
        // must return pointer to buffer
        virtual uchar* acquireBitStream(int* bufferSize) = 0;

        // callback function to signal that the encoded bitstream is ready to be written to file
        virtual void releaseBitStream(unsigned char* data, int size) = 0;

        // callback function to signal that the encoding operation on the frame has started
        virtual void onBeginFrame(int frameNumber, PicType picType) = 0;

        // callback function signals that the encoding operation on the frame has finished
        virtual void onEndFrame(int frameNumber, PicType picType) = 0;
    };

    class Impl;

private:
    cv::Ptr<Impl> impl_;
};

////////////////////////////////// Video Decoding //////////////////////////////////////////

namespace detail
{
    class FrameQueue;
    class VideoParser;
}

class CV_EXPORTS VideoReader_GPU
{
public:
    enum Codec
    {
        MPEG1 = 0,
        MPEG2,
        MPEG4,
        VC1,
        H264,
        JPEG,
        H264_SVC,
        H264_MVC,

        Uncompressed_YUV420 = (('I'<<24)|('Y'<<16)|('U'<<8)|('V')),   // Y,U,V (4:2:0)
        Uncompressed_YV12   = (('Y'<<24)|('V'<<16)|('1'<<8)|('2')),   // Y,V,U (4:2:0)
        Uncompressed_NV12   = (('N'<<24)|('V'<<16)|('1'<<8)|('2')),   // Y,UV  (4:2:0)
        Uncompressed_YUYV   = (('Y'<<24)|('U'<<16)|('Y'<<8)|('V')),   // YUYV/YUY2 (4:2:2)
        Uncompressed_UYVY   = (('U'<<24)|('Y'<<16)|('V'<<8)|('Y')),   // UYVY (4:2:2)
    };

    enum ChromaFormat
    {
        Monochrome=0,
        YUV420,
        YUV422,
        YUV444,
    };

    struct FormatInfo
    {
        Codec codec;
        ChromaFormat chromaFormat;
        int width;
        int height;
    };

    class VideoSource;

    VideoReader_GPU();
    explicit VideoReader_GPU(const String& filename);
    explicit VideoReader_GPU(const cv::Ptr<VideoSource>& source);

    ~VideoReader_GPU();

    void open(const String& filename);
    void open(const cv::Ptr<VideoSource>& source);
    bool isOpened() const;

    void close();

    bool read(GpuMat& image);

    FormatInfo format() const;
    void dumpFormat(std::ostream& st);

    class CV_EXPORTS VideoSource
    {
    public:
        VideoSource() : frameQueue_(0), videoParser_(0) {}
        virtual ~VideoSource() {}

        virtual FormatInfo format() const = 0;
        virtual void start() = 0;
        virtual void stop() = 0;
        virtual bool isStarted() const = 0;
        virtual bool hasError() const = 0;

        void setFrameQueue(detail::FrameQueue* frameQueue) { frameQueue_ = frameQueue; }
        void setVideoParser(detail::VideoParser* videoParser) { videoParser_ = videoParser; }

    protected:
        bool parseVideoData(const uchar* data, size_t size, bool endOfStream = false);

    private:
        VideoSource(const VideoSource&);
        VideoSource& operator =(const VideoSource&);

        detail::FrameQueue* frameQueue_;
        detail::VideoParser* videoParser_;
    };

    class Impl;

private:
    cv::Ptr<Impl> impl_;
};

}} // namespace cv { namespace gpu {

#endif /* __OPENCV_GPUCODEC_HPP__ */
