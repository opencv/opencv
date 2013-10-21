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

#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;
using namespace cv::cudacodec;

#if !defined(HAVE_NVCUVID) || !defined(WIN32)

cv::cudacodec::EncoderParams::EncoderParams() { throw_no_cuda(); }
cv::cudacodec::EncoderParams::EncoderParams(const String&) { throw_no_cuda(); }
void cv::cudacodec::EncoderParams::load(const String&) { throw_no_cuda(); }
void cv::cudacodec::EncoderParams::save(const String&) const { throw_no_cuda(); }

Ptr<VideoWriter> cv::cudacodec::createVideoWriter(const String&, Size, double, SurfaceFormat) { throw_no_cuda(); return Ptr<VideoWriter>(); }
Ptr<VideoWriter> cv::cudacodec::createVideoWriter(const String&, Size, double, const EncoderParams&, SurfaceFormat) { throw_no_cuda(); return Ptr<VideoWriter>(); }

Ptr<VideoWriter> cv::cudacodec::createVideoWriter(const Ptr<EncoderCallBack>&, Size, double, SurfaceFormat) { throw_no_cuda(); return Ptr<VideoWriter>(); }
Ptr<VideoWriter> cv::cudacodec::createVideoWriter(const Ptr<EncoderCallBack>&, Size, double, const EncoderParams&, SurfaceFormat) { throw_no_cuda(); return Ptr<VideoWriter>(); }

#else // !defined HAVE_CUDA || !defined WIN32

void RGB_to_YV12(const GpuMat& src, GpuMat& dst);

///////////////////////////////////////////////////////////////////////////
// VideoWriterImpl

namespace
{
    class NVEncoderWrapper
    {
    public:
        NVEncoderWrapper() : encoder_(0)
        {
            int err;

            err = NVGetHWEncodeCaps();
            if (err)
                CV_Error(Error::GpuNotSupported, "No CUDA capability present");

            // Create the Encoder API Interface
            err = NVCreateEncoder(&encoder_);
            CV_Assert( err == 0 );
        }

        ~NVEncoderWrapper()
        {
            if (encoder_)
                NVDestroyEncoder(encoder_);
        }

        operator NVEncoder() const
        {
            return encoder_;
        }

    private:
        NVEncoder encoder_;
    };

    enum CodecType
    {
        MPEG1, // not supported yet
        MPEG2, // not supported yet
        MPEG4, // not supported yet
        H264
    };

    class VideoWriterImpl : public VideoWriter
    {
    public:
        VideoWriterImpl(const Ptr<EncoderCallBack>& callback, Size frameSize, double fps, SurfaceFormat format, CodecType codec = H264);
        VideoWriterImpl(const Ptr<EncoderCallBack>& callback, Size frameSize, double fps, const EncoderParams& params, SurfaceFormat format, CodecType codec = H264);

        void write(InputArray frame, bool lastFrame = false);

        EncoderParams getEncoderParams() const;

    private:
        void initEncoder(double fps);
        void setEncodeParams(const EncoderParams& params);
        void initGpuMemory();
        void initCallBacks();
        void createHWEncoder();

        Ptr<EncoderCallBack> callback_;
        Size frameSize_;

        CodecType codec_;
        SurfaceFormat inputFormat_;
        NVVE_SurfaceFormat surfaceFormat_;

        NVEncoderWrapper encoder_;

        GpuMat videoFrame_;
        CUvideoctxlock cuCtxLock_;

        // CallBacks

        static unsigned char* NVENCAPI HandleAcquireBitStream(int* pBufferSize, void* pUserdata);
        static void NVENCAPI HandleReleaseBitStream(int nBytesInBuffer, unsigned char* cb, void* pUserdata);
        static void NVENCAPI HandleOnBeginFrame(const NVVE_BeginFrameInfo* pbfi, void* pUserdata);
        static void NVENCAPI HandleOnEndFrame(const NVVE_EndFrameInfo* pefi, void* pUserdata);
    };

    VideoWriterImpl::VideoWriterImpl(const Ptr<EncoderCallBack>& callback, Size frameSize, double fps, SurfaceFormat format, CodecType codec) :
        callback_(callback),
        frameSize_(frameSize),
        codec_(codec),
        inputFormat_(format),
        cuCtxLock_(0)
    {
        surfaceFormat_ = (inputFormat_ == SF_BGR ? YV12 : static_cast<NVVE_SurfaceFormat>(inputFormat_));

        initEncoder(fps);

        initGpuMemory();

        initCallBacks();

        createHWEncoder();
    }

    VideoWriterImpl::VideoWriterImpl(const Ptr<EncoderCallBack>& callback, Size frameSize, double fps, const EncoderParams& params, SurfaceFormat format, CodecType codec) :
        callback_(callback),
        frameSize_(frameSize),
        codec_(codec),
        inputFormat_(format),
        cuCtxLock_(0)
    {
        surfaceFormat_ = (inputFormat_ == SF_BGR ? YV12 : static_cast<NVVE_SurfaceFormat>(inputFormat_));

        initEncoder(fps);

        setEncodeParams(params);

        initGpuMemory();

        initCallBacks();

        createHWEncoder();
    }

    void VideoWriterImpl::initEncoder(double fps)
    {
        int err;

        // Set codec

        static const unsigned long codecs_id[] =
        {
            NV_CODEC_TYPE_MPEG1, NV_CODEC_TYPE_MPEG2, NV_CODEC_TYPE_MPEG4, NV_CODEC_TYPE_H264, NV_CODEC_TYPE_VC1
        };
        err = NVSetCodec(encoder_, codecs_id[codec_]);
        if (err)
            CV_Error(Error::StsNotImplemented, "Codec format is not supported");

        // Set default params

        err = NVSetDefaultParam(encoder_);
        CV_Assert( err == 0 );

        // Set some common params

        int inputSize[] = { frameSize_.width, frameSize_.height };
        err = NVSetParamValue(encoder_, NVVE_IN_SIZE, &inputSize);
        CV_Assert( err == 0 );
        err = NVSetParamValue(encoder_, NVVE_OUT_SIZE, &inputSize);
        CV_Assert( err == 0 );

        int aspectRatio[] = { frameSize_.width, frameSize_.height, ASPECT_RATIO_DAR };
        err = NVSetParamValue(encoder_, NVVE_ASPECT_RATIO, &aspectRatio);
        CV_Assert( err == 0 );

        // FPS

        int frame_rate = static_cast<int>(fps + 0.5);
        int frame_rate_base = 1;
        while (fabs(static_cast<double>(frame_rate) / frame_rate_base) - fps > 0.001)
        {
            frame_rate_base *= 10;
            frame_rate = static_cast<int>(fps*frame_rate_base + 0.5);
        }
        int FrameRate[] = { frame_rate, frame_rate_base };
        err = NVSetParamValue(encoder_, NVVE_FRAME_RATE, &FrameRate);
        CV_Assert( err == 0 );

        // Select device for encoding

        int gpuID = getDevice();
        err = NVSetParamValue(encoder_, NVVE_FORCE_GPU_SELECTION, &gpuID);
        CV_Assert( err == 0 );
    }

    void VideoWriterImpl::setEncodeParams(const EncoderParams& params)
    {
        int err;

        int P_Interval = params.P_Interval;
        err = NVSetParamValue(encoder_, NVVE_P_INTERVAL, &P_Interval);
        CV_Assert( err == 0 );

        int IDR_Period = params.IDR_Period;
        err = NVSetParamValue(encoder_, NVVE_IDR_PERIOD, &IDR_Period);
        CV_Assert( err == 0 );

        int DynamicGOP = params.DynamicGOP;
        err = NVSetParamValue(encoder_, NVVE_DYNAMIC_GOP, &DynamicGOP);
        CV_Assert( err == 0 );

        NVVE_RateCtrlType RCType = static_cast<NVVE_RateCtrlType>(params.RCType);
        err = NVSetParamValue(encoder_, NVVE_RC_TYPE, &RCType);
        CV_Assert( err == 0 );

        int AvgBitrate = params.AvgBitrate;
        err = NVSetParamValue(encoder_, NVVE_AVG_BITRATE, &AvgBitrate);
        CV_Assert( err == 0 );

        int PeakBitrate = params.PeakBitrate;
        err = NVSetParamValue(encoder_, NVVE_PEAK_BITRATE, &PeakBitrate);
        CV_Assert( err == 0 );

        int QP_Level_Intra = params.QP_Level_Intra;
        err = NVSetParamValue(encoder_, NVVE_QP_LEVEL_INTRA, &QP_Level_Intra);
        CV_Assert( err == 0 );

        int QP_Level_InterP = params.QP_Level_InterP;
        err = NVSetParamValue(encoder_, NVVE_QP_LEVEL_INTER_P, &QP_Level_InterP);
        CV_Assert( err == 0 );

        int QP_Level_InterB = params.QP_Level_InterB;
        err = NVSetParamValue(encoder_, NVVE_QP_LEVEL_INTER_B, &QP_Level_InterB);
        CV_Assert( err == 0 );

        int DeblockMode = params.DeblockMode;
        err = NVSetParamValue(encoder_, NVVE_DEBLOCK_MODE, &DeblockMode);
        CV_Assert( err == 0 );

        int ProfileLevel = params.ProfileLevel;
        err = NVSetParamValue(encoder_, NVVE_PROFILE_LEVEL, &ProfileLevel);
        CV_Assert( err == 0 );

        int ForceIntra = params.ForceIntra;
        err = NVSetParamValue(encoder_, NVVE_FORCE_INTRA, &ForceIntra);
        CV_Assert( err == 0 );

        int ForceIDR = params.ForceIDR;
        err = NVSetParamValue(encoder_, NVVE_FORCE_IDR, &ForceIDR);
        CV_Assert( err == 0 );

        int ClearStat = params.ClearStat;
        err = NVSetParamValue(encoder_, NVVE_CLEAR_STAT, &ClearStat);
        CV_Assert( err == 0 );

        NVVE_DI_MODE DIMode = static_cast<NVVE_DI_MODE>(params.DIMode);
        err = NVSetParamValue(encoder_, NVVE_SET_DEINTERLACE, &DIMode);
        CV_Assert( err == 0 );

        if (params.Presets != -1)
        {
            NVVE_PRESETS_TARGET Presets = static_cast<NVVE_PRESETS_TARGET>(params.Presets);
            err = NVSetParamValue(encoder_, NVVE_PRESETS, &Presets);
            CV_Assert( err == 0 );
        }

        int DisableCabac = params.DisableCabac;
        err = NVSetParamValue(encoder_, NVVE_DISABLE_CABAC, &DisableCabac);
        CV_Assert( err == 0 );

        int NaluFramingType = params.NaluFramingType;
        err = NVSetParamValue(encoder_, NVVE_CONFIGURE_NALU_FRAMING_TYPE, &NaluFramingType);
        CV_Assert( err == 0 );

        int DisableSPSPPS = params.DisableSPSPPS;
        err = NVSetParamValue(encoder_, NVVE_DISABLE_SPS_PPS, &DisableSPSPPS);
        CV_Assert( err == 0 );
    }

    EncoderParams VideoWriterImpl::getEncoderParams() const
    {
        int err;

        EncoderParams params;

        int P_Interval;
        err = NVGetParamValue(encoder_, NVVE_P_INTERVAL, &P_Interval);
        CV_Assert( err == 0 );
        params.P_Interval = P_Interval;

        int IDR_Period;
        err = NVGetParamValue(encoder_, NVVE_IDR_PERIOD, &IDR_Period);
        CV_Assert( err == 0 );
        params.IDR_Period = IDR_Period;

        int DynamicGOP;
        err = NVGetParamValue(encoder_, NVVE_DYNAMIC_GOP, &DynamicGOP);
        CV_Assert( err == 0 );
        params.DynamicGOP = DynamicGOP;

        NVVE_RateCtrlType RCType;
        err = NVGetParamValue(encoder_, NVVE_RC_TYPE, &RCType);
        CV_Assert( err == 0 );
        params.RCType = RCType;

        int AvgBitrate;
        err = NVGetParamValue(encoder_, NVVE_AVG_BITRATE, &AvgBitrate);
        CV_Assert( err == 0 );
        params.AvgBitrate = AvgBitrate;

        int PeakBitrate;
        err = NVGetParamValue(encoder_, NVVE_PEAK_BITRATE, &PeakBitrate);
        CV_Assert( err == 0 );
        params.PeakBitrate = PeakBitrate;

        int QP_Level_Intra;
        err = NVGetParamValue(encoder_, NVVE_QP_LEVEL_INTRA, &QP_Level_Intra);
        CV_Assert( err == 0 );
        params.QP_Level_Intra = QP_Level_Intra;

        int QP_Level_InterP;
        err = NVGetParamValue(encoder_, NVVE_QP_LEVEL_INTER_P, &QP_Level_InterP);
        CV_Assert( err == 0 );
        params.QP_Level_InterP = QP_Level_InterP;

        int QP_Level_InterB;
        err = NVGetParamValue(encoder_, NVVE_QP_LEVEL_INTER_B, &QP_Level_InterB);
        CV_Assert( err == 0 );
        params.QP_Level_InterB = QP_Level_InterB;

        int DeblockMode;
        err = NVGetParamValue(encoder_, NVVE_DEBLOCK_MODE, &DeblockMode);
        CV_Assert( err == 0 );
        params.DeblockMode = DeblockMode;

        int ProfileLevel;
        err = NVGetParamValue(encoder_, NVVE_PROFILE_LEVEL, &ProfileLevel);
        CV_Assert( err == 0 );
        params.ProfileLevel = ProfileLevel;

        int ForceIntra;
        err = NVGetParamValue(encoder_, NVVE_FORCE_INTRA, &ForceIntra);
        CV_Assert( err == 0 );
        params.ForceIntra = ForceIntra;

        int ForceIDR;
        err = NVGetParamValue(encoder_, NVVE_FORCE_IDR, &ForceIDR);
        CV_Assert( err == 0 );
        params.ForceIDR = ForceIDR;

        int ClearStat;
        err = NVGetParamValue(encoder_, NVVE_CLEAR_STAT, &ClearStat);
        CV_Assert( err == 0 );
        params.ClearStat = ClearStat;

        NVVE_DI_MODE DIMode;
        err = NVGetParamValue(encoder_, NVVE_SET_DEINTERLACE, &DIMode);
        CV_Assert( err == 0 );
        params.DIMode = DIMode;

        params.Presets = -1;

        int DisableCabac;
        err = NVGetParamValue(encoder_, NVVE_DISABLE_CABAC, &DisableCabac);
        CV_Assert( err == 0 );
        params.DisableCabac = DisableCabac;

        int NaluFramingType;
        err = NVGetParamValue(encoder_, NVVE_CONFIGURE_NALU_FRAMING_TYPE, &NaluFramingType);
        CV_Assert( err == 0 );
        params.NaluFramingType = NaluFramingType;

        int DisableSPSPPS;
        err = NVGetParamValue(encoder_, NVVE_DISABLE_SPS_PPS, &DisableSPSPPS);
        CV_Assert( err == 0 );
        params.DisableSPSPPS = DisableSPSPPS;

        return params;
    }

    void VideoWriterImpl::initGpuMemory()
    {
        int err;

        // initialize context
        GpuMat temp(1, 1, CV_8U);
        temp.release();

        static const int bpp[] =
        {
            16, // UYVY, 4:2:2
            16, // YUY2, 4:2:2
            12, // YV12, 4:2:0
            12, // NV12, 4:2:0
            12, // IYUV, 4:2:0
        };

        CUcontext cuContext;
        cuSafeCall( cuCtxGetCurrent(&cuContext) );

        // Allocate the CUDA memory Pitched Surface
        if (surfaceFormat_ == UYVY || surfaceFormat_ == YUY2)
            videoFrame_.create(frameSize_.height, (frameSize_.width * bpp[surfaceFormat_]) / 8, CV_8UC1);
        else
            videoFrame_.create((frameSize_.height * bpp[surfaceFormat_]) / 8, frameSize_.width, CV_8UC1);

        // Create the Video Context Lock (used for synchronization)
        cuSafeCall( cuvidCtxLockCreate(&cuCtxLock_, cuContext) );

        // If we are using GPU Device Memory with NVCUVENC, it is necessary to create a
        // CUDA Context with a Context Lock cuvidCtxLock.  The Context Lock needs to be passed to NVCUVENC

        int iUseDeviceMem = 1;
        err = NVSetParamValue(encoder_, NVVE_DEVICE_MEMORY_INPUT, &iUseDeviceMem);
        CV_Assert( err == 0 );

        err = NVSetParamValue(encoder_, NVVE_DEVICE_CTX_LOCK, &cuCtxLock_);
        CV_Assert( err == 0 );
    }

    void VideoWriterImpl::initCallBacks()
    {
        NVVE_CallbackParams cb;
        memset(&cb, 0, sizeof(NVVE_CallbackParams));

        cb.pfnacquirebitstream = HandleAcquireBitStream;
        cb.pfnonbeginframe     = HandleOnBeginFrame;
        cb.pfnonendframe       = HandleOnEndFrame;
        cb.pfnreleasebitstream = HandleReleaseBitStream;

        NVRegisterCB(encoder_, cb, this);
    }

    void VideoWriterImpl::createHWEncoder()
    {
        int err;

        // Create the NVIDIA HW resources for Encoding on NVIDIA hardware
        err = NVCreateHWEncoder(encoder_);
        CV_Assert( err == 0 );
    }

    // UYVY/YUY2 are both 4:2:2 formats (16bpc)
    // Luma, U, V are interleaved, chroma is subsampled (w/2,h)
    void copyUYVYorYUY2Frame(Size frameSize, const GpuMat& src, GpuMat& dst)
    {
        // Source is YUVY/YUY2 4:2:2, the YUV data in a packed and interleaved

        // YUV Copy setup
        CUDA_MEMCPY2D stCopyYUV422;
        memset(&stCopyYUV422, 0, sizeof(CUDA_MEMCPY2D));

        stCopyYUV422.srcXInBytes          = 0;
        stCopyYUV422.srcY                 = 0;
        stCopyYUV422.srcMemoryType        = CU_MEMORYTYPE_DEVICE;
        stCopyYUV422.srcHost              = 0;
        stCopyYUV422.srcDevice            = (CUdeviceptr) src.data;
        stCopyYUV422.srcArray             = 0;
        stCopyYUV422.srcPitch             = src.step;

        stCopyYUV422.dstXInBytes          = 0;
        stCopyYUV422.dstY                 = 0;
        stCopyYUV422.dstMemoryType        = CU_MEMORYTYPE_DEVICE;
        stCopyYUV422.dstHost              = 0;
        stCopyYUV422.dstDevice            = (CUdeviceptr) dst.data;
        stCopyYUV422.dstArray             = 0;
        stCopyYUV422.dstPitch             = dst.step;

        stCopyYUV422.WidthInBytes         = frameSize.width * 2;
        stCopyYUV422.Height               = frameSize.height;

        // DMA Luma/Chroma
        cuSafeCall( cuMemcpy2D(&stCopyYUV422) );
    }

    // YV12/IYUV are both 4:2:0 planar formats (12bpc)
    // Luma, U, V chroma planar (12bpc), chroma is subsampled (w/2,h/2)
    void copyYV12orIYUVFrame(Size frameSize, const GpuMat& src, GpuMat& dst)
    {
        // Source is YV12/IYUV, this native format is converted to NV12 format by the video encoder

        // (1) luma copy setup
        CUDA_MEMCPY2D stCopyLuma;
        memset(&stCopyLuma, 0, sizeof(CUDA_MEMCPY2D));

        stCopyLuma.srcXInBytes          = 0;
        stCopyLuma.srcY                 = 0;
        stCopyLuma.srcMemoryType        = CU_MEMORYTYPE_DEVICE;
        stCopyLuma.srcHost              = 0;
        stCopyLuma.srcDevice            = (CUdeviceptr) src.data;
        stCopyLuma.srcArray             = 0;
        stCopyLuma.srcPitch             = src.step;

        stCopyLuma.dstXInBytes          = 0;
        stCopyLuma.dstY                 = 0;
        stCopyLuma.dstMemoryType        = CU_MEMORYTYPE_DEVICE;
        stCopyLuma.dstHost              = 0;
        stCopyLuma.dstDevice            = (CUdeviceptr) dst.data;
        stCopyLuma.dstArray             = 0;
        stCopyLuma.dstPitch             = dst.step;

        stCopyLuma.WidthInBytes         = frameSize.width;
        stCopyLuma.Height               = frameSize.height;

        // (2) chroma copy setup, U/V can be done together
        CUDA_MEMCPY2D stCopyChroma;
        memset(&stCopyChroma, 0, sizeof(CUDA_MEMCPY2D));

        stCopyChroma.srcXInBytes        = 0;
        stCopyChroma.srcY               = frameSize.height << 1; // U/V chroma offset
        stCopyChroma.srcMemoryType      = CU_MEMORYTYPE_DEVICE;
        stCopyChroma.srcHost            = 0;
        stCopyChroma.srcDevice          = (CUdeviceptr) src.data;
        stCopyChroma.srcArray           = 0;
        stCopyChroma.srcPitch           = src.step >> 1; // chroma is subsampled by 2 (but it has U/V are next to each other)

        stCopyChroma.dstXInBytes        = 0;
        stCopyChroma.dstY               = frameSize.height << 1; // chroma offset (srcY*srcPitch now points to the chroma planes)
        stCopyChroma.dstMemoryType      = CU_MEMORYTYPE_DEVICE;
        stCopyChroma.dstHost            = 0;
        stCopyChroma.dstDevice          = (CUdeviceptr) dst.data;
        stCopyChroma.dstArray           = 0;
        stCopyChroma.dstPitch           = dst.step >> 1;

        stCopyChroma.WidthInBytes       = frameSize.width >> 1;
        stCopyChroma.Height             = frameSize.height; // U/V are sent together

        // DMA Luma
        cuSafeCall( cuMemcpy2D(&stCopyLuma) );

        // DMA Chroma channels (UV side by side)
        cuSafeCall( cuMemcpy2D(&stCopyChroma) );
    }

    // NV12 is 4:2:0 format (12bpc)
    // Luma followed by U/V chroma interleaved (12bpc), chroma is subsampled (w/2,h/2)
    void copyNV12Frame(Size frameSize, const GpuMat& src, GpuMat& dst)
    {
        // Source is NV12 in pitch linear memory
        // Because we are assume input is NV12 (if we take input in the native format), the encoder handles NV12 as a native format in pitch linear memory

        // Luma/Chroma can be done in a single transfer
        CUDA_MEMCPY2D stCopyNV12;
        memset(&stCopyNV12, 0, sizeof(CUDA_MEMCPY2D));

        stCopyNV12.srcXInBytes          = 0;
        stCopyNV12.srcY                 = 0;
        stCopyNV12.srcMemoryType        = CU_MEMORYTYPE_DEVICE;
        stCopyNV12.srcHost              = 0;
        stCopyNV12.srcDevice            = (CUdeviceptr) src.data;
        stCopyNV12.srcArray             = 0;
        stCopyNV12.srcPitch             = src.step;

        stCopyNV12.dstXInBytes          = 0;
        stCopyNV12.dstY                 = 0;
        stCopyNV12.dstMemoryType        = CU_MEMORYTYPE_DEVICE;
        stCopyNV12.dstHost              = 0;
        stCopyNV12.dstDevice            = (CUdeviceptr) dst.data;
        stCopyNV12.dstArray             = 0;
        stCopyNV12.dstPitch             = dst.step;

        stCopyNV12.WidthInBytes         = frameSize.width;
        stCopyNV12.Height               = (frameSize.height * 3) >> 1;

        // DMA Luma/Chroma
        cuSafeCall( cuMemcpy2D(&stCopyNV12) );
    }

    void VideoWriterImpl::write(InputArray _frame, bool lastFrame)
    {
        GpuMat frame = _frame.getGpuMat();

        if (inputFormat_ == SF_BGR)
        {
            CV_Assert( frame.size() == frameSize_ );
            CV_Assert( frame.type() == CV_8UC1 || frame.type() == CV_8UC3 || frame.type() == CV_8UC4 );
        }
        else
        {
            CV_Assert( frame.size() == videoFrame_.size() );
            CV_Assert( frame.type() == videoFrame_.type() );
        }

        NVVE_EncodeFrameParams efparams;
        efparams.Width = frameSize_.width;
        efparams.Height = frameSize_.height;
        efparams.Pitch = static_cast<int>(videoFrame_.step);
        efparams.SurfFmt = surfaceFormat_;
        efparams.PictureStruc = FRAME_PICTURE;
        efparams.topfieldfirst =  0;
        efparams.repeatFirstField = 0;
        efparams.progressiveFrame = (surfaceFormat_ == NV12) ? 1 : 0;
        efparams.bLast = lastFrame;
        efparams.picBuf = 0; // Must be set to NULL in order to support device memory input

        // Don't forget we need to lock/unlock between memcopies
        cuSafeCall( cuvidCtxLock(cuCtxLock_, 0) );

        if (inputFormat_ == SF_BGR)
        {
            RGB_to_YV12(frame, videoFrame_);
        }
        else
        {
            switch (surfaceFormat_)
            {
            case UYVY: // UYVY (4:2:2)
            case YUY2: // YUY2 (4:2:2)
                copyUYVYorYUY2Frame(frameSize_, frame, videoFrame_);
                break;

            case YV12: // YV12 (4:2:0), Y V U
            case IYUV: // IYUV (4:2:0), Y U V
                copyYV12orIYUVFrame(frameSize_, frame, videoFrame_);
                break;

            case NV12: // NV12 (4:2:0)
                copyNV12Frame(frameSize_, frame, videoFrame_);
                break;
            }
        }

        cuSafeCall( cuvidCtxUnlock(cuCtxLock_, 0) );

        int err = NVEncodeFrame(encoder_, &efparams, 0, videoFrame_.data);
        CV_Assert( err == 0 );
    }

    unsigned char* NVENCAPI VideoWriterImpl::HandleAcquireBitStream(int* pBufferSize, void* pUserdata)
    {
        VideoWriterImpl* thiz = static_cast<VideoWriterImpl*>(pUserdata);

        return thiz->callback_->acquireBitStream(pBufferSize);
    }

    void NVENCAPI VideoWriterImpl::HandleReleaseBitStream(int nBytesInBuffer, unsigned char* cb, void* pUserdata)
    {
        VideoWriterImpl* thiz = static_cast<VideoWriterImpl*>(pUserdata);

        thiz->callback_->releaseBitStream(cb, nBytesInBuffer);
    }

    void NVENCAPI VideoWriterImpl::HandleOnBeginFrame(const NVVE_BeginFrameInfo* pbfi, void* pUserdata)
    {
        VideoWriterImpl* thiz = static_cast<VideoWriterImpl*>(pUserdata);

        thiz->callback_->onBeginFrame(pbfi->nFrameNumber, static_cast<EncoderCallBack::PicType>(pbfi->nPicType));
    }

    void NVENCAPI VideoWriterImpl::HandleOnEndFrame(const NVVE_EndFrameInfo* pefi, void* pUserdata)
    {
        VideoWriterImpl* thiz = static_cast<VideoWriterImpl*>(pUserdata);

        thiz->callback_->onEndFrame(pefi->nFrameNumber, static_cast<EncoderCallBack::PicType>(pefi->nPicType));
    }

    ///////////////////////////////////////////////////////////////////////////
    // FFMPEG

    class EncoderCallBackFFMPEG : public EncoderCallBack
    {
    public:
        EncoderCallBackFFMPEG(const String& fileName, Size frameSize, double fps);
        ~EncoderCallBackFFMPEG();

        unsigned char* acquireBitStream(int* bufferSize);
        void releaseBitStream(unsigned char* data, int size);
        void onBeginFrame(int frameNumber, PicType picType);
        void onEndFrame(int frameNumber, PicType picType);

    private:
        static bool init_MediaStream_FFMPEG();

        struct OutputMediaStream_FFMPEG* stream_;
        std::vector<uchar> buf_;
        bool isKeyFrame_;

        static Create_OutputMediaStream_FFMPEG_Plugin create_OutputMediaStream_FFMPEG_p;
        static Release_OutputMediaStream_FFMPEG_Plugin release_OutputMediaStream_FFMPEG_p;
        static Write_OutputMediaStream_FFMPEG_Plugin write_OutputMediaStream_FFMPEG_p;
    };

    Create_OutputMediaStream_FFMPEG_Plugin EncoderCallBackFFMPEG::create_OutputMediaStream_FFMPEG_p = 0;
    Release_OutputMediaStream_FFMPEG_Plugin EncoderCallBackFFMPEG::release_OutputMediaStream_FFMPEG_p = 0;
    Write_OutputMediaStream_FFMPEG_Plugin EncoderCallBackFFMPEG::write_OutputMediaStream_FFMPEG_p = 0;

    bool EncoderCallBackFFMPEG::init_MediaStream_FFMPEG()
    {
        static bool initialized = false;

        if (!initialized)
        {
            #if defined(WIN32) || defined(_WIN32)
                const char* module_name = "opencv_ffmpeg"
                    CVAUX_STR(CV_VERSION_EPOCH) CVAUX_STR(CV_VERSION_MAJOR) CVAUX_STR(CV_VERSION_MINOR)
                #if (defined _MSC_VER && defined _M_X64) || (defined __GNUC__ && defined __x86_64__)
                    "_64"
                #endif
                    ".dll";

                static HMODULE cvFFOpenCV = LoadLibrary(module_name);

                if (cvFFOpenCV)
                {
                    create_OutputMediaStream_FFMPEG_p =
                        (Create_OutputMediaStream_FFMPEG_Plugin)GetProcAddress(cvFFOpenCV, "create_OutputMediaStream_FFMPEG");
                    release_OutputMediaStream_FFMPEG_p =
                        (Release_OutputMediaStream_FFMPEG_Plugin)GetProcAddress(cvFFOpenCV, "release_OutputMediaStream_FFMPEG");
                    write_OutputMediaStream_FFMPEG_p =
                        (Write_OutputMediaStream_FFMPEG_Plugin)GetProcAddress(cvFFOpenCV, "write_OutputMediaStream_FFMPEG");

                    initialized = create_OutputMediaStream_FFMPEG_p != 0 && release_OutputMediaStream_FFMPEG_p != 0 && write_OutputMediaStream_FFMPEG_p != 0;
                }
            #elif defined(HAVE_FFMPEG)
                create_OutputMediaStream_FFMPEG_p = create_OutputMediaStream_FFMPEG;
                release_OutputMediaStream_FFMPEG_p = release_OutputMediaStream_FFMPEG;
                write_OutputMediaStream_FFMPEG_p = write_OutputMediaStream_FFMPEG;

                initialized = true;
            #endif
        }

        return initialized;
    }

    EncoderCallBackFFMPEG::EncoderCallBackFFMPEG(const String& fileName, Size frameSize, double fps) :
        stream_(0), isKeyFrame_(false)
    {
        int buf_size = std::max(frameSize.area() * 4, 1024 * 1024);
        buf_.resize(buf_size);

        CV_Assert( init_MediaStream_FFMPEG() );

        stream_ = create_OutputMediaStream_FFMPEG_p(fileName.c_str(), frameSize.width, frameSize.height, fps);
        CV_Assert( stream_ != 0 );
    }

    EncoderCallBackFFMPEG::~EncoderCallBackFFMPEG()
    {
        release_OutputMediaStream_FFMPEG_p(stream_);
    }

    unsigned char* EncoderCallBackFFMPEG::acquireBitStream(int* bufferSize)
    {
        *bufferSize = static_cast<int>(buf_.size());
        return &buf_[0];
    }

    void EncoderCallBackFFMPEG::releaseBitStream(unsigned char* data, int size)
    {
        write_OutputMediaStream_FFMPEG_p(stream_, data, size, isKeyFrame_);
    }

    void EncoderCallBackFFMPEG::onBeginFrame(int frameNumber, PicType picType)
    {
        (void) frameNumber;
        isKeyFrame_ = (picType == IFRAME);
    }

    void EncoderCallBackFFMPEG::onEndFrame(int frameNumber, PicType picType)
    {
        (void) frameNumber;
        (void) picType;
    }
}

///////////////////////////////////////////////////////////////////////////
// EncoderParams

cv::cudacodec::EncoderParams::EncoderParams()
{
    P_Interval = 3;
    IDR_Period = 15;
    DynamicGOP = 0;
    RCType = 1;
    AvgBitrate = 4000000;
    PeakBitrate = 10000000;
    QP_Level_Intra = 25;
    QP_Level_InterP = 28;
    QP_Level_InterB = 31;
    DeblockMode = 1;
    ProfileLevel = 65357;
    ForceIntra = 0;
    ForceIDR = 0;
    ClearStat = 0;
    DIMode = 1;
    Presets = 2;
    DisableCabac = 0;
    NaluFramingType = 0;
    DisableSPSPPS = 0;
}

cv::cudacodec::EncoderParams::EncoderParams(const String& configFile)
{
    load(configFile);
}

void cv::cudacodec::EncoderParams::load(const String& configFile)
{
    FileStorage fs(configFile, FileStorage::READ);
    CV_Assert( fs.isOpened() );

    read(fs["P_Interval"     ], P_Interval, 3);
    read(fs["IDR_Period"     ], IDR_Period, 15);
    read(fs["DynamicGOP"     ], DynamicGOP, 0);
    read(fs["RCType"         ], RCType, 1);
    read(fs["AvgBitrate"     ], AvgBitrate, 4000000);
    read(fs["PeakBitrate"    ], PeakBitrate, 10000000);
    read(fs["QP_Level_Intra" ], QP_Level_Intra, 25);
    read(fs["QP_Level_InterP"], QP_Level_InterP, 28);
    read(fs["QP_Level_InterB"], QP_Level_InterB, 31);
    read(fs["DeblockMode"    ], DeblockMode, 1);
    read(fs["ProfileLevel"   ], ProfileLevel, 65357);
    read(fs["ForceIntra"     ], ForceIntra, 0);
    read(fs["ForceIDR"       ], ForceIDR, 0);
    read(fs["ClearStat"      ], ClearStat, 0);
    read(fs["DIMode"         ], DIMode, 1);
    read(fs["Presets"        ], Presets, 2);
    read(fs["DisableCabac"   ], DisableCabac, 0);
    read(fs["NaluFramingType"], NaluFramingType, 0);
    read(fs["DisableSPSPPS"  ], DisableSPSPPS, 0);
}

void cv::cudacodec::EncoderParams::save(const String& configFile) const
{
    FileStorage fs(configFile, FileStorage::WRITE);
    CV_Assert( fs.isOpened() );

    write(fs, "P_Interval"     , P_Interval);
    write(fs, "IDR_Period"     , IDR_Period);
    write(fs, "DynamicGOP"     , DynamicGOP);
    write(fs, "RCType"         , RCType);
    write(fs, "AvgBitrate"     , AvgBitrate);
    write(fs, "PeakBitrate"    , PeakBitrate);
    write(fs, "QP_Level_Intra" , QP_Level_Intra);
    write(fs, "QP_Level_InterP", QP_Level_InterP);
    write(fs, "QP_Level_InterB", QP_Level_InterB);
    write(fs, "DeblockMode"    , DeblockMode);
    write(fs, "ProfileLevel"   , ProfileLevel);
    write(fs, "ForceIntra"     , ForceIntra);
    write(fs, "ForceIDR"       , ForceIDR);
    write(fs, "ClearStat"      , ClearStat);
    write(fs, "DIMode"         , DIMode);
    write(fs, "Presets"        , Presets);
    write(fs, "DisableCabac"   , DisableCabac);
    write(fs, "NaluFramingType", NaluFramingType);
    write(fs, "DisableSPSPPS"  , DisableSPSPPS);
}

///////////////////////////////////////////////////////////////////////////
// createVideoWriter

Ptr<VideoWriter> cv::cudacodec::createVideoWriter(const String& fileName, Size frameSize, double fps, SurfaceFormat format)
{
    Ptr<EncoderCallBack> encoderCallback(new EncoderCallBackFFMPEG(fileName, frameSize, fps));
    return createVideoWriter(encoderCallback, frameSize, fps, format);
}

Ptr<VideoWriter> cv::cudacodec::createVideoWriter(const String& fileName, Size frameSize, double fps, const EncoderParams& params, SurfaceFormat format)
{
    Ptr<EncoderCallBack> encoderCallback(new EncoderCallBackFFMPEG(fileName, frameSize, fps));
    return createVideoWriter(encoderCallback, frameSize, fps, params, format);
}

Ptr<VideoWriter> cv::cudacodec::createVideoWriter(const Ptr<EncoderCallBack>& encoderCallback, Size frameSize, double fps, SurfaceFormat format)
{
    return makePtr<VideoWriterImpl>(encoderCallback, frameSize, fps, format);
}

Ptr<VideoWriter> cv::cudacodec::createVideoWriter(const Ptr<EncoderCallBack>& encoderCallback, Size frameSize, double fps, const EncoderParams& params, SurfaceFormat format)
{
    return makePtr<VideoWriterImpl>(encoderCallback, frameSize, fps, params, format);
}

#endif // !defined HAVE_CUDA || !defined WIN32
