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

#if !defined HAVE_CUDA || !defined WIN32

cv::gpu::VideoWriter_GPU::VideoWriter_GPU() { throw_nogpu(); }
cv::gpu::VideoWriter_GPU::VideoWriter_GPU(const std::string&, cv::Size, double, SurfaceFormat) { throw_nogpu(); }
cv::gpu::VideoWriter_GPU::VideoWriter_GPU(const std::string&, cv::Size, double, const EncoderParams&, SurfaceFormat) { throw_nogpu(); }
cv::gpu::VideoWriter_GPU::VideoWriter_GPU(const cv::Ptr<EncoderCallBack>&, cv::Size, double, SurfaceFormat) { throw_nogpu(); }
cv::gpu::VideoWriter_GPU::VideoWriter_GPU(const cv::Ptr<EncoderCallBack>&, cv::Size, double, const EncoderParams&, SurfaceFormat) { throw_nogpu(); }
cv::gpu::VideoWriter_GPU::~VideoWriter_GPU() {}
void cv::gpu::VideoWriter_GPU::open(const std::string&, cv::Size, double, SurfaceFormat) { throw_nogpu(); }
void cv::gpu::VideoWriter_GPU::open(const std::string&, cv::Size, double, const EncoderParams&, SurfaceFormat) { throw_nogpu(); }
void cv::gpu::VideoWriter_GPU::open(const cv::Ptr<EncoderCallBack>&, cv::Size, double, SurfaceFormat) { throw_nogpu(); }
void cv::gpu::VideoWriter_GPU::open(const cv::Ptr<EncoderCallBack>&, cv::Size, double, const EncoderParams&, SurfaceFormat) { throw_nogpu(); }
bool cv::gpu::VideoWriter_GPU::isOpened() const { return false; }
void cv::gpu::VideoWriter_GPU::close() {}
void cv::gpu::VideoWriter_GPU::write(const cv::gpu::GpuMat&, bool) { throw_nogpu(); }
cv::gpu::VideoWriter_GPU::EncoderParams cv::gpu::VideoWriter_GPU::getParams() const { EncoderParams params; throw_nogpu(); return params; }

cv::gpu::VideoWriter_GPU::EncoderParams::EncoderParams() { throw_nogpu(); }
cv::gpu::VideoWriter_GPU::EncoderParams::EncoderParams(const std::string&) { throw_nogpu(); }
void cv::gpu::VideoWriter_GPU::EncoderParams::load(const std::string&) { throw_nogpu(); }
void cv::gpu::VideoWriter_GPU::EncoderParams::save(const std::string&) const { throw_nogpu(); }

#else // !defined HAVE_CUDA || !defined WIN32

#ifdef HAVE_FFMPEG
    #ifdef NEW_FFMPEG
        #include "cap_ffmpeg_impl_v2.hpp"
    #else
        #include "cap_ffmpeg_impl.hpp"
    #endif
#else
    #include "cap_ffmpeg_api.hpp"
#endif


///////////////////////////////////////////////////////////////////////////
// VideoWriter_GPU::Impl

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
                CV_Error(CV_GpuNotSupported, "No CUDA capability present");

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
        MPEG1, //not supported yet
        MPEG2, //not supported yet
        MPEG4, //not supported yet
        H264
    };
}

class cv::gpu::VideoWriter_GPU::Impl
{
public:
    Impl(const cv::Ptr<EncoderCallBack>& callback, cv::Size frameSize, double fps, SurfaceFormat format, CodecType codec = H264);
    Impl(const cv::Ptr<EncoderCallBack>& callback, cv::Size frameSize, double fps, const EncoderParams& params, SurfaceFormat format, CodecType codec = H264);

    void write(const cv::gpu::GpuMat& image, bool lastFrame);

    EncoderParams getParams() const;

private:
    Impl(const Impl&);
    Impl& operator=(const Impl&);

    void initEncoder(double fps);
    void setEncodeParams(const EncoderParams& params);
    void initGpuMemory();
    void initCallBacks();
    void createHWEncoder();
    
    cv::Ptr<EncoderCallBack> callback_;
    cv::Size frameSize_;

    CodecType codec_;
    SurfaceFormat inputFormat_;
    NVVE_SurfaceFormat surfaceFormat_;

    NVEncoderWrapper encoder_;

    cv::gpu::GpuMat videoFrame_;
    CUvideoctxlock cuCtxLock_;

    // CallBacks

    static unsigned char* NVENCAPI HandleAcquireBitStream(int* pBufferSize, void* pUserdata);
    static void NVENCAPI HandleReleaseBitStream(int nBytesInBuffer, unsigned char* cb, void* pUserdata);
    static void NVENCAPI HandleOnBeginFrame(const NVVE_BeginFrameInfo* pbfi, void* pUserdata);
    static void NVENCAPI HandleOnEndFrame(const NVVE_EndFrameInfo* pefi, void* pUserdata);
};

cv::gpu::VideoWriter_GPU::Impl::Impl(const cv::Ptr<EncoderCallBack>& callback, cv::Size frameSize, double fps, SurfaceFormat format, CodecType codec) :
    callback_(callback),
    frameSize_(frameSize),
    codec_(codec),
    inputFormat_(format),
    cuCtxLock_(0)
{
    surfaceFormat_ = inputFormat_ == SF_BGR ? YV12 : static_cast<NVVE_SurfaceFormat>(inputFormat_);

    initEncoder(fps);

    initGpuMemory();

    initCallBacks();

    createHWEncoder();
}

cv::gpu::VideoWriter_GPU::Impl::Impl(const cv::Ptr<EncoderCallBack>& callback, cv::Size frameSize, double fps, const EncoderParams& params, SurfaceFormat format, CodecType codec) :
    callback_(callback),
    frameSize_(frameSize),
    codec_(codec),
    inputFormat_(format),
    cuCtxLock_(0)
{
    surfaceFormat_ = inputFormat_ == SF_BGR ? YV12 : static_cast<NVVE_SurfaceFormat>(inputFormat_);

    initEncoder(fps);

    setEncodeParams(params);

    initGpuMemory();

    initCallBacks();

    createHWEncoder();
}

void cv::gpu::VideoWriter_GPU::Impl::initEncoder(double fps)
{
    int err;

    // Set codec

    static const unsigned long codecs_id[] = 
    {
        NV_CODEC_TYPE_MPEG1, NV_CODEC_TYPE_MPEG2, NV_CODEC_TYPE_MPEG4, NV_CODEC_TYPE_H264, NV_CODEC_TYPE_VC1
    };
    err = NVSetCodec(encoder_, codecs_id[codec_]);
    if (err)
        CV_Error(CV_StsNotImplemented, "Codec format is not supported");

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

    int gpuID = cv::gpu::getDevice();
    err = NVSetParamValue(encoder_, NVVE_FORCE_GPU_SELECTION, &gpuID);
    CV_Assert( err == 0 );
}

void cv::gpu::VideoWriter_GPU::Impl::setEncodeParams(const EncoderParams& params)
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
        CV_Assert ( err == 0 );
    }

    int DisableCabac = params.DisableCabac;
    err = NVSetParamValue(encoder_, NVVE_DISABLE_CABAC, &DisableCabac);
    CV_Assert ( err == 0 );

    int NaluFramingType = params.NaluFramingType;
    err = NVSetParamValue(encoder_, NVVE_CONFIGURE_NALU_FRAMING_TYPE, &NaluFramingType);
    CV_Assert ( err == 0 );

    int DisableSPSPPS = params.DisableSPSPPS;
    err = NVSetParamValue(encoder_, NVVE_DISABLE_SPS_PPS, &DisableSPSPPS);
    CV_Assert ( err == 0 );
}

cv::gpu::VideoWriter_GPU::EncoderParams cv::gpu::VideoWriter_GPU::Impl::getParams() const
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
    CV_Assert ( err == 0 );
    params.DisableCabac = DisableCabac;

    int NaluFramingType;
    err = NVGetParamValue(encoder_, NVVE_CONFIGURE_NALU_FRAMING_TYPE, &NaluFramingType);
    CV_Assert ( err == 0 );
    params.NaluFramingType = NaluFramingType;

    int DisableSPSPPS;
    err = NVGetParamValue(encoder_, NVVE_DISABLE_SPS_PPS, &DisableSPSPPS);
    CV_Assert ( err == 0 );
    params.DisableSPSPPS = DisableSPSPPS;

    return params;
}

void cv::gpu::VideoWriter_GPU::Impl::initGpuMemory()
{
    int err;
    CUresult cuRes;

    // initialize context
    cv::gpu::GpuMat temp(1, 1, CV_8U);
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
    cuRes = cuCtxGetCurrent(&cuContext);
    CV_Assert( cuRes == CUDA_SUCCESS );

    // Allocate the CUDA memory Pitched Surface
    if (surfaceFormat_ == UYVY || surfaceFormat_ == YUY2)
        videoFrame_.create(frameSize_.height, (frameSize_.width * bpp[surfaceFormat_]) / 8, CV_8UC1);
    else
        videoFrame_.create((frameSize_.height * bpp[surfaceFormat_]) / 8, frameSize_.width, CV_8UC1);

    // Create the Video Context Lock (used for synchronization)
    cuRes = cuvidCtxLockCreate(&cuCtxLock_, cuContext);
    CV_Assert( cuRes == CUDA_SUCCESS );

    // If we are using GPU Device Memory with NVCUVENC, it is necessary to create a 
    // CUDA Context with a Context Lock cuvidCtxLock.  The Context Lock needs to be passed to NVCUVENC

    int iUseDeviceMem = 1;
    err = NVSetParamValue(encoder_, NVVE_DEVICE_MEMORY_INPUT, &iUseDeviceMem);
    CV_Assert ( err == 0 );

    err = NVSetParamValue(encoder_, NVVE_DEVICE_CTX_LOCK, &cuCtxLock_);
    CV_Assert ( err == 0 );
}

void cv::gpu::VideoWriter_GPU::Impl::initCallBacks()
{
    NVVE_CallbackParams cb;
    memset(&cb, 0, sizeof(NVVE_CallbackParams));

    cb.pfnacquirebitstream = HandleAcquireBitStream;
    cb.pfnonbeginframe     = HandleOnBeginFrame;
    cb.pfnonendframe       = HandleOnEndFrame;
    cb.pfnreleasebitstream = HandleReleaseBitStream;

    NVRegisterCB(encoder_, cb, this);
}

void cv::gpu::VideoWriter_GPU::Impl::createHWEncoder()
{
    int err;

    // Create the NVIDIA HW resources for Encoding on NVIDIA hardware
    err = NVCreateHWEncoder(encoder_);
    CV_Assert( err == 0 );
}

namespace cv { namespace gpu { namespace device 
{
    namespace video_encoding
    {
        void YV12_gpu(const DevMem2Db src, int cn, DevMem2Db dst);
    }
}}}

namespace
{
    // UYVY/YUY2 are both 4:2:2 formats (16bpc)
    // Luma, U, V are interleaved, chroma is subsampled (w/2,h)
    void copyUYVYorYUY2Frame(cv::Size frameSize, const cv::gpu::GpuMat& src, cv::gpu::GpuMat& dst)
    {
        CUresult res;

        // Source is YUVY/YUY2 4:2:2, the YUV data in a packed and interleaved

        // YUV Copy setup
        CUDA_MEMCPY2D stCopyYUV422;
        memset((void*)&stCopyYUV422, 0, sizeof(stCopyYUV422));
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
        res = cuMemcpy2D(&stCopyYUV422);
        CV_Assert( res == CUDA_SUCCESS );
    }

    // YV12/IYUV are both 4:2:0 planar formats (12bpc)
    // Luma, U, V chroma planar (12bpc), chroma is subsampled (w/2,h/2)
    void copyYV12orIYUVFrame(cv::Size frameSize, const cv::gpu::GpuMat& src, cv::gpu::GpuMat& dst)
    {
        CUresult res;

        // Source is YV12/IYUV, this native format is converted to NV12 format by the video encoder

        // (1) luma copy setup
        CUDA_MEMCPY2D stCopyLuma;
        memset((void*)&stCopyLuma, 0, sizeof(stCopyLuma));
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
        memset((void*)&stCopyChroma, 0, sizeof(stCopyChroma));
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
        res = cuMemcpy2D(&stCopyLuma);
        CV_Assert( res == CUDA_SUCCESS );

        // DMA Chroma channels (UV side by side)
        res = cuMemcpy2D(&stCopyChroma);
        CV_Assert( res == CUDA_SUCCESS );
    }

    // NV12 is 4:2:0 format (12bpc)
    // Luma followed by U/V chroma interleaved (12bpc), chroma is subsampled (w/2,h/2)
    void copyNV12Frame(cv::Size frameSize, const cv::gpu::GpuMat& src, cv::gpu::GpuMat& dst)
    {
        CUresult res;

        // Source is NV12 in pitch linear memory
        // Because we are assume input is NV12 (if we take input in the native format), the encoder handles NV12 as a native format in pitch linear memory

        // Luma/Chroma can be done in a single transfer
        CUDA_MEMCPY2D stCopyNV12;
        memset((void*)&stCopyNV12, 0, sizeof(stCopyNV12));
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
        stCopyNV12.Height               =(frameSize.height * 3) >> 1;

        // DMA Luma/Chroma
        res = cuMemcpy2D(&stCopyNV12);
        CV_Assert( res == CUDA_SUCCESS );
    }
}

void cv::gpu::VideoWriter_GPU::Impl::write(const cv::gpu::GpuMat& frame, bool lastFrame)
{
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
    CUresult res = cuvidCtxLock(cuCtxLock_, 0);
    CV_Assert( res == CUDA_SUCCESS );

    if (inputFormat_ == SF_BGR)
        cv::gpu::device::video_encoding::YV12_gpu(frame, frame.channels(), videoFrame_);
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

    res = cuvidCtxUnlock(cuCtxLock_, 0);
    CV_Assert( res == CUDA_SUCCESS );

    int err = NVEncodeFrame(encoder_, &efparams, 0, videoFrame_.data);
    CV_Assert( err == 0 );
}

unsigned char* NVENCAPI cv::gpu::VideoWriter_GPU::Impl::HandleAcquireBitStream(int* pBufferSize, void* pUserdata)
{
    Impl* thiz = static_cast<Impl*>(pUserdata);

    return thiz->callback_->acquireBitStream(pBufferSize);
}

void NVENCAPI cv::gpu::VideoWriter_GPU::Impl::HandleReleaseBitStream(int nBytesInBuffer, unsigned char* cb, void* pUserdata)
{
    Impl* thiz = static_cast<Impl*>(pUserdata);

    thiz->callback_->releaseBitStream(cb, nBytesInBuffer);
}

void NVENCAPI cv::gpu::VideoWriter_GPU::Impl::HandleOnBeginFrame(const NVVE_BeginFrameInfo* pbfi, void* pUserdata)
{
    Impl* thiz = static_cast<Impl*>(pUserdata);

    thiz->callback_->onBeginFrame(pbfi->nFrameNumber, static_cast<EncoderCallBack::PicType>(pbfi->nPicType));
}

void NVENCAPI cv::gpu::VideoWriter_GPU::Impl::HandleOnEndFrame(const NVVE_EndFrameInfo* pefi, void* pUserdata)
{
    Impl* thiz = static_cast<Impl*>(pUserdata);

    thiz->callback_->onEndFrame(pefi->nFrameNumber, static_cast<EncoderCallBack::PicType>(pefi->nPicType));
}

///////////////////////////////////////////////////////////////////////////
// FFMPEG

class EncoderCallBackFFMPEG : public cv::gpu::VideoWriter_GPU::EncoderCallBack
{
public:
    EncoderCallBackFFMPEG(const std::string& fileName, cv::Size frameSize, double fps);
    ~EncoderCallBackFFMPEG();

    unsigned char* acquireBitStream(int* bufferSize);
    void releaseBitStream(unsigned char* data, int size);
    void onBeginFrame(int frameNumber, PicType picType);
    void onEndFrame(int frameNumber, PicType picType);

private:
    EncoderCallBackFFMPEG(const EncoderCallBackFFMPEG&);
    EncoderCallBackFFMPEG& operator=(const EncoderCallBackFFMPEG&);

    struct OutputMediaStream_FFMPEG* stream_;
    std::vector<uchar> buf_;
    bool isKeyFrame_;
};

namespace
{
    Create_OutputMediaStream_FFMPEG_Plugin create_OutputMediaStream_FFMPEG_p = 0;
    Release_OutputMediaStream_FFMPEG_Plugin release_OutputMediaStream_FFMPEG_p = 0;
    Write_OutputMediaStream_FFMPEG_Plugin write_OutputMediaStream_FFMPEG_p = 0;

    bool init_MediaStream_FFMPEG()
    {
        static bool initialized = 0;

        if (!initialized)
        {
            #if defined WIN32 || defined _WIN32
                const char* module_name = "opencv_ffmpeg"
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
            #elif defined HAVE_FFMPEG
                create_OutputMediaStream_FFMPEG_p = create_OutputMediaStream_FFMPEG;
                release_OutputMediaStream_FFMPEG_p = release_OutputMediaStream_FFMPEG;
                write_OutputMediaStream_FFMPEG_p = write_OutputMediaStream_FFMPEG;

                initialized = true;
            #endif
        }

        return initialized;
    }
}

EncoderCallBackFFMPEG::EncoderCallBackFFMPEG(const std::string& fileName, cv::Size frameSize, double fps) :
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
    isKeyFrame_ = picType == IFRAME;
}

void EncoderCallBackFFMPEG::onEndFrame(int frameNumber, PicType picType)
{
}

///////////////////////////////////////////////////////////////////////////
// VideoWriter_GPU

cv::gpu::VideoWriter_GPU::VideoWriter_GPU()
{
}

cv::gpu::VideoWriter_GPU::VideoWriter_GPU(const std::string& fileName, cv::Size frameSize, double fps, SurfaceFormat format)
{
    open(fileName, frameSize, fps, format);
}

cv::gpu::VideoWriter_GPU::VideoWriter_GPU(const std::string& fileName, cv::Size frameSize, double fps, const EncoderParams& params, SurfaceFormat format)
{
    open(fileName, frameSize, fps, params, format);
}

cv::gpu::VideoWriter_GPU::VideoWriter_GPU(const cv::Ptr<EncoderCallBack>& encoderCallback, cv::Size frameSize, double fps, SurfaceFormat format)
{
    open(encoderCallback, frameSize, fps, format);
}

cv::gpu::VideoWriter_GPU::VideoWriter_GPU(const cv::Ptr<EncoderCallBack>& encoderCallback, cv::Size frameSize, double fps, const EncoderParams& params, SurfaceFormat format)
{
    open(encoderCallback, frameSize, fps, params, format);
}

cv::gpu::VideoWriter_GPU::~VideoWriter_GPU()
{
    close();
}

void cv::gpu::VideoWriter_GPU::open(const std::string& fileName, cv::Size frameSize, double fps, SurfaceFormat format)
{
    close();
    cv::Ptr<EncoderCallBack> encoderCallback(new EncoderCallBackFFMPEG(fileName, frameSize, fps));
    open(encoderCallback, frameSize, fps, format);
}

void cv::gpu::VideoWriter_GPU::open(const std::string& fileName, cv::Size frameSize, double fps, const EncoderParams& params, SurfaceFormat format)
{
    close();
    cv::Ptr<EncoderCallBack> encoderCallback(new EncoderCallBackFFMPEG(fileName, frameSize, fps));
    open(encoderCallback, frameSize, fps, params, format);
}

void cv::gpu::VideoWriter_GPU::open(const cv::Ptr<EncoderCallBack>& encoderCallback, cv::Size frameSize, double fps, SurfaceFormat format)
{
    close();
    impl_.reset(new Impl(encoderCallback, frameSize, fps, format));
}

void cv::gpu::VideoWriter_GPU::open(const cv::Ptr<EncoderCallBack>& encoderCallback, cv::Size frameSize, double fps, const EncoderParams& params, SurfaceFormat format)
{
    close();
    impl_.reset(new Impl(encoderCallback, frameSize, fps, params, format));
}

bool cv::gpu::VideoWriter_GPU::isOpened() const
{
    return impl_.get() != 0;
}

void cv::gpu::VideoWriter_GPU::close()
{
    impl_.reset();
}

void cv::gpu::VideoWriter_GPU::write(const cv::gpu::GpuMat& image, bool lastFrame)
{
    CV_Assert( isOpened() );

    impl_->write(image, lastFrame);
}

cv::gpu::VideoWriter_GPU::EncoderParams cv::gpu::VideoWriter_GPU::getParams() const 
{
    CV_Assert( isOpened() );

    return impl_->getParams();
}

///////////////////////////////////////////////////////////////////////////
// VideoWriter_GPU::EncoderParams

cv::gpu::VideoWriter_GPU::EncoderParams::EncoderParams()
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

cv::gpu::VideoWriter_GPU::EncoderParams::EncoderParams(const std::string& configFile)
{
    load(configFile);
}

void cv::gpu::VideoWriter_GPU::EncoderParams::load(const std::string& configFile)
{
    cv::FileStorage fs(configFile, cv::FileStorage::READ);
    CV_Assert( fs.isOpened() );

    cv::read(fs["P_Interval"     ], P_Interval, 3);
    cv::read(fs["IDR_Period"     ], IDR_Period, 15);
    cv::read(fs["DynamicGOP"     ], DynamicGOP, 0);
    cv::read(fs["RCType"         ], RCType, 1);
    cv::read(fs["AvgBitrate"     ], AvgBitrate, 4000000);
    cv::read(fs["PeakBitrate"    ], PeakBitrate, 10000000);
    cv::read(fs["QP_Level_Intra" ], QP_Level_Intra, 25);
    cv::read(fs["QP_Level_InterP"], QP_Level_InterP, 28);
    cv::read(fs["QP_Level_InterB"], QP_Level_InterB, 31);
    cv::read(fs["DeblockMode"    ], DeblockMode, 1);
    cv::read(fs["ProfileLevel"   ], ProfileLevel, 65357);
    cv::read(fs["ForceIntra"     ], ForceIntra, 0);
    cv::read(fs["ForceIDR"       ], ForceIDR, 0);
    cv::read(fs["ClearStat"      ], ClearStat, 0);
    cv::read(fs["DIMode"         ], DIMode, 1);
    cv::read(fs["Presets"        ], Presets, 2);
    cv::read(fs["DisableCabac"   ], DisableCabac, 0);
    cv::read(fs["NaluFramingType"], NaluFramingType, 0);
    cv::read(fs["DisableSPSPPS"  ], DisableSPSPPS, 0);
}

void cv::gpu::VideoWriter_GPU::EncoderParams::save(const std::string& configFile) const
{
    cv::FileStorage fs(configFile, cv::FileStorage::WRITE);
    CV_Assert( fs.isOpened() );

    cv::write(fs, "P_Interval"     , P_Interval);
    cv::write(fs, "IDR_Period"     , IDR_Period);
    cv::write(fs, "DynamicGOP"     , DynamicGOP);
    cv::write(fs, "RCType"         , RCType);
    cv::write(fs, "AvgBitrate"     , AvgBitrate);
    cv::write(fs, "PeakBitrate"    , PeakBitrate);
    cv::write(fs, "QP_Level_Intra" , QP_Level_Intra);
    cv::write(fs, "QP_Level_InterP", QP_Level_InterP);
    cv::write(fs, "QP_Level_InterB", QP_Level_InterB);
    cv::write(fs, "DeblockMode"    , DeblockMode);
    cv::write(fs, "ProfileLevel"   , ProfileLevel);
    cv::write(fs, "ForceIntra"     , ForceIntra);
    cv::write(fs, "ForceIDR"       , ForceIDR);
    cv::write(fs, "ClearStat"      , ClearStat);
    cv::write(fs, "DIMode"         , DIMode);
    cv::write(fs, "Presets"        , Presets);
    cv::write(fs, "DisableCabac"   , DisableCabac);
    cv::write(fs, "NaluFramingType", NaluFramingType);
    cv::write(fs, "DisableSPSPPS"  , DisableSPSPPS);
}

#endif // !defined HAVE_CUDA || !defined WIN32
