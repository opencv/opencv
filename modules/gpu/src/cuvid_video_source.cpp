#include "cuvid_video_source.h"
#include "cu_safe_call.h"

#if defined(HAVE_CUDA) && !defined(__APPLE__)

cv::gpu::detail::CuvidVideoSource::CuvidVideoSource(const std::string& fname)
{
    CUVIDSOURCEPARAMS params;
    std::memset(&params, 0, sizeof(CUVIDSOURCEPARAMS));

    // Fill parameter struct
    params.pUserData = this;                        // will be passed to data handlers
    params.pfnVideoDataHandler = HandleVideoData;   // our local video-handler callback
    params.pfnAudioDataHandler = 0;

    // now create the actual source
    CUresult res = cuvidCreateVideoSource(&videoSource_, fname.c_str(), &params);
    if (res == CUDA_ERROR_INVALID_SOURCE)
        throw std::runtime_error("Unsupported video source");
    cuSafeCall( res );

    CUVIDEOFORMAT vidfmt;
    cuSafeCall( cuvidGetSourceVideoFormat(videoSource_, &vidfmt, 0) );

    format_.codec = static_cast<VideoReader_GPU::Codec>(vidfmt.codec);
    format_.chromaFormat = static_cast<VideoReader_GPU::ChromaFormat>(vidfmt.chroma_format);
    format_.width = vidfmt.coded_width;
    format_.height = vidfmt.coded_height;
}

cv::gpu::VideoReader_GPU::FormatInfo cv::gpu::detail::CuvidVideoSource::format() const
{ 
    return format_;
}

void cv::gpu::detail::CuvidVideoSource::start()
{
    cuSafeCall( cuvidSetVideoSourceState(videoSource_, cudaVideoState_Started) );
}

void cv::gpu::detail::CuvidVideoSource::stop()
{
    cuSafeCall( cuvidSetVideoSourceState(videoSource_, cudaVideoState_Stopped) );
}

bool cv::gpu::detail::CuvidVideoSource::isStarted() const
{
    return (cuvidGetVideoSourceState(videoSource_) == cudaVideoState_Started);
}

bool cv::gpu::detail::CuvidVideoSource::hasError() const
{
    return (cuvidGetVideoSourceState(videoSource_) == cudaVideoState_Error);
}

int CUDAAPI cv::gpu::detail::CuvidVideoSource::HandleVideoData(void* userData, CUVIDSOURCEDATAPACKET* packet)
{
    CuvidVideoSource* thiz = static_cast<CuvidVideoSource*>(userData);

    return thiz->parseVideoData(packet->payload, packet->payload_size, (packet->flags & CUVID_PKT_ENDOFSTREAM) != 0);
}

#endif // defined(HAVE_CUDA) && !defined(__APPLE__)
