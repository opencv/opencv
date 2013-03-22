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

#include "video_parser.h"
#include "cu_safe_call.h"

#if defined(HAVE_CUDA) && defined(HAVE_NVCUVID)

cv::gpu::detail::VideoParser::VideoParser(VideoDecoder* videoDecoder, FrameQueue* frameQueue) :
    videoDecoder_(videoDecoder), frameQueue_(frameQueue), unparsedPackets_(0), hasError_(false)
{
    CUVIDPARSERPARAMS params;
    memset(&params, 0, sizeof(CUVIDPARSERPARAMS));

    params.CodecType              = videoDecoder->codec();
    params.ulMaxNumDecodeSurfaces = videoDecoder->maxDecodeSurfaces();
    params.ulMaxDisplayDelay      = 1; // this flag is needed so the parser will push frames out to the decoder as quickly as it can
    params.pUserData              = this;
    params.pfnSequenceCallback    = HandleVideoSequence;    // Called before decoding frames and/or whenever there is a format change
    params.pfnDecodePicture       = HandlePictureDecode;    // Called when a picture is ready to be decoded (decode order)
    params.pfnDisplayPicture      = HandlePictureDisplay;   // Called whenever a picture is ready to be displayed (display order)

    cuSafeCall( cuvidCreateVideoParser(&parser_, &params) );
}

bool cv::gpu::detail::VideoParser::parseVideoData(const unsigned char* data, size_t size, bool endOfStream)
{
    CUVIDSOURCEDATAPACKET packet;
    std::memset(&packet, 0, sizeof(CUVIDSOURCEDATAPACKET));

    if (endOfStream)
        packet.flags |= CUVID_PKT_ENDOFSTREAM;

    packet.payload_size = static_cast<unsigned long>(size);
    packet.payload = data;

    if (cuvidParseVideoData(parser_, &packet) != CUDA_SUCCESS)
    {
        hasError_ = true;
        frameQueue_->endDecode();
        return false;
    }

    const int maxUnparsedPackets = 15;

    ++unparsedPackets_;
    if (unparsedPackets_ > maxUnparsedPackets)
    {
        hasError_ = true;
        frameQueue_->endDecode();
        return false;
    }

    if (endOfStream)
        frameQueue_->endDecode();

    return !frameQueue_->isEndOfDecode();
}

int CUDAAPI cv::gpu::detail::VideoParser::HandleVideoSequence(void* userData, CUVIDEOFORMAT* format)
{
    VideoParser* thiz = static_cast<VideoParser*>(userData);

    thiz->unparsedPackets_ = 0;

    if (format->codec         != thiz->videoDecoder_->codec()       ||
        format->coded_width   != thiz->videoDecoder_->frameWidth()  ||
        format->coded_height  != thiz->videoDecoder_->frameHeight() ||
        format->chroma_format != thiz->videoDecoder_->chromaFormat())
    {
        VideoReader_GPU::FormatInfo newFormat;

        newFormat.codec = static_cast<VideoReader_GPU::Codec>(format->codec);
        newFormat.chromaFormat = static_cast<VideoReader_GPU::ChromaFormat>(format->chroma_format);
        newFormat.width = format->coded_width;
        newFormat.height = format->coded_height;

        try
        {
            thiz->videoDecoder_->create(newFormat);
        }
        catch (const cv::Exception&)
        {
            thiz->hasError_ = true;
            return false;
        }
    }

    return true;
}

int CUDAAPI cv::gpu::detail::VideoParser::HandlePictureDecode(void* userData, CUVIDPICPARAMS* picParams)
{
    VideoParser* thiz = static_cast<VideoParser*>(userData);

    thiz->unparsedPackets_ = 0;

    bool isFrameAvailable = thiz->frameQueue_->waitUntilFrameAvailable(picParams->CurrPicIdx);

    if (!isFrameAvailable)
        return false;

    if (!thiz->videoDecoder_->decodePicture(picParams))
    {
        thiz->hasError_ = true;
        return false;
    }

    return true;
}

int CUDAAPI cv::gpu::detail::VideoParser::HandlePictureDisplay(void* userData, CUVIDPARSERDISPINFO* picParams)
{
    VideoParser* thiz = static_cast<VideoParser*>(userData);

    thiz->unparsedPackets_ = 0;

    thiz->frameQueue_->enqueue(picParams);

    return true;
}

#endif // HAVE_CUDA
