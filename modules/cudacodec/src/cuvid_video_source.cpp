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

#ifdef HAVE_NVCUVID

using namespace cv;
using namespace cv::cudacodec;
using namespace cv::cudacodec::detail;

cv::cudacodec::detail::CuvidVideoSource::CuvidVideoSource(const String& fname)
{
    CUVIDSOURCEPARAMS params;
    std::memset(&params, 0, sizeof(CUVIDSOURCEPARAMS));

    // Fill parameter struct
    params.pUserData = this;                        // will be passed to data handlers
    params.pfnVideoDataHandler = HandleVideoData;   // our local video-handler callback
    params.pfnAudioDataHandler = 0;

    // now create the actual source
    CUresult cuRes = cuvidCreateVideoSource(&videoSource_, fname.c_str(), &params);
    if (cuRes == CUDA_ERROR_INVALID_SOURCE)
        throw std::runtime_error("");
    cuSafeCall( cuRes );

    CUVIDEOFORMAT vidfmt;
    cuSafeCall( cuvidGetSourceVideoFormat(videoSource_, &vidfmt, 0) );

    format_.codec = static_cast<Codec>(vidfmt.codec);
    format_.chromaFormat = static_cast<ChromaFormat>(vidfmt.chroma_format);
    format_.width = vidfmt.coded_width;
    format_.height = vidfmt.coded_height;
}

cv::cudacodec::detail::CuvidVideoSource::~CuvidVideoSource()
{
    cuvidDestroyVideoSource(videoSource_);
}

FormatInfo cv::cudacodec::detail::CuvidVideoSource::format() const
{
    return format_;
}

void cv::cudacodec::detail::CuvidVideoSource::start()
{
    cuSafeCall( cuvidSetVideoSourceState(videoSource_, cudaVideoState_Started) );
}

void cv::cudacodec::detail::CuvidVideoSource::stop()
{
    cuSafeCall( cuvidSetVideoSourceState(videoSource_, cudaVideoState_Stopped) );
}

bool cv::cudacodec::detail::CuvidVideoSource::isStarted() const
{
    return (cuvidGetVideoSourceState(videoSource_) == cudaVideoState_Started);
}

bool cv::cudacodec::detail::CuvidVideoSource::hasError() const
{
    return (cuvidGetVideoSourceState(videoSource_) == cudaVideoState_Error);
}

int CUDAAPI cv::cudacodec::detail::CuvidVideoSource::HandleVideoData(void* userData, CUVIDSOURCEDATAPACKET* packet)
{
    CuvidVideoSource* thiz = static_cast<CuvidVideoSource*>(userData);

    return thiz->parseVideoData(packet->payload, packet->payload_size, (packet->flags & CUVID_PKT_ENDOFSTREAM) != 0);
}

#endif // HAVE_NVCUVID
