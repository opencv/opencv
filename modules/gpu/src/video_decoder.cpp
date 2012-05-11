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

#include "video_decoder.h"
#include "frame_queue.h"

#if defined(HAVE_CUDA) && !defined(__APPLE__)

void cv::gpu::detail::VideoDecoder::create(const VideoReader_GPU::FormatInfo& videoFormat)
{
    release();

    cudaVideoCodec codec = static_cast<cudaVideoCodec>(videoFormat.codec);
    cudaVideoChromaFormat chromaFormat = static_cast<cudaVideoChromaFormat>(videoFormat.chromaFormat);

    cudaVideoCreateFlags videoCreateFlags = (codec == cudaVideoCodec_JPEG || codec == cudaVideoCodec_MPEG2) ?
                                            cudaVideoCreate_PreferCUDA :
                                            cudaVideoCreate_PreferCUVID;

    // Validate video format.  These are the currently supported formats via NVCUVID
    CV_Assert(cudaVideoCodec_MPEG1 == codec ||
              cudaVideoCodec_MPEG2 == codec ||
              cudaVideoCodec_MPEG4 == codec ||
              cudaVideoCodec_VC1   == codec ||
              cudaVideoCodec_H264  == codec ||
              cudaVideoCodec_JPEG  == codec ||
              cudaVideoCodec_YUV420== codec ||
              cudaVideoCodec_YV12  == codec ||
              cudaVideoCodec_NV12  == codec ||
              cudaVideoCodec_YUYV  == codec ||
              cudaVideoCodec_UYVY  == codec );

    CV_Assert(cudaVideoChromaFormat_Monochrome == chromaFormat ||
              cudaVideoChromaFormat_420        == chromaFormat ||
              cudaVideoChromaFormat_422        == chromaFormat ||
              cudaVideoChromaFormat_444        == chromaFormat);

    // Fill the decoder-create-info struct from the given video-format struct.
    std::memset(&createInfo_, 0, sizeof(CUVIDDECODECREATEINFO));

    // Create video decoder
    createInfo_.CodecType           = codec;
    createInfo_.ulWidth             = videoFormat.width;
    createInfo_.ulHeight            = videoFormat.height;
    createInfo_.ulNumDecodeSurfaces = FrameQueue::MaximumSize;

    // Limit decode memory to 24MB (16M pixels at 4:2:0 = 24M bytes)
    while (createInfo_.ulNumDecodeSurfaces * videoFormat.width * videoFormat.height > 16 * 1024 * 1024)
        createInfo_.ulNumDecodeSurfaces--;

    createInfo_.ChromaFormat    = chromaFormat;
    createInfo_.OutputFormat    = cudaVideoSurfaceFormat_NV12;
    createInfo_.DeinterlaceMode = cudaVideoDeinterlaceMode_Adaptive;

    // No scaling
    static const int MAX_FRAME_COUNT = 2;

    createInfo_.ulTargetWidth       = createInfo_.ulWidth;
    createInfo_.ulTargetHeight      = createInfo_.ulHeight;
    createInfo_.ulNumOutputSurfaces = MAX_FRAME_COUNT;  // We won't simultaneously map more than 8 surfaces
    createInfo_.ulCreationFlags     = videoCreateFlags;
    createInfo_.vidLock = lock_;

    // create the decoder
    cuSafeCall( cuvidCreateDecoder(&decoder_, &createInfo_) );
}

void cv::gpu::detail::VideoDecoder::release()
{
    if (decoder_)
    {
        cuvidDestroyDecoder(decoder_);
        decoder_ = 0;
    }
}

#endif // HAVE_CUDA
