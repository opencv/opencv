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

#ifndef __VIDEO_DECODER_HPP__
#define __VIDEO_DECODER_HPP__

#if CUDA_VERSION >= 9000
    #include <dynlink_nvcuvid.h>
#else
    #include <nvcuvid.h>
#endif

#include "opencv2/core/private.cuda.hpp"
#include "opencv2/cudacodec.hpp"

namespace cv { namespace cudacodec { namespace detail
{

class VideoDecoder
{
public:
    VideoDecoder(const FormatInfo& videoFormat, CUvideoctxlock lock) : lock_(lock), decoder_(0)
    {
        create(videoFormat);
    }

    ~VideoDecoder()
    {
        release();
    }

    void create(const FormatInfo& videoFormat);
    void release();

    // Get the code-type currently used.
    cudaVideoCodec codec() const { return createInfo_.CodecType; }
    unsigned long maxDecodeSurfaces() const { return createInfo_.ulNumDecodeSurfaces; }

    unsigned long frameWidth() const { return createInfo_.ulWidth; }
    unsigned long frameHeight() const { return createInfo_.ulHeight; }

    unsigned long targetWidth() const { return createInfo_.ulTargetWidth; }
    unsigned long targetHeight() const { return createInfo_.ulTargetHeight; }

    cudaVideoChromaFormat chromaFormat() const { return createInfo_.ChromaFormat; }

    bool decodePicture(CUVIDPICPARAMS* picParams)
    {
        return cuvidDecodePicture(decoder_, picParams) == CUDA_SUCCESS;
    }

    cuda::GpuMat mapFrame(int picIdx, CUVIDPROCPARAMS& videoProcParams)
    {
        CUdeviceptr ptr;
        unsigned int pitch;

        cuSafeCall( cuvidMapVideoFrame(decoder_, picIdx, &ptr, &pitch, &videoProcParams) );

        return cuda::GpuMat(targetHeight() * 3 / 2, targetWidth(), CV_8UC1, (void*) ptr, pitch);
    }

    void unmapFrame(cuda::GpuMat& frame)
    {
        cuSafeCall( cuvidUnmapVideoFrame(decoder_, (CUdeviceptr) frame.data) );
        frame.release();
    }

private:
    CUvideoctxlock lock_;
    CUVIDDECODECREATEINFO createInfo_;
    CUvideodecoder        decoder_;
};

}}}

#endif // __VIDEO_DECODER_HPP__
