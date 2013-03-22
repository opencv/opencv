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

#ifndef __VIDEO_DECODER_H__
#define __VIDEO_DECODER_H__

#include "precomp.hpp"
#include "cu_safe_call.h"

#if defined(HAVE_CUDA) && defined(HAVE_NVCUVID)

namespace cv { namespace gpu
{
    namespace detail
    {
        class VideoDecoder
        {
        public:
            VideoDecoder(const VideoReader_GPU::FormatInfo& videoFormat, CUvideoctxlock lock) : lock_(lock), decoder_(0)
            {
                create(videoFormat);
            }

            ~VideoDecoder()
            {
                release();
            }

            void create(const VideoReader_GPU::FormatInfo& videoFormat);
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

            cv::gpu::GpuMat mapFrame(int picIdx, CUVIDPROCPARAMS& videoProcParams)
            {
                CUdeviceptr ptr;
                unsigned int pitch;

                cuSafeCall( cuvidMapVideoFrame(decoder_, picIdx, &ptr, &pitch, &videoProcParams) );

                return GpuMat(targetHeight() * 3 / 2, targetWidth(), CV_8UC1, (void*) ptr, pitch);
            }

            void unmapFrame(cv::gpu::GpuMat& frame)
            {
                cuSafeCall( cuvidUnmapVideoFrame(decoder_, (CUdeviceptr) frame.data) );
                frame.release();
            }

        private:
            VideoDecoder(const VideoDecoder&);
            VideoDecoder& operator =(const VideoDecoder&);

            CUvideoctxlock lock_;
            CUVIDDECODECREATEINFO createInfo_;
            CUvideodecoder        decoder_;
        };
    }
}}

#endif // HAVE_CUDA

#endif // __VIDEO_DECODER_H__
