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

#ifndef __VIDEO_PARSER_HPP__
#define __VIDEO_PARSER_HPP__

#include "frame_queue.hpp"
#include "video_decoder.hpp"

namespace cv { namespace cudacodec { namespace detail {

class VideoParser
{
public:
    VideoParser(VideoDecoder* videoDecoder, FrameQueue* frameQueue);

    ~VideoParser()
    {
        cuvidDestroyVideoParser(parser_);
    }

    bool parseVideoData(const unsigned char* data, size_t size, bool endOfStream);

    bool hasError() const { return hasError_; }

private:
    VideoDecoder* videoDecoder_;
    FrameQueue* frameQueue_;
    CUvideoparser parser_;
    int unparsedPackets_;
    volatile bool hasError_;

    // Called when the decoder encounters a video format change (or initial sequence header)
    // This particular implementation of the callback returns 0 in case the video format changes
    // to something different than the original format. Returning 0 causes a stop of the app.
    static int CUDAAPI HandleVideoSequence(void* pUserData, CUVIDEOFORMAT* pFormat);

    // Called by the video parser to decode a single picture
    // Since the parser will deliver data as fast as it can, we need to make sure that the picture
    // index we're attempting to use for decode is no longer used for display
    static int CUDAAPI HandlePictureDecode(void* pUserData, CUVIDPICPARAMS* pPicParams);

    // Called by the video parser to display a video frame (in the case of field pictures, there may be
    // 2 decode calls per 1 display call, since two fields make up one frame)
    static int CUDAAPI HandlePictureDisplay(void* pUserData, CUVIDPARSERDISPINFO* pPicParams);
};

}}}

#endif // __VIDEO_PARSER_HPP__
