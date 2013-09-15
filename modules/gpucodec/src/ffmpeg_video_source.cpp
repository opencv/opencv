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
using namespace cv::gpucodec;
using namespace cv::gpucodec::detail;

namespace
{
    Create_InputMediaStream_FFMPEG_Plugin create_InputMediaStream_FFMPEG_p = 0;
    Release_InputMediaStream_FFMPEG_Plugin release_InputMediaStream_FFMPEG_p = 0;
    Read_InputMediaStream_FFMPEG_Plugin read_InputMediaStream_FFMPEG_p = 0;

    bool init_MediaStream_FFMPEG()
    {
        static bool initialized = 0;

        if (!initialized)
        {
            #if defined WIN32 || defined _WIN32
                const char* module_name = "opencv_ffmpeg"
                    CVAUX_STR(CV_VERSION_EPOCH) CVAUX_STR(CV_VERSION_MAJOR) CVAUX_STR(CV_VERSION_MINOR)
                #if (defined _MSC_VER && defined _M_X64) || (defined __GNUC__ && defined __x86_64__)
                    "_64"
                #endif
                    ".dll";

                static HMODULE cvFFOpenCV = LoadLibrary(module_name);

                if (cvFFOpenCV)
                {
                    create_InputMediaStream_FFMPEG_p =
                        (Create_InputMediaStream_FFMPEG_Plugin)GetProcAddress(cvFFOpenCV, "create_InputMediaStream_FFMPEG");
                    release_InputMediaStream_FFMPEG_p =
                        (Release_InputMediaStream_FFMPEG_Plugin)GetProcAddress(cvFFOpenCV, "release_InputMediaStream_FFMPEG");
                    read_InputMediaStream_FFMPEG_p =
                        (Read_InputMediaStream_FFMPEG_Plugin)GetProcAddress(cvFFOpenCV, "read_InputMediaStream_FFMPEG");

                    initialized = create_InputMediaStream_FFMPEG_p != 0 && release_InputMediaStream_FFMPEG_p != 0 && read_InputMediaStream_FFMPEG_p != 0;
                }
            #elif defined HAVE_FFMPEG
                create_InputMediaStream_FFMPEG_p = create_InputMediaStream_FFMPEG;
                release_InputMediaStream_FFMPEG_p = release_InputMediaStream_FFMPEG;
                read_InputMediaStream_FFMPEG_p = read_InputMediaStream_FFMPEG;

                initialized = true;
            #endif
        }

        return initialized;
    }
}

cv::gpucodec::detail::FFmpegVideoSource::FFmpegVideoSource(const String& fname) :
    stream_(0)
{
    CV_Assert( init_MediaStream_FFMPEG() );

    int codec;
    int chroma_format;
    int width;
    int height;

    stream_ = create_InputMediaStream_FFMPEG_p(fname.c_str(), &codec, &chroma_format, &width, &height);
    if (!stream_)
        CV_Error(Error::StsUnsupportedFormat, "Unsupported video source");

    format_.codec = static_cast<Codec>(codec);
    format_.chromaFormat = static_cast<ChromaFormat>(chroma_format);
    format_.width = width;
    format_.height = height;
}

cv::gpucodec::detail::FFmpegVideoSource::~FFmpegVideoSource()
{
    if (stream_)
        release_InputMediaStream_FFMPEG_p(stream_);
}

FormatInfo cv::gpucodec::detail::FFmpegVideoSource::format() const
{
    return format_;
}

bool cv::gpucodec::detail::FFmpegVideoSource::getNextPacket(unsigned char** data, int* size, bool* bEndOfFile)
{
    int endOfFile;

    int res = read_InputMediaStream_FFMPEG_p(stream_, data, size, &endOfFile);

    *bEndOfFile = (endOfFile != 0);
    return res != 0;
}

#endif // HAVE_CUDA
