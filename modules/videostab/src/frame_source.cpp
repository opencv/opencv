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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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
#include "opencv2/videostab/frame_source.hpp"
#include "opencv2/videostab/ring_buffer.hpp"

#include "opencv2/opencv_modules.hpp"
#ifdef HAVE_OPENCV_HIGHGUI
#  include "opencv2/highgui/highgui.hpp"
#endif

using namespace std;

namespace cv
{
namespace videostab
{

struct VideoFileSource::VideoReader
{
#ifdef HAVE_OPENCV_HIGHGUI
    mutable VideoCapture vc;
#endif
};

VideoFileSource::VideoFileSource(const string &path, bool volatileFrame)
    : path_(path), volatileFrame_(volatileFrame), reader_(VideoReader()) { reset(); }


void VideoFileSource::reset()
{
#ifdef HAVE_OPENCV_HIGHGUI
    reader_.vc.release();
    reader_.vc.open(path_);
    if (!reader_.vc.isOpened())
        throw runtime_error("can't open file: " + path_);
#else
    CV_Error(CV_StsNotImplemented, "OpenCV has been compiled without video I/O support");
#endif
}


Mat VideoFileSource::nextFrame()
{
    Mat frame;
#ifdef HAVE_OPENCV_HIGHGUI
    reader_.vc >> frame;
#endif
    return volatileFrame_ ? frame : frame.clone();
}

int VideoFileSource::width()
{
#ifdef HAVE_OPENCV_HIGHGUI
    return static_cast<int>(reader_.vc.get(CV_CAP_PROP_FRAME_WIDTH));
#else
    return 0;
#endif
}

int VideoFileSource::height()
{
#ifdef HAVE_OPENCV_HIGHGUI
    return static_cast<int>(reader_.vc.get(CV_CAP_PROP_FRAME_HEIGHT));
#else
    return 0;
#endif
}

int VideoFileSource::count()
{
#ifdef HAVE_OPENCV_HIGHGUI
    return static_cast<int>(reader_.vc.get(CV_CAP_PROP_FRAME_COUNT));
#else
    return 0;
#endif
}

double VideoFileSource::fps()
{
#ifdef HAVE_OPENCV_HIGHGUI
    return reader_.vc.get(CV_CAP_PROP_FPS);
#else
    return 0;
#endif
}

} // namespace videostab
} // namespace cv
