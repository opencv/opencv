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
#  include "opencv2/highgui.hpp"
#endif

namespace cv
{
namespace videostab
{

namespace {

class VideoFileSourceImpl : public IFrameSource
{
public:
    VideoFileSourceImpl(const String &path, bool volatileFrame)
        : path_(path), volatileFrame_(volatileFrame) { reset(); }

    virtual void reset()
    {
#ifdef HAVE_OPENCV_HIGHGUI
        vc.release();
        vc.open(path_);
        if (!vc.isOpened())
            CV_Error(0, "can't open file: " + path_);
#else
        CV_Error(CV_StsNotImplemented, "OpenCV has been compiled without video I/O support");
#endif
    }

    virtual Mat nextFrame()
    {
        Mat frame;
#ifdef HAVE_OPENCV_HIGHGUI
        vc >> frame;
#endif
        return volatileFrame_ ? frame : frame.clone();
    }

#ifdef HAVE_OPENCV_HIGHGUI
    int width() {return static_cast<int>(vc.get(CAP_PROP_FRAME_WIDTH));}
    int height() {return static_cast<int>(vc.get(CAP_PROP_FRAME_HEIGHT));}
    int count() {return static_cast<int>(vc.get(CAP_PROP_FRAME_COUNT));}
    double fps() {return vc.get(CAP_PROP_FPS);}
#else
    int width() {return 0;}
    int height() {return 0;}
    int count() {return 0;}
    double fps() {return 0;}
#endif

private:
    String path_;
    bool volatileFrame_;
#ifdef HAVE_OPENCV_HIGHGUI
    VideoCapture vc;
#endif
};

}//namespace

VideoFileSource::VideoFileSource(const String &path, bool volatileFrame)
    : impl(new VideoFileSourceImpl(path, volatileFrame)) {}

void VideoFileSource::reset() { impl->reset(); }
Mat VideoFileSource::nextFrame() { return impl->nextFrame(); }

int VideoFileSource::width() { return ((VideoFileSourceImpl*)impl.get())->width(); }
int VideoFileSource::height() { return ((VideoFileSourceImpl*)impl.get())->height(); }
int VideoFileSource::count() { return ((VideoFileSourceImpl*)impl.get())->count(); }
double VideoFileSource::fps() { return ((VideoFileSourceImpl*)impl.get())->fps(); }

} // namespace videostab
} // namespace cv
