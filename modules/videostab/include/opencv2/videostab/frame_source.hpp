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

#ifndef __OPENCV_VIDEOSTAB_FRAME_SOURCE_HPP__
#define __OPENCV_VIDEOSTAB_FRAME_SOURCE_HPP__

#include <vector>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace cv
{
namespace videostab
{

class CV_EXPORTS IFrameSource
{
public:
    virtual ~IFrameSource() {}
    virtual void reset() = 0;
    virtual Mat nextFrame() = 0;
};

class CV_EXPORTS NullFrameSource : public IFrameSource
{
public:
    virtual void reset() {}
    virtual Mat nextFrame() { return Mat(); }
};

class CV_EXPORTS VideoFileSource : public IFrameSource
{
public:
    VideoFileSource(const std::string &path, bool volatileFrame = false);

    virtual void reset();
    virtual Mat nextFrame();

    int frameCount() { return static_cast<int>(reader_.get(CV_CAP_PROP_FRAME_COUNT)); }
    double fps() { return reader_.get(CV_CAP_PROP_FPS); }

private:
    std::string path_;
    bool volatileFrame_;
    VideoCapture reader_;
};

} // namespace videostab
} // namespace cv

#endif
