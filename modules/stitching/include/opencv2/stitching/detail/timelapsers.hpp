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


#ifndef OPENCV_STITCHING_TIMELAPSERS_HPP
#define OPENCV_STITCHING_TIMELAPSERS_HPP

#include "opencv2/core.hpp"

namespace cv {
namespace detail {

//! @addtogroup stitching
//! @{

//  Base Timelapser class, takes a sequence of images, applies appropriate shift, stores result in dst_.

class CV_EXPORTS Timelapser
{
public:

    enum {AS_IS, CROP};

    virtual ~Timelapser() {}

    static Ptr<Timelapser> createDefault(int type);

    virtual void initialize(const std::vector<Point> &corners, const std::vector<Size> &sizes);
    virtual void process(InputArray img, InputArray mask, Point tl);
    virtual const UMat& getDst() {return dst_;}

protected:

    virtual bool test_point(Point pt);

    UMat dst_;
    Rect dst_roi_;
};


class CV_EXPORTS TimelapserCrop : public Timelapser
{
public:
    virtual void initialize(const std::vector<Point> &corners, const std::vector<Size> &sizes);
};

//! @}

} // namespace detail
} // namespace cv

#endif // OPENCV_STITCHING_TIMELAPSERS_HPP
