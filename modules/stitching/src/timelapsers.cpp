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

#include "precomp.hpp"
#include "opencl_kernels_stitching.hpp"

namespace cv {
namespace detail {

Ptr<Timelapser> Timelapser::createDefault(int type)
{
    if (type == AS_IS)
        return makePtr<Timelapser>();
    if (type == CROP)
        return makePtr<TimelapserCrop>();
    CV_Error(Error::StsBadArg, "unsupported timelapsing method");
    return Ptr<Timelapser>();
}


void Timelapser::initialize(const std::vector<Point> &corners, const std::vector<Size> &sizes)
{
    dst_roi_ = resultRoi(corners, sizes);
    dst_.create(dst_roi_.size(), CV_16SC3);
}

void Timelapser::process(InputArray _img, InputArray /*_mask*/, Point tl)
{
    CV_INSTRUMENT_REGION()

    dst_.setTo(Scalar::all(0));

    Mat img = _img.getMat();
    Mat dst = dst_.getMat(ACCESS_RW);

    CV_Assert(img.type() == CV_16SC3);
    int dx = tl.x - dst_roi_.x;
    int dy = tl.y - dst_roi_.y;

    for (int y = 0; y < img.rows; ++y)
    {
        const Point3_<short> *src_row = img.ptr<Point3_<short> >(y);

        for (int x = 0; x < img.cols; ++x)
        {
            if (test_point(Point(tl.x + x, tl.y + y)))
            {
                Point3_<short> *dst_row = dst.ptr<Point3_<short> >(dy + y);
                dst_row[dx + x] = src_row[x];
            }
        }
    }
}


bool Timelapser::test_point(Point pt)
{
    return dst_roi_.contains(pt);
}


void TimelapserCrop::initialize(const std::vector<Point> &corners, const std::vector<Size> &sizes)
{
    dst_roi_ = resultRoiIntersection(corners, sizes);
    dst_.create(dst_roi_.size(), CV_16SC3);
}


} // namespace detail
} // namespace cv
