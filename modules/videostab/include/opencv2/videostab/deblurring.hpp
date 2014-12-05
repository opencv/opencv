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

#ifndef __OPENCV_VIDEOSTAB_DEBLURRING_HPP__
#define __OPENCV_VIDEOSTAB_DEBLURRING_HPP__

#include <vector>
#include "opencv2/core.hpp"

namespace cv
{
namespace videostab
{

//! @addtogroup videostab
//! @{

CV_EXPORTS float calcBlurriness(const Mat &frame);

class CV_EXPORTS DeblurerBase
{
public:
    DeblurerBase() : radius_(0), frames_(0), motions_(0), blurrinessRates_(0) {}

    virtual ~DeblurerBase() {}

    virtual void setRadius(int val) { radius_ = val; }
    virtual int radius() const { return radius_; }

    virtual void deblur(int idx, Mat &frame) = 0;


    // data from stabilizer

    virtual void setFrames(const std::vector<Mat> &val) { frames_ = &val; }
    virtual const std::vector<Mat>& frames() const { return *frames_; }

    virtual void setMotions(const std::vector<Mat> &val) { motions_ = &val; }
    virtual const std::vector<Mat>& motions() const { return *motions_; }

    virtual void setBlurrinessRates(const std::vector<float> &val) { blurrinessRates_ = &val; }
    virtual const std::vector<float>& blurrinessRates() const { return *blurrinessRates_; }

protected:
    int radius_;
    const std::vector<Mat> *frames_;
    const std::vector<Mat> *motions_;
    const std::vector<float> *blurrinessRates_;
};

class CV_EXPORTS NullDeblurer : public DeblurerBase
{
public:
    virtual void deblur(int /*idx*/, Mat &/*frame*/) {}
};

class CV_EXPORTS WeightingDeblurer : public DeblurerBase
{
public:
    WeightingDeblurer();

    void setSensitivity(float val) { sensitivity_ = val; }
    float sensitivity() const { return sensitivity_; }

    virtual void deblur(int idx, Mat &frame);

private:
    float sensitivity_;
    Mat_<float> bSum_, gSum_, rSum_, wSum_;
};

//! @}

} // namespace videostab
} // namespace cv

#endif
