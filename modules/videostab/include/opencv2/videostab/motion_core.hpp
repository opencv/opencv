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

#ifndef __OPENCV_VIDEOSTAB_MOTION_CORE_HPP__
#define __OPENCV_VIDEOSTAB_MOTION_CORE_HPP__

#include <cmath>
#include "opencv2/core.hpp"

namespace cv
{
namespace videostab
{

//! @addtogroup videostab_motion
//! @{

/** @brief Describes motion model between two point clouds.
 */
enum MotionModel
{
    MM_TRANSLATION = 0,
    MM_TRANSLATION_AND_SCALE = 1,
    MM_ROTATION = 2,
    MM_RIGID = 3,
    MM_SIMILARITY = 4,
    MM_AFFINE = 5,
    MM_HOMOGRAPHY = 6,
    MM_UNKNOWN = 7
};

/** @brief Describes RANSAC method parameters.
 */
struct CV_EXPORTS RansacParams
{
    int size; //!< subset size
    float thresh; //!< max error to classify as inlier
    float eps; //!< max outliers ratio
    float prob; //!< probability of success

    RansacParams() : size(0), thresh(0), eps(0), prob(0) {}
    /** @brief Constructor
    @param size Subset size.
    @param thresh Maximum re-projection error value to classify as inlier.
    @param eps Maximum ratio of incorrect correspondences.
    @param prob Required success probability.
     */
    RansacParams(int size, float thresh, float eps, float prob);

    /**
    @return Number of iterations that'll be performed by RANSAC method.
    */
    int niters() const
    {
        return static_cast<int>(
                std::ceil(std::log(1 - prob) / std::log(1 - std::pow(1 - eps, size))));
    }

    /**
    @param model Motion model. See cv::videostab::MotionModel.
    @return Default RANSAC method parameters for the given motion model.
    */
    static RansacParams default2dMotion(MotionModel model)
    {
        CV_Assert(model < MM_UNKNOWN);
        if (model == MM_TRANSLATION)
            return RansacParams(1, 0.5f, 0.5f, 0.99f);
        if (model == MM_TRANSLATION_AND_SCALE)
            return RansacParams(2, 0.5f, 0.5f, 0.99f);
        if (model == MM_ROTATION)
            return RansacParams(1, 0.5f, 0.5f, 0.99f);
        if (model == MM_RIGID)
            return RansacParams(2, 0.5f, 0.5f, 0.99f);
        if (model == MM_SIMILARITY)
            return RansacParams(2, 0.5f, 0.5f, 0.99f);
        if (model == MM_AFFINE)
            return RansacParams(3, 0.5f, 0.5f, 0.99f);
        return RansacParams(4, 0.5f, 0.5f, 0.99f);
    }
};

inline RansacParams::RansacParams(int _size, float _thresh, float _eps, float _prob)
    : size(_size), thresh(_thresh), eps(_eps), prob(_prob) {}

//! @}

} // namespace videostab
} // namespace cv

#endif
