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

#ifndef OPENCV_STITCHING_CAMERA_HPP
#define OPENCV_STITCHING_CAMERA_HPP

#include "opencv2/core.hpp"

namespace cv {
namespace detail {

//! @addtogroup stitching
//! @{

/** @brief Describes camera parameters.

@note Translation is assumed to be zero during the whole stitching pipeline. :
 */
struct CV_EXPORTS_W_SIMPLE CameraParams
{
    CameraParams();
    CameraParams(const CameraParams& other);
    CameraParams& operator =(const CameraParams& other);
    CV_WRAP Mat K() const;

    CV_PROP_RW double focal; // Focal length
    CV_PROP_RW double aspect; // Aspect ratio
    CV_PROP_RW double ppx; // Principal point X
    CV_PROP_RW double ppy; // Principal point Y
    CV_PROP_RW Mat R; // Rotation
    CV_PROP_RW Mat t; // Translation
};

//! @}

} // namespace detail
} // namespace cv

#endif // #ifndef OPENCV_STITCHING_CAMERA_HPP
