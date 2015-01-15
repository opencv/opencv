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

#ifndef __OPENCV_CUDA_HPP__
#define __OPENCV_CUDA_HPP__

#ifndef __cplusplus
#  error cuda.hpp header must be compiled as C++
#endif

#include "opencv2/core/cuda.hpp"

/**
    @addtogroup cuda
    @{
        @defgroup cuda_calib3d Camera Calibration and 3D Reconstruction
    @}
 */

namespace cv { namespace cuda {

//////////////////////////// Calib3d ////////////////////////////

//! @addtogroup cuda_calib3d
//! @{

CV_EXPORTS void transformPoints(const GpuMat& src, const Mat& rvec, const Mat& tvec,
                                GpuMat& dst, Stream& stream = Stream::Null());

CV_EXPORTS void projectPoints(const GpuMat& src, const Mat& rvec, const Mat& tvec,
                              const Mat& camera_mat, const Mat& dist_coef, GpuMat& dst,
                              Stream& stream = Stream::Null());

/** @brief Finds the object pose from 3D-2D point correspondences.

@param object Single-row matrix of object points.
@param image Single-row matrix of image points.
@param camera_mat 3x3 matrix of intrinsic camera parameters.
@param dist_coef Distortion coefficients. See undistortPoints for details.
@param rvec Output 3D rotation vector.
@param tvec Output 3D translation vector.
@param use_extrinsic_guess Flag to indicate that the function must use rvec and tvec as an
initial transformation guess. It is not supported for now.
@param num_iters Maximum number of RANSAC iterations.
@param max_dist Euclidean distance threshold to detect whether point is inlier or not.
@param min_inlier_count Flag to indicate that the function must stop if greater or equal number
of inliers is achieved. It is not supported for now.
@param inliers Output vector of inlier indices.
 */
CV_EXPORTS void solvePnPRansac(const Mat& object, const Mat& image, const Mat& camera_mat,
                               const Mat& dist_coef, Mat& rvec, Mat& tvec, bool use_extrinsic_guess=false,
                               int num_iters=100, float max_dist=8.0, int min_inlier_count=100,
                               std::vector<int>* inliers=NULL);

//! @}

}} // namespace cv { namespace cuda {

#endif /* __OPENCV_CUDA_HPP__ */
