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

#ifndef __OPENCV_GPU_HPP__
#define __OPENCV_GPU_HPP__

#include "opencv2/core/gpumat.hpp"
#include "opencv2/gpuarithm.hpp"
#include "opencv2/gpufilters.hpp"
#include "opencv2/gpuwarping.hpp"
#include "opencv2/gpuimgproc.hpp"
#include "opencv2/gpufeatures2d.hpp"
#include "opencv2/gpuvideo.hpp"
#include "opencv2/gpucalib3d.hpp"
#include "opencv2/gpuobjdetect.hpp"

namespace cv { namespace gpu {

//!performs labeling via graph cuts of a 2D regular 4-connected graph.
CV_EXPORTS void graphcut(GpuMat& terminals, GpuMat& leftTransp, GpuMat& rightTransp, GpuMat& top, GpuMat& bottom, GpuMat& labels,
                         GpuMat& buf, Stream& stream = Stream::Null());

//!performs labeling via graph cuts of a 2D regular 8-connected graph.
CV_EXPORTS void graphcut(GpuMat& terminals, GpuMat& leftTransp, GpuMat& rightTransp, GpuMat& top, GpuMat& topLeft, GpuMat& topRight,
                         GpuMat& bottom, GpuMat& bottomLeft, GpuMat& bottomRight,
                         GpuMat& labels,
                         GpuMat& buf, Stream& stream = Stream::Null());

//! compute mask for Generalized Flood fill componetns labeling.
CV_EXPORTS void connectivityMask(const GpuMat& image, GpuMat& mask, const cv::Scalar& lo, const cv::Scalar& hi, Stream& stream = Stream::Null());

//! performs connected componnents labeling.
CV_EXPORTS void labelComponents(const GpuMat& mask, GpuMat& components, int flags = 0, Stream& stream = Stream::Null());

//! removes points (CV_32FC2, single row matrix) with zero mask value
CV_EXPORTS void compactPoints(GpuMat &points0, GpuMat &points1, const GpuMat &mask);

CV_EXPORTS void calcWobbleSuppressionMaps(
        int left, int idx, int right, Size size, const Mat &ml, const Mat &mr,
        GpuMat &mapx, GpuMat &mapy);

}} // namespace cv { namespace gpu {

#endif /* __OPENCV_GPU_HPP__ */
