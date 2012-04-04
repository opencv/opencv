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
#include "opencv2/video/video.hpp"
#include "opencv2/videostab/optical_flow.hpp"
#include "opencv2/videostab/ring_buffer.hpp"

using namespace std;

namespace cv
{
namespace videostab
{

void SparsePyrLkOptFlowEstimator::run(
        InputArray frame0, InputArray frame1, InputArray points0, InputOutputArray points1,
        OutputArray status, OutputArray errors)
{
    calcOpticalFlowPyrLK(frame0, frame1, points0, points1, status, errors, winSize_, maxLevel_);
}


#if HAVE_OPENCV_GPU
DensePyrLkOptFlowEstimatorGpu::DensePyrLkOptFlowEstimatorGpu()
{
    CV_Assert(gpu::getCudaEnabledDeviceCount() > 0);
}


void DensePyrLkOptFlowEstimatorGpu::run(
        InputArray frame0, InputArray frame1, InputOutputArray flowX, InputOutputArray flowY,
        OutputArray errors)
{
    frame0_.upload(frame0.getMat());
    frame1_.upload(frame1.getMat());

    optFlowEstimator_.winSize = winSize_;
    optFlowEstimator_.maxLevel = maxLevel_;
    if (errors.needed())
    {
        optFlowEstimator_.dense(frame0_, frame1_, flowX_, flowY_, &errors_);
        errors_.download(errors.getMatRef());
    }
    else
        optFlowEstimator_.dense(frame0_, frame1_, flowX_, flowY_);

    flowX_.download(flowX.getMatRef());
    flowY_.download(flowY.getMatRef());
}
#endif


} // namespace videostab
} // namespace cv
