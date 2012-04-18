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

#ifndef __OPENCV_VIDEOSTAB_OPTICAL_FLOW_HPP__
#define __OPENCV_VIDEOSTAB_OPTICAL_FLOW_HPP__

#include "opencv2/core/core.hpp"
#include "opencv2/opencv_modules.hpp"

#if HAVE_OPENCV_GPU
  #include "opencv2/gpu/gpu.hpp"
#endif

namespace cv
{
namespace videostab
{

class CV_EXPORTS ISparseOptFlowEstimator
{
public:
    virtual ~ISparseOptFlowEstimator() {}
    virtual void run(
            InputArray frame0, InputArray frame1, InputArray points0, InputOutputArray points1,
            OutputArray status, OutputArray errors) = 0;
};

class CV_EXPORTS IDenseOptFlowEstimator
{
public:
    virtual ~IDenseOptFlowEstimator() {}
    virtual void run(
            InputArray frame0, InputArray frame1, InputOutputArray flowX, InputOutputArray flowY,
            OutputArray errors) = 0;
};

class CV_EXPORTS PyrLkOptFlowEstimatorBase
{
public:
    PyrLkOptFlowEstimatorBase() { setWinSize(Size(21, 21)); setMaxLevel(3); }

    void setWinSize(Size val) { winSize_ = val; }
    Size winSize() const { return winSize_; }

    void setMaxLevel(int val) { maxLevel_ = val; }
    int maxLevel() const { return maxLevel_; }

protected:
    Size winSize_;
    int maxLevel_;
};

class CV_EXPORTS SparsePyrLkOptFlowEstimator
        : public PyrLkOptFlowEstimatorBase, public ISparseOptFlowEstimator
{
public:
    virtual void run(
            InputArray frame0, InputArray frame1, InputArray points0, InputOutputArray points1,
            OutputArray status, OutputArray errors);
};

#if HAVE_OPENCV_GPU
class CV_EXPORTS SparsePyrLkOptFlowEstimatorGpu
        : public PyrLkOptFlowEstimatorBase, public ISparseOptFlowEstimator
{
public:
    SparsePyrLkOptFlowEstimatorGpu();

    virtual void run(
            InputArray frame0, InputArray frame1, InputArray points0, InputOutputArray points1,
            OutputArray status, OutputArray errors);

    void run(const gpu::GpuMat &frame0, const gpu::GpuMat &frame1, const gpu::GpuMat &points0, gpu::GpuMat &points1,
             gpu::GpuMat &status, gpu::GpuMat &errors);

    void run(const gpu::GpuMat &frame0, const gpu::GpuMat &frame1, const gpu::GpuMat &points0, gpu::GpuMat &points1,
             gpu::GpuMat &status);

private:
    gpu::PyrLKOpticalFlow optFlowEstimator_;
    gpu::GpuMat frame0_, frame1_, points0_, points1_, status_, errors_;
};

class CV_EXPORTS DensePyrLkOptFlowEstimatorGpu
        : public PyrLkOptFlowEstimatorBase, public IDenseOptFlowEstimator
{
public:
    DensePyrLkOptFlowEstimatorGpu();

    virtual void run(
            InputArray frame0, InputArray frame1, InputOutputArray flowX, InputOutputArray flowY,
            OutputArray errors);

private:
    gpu::PyrLKOpticalFlow optFlowEstimator_;
    gpu::GpuMat frame0_, frame1_, flowX_, flowY_, errors_;
};
#endif

} // namespace videostab
} // namespace cv

#endif
