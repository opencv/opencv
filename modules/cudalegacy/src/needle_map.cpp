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

#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;

#if !defined (HAVE_CUDA) || !defined(HAVE_OPENCV_CUDAIMGPROC) || defined (CUDA_DISABLER)

void cv::cuda::createOpticalFlowNeedleMap(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&) { throw_no_cuda(); }

#else

namespace cv { namespace cuda { namespace device
{
    namespace optical_flow
    {
        void NeedleMapAverage_gpu(PtrStepSzf u, PtrStepSzf v, PtrStepSzf u_avg, PtrStepSzf v_avg);
        void CreateOpticalFlowNeedleMap_gpu(PtrStepSzf u_avg, PtrStepSzf v_avg, float* vertex_buffer, float* color_data, float max_flow, float xscale, float yscale);
    }
}}}

void cv::cuda::createOpticalFlowNeedleMap(const GpuMat& u, const GpuMat& v, GpuMat& vertex, GpuMat& colors)
{
    using namespace cv::cuda::device::optical_flow;

    CV_Assert(u.type() == CV_32FC1);
    CV_Assert(v.type() == u.type() && v.size() == u.size());

    const int NEEDLE_MAP_SCALE = 16;

    const int x_needles = u.cols / NEEDLE_MAP_SCALE;
    const int y_needles = u.rows / NEEDLE_MAP_SCALE;

    GpuMat u_avg(y_needles, x_needles, CV_32FC1);
    GpuMat v_avg(y_needles, x_needles, CV_32FC1);

    NeedleMapAverage_gpu(u, v, u_avg, v_avg);

    const int NUM_VERTS_PER_ARROW = 6;

    const int num_arrows = x_needles * y_needles * NUM_VERTS_PER_ARROW;

    vertex.create(1, num_arrows, CV_32FC3);
    colors.create(1, num_arrows, CV_32FC3);

    colors.setTo(Scalar::all(1.0));

    double uMax, vMax;
    cuda::minMax(u_avg, 0, &uMax);
    cuda::minMax(v_avg, 0, &vMax);

    float max_flow = static_cast<float>(std::sqrt(uMax * uMax + vMax * vMax));

    CreateOpticalFlowNeedleMap_gpu(u_avg, v_avg, vertex.ptr<float>(), colors.ptr<float>(), max_flow, 1.0f / u.cols, 1.0f / u.rows);

    cuda::cvtColor(colors, colors, COLOR_HSV2RGB);
}

#endif /* HAVE_CUDA */
