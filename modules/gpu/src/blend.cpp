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
//     and/or other GpuMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or bpied warranties, including, but not limited to, the bpied
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

using namespace std;
using namespace cv;
using namespace cv::gpu;

#if !defined (HAVE_CUDA)

void cv::gpu::blendLinear(const GpuMat&, const GpuMat&, const GpuMat&, const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }

#else

BEGIN_OPENCV_DEVICE_NAMESPACE

namespace blend
{
    template <typename T>
    void blendLinearCaller(int rows, int cols, int cn, PtrStep<T> img1, PtrStep<T> img2, PtrStepf weights1, PtrStepf weights2, PtrStep<T> result, cudaStream_t stream);

    void blendLinearCaller8UC4(int rows, int cols, PtrStepb img1, PtrStepb img2, PtrStepf weights1, PtrStepf weights2, PtrStepb result, cudaStream_t stream);
}

END_OPENCV_DEVICE_NAMESPACE

using namespace OPENCV_DEVICE_NAMESPACE_ blend;

void cv::gpu::blendLinear(const GpuMat& img1, const GpuMat& img2, const GpuMat& weights1, const GpuMat& weights2, 
                          GpuMat& result, Stream& stream)
{
    CV_Assert(img1.size() == img2.size());
    CV_Assert(img1.type() == img2.type());
    CV_Assert(weights1.size() == img1.size());
    CV_Assert(weights2.size() == img2.size());
    CV_Assert(weights1.type() == CV_32F);
    CV_Assert(weights2.type() == CV_32F);

    const Size size = img1.size();
    const int depth = img1.depth();
    const int cn = img1.channels();

    result.create(size, CV_MAKE_TYPE(depth, cn));

    switch (depth)
    {
    case CV_8U:
        if (cn != 4)
            blendLinearCaller<uchar>(size.height, size.width, cn, img1, img2, weights1, weights2, result, StreamAccessor::getStream(stream));
        else
            blendLinearCaller8UC4(size.height, size.width, img1, img2, weights1, weights2, result, StreamAccessor::getStream(stream));
        break;
    case CV_32F:
        blendLinearCaller<float>(size.height, size.width, cn, img1, img2, weights1, weights2, result, StreamAccessor::getStream(stream));
        break;
    default:
        CV_Error(CV_StsUnsupportedFormat, "bad image depth in linear blending function");
    }
}

#endif
