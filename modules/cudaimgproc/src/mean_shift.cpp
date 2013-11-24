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

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

void cv::cuda::meanShiftFiltering(InputArray, OutputArray, int, int, TermCriteria, Stream&) { throw_no_cuda(); }
void cv::cuda::meanShiftProc(InputArray, OutputArray, OutputArray, int, int, TermCriteria, Stream&) { throw_no_cuda(); }

#else /* !defined (HAVE_CUDA) */

////////////////////////////////////////////////////////////////////////
// meanShiftFiltering

namespace cv { namespace cuda { namespace device
{
    namespace imgproc
    {
        void meanShiftFiltering_gpu(const PtrStepSzb& src, PtrStepSzb dst, int sp, int sr, int maxIter, float eps, cudaStream_t stream);
    }
}}}

void cv::cuda::meanShiftFiltering(InputArray _src, OutputArray _dst, int sp, int sr, TermCriteria criteria, Stream& stream)
{
    using namespace ::cv::cuda::device::imgproc;

    GpuMat src = _src.getGpuMat();

    CV_Assert( src.type() == CV_8UC4 );

    _dst.create(src.size(), CV_8UC4);
    GpuMat dst = _dst.getGpuMat();

    if (!(criteria.type & TermCriteria::MAX_ITER))
        criteria.maxCount = 5;

    int maxIter = std::min(std::max(criteria.maxCount, 1), 100);

    if (!(criteria.type & TermCriteria::EPS))
        criteria.epsilon = 1.f;

    float eps = (float) std::max(criteria.epsilon, 0.0);

    meanShiftFiltering_gpu(src, dst, sp, sr, maxIter, eps, StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// meanShiftProc_CUDA

namespace cv { namespace cuda { namespace device
{
    namespace imgproc
    {
        void meanShiftProc_gpu(const PtrStepSzb& src, PtrStepSzb dstr, PtrStepSzb dstsp, int sp, int sr, int maxIter, float eps, cudaStream_t stream);
    }
}}}

void cv::cuda::meanShiftProc(InputArray _src, OutputArray _dstr, OutputArray _dstsp, int sp, int sr, TermCriteria criteria, Stream& stream)
{
    using namespace ::cv::cuda::device::imgproc;

    GpuMat src = _src.getGpuMat();

    CV_Assert( src.type() == CV_8UC4 );

    _dstr.create(src.size(), CV_8UC4);
    _dstsp.create(src.size(), CV_16SC2);

    GpuMat dstr = _dstr.getGpuMat();
    GpuMat dstsp = _dstsp.getGpuMat();

    if (!(criteria.type & TermCriteria::MAX_ITER))
        criteria.maxCount = 5;

    int maxIter = std::min(std::max(criteria.maxCount, 1), 100);

    if (!(criteria.type & TermCriteria::EPS))
        criteria.epsilon = 1.f;

    float eps = (float) std::max(criteria.epsilon, 0.0);

    meanShiftProc_gpu(src, dstr, dstsp, sp, sr, maxIter, eps, StreamAccessor::getStream(stream));
}

#endif /* !defined (HAVE_CUDA) */
