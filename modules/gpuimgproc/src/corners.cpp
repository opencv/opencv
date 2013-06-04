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
using namespace cv::gpu;

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

void cv::gpu::cornerHarris(const GpuMat&, GpuMat&, int, int, double, int) { throw_no_cuda(); }
void cv::gpu::cornerHarris(const GpuMat&, GpuMat&, GpuMat&, GpuMat&, int, int, double, int) { throw_no_cuda(); }
void cv::gpu::cornerHarris(const GpuMat&, GpuMat&, GpuMat&, GpuMat&, GpuMat&, int, int, double, int, Stream&) { throw_no_cuda(); }

void cv::gpu::cornerMinEigenVal(const GpuMat&, GpuMat&, int, int, int) { throw_no_cuda(); }
void cv::gpu::cornerMinEigenVal(const GpuMat&, GpuMat&, GpuMat&, GpuMat&, int, int, int) { throw_no_cuda(); }
void cv::gpu::cornerMinEigenVal(const GpuMat&, GpuMat&, GpuMat&, GpuMat&, GpuMat&, int, int, int, Stream&) { throw_no_cuda(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace gpu { namespace cudev
{
    namespace imgproc
    {
        void cornerHarris_gpu(int block_size, float k, PtrStepSzf Dx, PtrStepSzf Dy, PtrStepSzf dst, int border_type, cudaStream_t stream);
        void cornerMinEigenVal_gpu(int block_size, PtrStepSzf Dx, PtrStepSzf Dy, PtrStepSzf dst, int border_type, cudaStream_t stream);
    }
}}}

namespace
{
    void extractCovData(const GpuMat& src, GpuMat& Dx, GpuMat& Dy, GpuMat& buf, int blockSize, int ksize, int borderType, Stream& stream)
    {
        double scale = static_cast<double>(1 << ((ksize > 0 ? ksize : 3) - 1)) * blockSize;

        if (ksize < 0)
            scale *= 2.;

        if (src.depth() == CV_8U)
            scale *= 255.;

        scale = 1./scale;

        Dx.create(src.size(), CV_32F);
        Dy.create(src.size(), CV_32F);

        if (ksize > 0)
        {
            Sobel(src, Dx, CV_32F, 1, 0, buf, ksize, scale, borderType, -1, stream);
            Sobel(src, Dy, CV_32F, 0, 1, buf, ksize, scale, borderType, -1, stream);
        }
        else
        {
            Scharr(src, Dx, CV_32F, 1, 0, buf, scale, borderType, -1, stream);
            Scharr(src, Dy, CV_32F, 0, 1, buf, scale, borderType, -1, stream);
        }
    }
}

void cv::gpu::cornerHarris(const GpuMat& src, GpuMat& dst, int blockSize, int ksize, double k, int borderType)
{
    GpuMat Dx, Dy;
    cornerHarris(src, dst, Dx, Dy, blockSize, ksize, k, borderType);
}

void cv::gpu::cornerHarris(const GpuMat& src, GpuMat& dst, GpuMat& Dx, GpuMat& Dy, int blockSize, int ksize, double k, int borderType)
{
    GpuMat buf;
    cornerHarris(src, dst, Dx, Dy, buf, blockSize, ksize, k, borderType);
}

void cv::gpu::cornerHarris(const GpuMat& src, GpuMat& dst, GpuMat& Dx, GpuMat& Dy, GpuMat& buf, int blockSize, int ksize, double k, int borderType, Stream& stream)
{
    using namespace cv::gpu::cudev::imgproc;

    CV_Assert(borderType == cv::BORDER_REFLECT101 || borderType == cv::BORDER_REPLICATE || borderType == cv::BORDER_REFLECT);

    extractCovData(src, Dx, Dy, buf, blockSize, ksize, borderType, stream);

    dst.create(src.size(), CV_32F);

    cornerHarris_gpu(blockSize, static_cast<float>(k), Dx, Dy, dst, borderType, StreamAccessor::getStream(stream));
}

void cv::gpu::cornerMinEigenVal(const GpuMat& src, GpuMat& dst, int blockSize, int ksize, int borderType)
{
    GpuMat Dx, Dy;
    cornerMinEigenVal(src, dst, Dx, Dy, blockSize, ksize, borderType);
}

void cv::gpu::cornerMinEigenVal(const GpuMat& src, GpuMat& dst, GpuMat& Dx, GpuMat& Dy, int blockSize, int ksize, int borderType)
{
    GpuMat buf;
    cornerMinEigenVal(src, dst, Dx, Dy, buf, blockSize, ksize, borderType);
}

void cv::gpu::cornerMinEigenVal(const GpuMat& src, GpuMat& dst, GpuMat& Dx, GpuMat& Dy, GpuMat& buf, int blockSize, int ksize, int borderType, Stream& stream)
{
    using namespace ::cv::gpu::cudev::imgproc;

    CV_Assert(borderType == cv::BORDER_REFLECT101 || borderType == cv::BORDER_REPLICATE || borderType == cv::BORDER_REFLECT);

    extractCovData(src, Dx, Dy, buf, blockSize, ksize, borderType, stream);

    dst.create(src.size(), CV_32F);

    cornerMinEigenVal_gpu(blockSize, Dx, Dy, dst, borderType, StreamAccessor::getStream(stream));
}

#endif /* !defined (HAVE_CUDA) */
