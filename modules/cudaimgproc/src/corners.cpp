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

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER) || !defined(HAVE_OPENCV_CUDAFILTERS)

Ptr<cuda::CornernessCriteria> cv::cuda::createHarrisCorner(int, int, int, double, int) { throw_no_cuda(); return Ptr<cuda::CornernessCriteria>(); }
Ptr<cuda::CornernessCriteria> cv::cuda::createMinEigenValCorner(int, int, int, int) { throw_no_cuda(); return Ptr<cuda::CornernessCriteria>(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace cuda { namespace device
{
    namespace imgproc
    {
        void cornerHarris_gpu(int block_size, float k, PtrStepSzf Dx, PtrStepSzf Dy, PtrStepSzf dst, int border_type, cudaStream_t stream);
        void cornerMinEigenVal_gpu(int block_size, PtrStepSzf Dx, PtrStepSzf Dy, PtrStepSzf dst, int border_type, cudaStream_t stream);
    }
}}}

namespace
{
    class CornerBase : public CornernessCriteria
    {
    protected:
        CornerBase(int srcType, int blockSize, int ksize, int borderType);

        void extractCovData(const GpuMat& src, Stream& stream);

        int srcType_;
        int blockSize_;
        int ksize_;
        int borderType_;
        GpuMat Dx_, Dy_;

    private:
        Ptr<cuda::Filter> filterDx_, filterDy_;
    };

    CornerBase::CornerBase(int srcType, int blockSize, int ksize, int borderType) :
        srcType_(srcType), blockSize_(blockSize), ksize_(ksize), borderType_(borderType)
    {
        CV_Assert( borderType_ == BORDER_REFLECT101 || borderType_ == BORDER_REPLICATE || borderType_ == BORDER_REFLECT );

        const int sdepth = CV_MAT_DEPTH(srcType_);
        const int cn = CV_MAT_CN(srcType_);

        CV_Assert( cn == 1 );

        double scale = static_cast<double>(1 << ((ksize_ > 0 ? ksize_ : 3) - 1)) * blockSize_;

        if (ksize_ < 0)
            scale *= 2.;

        if (sdepth == CV_8U)
            scale *= 255.;

        scale = 1./scale;

        if (ksize_ > 0)
        {
            filterDx_ = cuda::createSobelFilter(srcType, CV_32F, 1, 0, ksize_, scale, borderType_);
            filterDy_ = cuda::createSobelFilter(srcType, CV_32F, 0, 1, ksize_, scale, borderType_);
        }
        else
        {
            filterDx_ = cuda::createScharrFilter(srcType, CV_32F, 1, 0, scale, borderType_);
            filterDy_ = cuda::createScharrFilter(srcType, CV_32F, 0, 1, scale, borderType_);
        }
    }

    void CornerBase::extractCovData(const GpuMat& src, Stream& stream)
    {
        CV_Assert( src.type() == srcType_ );
        filterDx_->apply(src, Dx_, stream);
        filterDy_->apply(src, Dy_, stream);
    }

    class Harris : public CornerBase
    {
    public:
        Harris(int srcType, int blockSize, int ksize, double k, int borderType) :
            CornerBase(srcType, blockSize, ksize, borderType), k_(static_cast<float>(k))
        {
        }

        void compute(InputArray src, OutputArray dst, Stream& stream = Stream::Null());

    private:
        float k_;
    };

    void Harris::compute(InputArray _src, OutputArray _dst, Stream& stream)
    {
        using namespace cv::cuda::device::imgproc;

        GpuMat src = _src.getGpuMat();

        extractCovData(src, stream);

        _dst.create(src.size(), CV_32FC1);
        GpuMat dst = _dst.getGpuMat();

        cornerHarris_gpu(blockSize_, k_, Dx_, Dy_, dst, borderType_, StreamAccessor::getStream(stream));
    }

    class MinEigenVal : public CornerBase
    {
    public:
        MinEigenVal(int srcType, int blockSize, int ksize, int borderType) :
            CornerBase(srcType, blockSize, ksize, borderType)
        {
        }

        void compute(InputArray src, OutputArray dst, Stream& stream = Stream::Null());

    private:
        float k_;
    };

    void MinEigenVal::compute(InputArray _src, OutputArray _dst, Stream& stream)
    {
        using namespace cv::cuda::device::imgproc;

        GpuMat src = _src.getGpuMat();

        extractCovData(src, stream);

        _dst.create(src.size(), CV_32FC1);
        GpuMat dst = _dst.getGpuMat();

        cornerMinEigenVal_gpu(blockSize_, Dx_, Dy_, dst, borderType_, StreamAccessor::getStream(stream));
    }
}

Ptr<cuda::CornernessCriteria> cv::cuda::createHarrisCorner(int srcType, int blockSize, int ksize, double k, int borderType)
{
    return makePtr<Harris>(srcType, blockSize, ksize, k, borderType);
}

Ptr<cuda::CornernessCriteria> cv::cuda::createMinEigenValCorner(int srcType, int blockSize, int ksize, int borderType)
{
    return makePtr<MinEigenVal>(srcType, blockSize, ksize, borderType);
}

#endif /* !defined (HAVE_CUDA) */
