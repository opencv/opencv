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

Ptr<cuda::StereoBM> cv::cuda::createStereoBM(int, int) { throw_no_cuda(); return Ptr<cuda::StereoBM>(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace cuda { namespace device
{
    namespace stereobm
    {
        void stereoBM_CUDA(const PtrStepSzb& left, const PtrStepSzb& right, const PtrStepSzb& disp, int ndisp, int winsz, const PtrStepSz<unsigned int>& minSSD_buf, cudaStream_t & stream);
        void prefilter_xsobel(const PtrStepSzb& input, const PtrStepSzb& output, int prefilterCap /*= 31*/, cudaStream_t & stream);
        void postfilter_textureness(const PtrStepSzb& input, int winsz, float avgTexturenessThreshold, const PtrStepSzb& disp, cudaStream_t & stream);
    }
}}}

namespace
{
    class StereoBMImpl : public cuda::StereoBM
    {
    public:
        StereoBMImpl(int numDisparities, int blockSize);

        void compute(InputArray left, InputArray right, OutputArray disparity);
        void compute(InputArray left, InputArray right, OutputArray disparity, Stream& stream);

        int getMinDisparity() const { return 0; }
        void setMinDisparity(int /*minDisparity*/) {}

        int getNumDisparities() const { return ndisp_; }
        void setNumDisparities(int numDisparities) { ndisp_ = numDisparities; }

        int getBlockSize() const { return winSize_; }
        void setBlockSize(int blockSize) { winSize_ = blockSize; }

        int getSpeckleWindowSize() const { return 0; }
        void setSpeckleWindowSize(int /*speckleWindowSize*/) {}

        int getSpeckleRange() const { return 0; }
        void setSpeckleRange(int /*speckleRange*/) {}

        int getDisp12MaxDiff() const { return 0; }
        void setDisp12MaxDiff(int /*disp12MaxDiff*/) {}

        int getPreFilterType() const { return preset_; }
        void setPreFilterType(int preFilterType) { preset_ = preFilterType; }

        int getPreFilterSize() const { return 0; }
        void setPreFilterSize(int /*preFilterSize*/) {}

        int getPreFilterCap() const { return preFilterCap_; }
        void setPreFilterCap(int preFilterCap) { preFilterCap_ = preFilterCap; }

        int getTextureThreshold() const { return static_cast<int>(avergeTexThreshold_); }
        void setTextureThreshold(int textureThreshold) { avergeTexThreshold_ = static_cast<float>(textureThreshold); }

        int getUniquenessRatio() const { return 0; }
        void setUniquenessRatio(int /*uniquenessRatio*/) {}

        int getSmallerBlockSize() const { return 0; }
        void setSmallerBlockSize(int /*blockSize*/){}

        Rect getROI1() const { return Rect(); }
        void setROI1(Rect /*roi1*/) {}

        Rect getROI2() const { return Rect(); }
        void setROI2(Rect /*roi2*/) {}

    private:
        int preset_;
        int ndisp_;
        int winSize_;
        int preFilterCap_;
        float avergeTexThreshold_;

        GpuMat minSSD_, leBuf_, riBuf_;
    };

    StereoBMImpl::StereoBMImpl(int numDisparities, int blockSize)
        : preset_(0), ndisp_(numDisparities), winSize_(blockSize), preFilterCap_(31), avergeTexThreshold_(3)
    {
    }

    void StereoBMImpl::compute(InputArray left, InputArray right, OutputArray disparity)
    {
        compute(left, right, disparity, Stream::Null());
    }

    void StereoBMImpl::compute(InputArray _left, InputArray _right, OutputArray _disparity, Stream& _stream)
    {
        using namespace ::cv::cuda::device::stereobm;

        const int max_supported_ndisp = 1 << (sizeof(unsigned char) * 8);
        CV_Assert( 0 < ndisp_ && ndisp_ <= max_supported_ndisp );
        CV_Assert( ndisp_ % 8 == 0 );
        CV_Assert( winSize_ % 2 == 1 );

        GpuMat left = _left.getGpuMat();
        GpuMat right = _right.getGpuMat();

        CV_Assert( left.type() == CV_8UC1 );
        CV_Assert( left.size() == right.size() && left.type() == right.type() );

        _disparity.create(left.size(), CV_8UC1);
        GpuMat disparity = _disparity.getGpuMat();

        cudaStream_t stream = StreamAccessor::getStream(_stream);

        cuda::ensureSizeIsEnough(left.size(), CV_32SC1, minSSD_);

        PtrStepSzb le_for_bm =  left;
        PtrStepSzb ri_for_bm = right;

        if (preset_ == cv::StereoBM::PREFILTER_XSOBEL)
        {
            cuda::ensureSizeIsEnough(left.size(), left.type(), leBuf_);
            cuda::ensureSizeIsEnough(right.size(), right.type(), riBuf_);

            prefilter_xsobel( left, leBuf_, preFilterCap_, stream);
            prefilter_xsobel(right, riBuf_, preFilterCap_, stream);

            le_for_bm = leBuf_;
            ri_for_bm = riBuf_;
        }

        stereoBM_CUDA(le_for_bm, ri_for_bm, disparity, ndisp_, winSize_, minSSD_, stream);

        if (avergeTexThreshold_ > 0)
            postfilter_textureness(le_for_bm, winSize_, avergeTexThreshold_, disparity, stream);
    }
}

Ptr<cuda::StereoBM> cv::cuda::createStereoBM(int numDisparities, int blockSize)
{
    return makePtr<StereoBMImpl>(numDisparities, blockSize);
}

#endif /* !defined (HAVE_CUDA) */
