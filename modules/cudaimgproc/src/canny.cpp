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

Ptr<CannyEdgeDetector> cv::cuda::createCannyEdgeDetector(double, double, int, bool) { throw_no_cuda(); return Ptr<CannyEdgeDetector>(); }

#else /* !defined (HAVE_CUDA) */

namespace canny
{
    void calcMagnitude(PtrStepSzb srcWhole, int xoff, int yoff, PtrStepSzi dx, PtrStepSzi dy, PtrStepSzf mag, bool L2Grad, cudaStream_t stream);
    void calcMagnitude(PtrStepSzi dx, PtrStepSzi dy, PtrStepSzf mag, bool L2Grad, cudaStream_t stream);

    void calcMap(PtrStepSzi dx, PtrStepSzi dy, PtrStepSzf mag, PtrStepSzi map, float low_thresh, float high_thresh, cudaStream_t stream);

    void edgesHysteresisLocal(PtrStepSzi map, short2* st1, cudaStream_t stream);

    void edgesHysteresisGlobal(PtrStepSzi map, short2* st1, short2* st2, cudaStream_t stream);

    void getEdges(PtrStepSzi map, PtrStepSzb dst, cudaStream_t stream);
}

namespace
{
    class CannyImpl : public CannyEdgeDetector
    {
    public:
        CannyImpl(double low_thresh, double high_thresh, int apperture_size, bool L2gradient) :
            low_thresh_(low_thresh), high_thresh_(high_thresh), apperture_size_(apperture_size), L2gradient_(L2gradient)
        {
            old_apperture_size_ = -1;
        }

        void detect(InputArray image, OutputArray edges, Stream& stream);
        void detect(InputArray dx, InputArray dy, OutputArray edges, Stream& stream);

        void setLowThreshold(double low_thresh) { low_thresh_ = low_thresh; }
        double getLowThreshold() const { return low_thresh_; }

        void setHighThreshold(double high_thresh) { high_thresh_ = high_thresh; }
        double getHighThreshold() const { return high_thresh_; }

        void setAppertureSize(int apperture_size) { apperture_size_ = apperture_size; }
        int getAppertureSize() const { return apperture_size_; }

        void setL2Gradient(bool L2gradient) { L2gradient_ = L2gradient; }
        bool getL2Gradient() const { return L2gradient_; }

        void write(FileStorage& fs) const
        {
            fs << "name" << "Canny_CUDA"
            << "low_thresh" << low_thresh_
            << "high_thresh" << high_thresh_
            << "apperture_size" << apperture_size_
            << "L2gradient" << L2gradient_;
        }

        void read(const FileNode& fn)
        {
            CV_Assert( String(fn["name"]) == "Canny_CUDA" );
            low_thresh_ = (double)fn["low_thresh"];
            high_thresh_ = (double)fn["high_thresh"];
            apperture_size_ = (int)fn["apperture_size"];
            L2gradient_ = (int)fn["L2gradient"] != 0;
        }

    private:
        void createBuf(Size image_size);
        void CannyCaller(GpuMat& edges, Stream& stream);

        double low_thresh_;
        double high_thresh_;
        int apperture_size_;
        bool L2gradient_;

        GpuMat dx_, dy_;
        GpuMat mag_;
        GpuMat map_;
        GpuMat st1_, st2_;
#ifdef HAVE_OPENCV_CUDAFILTERS
        Ptr<Filter> filterDX_, filterDY_;
#endif
        int old_apperture_size_;
    };

    void CannyImpl::detect(InputArray _image, OutputArray _edges, Stream& stream)
    {
        GpuMat image = _image.getGpuMat();

        CV_Assert( image.type() == CV_8UC1 );
        CV_Assert( deviceSupports(SHARED_ATOMICS) );

        if (low_thresh_ > high_thresh_)
            std::swap(low_thresh_, high_thresh_);

        createBuf(image.size());

        _edges.create(image.size(), CV_8UC1);
        GpuMat edges = _edges.getGpuMat();

        if (apperture_size_ == 3)
        {
            Size wholeSize;
            Point ofs;
            image.locateROI(wholeSize, ofs);
            GpuMat srcWhole(wholeSize, image.type(), image.datastart, image.step);

            canny::calcMagnitude(srcWhole, ofs.x, ofs.y, dx_, dy_, mag_, L2gradient_, StreamAccessor::getStream(stream));
        }
        else
        {
#ifndef HAVE_OPENCV_CUDAFILTERS
            throw_no_cuda();
#else
            filterDX_->apply(image, dx_, stream);
            filterDY_->apply(image, dy_, stream);

            canny::calcMagnitude(dx_, dy_, mag_, L2gradient_, StreamAccessor::getStream(stream));
#endif
        }

        CannyCaller(edges, stream);
    }

    void CannyImpl::detect(InputArray _dx, InputArray _dy, OutputArray _edges, Stream& stream)
    {
        GpuMat dx = _dx.getGpuMat();
        GpuMat dy = _dy.getGpuMat();

        CV_Assert( dx.type() == CV_32SC1 );
        CV_Assert( dy.type() == dx.type() && dy.size() == dx.size() );
        CV_Assert( deviceSupports(SHARED_ATOMICS) );

        dx.copyTo(dx_, stream);
        dy.copyTo(dy_, stream);

        if (low_thresh_ > high_thresh_)
            std::swap(low_thresh_, high_thresh_);

        createBuf(dx.size());

        _edges.create(dx.size(), CV_8UC1);
        GpuMat edges = _edges.getGpuMat();

        canny::calcMagnitude(dx_, dy_, mag_, L2gradient_, StreamAccessor::getStream(stream));

        CannyCaller(edges, stream);
    }

    void CannyImpl::createBuf(Size image_size)
    {
        CV_Assert(image_size.width < std::numeric_limits<short>::max() && image_size.height < std::numeric_limits<short>::max());

        ensureSizeIsEnough(image_size, CV_32SC1, dx_);
        ensureSizeIsEnough(image_size, CV_32SC1, dy_);

#ifdef HAVE_OPENCV_CUDAFILTERS
        if (apperture_size_ != 3 && apperture_size_ != old_apperture_size_)
        {
            filterDX_ = cuda::createDerivFilter(CV_8UC1, CV_32S, 1, 0, apperture_size_, false, 1, BORDER_REPLICATE);
            filterDY_ = cuda::createDerivFilter(CV_8UC1, CV_32S, 0, 1, apperture_size_, false, 1, BORDER_REPLICATE);
            old_apperture_size_ = apperture_size_;
        }
#endif

        ensureSizeIsEnough(image_size, CV_32FC1, mag_);
        ensureSizeIsEnough(image_size, CV_32SC1, map_);

        ensureSizeIsEnough(1, image_size.area(), CV_16SC2, st1_);
        ensureSizeIsEnough(1, image_size.area(), CV_16SC2, st2_);
    }

    void CannyImpl::CannyCaller(GpuMat& edges, Stream& stream)
    {
        map_.setTo(Scalar::all(0));
        canny::calcMap(dx_, dy_, mag_, map_, static_cast<float>(low_thresh_), static_cast<float>(high_thresh_), StreamAccessor::getStream(stream));

        canny::edgesHysteresisLocal(map_, st1_.ptr<short2>(), StreamAccessor::getStream(stream));

        canny::edgesHysteresisGlobal(map_, st1_.ptr<short2>(), st2_.ptr<short2>(), StreamAccessor::getStream(stream));

        canny::getEdges(map_, edges, StreamAccessor::getStream(stream));
    }
}

Ptr<CannyEdgeDetector> cv::cuda::createCannyEdgeDetector(double low_thresh, double high_thresh, int apperture_size, bool L2gradient)
{
    return makePtr<CannyImpl>(low_thresh, high_thresh, apperture_size, L2gradient);
}

#endif /* !defined (HAVE_CUDA) */
