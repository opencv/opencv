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

Ptr<cv::cuda::FastFeatureDetector> cv::cuda::FastFeatureDetector::create(int, bool, int, int) { throw_no_cuda(); return Ptr<cv::cuda::FastFeatureDetector>(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace cuda { namespace device
{
    namespace fast
    {
        int calcKeypoints_gpu(PtrStepSzb img, PtrStepSzb mask, short2* kpLoc, int maxKeypoints, PtrStepSzi score, int threshold, unsigned int* d_counter, cudaStream_t stream);
        int nonmaxSuppression_gpu(const short2* kpLoc, int count, PtrStepSzi score, short2* loc, float* response, unsigned int* d_counter, cudaStream_t stream);
    }
}}}

namespace
{
    class FAST_Impl : public cv::cuda::FastFeatureDetector
    {
    public:
        FAST_Impl(int threshold, bool nonmaxSuppression, int max_npoints);

        virtual void detect(InputArray _image, std::vector<KeyPoint>& keypoints, InputArray _mask);
        virtual void detectAsync(InputArray _image, OutputArray _keypoints, InputArray _mask, Stream& stream);

        virtual void convert(InputArray _gpu_keypoints, std::vector<KeyPoint>& keypoints);

        virtual void setThreshold(int threshold) { threshold_ = threshold; }
        virtual int getThreshold() const { return threshold_; }

        virtual void setNonmaxSuppression(bool f) { nonmaxSuppression_ = f; }
        virtual bool getNonmaxSuppression() const { return nonmaxSuppression_; }

        virtual void setMaxNumPoints(int max_npoints) { max_npoints_ = max_npoints; }
        virtual int getMaxNumPoints() const { return max_npoints_; }

        virtual void setType(int type) { CV_Assert( type == TYPE_9_16 ); }
        virtual int getType() const { return TYPE_9_16; }

    private:
        int threshold_;
        bool nonmaxSuppression_;
        int max_npoints_;

        unsigned int* d_counter;
    };

    FAST_Impl::FAST_Impl(int threshold, bool nonmaxSuppression, int max_npoints) :
        threshold_(threshold), nonmaxSuppression_(nonmaxSuppression), max_npoints_(max_npoints)
    {
    }

    void FAST_Impl::detect(InputArray _image, std::vector<KeyPoint>& keypoints, InputArray _mask)
    {
        if (_image.empty())
        {
            keypoints.clear();
            return;
        }

        BufferPool pool(Stream::Null());
        GpuMat d_keypoints = pool.getBuffer(ROWS_COUNT, max_npoints_, CV_32FC1);

        detectAsync(_image, d_keypoints, _mask, Stream::Null());
        convert(d_keypoints, keypoints);
    }

    void FAST_Impl::detectAsync(InputArray _image, OutputArray _keypoints, InputArray _mask, Stream& stream)
    {
        using namespace cv::cuda::device::fast;

        cudaSafeCall( cudaMalloc(&d_counter, sizeof(unsigned int)) );

        const GpuMat img = _image.getGpuMat();
        const GpuMat mask = _mask.getGpuMat();

        CV_Assert( img.type() == CV_8UC1 );
        CV_Assert( mask.empty() || (mask.type() == CV_8UC1 && mask.size() == img.size()) );

        BufferPool pool(stream);

        GpuMat kpLoc = pool.getBuffer(1, max_npoints_, CV_16SC2);

        GpuMat score;
        if (nonmaxSuppression_)
        {
            score = pool.getBuffer(img.size(), CV_32SC1);
            score.setTo(Scalar::all(0), stream);
        }

        int count = calcKeypoints_gpu(img, mask, kpLoc.ptr<short2>(), max_npoints_, score, threshold_, d_counter, StreamAccessor::getStream(stream));
        count = std::min(count, max_npoints_);

        if (count == 0)
        {
            _keypoints.release();
            return;
        }

        ensureSizeIsEnough(ROWS_COUNT, count, CV_32FC1, _keypoints);
        GpuMat& keypoints = _keypoints.getGpuMatRef();

        if (nonmaxSuppression_)
        {
            count = nonmaxSuppression_gpu(kpLoc.ptr<short2>(), count, score, keypoints.ptr<short2>(LOCATION_ROW), keypoints.ptr<float>(RESPONSE_ROW), d_counter, StreamAccessor::getStream(stream));
            if (count == 0)
            {
                keypoints.release();
            }
            else
            {
                keypoints.cols = count;
            }
        }
        else
        {
            GpuMat locRow(1, count, kpLoc.type(), keypoints.ptr(0));
            kpLoc.colRange(0, count).copyTo(locRow, stream);
            keypoints.row(1).setTo(Scalar::all(0), stream);
        }

        cudaSafeCall( cudaFree(d_counter) );
    }

    void FAST_Impl::convert(InputArray _gpu_keypoints, std::vector<KeyPoint>& keypoints)
    {
        if (_gpu_keypoints.empty())
        {
            keypoints.clear();
            return;
        }

        Mat h_keypoints;
        if (_gpu_keypoints.kind() == _InputArray::CUDA_GPU_MAT)
        {
            _gpu_keypoints.getGpuMat().download(h_keypoints);
        }
        else
        {
            h_keypoints = _gpu_keypoints.getMat();
        }

        CV_Assert( h_keypoints.rows == ROWS_COUNT );
        CV_Assert( h_keypoints.elemSize() == 4 );

        const int npoints = h_keypoints.cols;

        keypoints.resize(npoints);

        const short2* loc_row = h_keypoints.ptr<short2>(LOCATION_ROW);
        const float* response_row = h_keypoints.ptr<float>(RESPONSE_ROW);

        for (int i = 0; i < npoints; ++i)
        {
            KeyPoint kp(loc_row[i].x, loc_row[i].y, static_cast<float>(FEATURE_SIZE), -1, response_row[i]);
            keypoints[i] = kp;
        }
    }
}

Ptr<cv::cuda::FastFeatureDetector> cv::cuda::FastFeatureDetector::create(int threshold, bool nonmaxSuppression, int type, int max_npoints)
{
    CV_Assert( type == TYPE_9_16 );
    return makePtr<FAST_Impl>(threshold, nonmaxSuppression, max_npoints);
}

#endif /* !defined (HAVE_CUDA) */
