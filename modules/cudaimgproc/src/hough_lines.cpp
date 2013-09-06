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

Ptr<cuda::HoughLinesDetector> cv::cuda::createHoughLinesDetector(float, float, int, bool, int) { throw_no_cuda(); return Ptr<HoughLinesDetector>(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace cuda { namespace device
{
    namespace hough
    {
        int buildPointList_gpu(PtrStepSzb src, unsigned int* list);
    }

    namespace hough_lines
    {
        void linesAccum_gpu(const unsigned int* list, int count, PtrStepSzi accum, float rho, float theta, size_t sharedMemPerBlock, bool has20);
        int linesGetResult_gpu(PtrStepSzi accum, float2* out, int* votes, int maxSize, float rho, float theta, int threshold, bool doSort);
    }
}}}

namespace
{
    class HoughLinesDetectorImpl : public HoughLinesDetector
    {
    public:
        HoughLinesDetectorImpl(float rho, float theta, int threshold, bool doSort, int maxLines) :
            rho_(rho), theta_(theta), threshold_(threshold), doSort_(doSort), maxLines_(maxLines)
        {
        }

        void detect(InputArray src, OutputArray lines);
        void downloadResults(InputArray d_lines, OutputArray h_lines, OutputArray h_votes = noArray());

        void setRho(float rho) { rho_ = rho; }
        float getRho() const { return rho_; }

        void setTheta(float theta) { theta_ = theta; }
        float getTheta() const { return theta_; }

        void setThreshold(int threshold) { threshold_ = threshold; }
        int getThreshold() const { return threshold_; }

        void setDoSort(bool doSort) { doSort_ = doSort; }
        bool getDoSort() const { return doSort_; }

        void setMaxLines(int maxLines) { maxLines_ = maxLines; }
        int getMaxLines() const { return maxLines_; }

        void write(FileStorage& fs) const
        {
            fs << "name" << "HoughLinesDetector_CUDA"
            << "rho" << rho_
            << "theta" << theta_
            << "threshold" << threshold_
            << "doSort" << doSort_
            << "maxLines" << maxLines_;
        }

        void read(const FileNode& fn)
        {
            CV_Assert( String(fn["name"]) == "HoughLinesDetector_CUDA" );
            rho_ = (float)fn["rho"];
            theta_ = (float)fn["theta"];
            threshold_ = (int)fn["threshold"];
            doSort_ = (int)fn["doSort"] != 0;
            maxLines_ = (int)fn["maxLines"];
        }

    private:
        float rho_;
        float theta_;
        int threshold_;
        bool doSort_;
        int maxLines_;

        GpuMat accum_;
        GpuMat list_;
        GpuMat result_;
    };

    void HoughLinesDetectorImpl::detect(InputArray _src, OutputArray lines)
    {
        using namespace cv::cuda::device::hough;
        using namespace cv::cuda::device::hough_lines;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.type() == CV_8UC1 );
        CV_Assert( src.cols < std::numeric_limits<unsigned short>::max() );
        CV_Assert( src.rows < std::numeric_limits<unsigned short>::max() );

        ensureSizeIsEnough(1, src.size().area(), CV_32SC1, list_);
        unsigned int* srcPoints = list_.ptr<unsigned int>();

        const int pointsCount = buildPointList_gpu(src, srcPoints);
        if (pointsCount == 0)
        {
            lines.release();
            return;
        }

        const int numangle = cvRound(CV_PI / theta_);
        const int numrho = cvRound(((src.cols + src.rows) * 2 + 1) / rho_);
        CV_Assert( numangle > 0 && numrho > 0 );

        ensureSizeIsEnough(numangle + 2, numrho + 2, CV_32SC1, accum_);
        accum_.setTo(Scalar::all(0));

        DeviceInfo devInfo;
        linesAccum_gpu(srcPoints, pointsCount, accum_, rho_, theta_, devInfo.sharedMemPerBlock(), devInfo.supports(FEATURE_SET_COMPUTE_20));

        ensureSizeIsEnough(2, maxLines_, CV_32FC2, result_);

        int linesCount = linesGetResult_gpu(accum_, result_.ptr<float2>(0), result_.ptr<int>(1), maxLines_, rho_, theta_, threshold_, doSort_);

        if (linesCount == 0)
        {
            lines.release();
            return;
        }

        result_.cols = linesCount;
        result_.copyTo(lines);
    }

    void HoughLinesDetectorImpl::downloadResults(InputArray _d_lines, OutputArray h_lines, OutputArray h_votes)
    {
        GpuMat d_lines = _d_lines.getGpuMat();

        if (d_lines.empty())
        {
            h_lines.release();
            if (h_votes.needed())
                h_votes.release();
            return;
        }

        CV_Assert( d_lines.rows == 2 && d_lines.type() == CV_32FC2 );

        d_lines.row(0).download(h_lines);

        if (h_votes.needed())
        {
            GpuMat d_votes(1, d_lines.cols, CV_32SC1, d_lines.ptr<int>(1));
            d_votes.download(h_votes);
        }
    }
}

Ptr<HoughLinesDetector> cv::cuda::createHoughLinesDetector(float rho, float theta, int threshold, bool doSort, int maxLines)
{
    return makePtr<HoughLinesDetectorImpl>(rho, theta, threshold, doSort, maxLines);
}

#endif /* !defined (HAVE_CUDA) */
