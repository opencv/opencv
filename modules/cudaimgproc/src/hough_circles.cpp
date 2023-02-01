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

Ptr<cuda::HoughCirclesDetector> cv::cuda::createHoughCirclesDetector(float, float, int, int, int, int, int) { throw_no_cuda(); return Ptr<HoughCirclesDetector>(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace cuda { namespace device
{
    namespace hough
    {
        int buildPointList_gpu(PtrStepSzb src, unsigned int* list);
    }

    namespace hough_circles
    {
        void circlesAccumCenters_gpu(const unsigned int* list, int count, PtrStepi dx, PtrStepi dy, PtrStepSzi accum, int minRadius, int maxRadius, float idp);
        int buildCentersList_gpu(PtrStepSzi accum, unsigned int* centers, int threshold);
        int circlesAccumRadius_gpu(const unsigned int* centers, int centersCount, const unsigned int* list, int count,
                                   float3* circles, int maxCircles, float dp, int minRadius, int maxRadius, int threshold, bool has20);
    }
}}}

namespace
{
    class HoughCirclesDetectorImpl : public HoughCirclesDetector
    {
    public:
        HoughCirclesDetectorImpl(float dp, float minDist, int cannyThreshold, int votesThreshold, int minRadius, int maxRadius, int maxCircles);

        void detect(InputArray src, OutputArray circles, Stream& stream);

        void setDp(float dp) { dp_ = dp; }
        float getDp() const { return dp_; }

        void setMinDist(float minDist) { minDist_ = minDist; }
        float getMinDist() const { return minDist_; }

        void setCannyThreshold(int cannyThreshold) { cannyThreshold_ = cannyThreshold; }
        int getCannyThreshold() const { return cannyThreshold_; }

        void setVotesThreshold(int votesThreshold) { votesThreshold_ = votesThreshold; }
        int getVotesThreshold() const { return votesThreshold_; }

        void setMinRadius(int minRadius) { minRadius_ = minRadius; }
        int getMinRadius() const { return minRadius_; }

        void setMaxRadius(int maxRadius) { maxRadius_ = maxRadius; }
        int getMaxRadius() const { return maxRadius_; }

        void setMaxCircles(int maxCircles) { maxCircles_ = maxCircles; }
        int getMaxCircles() const { return maxCircles_; }

        void write(FileStorage& fs) const
        {
            writeFormat(fs);
            fs << "name" << "HoughCirclesDetector_CUDA"
            << "dp" << dp_
            << "minDist" << minDist_
            << "cannyThreshold" << cannyThreshold_
            << "votesThreshold" << votesThreshold_
            << "minRadius" << minRadius_
            << "maxRadius" << maxRadius_
            << "maxCircles" << maxCircles_;
        }

        void read(const FileNode& fn)
        {
            CV_Assert( String(fn["name"]) == "HoughCirclesDetector_CUDA" );
            dp_ = (float)fn["dp"];
            minDist_ = (float)fn["minDist"];
            cannyThreshold_ = (int)fn["cannyThreshold"];
            votesThreshold_ = (int)fn["votesThreshold"];
            minRadius_ = (int)fn["minRadius"];
            maxRadius_ = (int)fn["maxRadius"];
            maxCircles_ = (int)fn["maxCircles"];
        }

    private:
        float dp_;
        float minDist_;
        int cannyThreshold_;
        int votesThreshold_;
        int minRadius_;
        int maxRadius_;
        int maxCircles_;

        GpuMat dx_, dy_;
        GpuMat edges_;
        GpuMat accum_;
        Mat tt; //CPU copy of accum_
        GpuMat list_;
        GpuMat result_;
        Ptr<cuda::Filter> filterDx_;
        Ptr<cuda::Filter> filterDy_;
        Ptr<cuda::CannyEdgeDetector> canny_;
    };

    bool centersCompare(Vec3f a, Vec3f b) {return (a[2] > b[2]);}

    HoughCirclesDetectorImpl::HoughCirclesDetectorImpl(float dp, float minDist, int cannyThreshold, int votesThreshold,
                                                       int minRadius, int maxRadius, int maxCircles) :
        dp_(dp), minDist_(minDist), cannyThreshold_(cannyThreshold), votesThreshold_(votesThreshold),
        minRadius_(minRadius), maxRadius_(maxRadius), maxCircles_(maxCircles)
    {
        canny_ = cuda::createCannyEdgeDetector(std::max(cannyThreshold_ / 2, 1), cannyThreshold_);

        filterDx_ = cuda::createSobelFilter(CV_8UC1, CV_32S, 1, 0);
        filterDy_ = cuda::createSobelFilter(CV_8UC1, CV_32S, 0, 1);
    }

    void HoughCirclesDetectorImpl::detect(InputArray _src, OutputArray circles, Stream& stream)
    {
        // TODO : implement async version
        CV_UNUSED(stream);

        using namespace cv::cuda::device::hough;
        using namespace cv::cuda::device::hough_circles;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.type() == CV_8UC1 );
        CV_Assert( src.cols < std::numeric_limits<unsigned short>::max() );
        CV_Assert( src.rows < std::numeric_limits<unsigned short>::max() );
        CV_Assert( dp_ > 0 );
        CV_Assert( minRadius_ > 0 && maxRadius_ > minRadius_ );
        CV_Assert( cannyThreshold_ > 0 );
        CV_Assert( votesThreshold_ > 0 );
        CV_Assert( maxCircles_ > 0 );

        const float idp = 1.0f / dp_;

        filterDx_->apply(src, dx_);
        filterDy_->apply(src, dy_);

        canny_->setLowThreshold(std::max(cannyThreshold_ / 2, 1));
        canny_->setHighThreshold(cannyThreshold_);

        canny_->detect(dx_, dy_, edges_);

        ensureSizeIsEnough(2, src.size().area(), CV_32SC1, list_);
        unsigned int* srcPoints = list_.ptr<unsigned int>(0);
        unsigned int* centers = list_.ptr<unsigned int>(1);

        const int pointsCount = buildPointList_gpu(edges_, srcPoints);
        if (pointsCount == 0)
        {
            circles.release();
            return;
        }

        ensureSizeIsEnough(cvCeil(src.rows * idp) + 2, cvCeil(src.cols * idp) + 2, CV_32SC1, accum_);
        accum_.setTo(Scalar::all(0));

        circlesAccumCenters_gpu(srcPoints, pointsCount, dx_, dy_, accum_, minRadius_, maxRadius_, idp);

        accum_.download(tt);

        int centersCount = buildCentersList_gpu(accum_, centers, votesThreshold_);
        if (centersCount == 0)
        {
            circles.release();
            return;
        }

        if (minDist_ > 1)
        {
            AutoBuffer<ushort2> oldBuf_(centersCount);
            AutoBuffer<ushort2> newBuf_(centersCount);
            int newCount = 0;

            ushort2* oldBuf = oldBuf_.data();
            ushort2* newBuf = newBuf_.data();

            cudaSafeCall( cudaMemcpy(oldBuf, centers, centersCount * sizeof(ushort2), cudaMemcpyDeviceToHost) );

            const int cellSize = cvRound(minDist_);
            const int gridWidth = (src.cols + cellSize - 1) / cellSize;
            const int gridHeight = (src.rows + cellSize - 1) / cellSize;

            std::vector< std::vector<ushort2> > grid(gridWidth * gridHeight);

            const float minDist2 = minDist_ * minDist_;

            std::vector<Vec3f> sortBuf;
            for(int i=0; i<centersCount; i++){
                Vec3f temp;
                temp[0] = oldBuf[i].x;
                temp[1] = oldBuf[i].y;
                temp[2] = tt.at<int>(temp[1]+1, temp[0]+1);
                sortBuf.push_back(temp);
            }
            std::sort(sortBuf.begin(), sortBuf.end(), centersCompare);

            for (int i = 0; i < centersCount; ++i)
            {
                ushort2 p;
                p.x = sortBuf[i][0];
                p.y = sortBuf[i][1];

                bool good = true;

                int xCell = static_cast<int>(p.x / cellSize);
                int yCell = static_cast<int>(p.y / cellSize);

                int x1 = xCell - 1;
                int y1 = yCell - 1;
                int x2 = xCell + 1;
                int y2 = yCell + 1;

                // boundary check
                x1 = std::max(0, x1);
                y1 = std::max(0, y1);
                x2 = std::min(gridWidth - 1, x2);
                y2 = std::min(gridHeight - 1, y2);

                for (int yy = y1; yy <= y2; ++yy)
                {
                    for (int xx = x1; xx <= x2; ++xx)
                    {
                        std::vector<ushort2>& m = grid[yy * gridWidth + xx];

                        for(size_t j = 0; j < m.size(); ++j)
                        {
                            float dx = (float)(p.x - m[j].x);
                            float dy = (float)(p.y - m[j].y);

                            if (dx * dx + dy * dy < minDist2)
                            {
                                good = false;
                                goto break_out;
                            }
                        }
                    }
                }

                break_out:

                if(good)
                {
                    grid[yCell * gridWidth + xCell].push_back(p);

                    newBuf[newCount++] = p;
                }
            }

            cudaSafeCall( cudaMemcpy(centers, newBuf, newCount * sizeof(unsigned int), cudaMemcpyHostToDevice) );
            centersCount = newCount;
        }

        ensureSizeIsEnough(1, maxCircles_, CV_32FC3, result_);

        int circlesCount = circlesAccumRadius_gpu(centers, centersCount, srcPoints, pointsCount, result_.ptr<float3>(), maxCircles_,
                                                  dp_, minRadius_, maxRadius_, votesThreshold_, deviceSupports(FEATURE_SET_COMPUTE_20));

        if (circlesCount == 0)
        {
            circles.release();
            return;
        }

        result_.cols = circlesCount;
        result_.copyTo(circles);
    }
}

Ptr<HoughCirclesDetector> cv::cuda::createHoughCirclesDetector(float dp, float minDist, int cannyThreshold, int votesThreshold, int minRadius, int maxRadius, int maxCircles)
{
    return makePtr<HoughCirclesDetectorImpl>(dp, minDist, cannyThreshold, votesThreshold, minRadius, maxRadius, maxCircles);
}

#endif /* !defined (HAVE_CUDA) */
