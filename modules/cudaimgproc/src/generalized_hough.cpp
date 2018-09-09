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

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER) || !defined(HAVE_OPENCV_CUDAARITHM)

Ptr<GeneralizedHoughBallard> cv::cuda::createGeneralizedHoughBallard() { throw_no_cuda(); return Ptr<GeneralizedHoughBallard>(); }

Ptr<GeneralizedHoughGuil> cv::cuda::createGeneralizedHoughGuil() { throw_no_cuda(); return Ptr<GeneralizedHoughGuil>(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace cuda { namespace device
{
    namespace ght
    {
        template <typename T>
        int buildEdgePointList_gpu(PtrStepSzb edges, PtrStepSzb dx, PtrStepSzb dy, unsigned int* coordList, float* thetaList);
        void buildRTable_gpu(const unsigned int* coordList, const float* thetaList, int pointsCount,
                             PtrStepSz<short2> r_table, int* r_sizes,
                             short2 templCenter, int levels);

        void Ballard_Pos_calcHist_gpu(const unsigned int* coordList, const float* thetaList, int pointsCount,
                                      PtrStepSz<short2> r_table, const int* r_sizes,
                                      PtrStepSzi hist,
                                      float dp, int levels);
        int Ballard_Pos_findPosInHist_gpu(PtrStepSzi hist, float4* out, int3* votes, int maxSize, float dp, int threshold);

        void Guil_Full_setTemplFeatures(PtrStepb p1_pos, PtrStepb p1_theta, PtrStepb p2_pos, PtrStepb d12, PtrStepb r1, PtrStepb r2);
        void Guil_Full_setImageFeatures(PtrStepb p1_pos, PtrStepb p1_theta, PtrStepb p2_pos, PtrStepb d12, PtrStepb r1, PtrStepb r2);
        void Guil_Full_buildTemplFeatureList_gpu(const unsigned int* coordList, const float* thetaList, int pointsCount,
                                                 int* sizes, int maxSize,
                                                 float xi, float angleEpsilon, int levels,
                                                 float2 center, float maxDist);
        void Guil_Full_buildImageFeatureList_gpu(const unsigned int* coordList, const float* thetaList, int pointsCount,
                                                 int* sizes, int maxSize,
                                                 float xi, float angleEpsilon, int levels,
                                                 float2 center, float maxDist);
        void Guil_Full_calcOHist_gpu(const int* templSizes, const int* imageSizes, int* OHist,
                                     float minAngle, float maxAngle, float angleStep, int angleRange,
                                     int levels, int tMaxSize);
        void Guil_Full_calcSHist_gpu(const int* templSizes, const int* imageSizes, int* SHist,
                                     float angle, float angleEpsilon,
                                     float minScale, float maxScale, float iScaleStep, int scaleRange,
                                     int levels, int tMaxSize);
        void Guil_Full_calcPHist_gpu(const int* templSizes, const int* imageSizes, PtrStepSzi PHist,
                                     float angle, float angleEpsilon, float scale,
                                     float dp,
                                     int levels, int tMaxSize);
        int Guil_Full_findPosInHist_gpu(PtrStepSzi hist, float4* out, int3* votes, int curSize, int maxSize,
                                        float angle, int angleVotes, float scale, int scaleVotes,
                                        float dp, int threshold);
    }
}}}

// common

namespace
{
    class GeneralizedHoughBase
    {
    protected:
        GeneralizedHoughBase();
        virtual ~GeneralizedHoughBase() {}

        void setTemplateImpl(InputArray templ, Point templCenter);
        void setTemplateImpl(InputArray edges, InputArray dx, InputArray dy, Point templCenter);

        void detectImpl(InputArray image, OutputArray positions, OutputArray votes);
        void detectImpl(InputArray edges, InputArray dx, InputArray dy, OutputArray positions, OutputArray votes);

        void buildEdgePointList(const GpuMat& edges, const GpuMat& dx, const GpuMat& dy);

        virtual void processTempl() = 0;
        virtual void processImage() = 0;

        int cannyLowThresh_;
        int cannyHighThresh_;
        double minDist_;
        double dp_;
        int maxBufferSize_;

        Size templSize_;
        Point templCenter_;
        GpuMat templEdges_;
        GpuMat templDx_;
        GpuMat templDy_;

        Size imageSize_;
        GpuMat imageEdges_;
        GpuMat imageDx_;
        GpuMat imageDy_;

        GpuMat edgePointList_;

        GpuMat outBuf_;
        int posCount_;

    private:
#ifdef HAVE_OPENCV_CUDAFILTERS
        void calcEdges(InputArray src, GpuMat& edges, GpuMat& dx, GpuMat& dy);
#endif

        void filterMinDist();
        void convertTo(OutputArray positions, OutputArray votes);

#ifdef HAVE_OPENCV_CUDAFILTERS
        Ptr<cuda::CannyEdgeDetector> canny_;
        Ptr<cuda::Filter> filterDx_;
        Ptr<cuda::Filter> filterDy_;
#endif

        std::vector<float4> oldPosBuf_;
        std::vector<int3> oldVoteBuf_;
        std::vector<float4> newPosBuf_;
        std::vector<int3> newVoteBuf_;
        std::vector<int> indexies_;
    };

    GeneralizedHoughBase::GeneralizedHoughBase()
    {
        cannyLowThresh_ = 50;
        cannyHighThresh_ = 100;
        minDist_ = 1.0;
        dp_ = 1.0;

        maxBufferSize_ = 10000;

#ifdef HAVE_OPENCV_CUDAFILTERS
        canny_ = cuda::createCannyEdgeDetector(cannyLowThresh_, cannyHighThresh_);
        filterDx_ = cuda::createSobelFilter(CV_8UC1, CV_32S, 1, 0);
        filterDy_ = cuda::createSobelFilter(CV_8UC1, CV_32S, 0, 1);
#endif
    }

#ifdef HAVE_OPENCV_CUDAFILTERS
    void GeneralizedHoughBase::calcEdges(InputArray _src, GpuMat& edges, GpuMat& dx, GpuMat& dy)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.type() == CV_8UC1 );
        CV_Assert( cannyLowThresh_ > 0 && cannyLowThresh_ < cannyHighThresh_ );

        ensureSizeIsEnough(src.size(), CV_32SC1, dx);
        ensureSizeIsEnough(src.size(), CV_32SC1, dy);

        filterDx_->apply(src, dx);
        filterDy_->apply(src, dy);

        ensureSizeIsEnough(src.size(), CV_8UC1, edges);

        canny_->setLowThreshold(cannyLowThresh_);
        canny_->setHighThreshold(cannyHighThresh_);
        canny_->detect(dx, dy, edges);
    }
#endif

    void GeneralizedHoughBase::setTemplateImpl(InputArray templ, Point templCenter)
    {
#ifndef HAVE_OPENCV_CUDAFILTERS
        CV_UNUSED(templ);
        CV_UNUSED(templCenter);
        throw_no_cuda();
#else
        calcEdges(templ, templEdges_, templDx_, templDy_);

        if (templCenter == Point(-1, -1))
            templCenter = Point(templEdges_.cols / 2, templEdges_.rows / 2);

        templSize_ = templEdges_.size();
        templCenter_ = templCenter;

        processTempl();
#endif
    }

    void GeneralizedHoughBase::setTemplateImpl(InputArray edges, InputArray dx, InputArray dy, Point templCenter)
    {
        edges.getGpuMat().copyTo(templEdges_);
        dx.getGpuMat().copyTo(templDx_);
        dy.getGpuMat().copyTo(templDy_);

        CV_Assert( templEdges_.type() == CV_8UC1 );
        CV_Assert( templDx_.type() == CV_32FC1 && templDx_.size() == templEdges_.size() );
        CV_Assert( templDy_.type() == templDx_.type() && templDy_.size() == templEdges_.size() );

        if (templCenter == Point(-1, -1))
            templCenter = Point(templEdges_.cols / 2, templEdges_.rows / 2);

        templSize_ = templEdges_.size();
        templCenter_ = templCenter;

        processTempl();
    }

    void GeneralizedHoughBase::detectImpl(InputArray image, OutputArray positions, OutputArray votes)
    {
#ifndef HAVE_OPENCV_CUDAFILTERS
        CV_UNUSED(image);
        CV_UNUSED(positions);
        CV_UNUSED(votes);
        throw_no_cuda();
#else
        calcEdges(image, imageEdges_, imageDx_, imageDy_);

        imageSize_ = imageEdges_.size();

        posCount_ = 0;

        processImage();

        if (posCount_ == 0)
        {
            positions.release();
            if (votes.needed())
                votes.release();
        }
        else
        {
            if (minDist_ > 1)
                filterMinDist();
            convertTo(positions, votes);
        }
#endif
    }

    void GeneralizedHoughBase::detectImpl(InputArray edges, InputArray dx, InputArray dy, OutputArray positions, OutputArray votes)
    {
        edges.getGpuMat().copyTo(imageEdges_);
        dx.getGpuMat().copyTo(imageDx_);
        dy.getGpuMat().copyTo(imageDy_);

        CV_Assert( imageEdges_.type() == CV_8UC1 );
        CV_Assert( imageDx_.type() == CV_32FC1 && imageDx_.size() == imageEdges_.size() );
        CV_Assert( imageDy_.type() == imageDx_.type() && imageDy_.size() == imageEdges_.size() );

        imageSize_ = imageEdges_.size();

        posCount_ = 0;

        processImage();

        if (posCount_ == 0)
        {
            positions.release();
            if (votes.needed())
                votes.release();
        }
        else
        {
            if (minDist_ > 1)
                filterMinDist();
            convertTo(positions, votes);
        }
    }

    void GeneralizedHoughBase::buildEdgePointList(const GpuMat& edges, const GpuMat& dx, const GpuMat& dy)
    {
        using namespace cv::cuda::device::ght;

        typedef int (*func_t)(PtrStepSzb edges, PtrStepSzb dx, PtrStepSzb dy, unsigned int* coordList, float* thetaList);
        static const func_t funcs[] =
        {
            0,
            0,
            0,
            buildEdgePointList_gpu<short>,
            buildEdgePointList_gpu<int>,
            buildEdgePointList_gpu<float>,
            0
        };

        CV_Assert( edges.type() == CV_8UC1 );
        CV_Assert( dx.size() == edges.size() );
        CV_Assert( dy.type() == dx.type() && dy.size() == edges.size() );

        const func_t func = funcs[dx.depth()];
        CV_Assert( func != 0 );

        edgePointList_.cols = (int) (edgePointList_.step / sizeof(int));
        ensureSizeIsEnough(2, edges.size().area(), CV_32SC1, edgePointList_);

        edgePointList_.cols = func(edges, dx, dy, edgePointList_.ptr<unsigned int>(0), edgePointList_.ptr<float>(1));
    }

    struct IndexCmp
    {
        const int3* aux;

        explicit IndexCmp(const int3* _aux) : aux(_aux) {}

        bool operator ()(int l1, int l2) const
        {
            return aux[l1].x > aux[l2].x;
        }
    };

    void GeneralizedHoughBase::filterMinDist()
    {
        oldPosBuf_.resize(posCount_);
        oldVoteBuf_.resize(posCount_);

        cudaSafeCall( cudaMemcpy(&oldPosBuf_[0], outBuf_.ptr(0), posCount_ * sizeof(float4), cudaMemcpyDeviceToHost) );
        cudaSafeCall( cudaMemcpy(&oldVoteBuf_[0], outBuf_.ptr(1), posCount_ * sizeof(int3), cudaMemcpyDeviceToHost) );

        indexies_.resize(posCount_);
        for (int i = 0; i < posCount_; ++i)
            indexies_[i] = i;
        std::sort(indexies_.begin(), indexies_.end(), IndexCmp(&oldVoteBuf_[0]));

        newPosBuf_.clear();
        newVoteBuf_.clear();
        newPosBuf_.reserve(posCount_);
        newVoteBuf_.reserve(posCount_);

        const int cellSize = cvRound(minDist_);
        const int gridWidth = (imageSize_.width + cellSize - 1) / cellSize;
        const int gridHeight = (imageSize_.height + cellSize - 1) / cellSize;

        std::vector< std::vector<Point2f> > grid(gridWidth * gridHeight);

        const double minDist2 = minDist_ * minDist_;

        for (int i = 0; i < posCount_; ++i)
        {
            const int ind = indexies_[i];

            Point2f p(oldPosBuf_[ind].x, oldPosBuf_[ind].y);

            bool good = true;

            const int xCell = static_cast<int>(p.x / cellSize);
            const int yCell = static_cast<int>(p.y / cellSize);

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
                    const std::vector<Point2f>& m = grid[yy * gridWidth + xx];

                    for(size_t j = 0; j < m.size(); ++j)
                    {
                        const Point2f d = p - m[j];

                        if (d.ddot(d) < minDist2)
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

                newPosBuf_.push_back(oldPosBuf_[ind]);
                newVoteBuf_.push_back(oldVoteBuf_[ind]);
            }
        }

        posCount_ = static_cast<int>(newPosBuf_.size());
        cudaSafeCall( cudaMemcpy(outBuf_.ptr(0), &newPosBuf_[0], posCount_ * sizeof(float4), cudaMemcpyHostToDevice) );
        cudaSafeCall( cudaMemcpy(outBuf_.ptr(1), &newVoteBuf_[0], posCount_ * sizeof(int3), cudaMemcpyHostToDevice) );
    }

    void GeneralizedHoughBase::convertTo(OutputArray positions, OutputArray votes)
    {
        ensureSizeIsEnough(1, posCount_, CV_32FC4, positions);
        GpuMat(1, posCount_, CV_32FC4, outBuf_.ptr(0), outBuf_.step).copyTo(positions);

        if (votes.needed())
        {
            ensureSizeIsEnough(1, posCount_, CV_32FC3, votes);
            GpuMat(1, posCount_, CV_32FC4, outBuf_.ptr(1), outBuf_.step).copyTo(votes);
        }
    }
}

// GeneralizedHoughBallard

namespace
{
    class GeneralizedHoughBallardImpl : public GeneralizedHoughBallard, private GeneralizedHoughBase
    {
    public:
        GeneralizedHoughBallardImpl();

        void setTemplate(InputArray templ, Point templCenter) { setTemplateImpl(templ, templCenter); }
        void setTemplate(InputArray edges, InputArray dx, InputArray dy, Point templCenter) { setTemplateImpl(edges, dx, dy, templCenter); }

        void detect(InputArray image, OutputArray positions, OutputArray votes) { detectImpl(image, positions, votes); }
        void detect(InputArray edges, InputArray dx, InputArray dy, OutputArray positions, OutputArray votes) { detectImpl(edges, dx, dy, positions, votes); }

        void setCannyLowThresh(int cannyLowThresh) { cannyLowThresh_ = cannyLowThresh; }
        int getCannyLowThresh() const { return cannyLowThresh_; }

        void setCannyHighThresh(int cannyHighThresh) { cannyHighThresh_ = cannyHighThresh; }
        int getCannyHighThresh() const { return cannyHighThresh_; }

        void setMinDist(double minDist) { minDist_ = minDist; }
        double getMinDist() const { return minDist_; }

        void setDp(double dp) { dp_ = dp; }
        double getDp() const { return dp_; }

        void setMaxBufferSize(int maxBufferSize) { maxBufferSize_ = maxBufferSize; }
        int getMaxBufferSize() const { return maxBufferSize_; }

        void setLevels(int levels) { levels_ = levels; }
        int getLevels() const { return levels_; }

        void setVotesThreshold(int votesThreshold) { votesThreshold_ = votesThreshold; }
        int getVotesThreshold() const { return votesThreshold_; }

    private:
        void processTempl();
        void processImage();

        void calcHist();
        void findPosInHist();

        int levels_;
        int votesThreshold_;

        GpuMat r_table_;
        GpuMat r_sizes_;

        GpuMat hist_;
    };

    GeneralizedHoughBallardImpl::GeneralizedHoughBallardImpl()
    {
        levels_ = 360;
        votesThreshold_ = 100;
    }

    void GeneralizedHoughBallardImpl::processTempl()
    {
        using namespace cv::cuda::device::ght;

        CV_Assert( levels_ > 0 );

        buildEdgePointList(templEdges_, templDx_, templDy_);

        ensureSizeIsEnough(levels_ + 1, maxBufferSize_, CV_16SC2, r_table_);
        ensureSizeIsEnough(1, levels_ + 1, CV_32SC1, r_sizes_);
        r_sizes_.setTo(Scalar::all(0));

        if (edgePointList_.cols > 0)
        {
            buildRTable_gpu(edgePointList_.ptr<unsigned int>(0), edgePointList_.ptr<float>(1), edgePointList_.cols,
                            r_table_, r_sizes_.ptr<int>(), make_short2(templCenter_.x, templCenter_.y), levels_);
            cuda::min(r_sizes_, maxBufferSize_, r_sizes_);
        }
    }

    void GeneralizedHoughBallardImpl::processImage()
    {
        calcHist();
        findPosInHist();
    }

    void GeneralizedHoughBallardImpl::calcHist()
    {
        using namespace cv::cuda::device::ght;

        CV_Assert( levels_ > 0 && r_table_.rows == (levels_ + 1) && r_sizes_.cols == (levels_ + 1) );
        CV_Assert( dp_ > 0.0);

        const double idp = 1.0 / dp_;

        buildEdgePointList(imageEdges_, imageDx_, imageDy_);

        ensureSizeIsEnough(cvCeil(imageSize_.height * idp) + 2, cvCeil(imageSize_.width * idp) + 2, CV_32SC1, hist_);
        hist_.setTo(Scalar::all(0));

        if (edgePointList_.cols > 0)
        {
            Ballard_Pos_calcHist_gpu(edgePointList_.ptr<unsigned int>(0), edgePointList_.ptr<float>(1), edgePointList_.cols,
                                     r_table_, r_sizes_.ptr<int>(),
                                     hist_,
                                     (float)dp_, levels_);
        }
    }

    void GeneralizedHoughBallardImpl::findPosInHist()
    {
        using namespace cv::cuda::device::ght;

        CV_Assert( votesThreshold_ > 0 );

        ensureSizeIsEnough(2, maxBufferSize_, CV_32FC4, outBuf_);

        posCount_ = Ballard_Pos_findPosInHist_gpu(hist_, outBuf_.ptr<float4>(0), outBuf_.ptr<int3>(1), maxBufferSize_, (float)dp_, votesThreshold_);
    }
}

Ptr<GeneralizedHoughBallard> cv::cuda::createGeneralizedHoughBallard()
{
    return makePtr<GeneralizedHoughBallardImpl>();
}

// GeneralizedHoughGuil

namespace
{
    class GeneralizedHoughGuilImpl : public GeneralizedHoughGuil, private GeneralizedHoughBase
    {
    public:
        GeneralizedHoughGuilImpl();

        void setTemplate(InputArray templ, Point templCenter) { setTemplateImpl(templ, templCenter); }
        void setTemplate(InputArray edges, InputArray dx, InputArray dy, Point templCenter) { setTemplateImpl(edges, dx, dy, templCenter); }

        void detect(InputArray image, OutputArray positions, OutputArray votes) { detectImpl(image, positions, votes); }
        void detect(InputArray edges, InputArray dx, InputArray dy, OutputArray positions, OutputArray votes) { detectImpl(edges, dx, dy, positions, votes); }

        void setCannyLowThresh(int cannyLowThresh) { cannyLowThresh_ = cannyLowThresh; }
        int getCannyLowThresh() const { return cannyLowThresh_; }

        void setCannyHighThresh(int cannyHighThresh) { cannyHighThresh_ = cannyHighThresh; }
        int getCannyHighThresh() const { return cannyHighThresh_; }

        void setMinDist(double minDist) { minDist_ = minDist; }
        double getMinDist() const { return minDist_; }

        void setDp(double dp) { dp_ = dp; }
        double getDp() const { return dp_; }

        void setMaxBufferSize(int maxBufferSize) { maxBufferSize_ = maxBufferSize; }
        int getMaxBufferSize() const { return maxBufferSize_; }

        void setXi(double xi) { xi_ = xi; }
        double getXi() const { return xi_; }

        void setLevels(int levels) { levels_ = levels; }
        int getLevels() const { return levels_; }

        void setAngleEpsilon(double angleEpsilon) { angleEpsilon_ = angleEpsilon; }
        double getAngleEpsilon() const { return angleEpsilon_; }

        void setMinAngle(double minAngle) { minAngle_ = minAngle; }
        double getMinAngle() const { return minAngle_; }

        void setMaxAngle(double maxAngle) { maxAngle_ = maxAngle; }
        double getMaxAngle() const { return maxAngle_; }

        void setAngleStep(double angleStep) { angleStep_ = angleStep; }
        double getAngleStep() const { return angleStep_; }

        void setAngleThresh(int angleThresh) { angleThresh_ = angleThresh; }
        int getAngleThresh() const { return angleThresh_; }

        void setMinScale(double minScale) { minScale_ = minScale; }
        double getMinScale() const { return minScale_; }

        void setMaxScale(double maxScale) { maxScale_ = maxScale; }
        double getMaxScale() const { return maxScale_; }

        void setScaleStep(double scaleStep) { scaleStep_ = scaleStep; }
        double getScaleStep() const { return scaleStep_; }

        void setScaleThresh(int scaleThresh) { scaleThresh_ = scaleThresh; }
        int getScaleThresh() const { return scaleThresh_; }

        void setPosThresh(int posThresh) { posThresh_ = posThresh; }
        int getPosThresh() const { return posThresh_; }

    private:
        void processTempl();
        void processImage();

        double xi_;
        int levels_;
        double angleEpsilon_;

        double minAngle_;
        double maxAngle_;
        double angleStep_;
        int angleThresh_;

        double minScale_;
        double maxScale_;
        double scaleStep_;
        int scaleThresh_;

        int posThresh_;

        struct Feature
        {
            GpuMat p1_pos;
            GpuMat p1_theta;
            GpuMat p2_pos;

            GpuMat d12;

            GpuMat r1;
            GpuMat r2;

            GpuMat sizes;
            int maxSize;

            void create(int levels, int maxCapacity, bool isTempl);
        };

        typedef void (*set_func_t)(PtrStepb p1_pos, PtrStepb p1_theta, PtrStepb p2_pos, PtrStepb d12, PtrStepb r1, PtrStepb r2);
        typedef void (*build_func_t)(const unsigned int* coordList, const float* thetaList, int pointsCount,
                                     int* sizes, int maxSize,
                                     float xi, float angleEpsilon, int levels,
                                     float2 center, float maxDist);

        void buildFeatureList(const GpuMat& edges, const GpuMat& dx, const GpuMat& dy, Feature& features,
                              set_func_t set_func, build_func_t build_func, bool isTempl, Point2d center = Point2d());

        void calcOrientation();
        void calcScale(double angle);
        void calcPosition(double angle, int angleVotes, double scale, int scaleVotes);

        Feature templFeatures_;
        Feature imageFeatures_;

        std::vector< std::pair<double, int> > angles_;
        std::vector< std::pair<double, int> > scales_;

        GpuMat hist_;
        std::vector<int> h_buf_;
    };

    GeneralizedHoughGuilImpl::GeneralizedHoughGuilImpl()
    {
        maxBufferSize_ = 1000;

        xi_ = 90.0;
        levels_ = 360;
        angleEpsilon_ = 1.0;

        minAngle_ = 0.0;
        maxAngle_ = 360.0;
        angleStep_ = 1.0;
        angleThresh_ = 15000;

        minScale_ = 0.5;
        maxScale_ = 2.0;
        scaleStep_ = 0.05;
        scaleThresh_ = 1000;

        posThresh_ = 100;
    }

    void GeneralizedHoughGuilImpl::processTempl()
    {
        using namespace cv::cuda::device::ght;

        buildFeatureList(templEdges_, templDx_, templDy_, templFeatures_,
            Guil_Full_setTemplFeatures, Guil_Full_buildTemplFeatureList_gpu,
            true, templCenter_);

        h_buf_.resize(templFeatures_.sizes.cols);
        cudaSafeCall( cudaMemcpy(&h_buf_[0], templFeatures_.sizes.data, h_buf_.size() * sizeof(int), cudaMemcpyDeviceToHost) );
        templFeatures_.maxSize = *std::max_element(h_buf_.begin(), h_buf_.end());
    }

    void GeneralizedHoughGuilImpl::processImage()
    {
        using namespace cv::cuda::device::ght;

        CV_Assert( levels_ > 0 );
        CV_Assert( templFeatures_.sizes.cols == levels_ + 1 );
        CV_Assert( minAngle_ >= 0.0 && minAngle_ < maxAngle_ && maxAngle_ <= 360.0 );
        CV_Assert( angleStep_ > 0.0 && angleStep_ < 360.0 );
        CV_Assert( angleThresh_ > 0 );
        CV_Assert( minScale_ > 0.0 && minScale_ < maxScale_ );
        CV_Assert( scaleStep_ > 0.0 );
        CV_Assert( scaleThresh_ > 0 );
        CV_Assert( dp_ > 0.0 );
        CV_Assert( posThresh_ > 0 );

        const double iAngleStep = 1.0 / angleStep_;
        const int angleRange = cvCeil((maxAngle_ - minAngle_) * iAngleStep);

        const double iScaleStep = 1.0 / scaleStep_;
        const int scaleRange = cvCeil((maxScale_ - minScale_) * iScaleStep);

        const double idp = 1.0 / dp_;
        const int histRows = cvCeil(imageSize_.height * idp);
        const int histCols = cvCeil(imageSize_.width * idp);

        ensureSizeIsEnough(histRows + 2, std::max(angleRange + 1, std::max(scaleRange + 1, histCols + 2)), CV_32SC1, hist_);
        h_buf_.resize(std::max(angleRange + 1, scaleRange + 1));

        ensureSizeIsEnough(2, maxBufferSize_, CV_32FC4, outBuf_);

        buildFeatureList(imageEdges_, imageDx_, imageDy_, imageFeatures_,
            Guil_Full_setImageFeatures, Guil_Full_buildImageFeatureList_gpu,
            false);

        calcOrientation();

        for (size_t i = 0; i < angles_.size(); ++i)
        {
            const double angle = angles_[i].first;
            const int angleVotes = angles_[i].second;

            calcScale(angle);

            for (size_t j = 0; j < scales_.size(); ++j)
            {
                const double scale = scales_[j].first;
                const int scaleVotes = scales_[j].second;

                calcPosition(angle, angleVotes, scale, scaleVotes);
            }
        }
    }

    void GeneralizedHoughGuilImpl::Feature::create(int levels, int maxCapacity, bool isTempl)
    {
        if (!isTempl)
        {
            ensureSizeIsEnough(levels + 1, maxCapacity, CV_32FC2, p1_pos);
            ensureSizeIsEnough(levels + 1, maxCapacity, CV_32FC2, p2_pos);
        }

        ensureSizeIsEnough(levels + 1, maxCapacity, CV_32FC1, p1_theta);

        ensureSizeIsEnough(levels + 1, maxCapacity, CV_32FC1, d12);

        if (isTempl)
        {
            ensureSizeIsEnough(levels + 1, maxCapacity, CV_32FC2, r1);
            ensureSizeIsEnough(levels + 1, maxCapacity, CV_32FC2, r2);
        }

        ensureSizeIsEnough(1, levels + 1, CV_32SC1, sizes);
        sizes.setTo(Scalar::all(0));

        maxSize = 0;
    }

    void GeneralizedHoughGuilImpl::buildFeatureList(const GpuMat& edges, const GpuMat& dx, const GpuMat& dy, Feature& features,
                                                    set_func_t set_func, build_func_t build_func, bool isTempl, Point2d center)
    {
        CV_Assert( levels_ > 0 );

        const double maxDist = sqrt((double) templSize_.width * templSize_.width + templSize_.height * templSize_.height) * maxScale_;

        features.create(levels_, maxBufferSize_, isTempl);
        set_func(features.p1_pos, features.p1_theta, features.p2_pos, features.d12, features.r1, features.r2);

        buildEdgePointList(edges, dx, dy);

        if (edgePointList_.cols > 0)
        {
            build_func(edgePointList_.ptr<unsigned int>(0), edgePointList_.ptr<float>(1), edgePointList_.cols,
                features.sizes.ptr<int>(), maxBufferSize_, (float)xi_, (float)angleEpsilon_, levels_, make_float2((float)center.x, (float)center.y), (float)maxDist);
        }
    }

    void GeneralizedHoughGuilImpl::calcOrientation()
    {
        using namespace cv::cuda::device::ght;

        const double iAngleStep = 1.0 / angleStep_;
        const int angleRange = cvCeil((maxAngle_ - minAngle_) * iAngleStep);

        hist_.setTo(Scalar::all(0));
        Guil_Full_calcOHist_gpu(templFeatures_.sizes.ptr<int>(), imageFeatures_.sizes.ptr<int>(0), hist_.ptr<int>(),
                                (float)minAngle_, (float)maxAngle_, (float)angleStep_, angleRange, levels_, templFeatures_.maxSize);
        cudaSafeCall( cudaMemcpy(&h_buf_[0], hist_.data, h_buf_.size() * sizeof(int), cudaMemcpyDeviceToHost) );

        angles_.clear();

        for (int n = 0; n < angleRange; ++n)
        {
            if (h_buf_[n] >= angleThresh_)
            {
                const double angle = minAngle_ + n * angleStep_;
                angles_.push_back(std::make_pair(angle, h_buf_[n]));
            }
        }
    }

    void GeneralizedHoughGuilImpl::calcScale(double angle)
    {
        using namespace cv::cuda::device::ght;

        const double iScaleStep = 1.0 / scaleStep_;
        const int scaleRange = cvCeil((maxScale_ - minScale_) * iScaleStep);

        hist_.setTo(Scalar::all(0));
        Guil_Full_calcSHist_gpu(templFeatures_.sizes.ptr<int>(), imageFeatures_.sizes.ptr<int>(0), hist_.ptr<int>(),
                                (float)angle, (float)angleEpsilon_, (float)minScale_, (float)maxScale_,
                                (float)iScaleStep, scaleRange, levels_, templFeatures_.maxSize);
        cudaSafeCall( cudaMemcpy(&h_buf_[0], hist_.data, h_buf_.size() * sizeof(int), cudaMemcpyDeviceToHost) );

        scales_.clear();

        for (int s = 0; s < scaleRange; ++s)
        {
            if (h_buf_[s] >= scaleThresh_)
            {
                const double scale = minScale_ + s * scaleStep_;
                scales_.push_back(std::make_pair(scale, h_buf_[s]));
            }
        }
    }

    void GeneralizedHoughGuilImpl::calcPosition(double angle, int angleVotes, double scale, int scaleVotes)
    {
        using namespace cv::cuda::device::ght;

        hist_.setTo(Scalar::all(0));
        Guil_Full_calcPHist_gpu(templFeatures_.sizes.ptr<int>(), imageFeatures_.sizes.ptr<int>(0), hist_,
                                (float)angle, (float)angleEpsilon_, (float)scale, (float)dp_, levels_, templFeatures_.maxSize);

        posCount_ = Guil_Full_findPosInHist_gpu(hist_, outBuf_.ptr<float4>(0), outBuf_.ptr<int3>(1),
                                                posCount_, maxBufferSize_, (float)angle, angleVotes,
                                                (float)scale, scaleVotes, (float)dp_, posThresh_);
    }
}

Ptr<GeneralizedHoughGuil> cv::cuda::createGeneralizedHoughGuil()
{
    return makePtr<GeneralizedHoughGuilImpl>();
}

#endif /* !defined (HAVE_CUDA) */
