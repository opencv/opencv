/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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
#include "opencv2/core/hal/intrin.hpp"
#include <functional>
#include <limits>

using namespace cv;

// common

namespace
{
    double toRad(double a)
    {
        return a * CV_PI / 180.0;
    }

    bool notNull(float v)
    {
        return fabs(v) > std::numeric_limits<float>::epsilon();
    }

    class GeneralizedHoughBase
    {
    protected:
        GeneralizedHoughBase();
        virtual ~GeneralizedHoughBase() {}

        void setTemplateImpl(InputArray templ, Point templCenter);
        void setTemplateImpl(InputArray edges, InputArray dx, InputArray dy, Point templCenter);

        void detectImpl(InputArray image, OutputArray positions, OutputArray votes);
        void detectImpl(InputArray edges, InputArray dx, InputArray dy, OutputArray positions, OutputArray votes);

        virtual void processTempl() = 0;
        virtual void processImage() = 0;

        int cannyLowThresh_;
        int cannyHighThresh_;
        double minDist_;
        double dp_;

        Size templSize_;
        Point templCenter_;
        Mat templEdges_;
        Mat templDx_;
        Mat templDy_;

        Size imageSize_;
        Mat imageEdges_;
        Mat imageDx_;
        Mat imageDy_;

        std::vector<Vec4f> posOutBuf_;
        std::vector<Vec3i> voteOutBuf_;

    private:
        void calcEdges(InputArray src, Mat& edges, Mat& dx, Mat& dy);
        void filterMinDist();
        void convertTo(OutputArray positions, OutputArray votes);
    };

    GeneralizedHoughBase::GeneralizedHoughBase()
    {
        cannyLowThresh_ = 50;
        cannyHighThresh_ = 100;
        minDist_ = 1.0;
        dp_ = 1.0;
    }

    void GeneralizedHoughBase::calcEdges(InputArray _src, Mat& edges, Mat& dx, Mat& dy)
    {
        Mat src = _src.getMat();

        CV_Assert( src.type() == CV_8UC1 );
        CV_Assert( cannyLowThresh_ > 0 && cannyLowThresh_ < cannyHighThresh_ );

        Canny(src, edges, cannyLowThresh_, cannyHighThresh_);
        spatialGradient(src, dx, dy, 3, BORDER_DEFAULT, CV_32F);
    }

    void GeneralizedHoughBase::setTemplateImpl(InputArray templ, Point templCenter)
    {
        calcEdges(templ, templEdges_, templDx_, templDy_);

        if (templCenter == Point(-1, -1))
            templCenter = Point(templEdges_.cols / 2, templEdges_.rows / 2);

        templSize_ = templEdges_.size();
        templCenter_ = templCenter;

        processTempl();
    }

    void GeneralizedHoughBase::setTemplateImpl(InputArray edges, InputArray dx, InputArray dy, Point templCenter)
    {
        edges.getMat().copyTo(templEdges_);
        dx.getMat().copyTo(templDx_);
        dy.getMat().copyTo(templDy_);

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
        calcEdges(image, imageEdges_, imageDx_, imageDy_);

        imageSize_ = imageEdges_.size();

        posOutBuf_.clear();
        voteOutBuf_.clear();

        processImage();

        if (!posOutBuf_.empty())
        {
            if (minDist_ > 1)
                filterMinDist();
            convertTo(positions, votes);
        }
        else
        {
            positions.release();
            if (votes.needed())
                votes.release();
        }
    }

    void GeneralizedHoughBase::detectImpl(InputArray edges, InputArray dx, InputArray dy, OutputArray positions, OutputArray votes)
    {
        edges.getMat().copyTo(imageEdges_);
        dx.getMat().copyTo(imageDx_);
        dy.getMat().copyTo(imageDy_);

        CV_Assert( imageEdges_.type() == CV_8UC1 );
        CV_Assert( imageDx_.type() == CV_32FC1 && imageDx_.size() == imageEdges_.size() );
        CV_Assert( imageDy_.type() == imageDx_.type() && imageDy_.size() == imageEdges_.size() );

        imageSize_ = imageEdges_.size();

        posOutBuf_.clear();
        voteOutBuf_.clear();

        processImage();

        if (!posOutBuf_.empty())
        {
            if (minDist_ > 1)
                filterMinDist();
            convertTo(positions, votes);
        }
        else
        {
            positions.release();
            if (votes.needed())
                votes.release();
        }
    }

    class Vec3iGreaterThanIdx
    {
    public:
        Vec3iGreaterThanIdx( const Vec3i* _arr ) : arr(_arr) {}
        bool operator()(size_t a, size_t b) const { return arr[a][0] > arr[b][0]; }
        const Vec3i* arr;
    };

    void GeneralizedHoughBase::filterMinDist()
    {
        size_t oldSize = posOutBuf_.size();
        const bool hasVotes = !voteOutBuf_.empty();

        CV_Assert( !hasVotes || voteOutBuf_.size() == oldSize );

        std::vector<Vec4f> oldPosBuf(posOutBuf_);
        std::vector<Vec3i> oldVoteBuf(voteOutBuf_);

        std::vector<size_t> indexies(oldSize);
        for (size_t i = 0; i < oldSize; ++i)
            indexies[i] = i;
        std::sort(indexies.begin(), indexies.end(), Vec3iGreaterThanIdx(&oldVoteBuf[0]));

        posOutBuf_.clear();
        voteOutBuf_.clear();

        const int cellSize = cvRound(minDist_);
        const int gridWidth = (imageSize_.width + cellSize - 1) / cellSize;
        const int gridHeight = (imageSize_.height + cellSize - 1) / cellSize;

        std::vector< std::vector<Point2f> > grid(gridWidth * gridHeight);

        const double minDist2 = minDist_ * minDist_;

        for (size_t i = 0; i < oldSize; ++i)
        {
            const size_t ind = indexies[i];

            Point2f p(oldPosBuf[ind][0], oldPosBuf[ind][1]);

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

                posOutBuf_.push_back(oldPosBuf[ind]);
                if (hasVotes)
                    voteOutBuf_.push_back(oldVoteBuf[ind]);
            }
        }
    }

    void GeneralizedHoughBase::convertTo(OutputArray _positions, OutputArray _votes)
    {
        const int total = static_cast<int>(posOutBuf_.size());
        const bool hasVotes = !voteOutBuf_.empty();

        CV_Assert( !hasVotes || voteOutBuf_.size() == posOutBuf_.size() );

        _positions.create(1, total, CV_32FC4);
        Mat positions = _positions.getMat();
        Mat(1, total, CV_32FC4, &posOutBuf_[0]).copyTo(positions);

        if (_votes.needed())
        {
            if (!hasVotes)
            {
                _votes.release();
            }
            else
            {
                _votes.create(1, total, CV_32SC3);
                Mat votes = _votes.getMat();
                Mat(1, total, CV_32SC3, &voteOutBuf_[0]).copyTo(votes);
            }
        }
    }
}

// GeneralizedHoughBallard

namespace
{
    class GeneralizedHoughBallardImpl CV_FINAL : public GeneralizedHoughBallard, private GeneralizedHoughBase
    {
    public:
        GeneralizedHoughBallardImpl();

        void setTemplate(InputArray templ, Point templCenter) CV_OVERRIDE { setTemplateImpl(templ, templCenter); }
        void setTemplate(InputArray edges, InputArray dx, InputArray dy, Point templCenter) CV_OVERRIDE { setTemplateImpl(edges, dx, dy, templCenter); }

        void detect(InputArray image, OutputArray positions, OutputArray votes) CV_OVERRIDE { detectImpl(image, positions, votes); }
        void detect(InputArray edges, InputArray dx, InputArray dy, OutputArray positions, OutputArray votes) CV_OVERRIDE { detectImpl(edges, dx, dy, positions, votes); }

        void setCannyLowThresh(int cannyLowThresh) CV_OVERRIDE { cannyLowThresh_ = cannyLowThresh; }
        int getCannyLowThresh() const CV_OVERRIDE { return cannyLowThresh_; }

        void setCannyHighThresh(int cannyHighThresh) CV_OVERRIDE { cannyHighThresh_ = cannyHighThresh; }
        int getCannyHighThresh() const CV_OVERRIDE { return cannyHighThresh_; }

        void setMinDist(double minDist) CV_OVERRIDE { minDist_ = minDist; }
        double getMinDist() const CV_OVERRIDE { return minDist_; }

        void setDp(double dp) CV_OVERRIDE { dp_ = dp; }
        double getDp() const CV_OVERRIDE { return dp_; }

        void setMaxBufferSize(int) CV_OVERRIDE {  }
        int getMaxBufferSize() const CV_OVERRIDE { return 0; }

        void setLevels(int levels) CV_OVERRIDE { levels_ = levels; }
        int getLevels() const CV_OVERRIDE { return levels_; }

        void setVotesThreshold(int votesThreshold) CV_OVERRIDE { votesThreshold_ = votesThreshold; }
        int getVotesThreshold() const CV_OVERRIDE { return votesThreshold_; }

    private:
        void processTempl() CV_OVERRIDE;
        void processImage() CV_OVERRIDE;

        void calcHist();
        void findPosInHist();

        int levels_;
        int votesThreshold_;

        std::vector< std::vector<Point> > r_table_;
        Mat hist_;
    };

    GeneralizedHoughBallardImpl::GeneralizedHoughBallardImpl()
    {
        levels_ = 360;
        votesThreshold_ = 100;
    }

    void GeneralizedHoughBallardImpl::processTempl()
    {
        CV_Assert( levels_ > 0 );

        const double thetaScale = levels_ / 360.0;

        r_table_.resize(levels_ + 1);
        std::for_each(r_table_.begin(), r_table_.end(), [](std::vector<Point>& e)->void { e.clear(); });

        for (int y = 0; y < templSize_.height; ++y)
        {
            const uchar* edgesRow = templEdges_.ptr(y);
            const float* dxRow = templDx_.ptr<float>(y);
            const float* dyRow = templDy_.ptr<float>(y);

            for (int x = 0; x < templSize_.width; ++x)
            {
                const Point p(x, y);

                if (edgesRow[x] && (notNull(dyRow[x]) || notNull(dxRow[x])))
                {
                    const float theta = fastAtan2(dyRow[x], dxRow[x]);
                    const int n = cvRound(theta * thetaScale);
                    r_table_[n].push_back(p - templCenter_);
                }
            }
        }
    }

    void GeneralizedHoughBallardImpl::processImage()
    {
        calcHist();
        findPosInHist();
    }

    void GeneralizedHoughBallardImpl::calcHist()
    {
        CV_INSTRUMENT_REGION();

        CV_Assert( imageEdges_.type() == CV_8UC1 );
        CV_Assert( imageDx_.type() == CV_32FC1 && imageDx_.size() == imageSize_);
        CV_Assert( imageDy_.type() == imageDx_.type() && imageDy_.size() == imageSize_);
        CV_Assert( levels_ > 0 && r_table_.size() == static_cast<size_t>(levels_ + 1) );
        CV_Assert( dp_ > 0.0 );

        const double thetaScale = levels_ / 360.0;
        const double idp = 1.0 / dp_;

        hist_.create(cvCeil(imageSize_.height * idp) + 2, cvCeil(imageSize_.width * idp) + 2, CV_32SC1);
        hist_.setTo(0);

        const int rows = hist_.rows - 2;
        const int cols = hist_.cols - 2;

        for (int y = 0; y < imageSize_.height; ++y)
        {
            const uchar* edgesRow = imageEdges_.ptr(y);
            const float* dxRow = imageDx_.ptr<float>(y);
            const float* dyRow = imageDy_.ptr<float>(y);

            for (int x = 0; x < imageSize_.width; ++x)
            {
                const Point p(x, y);

                if (edgesRow[x] && (notNull(dyRow[x]) || notNull(dxRow[x])))
                {
                    const float theta = fastAtan2(dyRow[x], dxRow[x]);
                    const int n = cvRound(theta * thetaScale);

                    const std::vector<Point>& r_row = r_table_[n];

                    for (size_t j = 0; j < r_row.size(); ++j)
                    {
                        Point c = p - r_row[j];

                        c.x = cvRound(c.x * idp);
                        c.y = cvRound(c.y * idp);

                        if (c.x >= 0 && c.x < cols && c.y >= 0 && c.y < rows)
                            ++hist_.at<int>(c.y + 1, c.x + 1);
                    }
                }
            }
        }
    }

    void GeneralizedHoughBallardImpl::findPosInHist()
    {
        CV_Assert( votesThreshold_ > 0 );

        const int histRows = hist_.rows - 2;
        const int histCols = hist_.cols - 2;

        for(int y = 0; y < histRows; ++y)
        {
            const int* prevRow = hist_.ptr<int>(y);
            const int* curRow = hist_.ptr<int>(y + 1);
            const int* nextRow = hist_.ptr<int>(y + 2);

            for(int x = 0; x < histCols; ++x)
            {
                const int votes = curRow[x + 1];

                if (votes > votesThreshold_ && votes > curRow[x] && votes >= curRow[x + 2] && votes > prevRow[x + 1] && votes >= nextRow[x + 1])
                {
                    posOutBuf_.push_back(Vec4f(static_cast<float>(x * dp_), static_cast<float>(y * dp_), 1.0f, 0.0f));
                    voteOutBuf_.push_back(Vec3i(votes, 0, 0));
                }
            }
        }
    }
}

Ptr<GeneralizedHoughBallard> cv::createGeneralizedHoughBallard()
{
    return makePtr<GeneralizedHoughBallardImpl>();
}

// GeneralizedHoughGuil

namespace
{
    class GeneralizedHoughGuilImpl CV_FINAL : public GeneralizedHoughGuil, private GeneralizedHoughBase
    {
    public:
        GeneralizedHoughGuilImpl();

        void setTemplate(InputArray templ, Point templCenter) CV_OVERRIDE { setTemplateImpl(templ, templCenter); }
        void setTemplate(InputArray edges, InputArray dx, InputArray dy, Point templCenter) CV_OVERRIDE { setTemplateImpl(edges, dx, dy, templCenter); }

        void detect(InputArray image, OutputArray positions, OutputArray votes) CV_OVERRIDE { detectImpl(image, positions, votes); }
        void detect(InputArray edges, InputArray dx, InputArray dy, OutputArray positions, OutputArray votes) CV_OVERRIDE { detectImpl(edges, dx, dy, positions, votes); }

        void setCannyLowThresh(int cannyLowThresh) CV_OVERRIDE { cannyLowThresh_ = cannyLowThresh; }
        int getCannyLowThresh() const CV_OVERRIDE { return cannyLowThresh_; }

        void setCannyHighThresh(int cannyHighThresh) CV_OVERRIDE { cannyHighThresh_ = cannyHighThresh; }
        int getCannyHighThresh() const CV_OVERRIDE { return cannyHighThresh_; }

        void setMinDist(double minDist) CV_OVERRIDE { minDist_ = minDist; }
        double getMinDist() const CV_OVERRIDE { return minDist_; }

        void setDp(double dp) CV_OVERRIDE { dp_ = dp; }
        double getDp() const CV_OVERRIDE { return dp_; }

        void setMaxBufferSize(int maxBufferSize) CV_OVERRIDE { maxBufferSize_ = maxBufferSize; }
        int getMaxBufferSize() const CV_OVERRIDE { return maxBufferSize_; }

        void setXi(double xi) CV_OVERRIDE { xi_ = xi; }
        double getXi() const CV_OVERRIDE { return xi_; }

        void setLevels(int levels) CV_OVERRIDE { levels_ = levels; }
        int getLevels() const CV_OVERRIDE { return levels_; }

        void setAngleEpsilon(double angleEpsilon) CV_OVERRIDE { angleEpsilon_ = angleEpsilon; }
        double getAngleEpsilon() const CV_OVERRIDE { return angleEpsilon_; }

        void setMinAngle(double minAngle) CV_OVERRIDE { minAngle_ = minAngle; }
        double getMinAngle() const CV_OVERRIDE { return minAngle_; }

        void setMaxAngle(double maxAngle) CV_OVERRIDE { maxAngle_ = maxAngle; }
        double getMaxAngle() const CV_OVERRIDE { return maxAngle_; }

        void setAngleStep(double angleStep) CV_OVERRIDE { angleStep_ = angleStep; }
        double getAngleStep() const CV_OVERRIDE { return angleStep_; }

        void setAngleThresh(int angleThresh) CV_OVERRIDE { angleThresh_ = angleThresh; }
        int getAngleThresh() const CV_OVERRIDE { return angleThresh_; }

        void setMinScale(double minScale) CV_OVERRIDE { minScale_ = minScale; }
        double getMinScale() const CV_OVERRIDE { return minScale_; }

        void setMaxScale(double maxScale) CV_OVERRIDE { maxScale_ = maxScale; }
        double getMaxScale() const CV_OVERRIDE { return maxScale_; }

        void setScaleStep(double scaleStep) CV_OVERRIDE { scaleStep_ = scaleStep; }
        double getScaleStep() const CV_OVERRIDE { return scaleStep_; }

        void setScaleThresh(int scaleThresh) CV_OVERRIDE { scaleThresh_ = scaleThresh; }
        int getScaleThresh() const CV_OVERRIDE { return scaleThresh_; }

        void setPosThresh(int posThresh) CV_OVERRIDE { posThresh_ = posThresh; }
        int getPosThresh() const CV_OVERRIDE { return posThresh_; }

    private:
        void processTempl() CV_OVERRIDE;
        void processImage() CV_OVERRIDE;

        int maxBufferSize_;
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

        struct ContourPoint
        {
            Point2d pos;
            double theta;
        };

        struct Feature
        {
            ContourPoint p1;
            ContourPoint p2;

            double alpha12;
            double d12;

            Point2d r1;
            Point2d r2;
        };

        void buildFeatureList(const Mat& edges, const Mat& dx, const Mat& dy, std::vector< std::vector<Feature> >& features, Point2d center = Point2d());
        void getContourPoints(const Mat& edges, const Mat& dx, const Mat& dy, std::vector<ContourPoint>& points);

        void calcOrientation();
        void calcScale(double angle);
        void calcPosition(double angle, int angleVotes, double scale, int scaleVotes);

        void prepareVoting();

        std::vector< std::vector<Feature> > templFeatures_;
        std::vector< std::vector<Feature> > imageFeatures_;

        std::vector<size_t> imageLevelOfs_;
        std::vector<double> imageTheta_;
        std::vector<double> imageD12_;
        std::vector<double> imageP1x_, imageP1y_, imageP2x_, imageP2y_;
        bool useParallel_;

        std::vector< std::pair<double, int> > angles_;
        std::vector< std::pair<double, int> > scales_;
    };

    double clampAngle(double a)
    {
        double res = a;

        while (res > 360.0)
            res -= 360.0;
        while (res < 0)
            res += 360.0;

        return res;
    }

    bool angleEq(double a, double b, double eps = 1.0)
    {
        return (fabs(clampAngle(a - b)) <= eps);
    }

#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
    inline v_float64 clampAngle(v_float64 r)
    {
        const v_float64 v360 = vx_setall_f64(360.0);
        const v_float64 vZero = vx_setzero_f64();
        v_float64 m = v_gt(r, v360);
        while (v_check_any(m))
        {
            r = v_select(m, v_sub(r, v360), r);
            m = v_gt(r, v360);
        }
        m = v_lt(r, vZero);
        while (v_check_any(m))
        {
            r = v_select(m, v_add(r, v360), r);
            m = v_lt(r, vZero);
        }
        return r;
    }

    inline v_float64 angleEq(const v_float64& a, const v_float64& b, const v_float64& eps)
    {
        return v_le(v_abs(clampAngle(v_sub(a, b))), eps);
    }
#endif

    void voteOrientation(const double* theta, size_t n, double tTheta,
                         double minAngle, double maxAngle, double iAngleStep, int* hist)
    {
        size_t k = 0;
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
        const int vlanes = VTraits<v_float64>::vlanes();
        uint64 maskBuf[VTraits<v_float64>::max_nlanes];
        int idxBuf[VTraits<v_int32>::max_nlanes];
        const v_float64 vTheta = vx_setall_f64(tTheta);
        const v_float64 vMin = vx_setall_f64(minAngle);
        const v_float64 vMax = vx_setall_f64(maxAngle);
        const v_float64 vInvStep = vx_setall_f64(iAngleStep);
        for (; k + vlanes <= n; k += vlanes)
        {
            const v_float64 a = clampAngle(v_sub(vx_load(theta + k), vTheta));
            const v_float64 m = v_and(v_ge(a, vMin), v_le(a, vMax));
            if (!v_check_any(m))
                continue;
            v_store(idxBuf, v_round(v_mul(v_sub(a, vMin), vInvStep)));
            v_store(maskBuf, v_reinterpret_as_u64(m));
            for (int l = 0; l < vlanes; ++l)
                if (maskBuf[l])
                    ++hist[idxBuf[l]];
        }
        vx_cleanup();
#endif
        for (; k < n; ++k)
        {
            const double angle = clampAngle(theta[k] - tTheta);
            if (angle >= minAngle && angle <= maxAngle)
            {
                const int b = cvRound((angle - minAngle) * iAngleStep);
                ++hist[b];
            }
        }
    }

    void voteScale(const double* theta, const double* d12, size_t n, double tTheta, double tD12,
                   double eps, double minScale, double maxScale, double iScaleStep, int* hist)
    {
        size_t k = 0;
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
        const int vlanes = VTraits<v_float64>::vlanes();
        uint64 maskBuf[VTraits<v_float64>::max_nlanes];
        int idxBuf[VTraits<v_int32>::max_nlanes];
        const v_float64 vTheta = vx_setall_f64(tTheta);
        const v_float64 vEps = vx_setall_f64(eps);
        const v_float64 vD12 = vx_setall_f64(tD12);
        const v_float64 vMin = vx_setall_f64(minScale);
        const v_float64 vMax = vx_setall_f64(maxScale);
        const v_float64 vInvStep = vx_setall_f64(iScaleStep);
        for (; k + 2 * vlanes <= n; k += 2 * vlanes)
        {
            const v_float64 e0 = angleEq(vx_load(theta + k), vTheta, vEps);
            const v_float64 e1 = angleEq(vx_load(theta + k + vlanes), vTheta, vEps);
            if (!v_check_any(v_or(e0, e1)))
                continue;
            for (int h = 0; h < 2; ++h)
            {
                const size_t kk = k + h * vlanes;
                const v_float64 s = v_div(vx_load(d12 + kk), vD12);
                const v_float64 m = v_and(h == 0 ? e0 : e1, v_and(v_ge(s, vMin), v_le(s, vMax)));
                if (!v_check_any(m))
                    continue;
                v_store(idxBuf, v_round(v_mul(v_sub(s, vMin), vInvStep)));
                v_store(maskBuf, v_reinterpret_as_u64(m));
                for (int l = 0; l < vlanes; ++l)
                    if (maskBuf[l])
                        ++hist[idxBuf[l]];
            }
        }
        vx_cleanup();
#endif
        for (; k < n; ++k)
        {
            if (angleEq(theta[k], tTheta, eps))
            {
                const double scale = d12[k] / tD12;
                if (scale >= minScale && scale <= maxScale)
                {
                    const int s = cvRound((scale - minScale) * iScaleStep);
                    ++hist[s];
                }
            }
        }
    }

    struct PositionVoteParams
    {
        double tTheta, eps, idp;
        Point2d r1, r2;
        int histRows, histCols;
        int* hist;
        size_t histStep;
    };

    template <bool Atomic>
    inline void addPositionVote(const PositionVoteParams& params, double c1x, double c1y)
    {
        int* cell = params.hist + (size_t)(cvRound(c1y) + 1) * params.histStep + (cvRound(c1x) + 1);
        if (Atomic)
            CV_XADD(cell, 1);
        else
            ++*cell;
    }

    template <bool Atomic>
    void votePosition(const double* theta, const double* p1x, const double* p1y,
                      const double* p2x, const double* p2y, size_t n, const PositionVoteParams& params)
    {
        size_t k = 0;
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
        const int vlanes = VTraits<v_float64>::vlanes();
        uint64 maskBuf[VTraits<v_float64>::max_nlanes];
        double xBuf[VTraits<v_float64>::max_nlanes], yBuf[VTraits<v_float64>::max_nlanes];
        const v_float64 vTheta = vx_setall_f64(params.tTheta);
        const v_float64 vEps = vx_setall_f64(params.eps);
        const v_float64 vIdp = vx_setall_f64(params.idp);
        const v_float64 vR1X = vx_setall_f64(params.r1.x), vR1Y = vx_setall_f64(params.r1.y);
        const v_float64 vR2X = vx_setall_f64(params.r2.x), vR2Y = vx_setall_f64(params.r2.y);
        const v_float64 vOne = vx_setall_f64(1.0), vZero = vx_setzero_f64();
        const v_float64 vRows = vx_setall_f64((double)params.histRows), vCols = vx_setall_f64((double)params.histCols);
        for (; k + 2 * vlanes <= n; k += 2 * vlanes)
        {
            const v_float64 e0 = angleEq(vx_load(theta + k), vTheta, vEps);
            const v_float64 e1 = angleEq(vx_load(theta + k + vlanes), vTheta, vEps);
            if (!v_check_any(v_or(e0, e1)))
                continue;
            for (int h = 0; h < 2; ++h)
            {
                const v_float64 e = h == 0 ? e0 : e1;
                if (!v_check_any(e))
                    continue;
                const size_t kk = k + h * vlanes;
                const v_float64 c1x = v_mul(v_sub(vx_load(p1x + kk), vR1X), vIdp);
                const v_float64 c1y = v_mul(v_sub(vx_load(p1y + kk), vR1Y), vIdp);
                const v_float64 c2x = v_mul(v_sub(vx_load(p2x + kk), vR2X), vIdp);
                const v_float64 c2y = v_mul(v_sub(vx_load(p2y + kk), vR2Y), vIdp);
                const v_float64 tooFar = v_or(v_gt(v_abs(v_sub(c1x, c2x)), vOne),
                                              v_gt(v_abs(v_sub(c1y, c2y)), vOne));
                const v_float64 inBounds = v_and(v_and(v_ge(c1y, vZero), v_lt(c1y, vRows)),
                                            v_and(v_ge(c1x, vZero), v_lt(c1x, vCols)));
                const v_float64 m = v_and(v_and(e, v_not(tooFar)), inBounds);
                if (!v_check_any(m))
                    continue;
                v_store(maskBuf, v_reinterpret_as_u64(m));
                v_store(xBuf, c1x);
                v_store(yBuf, c1y);
                for (int l = 0; l < vlanes; ++l)
                    if (maskBuf[l])
                        addPositionVote<Atomic>(params, xBuf[l], yBuf[l]);
            }
        }
        vx_cleanup();
#endif
        for (; k < n; ++k)
        {
            if (angleEq(theta[k], params.tTheta, params.eps))
            {
                Point2d c1(p1x[k], p1y[k]), c2(p2x[k], p2y[k]);

                c1 = c1 - params.r1;
                c1 *= params.idp;

                c2 = c2 - params.r2;
                c2 *= params.idp;

                if (fabs(c1.x - c2.x) > 1 || fabs(c1.y - c2.y) > 1)
                    continue;

                if (c1.y >= 0 && c1.y < params.histRows && c1.x >= 0 && c1.x < params.histCols)
                    addPositionVote<Atomic>(params, c1.x, c1.y);
            }
        }
    }

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
        buildFeatureList(templEdges_, templDx_, templDy_, templFeatures_, templCenter_);
    }

    void GeneralizedHoughGuilImpl::processImage()
    {
        buildFeatureList(imageEdges_, imageDx_, imageDy_, imageFeatures_);

        prepareVoting();

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

    void GeneralizedHoughGuilImpl::buildFeatureList(const Mat& edges, const Mat& dx, const Mat& dy, std::vector< std::vector<Feature> >& features, Point2d center)
    {
        CV_Assert( levels_ > 0 );

        const double maxDist = sqrt((double) templSize_.width * templSize_.width + templSize_.height * templSize_.height) * maxScale_;

        const double alphaScale = levels_ / 360.0;

        std::vector<ContourPoint> points;
        getContourPoints(edges, dx, dy, points);

        features.resize(levels_ + 1);
        const size_t maxBufferSize = maxBufferSize_;
        std::for_each(features.begin(), features.end(), [maxBufferSize](std::vector<Feature>& e) {
            e.clear();
            e.reserve(maxBufferSize);
        });

        auto addPair = [&](const ContourPoint& p1, const ContourPoint& p2)
        {
            const Point2d d = p1.pos - p2.pos;

            Feature f;

            f.p1 = p1;
            f.p2 = p2;

            f.alpha12 = clampAngle(fastAtan2((float)d.y, (float)d.x) - p1.theta);
            f.d12 = norm(d);

            if (f.d12 > maxDist)
                return;

            f.r1 = p1.pos - center;
            f.r2 = p2.pos - center;

            const int n = cvRound(f.alpha12 * alphaScale);

            if (features[n].size() < static_cast<size_t>(maxBufferSize_))
                features[n].push_back(f);
        };

        const size_t npoints = points.size();
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
        std::vector<double> thetas(npoints);
        for (size_t i = 0; i < npoints; ++i)
            thetas[i] = points[i].theta;
        const int vlanes = VTraits<v_float64>::vlanes();
        uint64 maskBuf[2 * VTraits<v_float64>::max_nlanes];
        const v_float64 vXi = vx_setall_f64(xi_);
        const v_float64 vEps = vx_setall_f64(angleEpsilon_);
#endif

        for (size_t i = 0; i < npoints; ++i)
        {
            const ContourPoint p1 = points[i];
            size_t j = 0;

#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
            const v_float64 vTheta1 = vx_setall_f64(p1.theta);
            for (; j + 2 * vlanes <= npoints; j += 2 * vlanes)
            {
                const v_float64 e0 = angleEq(v_sub(vTheta1, vx_load(&thetas[j])), vXi, vEps);
                const v_float64 e1 = angleEq(v_sub(vTheta1, vx_load(&thetas[j + vlanes])), vXi, vEps);
                if (!v_check_any(v_or(e0, e1)))
                    continue;
                v_store(maskBuf, v_reinterpret_as_u64(e0));
                v_store(maskBuf + vlanes, v_reinterpret_as_u64(e1));
                for (int l = 0; l < 2 * vlanes; ++l)
                    if (maskBuf[l])
                        addPair(p1, points[j + l]);
            }
#endif
            for (; j < npoints; ++j)
            {
                const ContourPoint& p2 = points[j];

                if (angleEq(p1.theta - p2.theta, xi_, angleEpsilon_))
                    addPair(p1, p2);
            }
        }
    }

    void GeneralizedHoughGuilImpl::getContourPoints(const Mat& edges, const Mat& dx, const Mat& dy, std::vector<ContourPoint>& points)
    {
        CV_Assert( edges.type() == CV_8UC1 );
        CV_Assert( dx.type() == CV_32FC1 && dx.size == edges.size );
        CV_Assert( dy.type() == dx.type() && dy.size == edges.size );

        points.clear();
        points.reserve(edges.size().area());

        for (int y = 0; y < edges.rows; ++y)
        {
            const uchar* edgesRow = edges.ptr(y);
            const float* dxRow = dx.ptr<float>(y);
            const float* dyRow = dy.ptr<float>(y);

            for (int x = 0; x < edges.cols; ++x)
            {
                if (edgesRow[x] && (notNull(dyRow[x]) || notNull(dxRow[x])))
                {
                    ContourPoint p;

                    p.pos = Point2d(x, y);
                    p.theta = fastAtan2(dyRow[x], dxRow[x]);

                    points.push_back(p);
                }
            }
        }
    }

    void GeneralizedHoughGuilImpl::prepareVoting()
    {
        CV_Assert( templFeatures_.size() == static_cast<size_t>(levels_ + 1) );
        CV_Assert( imageFeatures_.size() == templFeatures_.size() );

        const int nlevels = levels_ + 1;
        uint64 totalWork = 0;
        imageLevelOfs_.assign(nlevels + 1, 0);
        for (int i = 0; i < nlevels; ++i)
        {
            imageLevelOfs_[i + 1] = imageLevelOfs_[i] + imageFeatures_[i].size();
            totalWork += (uint64)templFeatures_[i].size() * imageFeatures_[i].size();
        }

        const size_t total = imageLevelOfs_[nlevels];
        imageTheta_.resize(total);
        imageD12_.resize(total);
        imageP1x_.resize(total);
        imageP1y_.resize(total);
        imageP2x_.resize(total);
        imageP2y_.resize(total);
        for (int i = 0; i < nlevels; ++i)
        {
            for (size_t k = 0, idx = imageLevelOfs_[i]; k < imageFeatures_[i].size(); ++k, ++idx)
            {
                const Feature& f = imageFeatures_[i][k];
                imageTheta_[idx] = f.p1.theta;
                imageD12_[idx] = f.d12;
                imageP1x_[idx] = f.p1.pos.x;
                imageP1y_[idx] = f.p1.pos.y;
                imageP2x_[idx] = f.p2.pos.x;
                imageP2y_[idx] = f.p2.pos.y;
            }
        }

        useParallel_ = getNumThreads() > 1 && totalWork >= 8192;
    }

    void GeneralizedHoughGuilImpl::calcOrientation()
    {
        CV_Assert( levels_ > 0 );
        CV_Assert( templFeatures_.size() == static_cast<size_t>(levels_ + 1) );
        CV_Assert( imageFeatures_.size() == templFeatures_.size() );
        CV_Assert( minAngle_ >= 0.0 && minAngle_ < maxAngle_ && maxAngle_ <= 360.0 );
        CV_Assert( angleStep_ > 0.0 && angleStep_ < 360.0 );
        CV_Assert( angleThresh_ > 0 );

        const double iAngleStep = 1.0 / angleStep_;
        const int angleRange = cvCeil((maxAngle_ - minAngle_) * iAngleStep);

        std::vector<int> OHist(angleRange + 1, 0);

        auto voteLevels = [&](const Range& r, int* hist)
        {
            for (int i = r.start; i < r.end; ++i)
            {
                const size_t ofs = imageLevelOfs_[i], n = imageLevelOfs_[i + 1] - ofs;
                for (size_t j = 0; j < templFeatures_[i].size(); ++j)
                    voteOrientation(imageTheta_.data() + ofs, n, templFeatures_[i][j].p1.theta,
                                    minAngle_, maxAngle_, iAngleStep, hist);
            }
        };

        if (useParallel_)
        {
            parallel_for_(Range(0, levels_ + 1), [&](const Range& r)
            {
                std::vector<int> localHist(OHist.size(), 0);
                voteLevels(r, localHist.data());
                for (size_t b = 0; b < localHist.size(); ++b)
                    if (localHist[b])
                        CV_XADD(&OHist[b], localHist[b]);
            });
        }
        else
            voteLevels(Range(0, levels_ + 1), OHist.data());

        angles_.clear();

        for (int n = 0; n < angleRange; ++n)
        {
            if (OHist[n] >= angleThresh_)
            {
                const double angle = minAngle_ + n * angleStep_;
                angles_.push_back(std::make_pair(angle, OHist[n]));
            }
        }
    }

    void GeneralizedHoughGuilImpl::calcScale(double angle)
    {
        CV_Assert( levels_ > 0 );
        CV_Assert( templFeatures_.size() == static_cast<size_t>(levels_ + 1) );
        CV_Assert( imageFeatures_.size() == templFeatures_.size() );
        CV_Assert( minScale_ > 0.0 && minScale_ < maxScale_ );
        CV_Assert( scaleStep_ > 0.0 );
        CV_Assert( scaleThresh_ > 0 );

        const double iScaleStep = 1.0 / scaleStep_;
        const int scaleRange = cvCeil((maxScale_ - minScale_) * iScaleStep);

        std::vector<int> SHist(scaleRange + 1, 0);

        auto voteLevels = [&](const Range& r, int* hist)
        {
            for (int i = r.start; i < r.end; ++i)
            {
                const size_t ofs = imageLevelOfs_[i], n = imageLevelOfs_[i + 1] - ofs;
                for (size_t j = 0; j < templFeatures_[i].size(); ++j)
                {
                    const Feature& templF = templFeatures_[i][j];
                    voteScale(imageTheta_.data() + ofs, imageD12_.data() + ofs, n, templF.p1.theta + angle, templF.d12,
                              angleEpsilon_, minScale_, maxScale_, iScaleStep, hist);
                }
            }
        };

        if (useParallel_)
        {
            parallel_for_(Range(0, levels_ + 1), [&](const Range& r)
            {
                std::vector<int> localHist(SHist.size(), 0);
                voteLevels(r, localHist.data());
                for (size_t b = 0; b < localHist.size(); ++b)
                    if (localHist[b])
                        CV_XADD(&SHist[b], localHist[b]);
            });
        }
        else
            voteLevels(Range(0, levels_ + 1), SHist.data());

        scales_.clear();

        for (int s = 0; s < scaleRange; ++s)
        {
            if (SHist[s] >= scaleThresh_)
            {
                const double scale = minScale_ + s * scaleStep_;
                scales_.push_back(std::make_pair(scale, SHist[s]));
            }
        }
    }

    void GeneralizedHoughGuilImpl::calcPosition(double angle, int angleVotes, double scale, int scaleVotes)
    {
        CV_Assert( levels_ > 0 );
        CV_Assert( templFeatures_.size() == static_cast<size_t>(levels_ + 1) );
        CV_Assert( imageFeatures_.size() == templFeatures_.size() );
        CV_Assert( dp_ > 0.0 );
        CV_Assert( posThresh_ > 0 );

        const double sinVal = sin(toRad(angle));
        const double cosVal = cos(toRad(angle));
        const double idp = 1.0 / dp_;

        const int histRows = cvCeil(imageSize_.height * idp);
        const int histCols = cvCeil(imageSize_.width * idp);

        Mat DHist(histRows + 2, histCols + 2, CV_32SC1, Scalar::all(0));

        auto voteLevels = [&](const Range& r, bool atomic)
        {
            PositionVoteParams params;
            params.eps = angleEpsilon_;
            params.idp = idp;
            params.histRows = histRows;
            params.histCols = histCols;
            params.hist = DHist.ptr<int>(0);
            params.histStep = DHist.step1();

            for (int i = r.start; i < r.end; ++i)
            {
                const size_t ofs = imageLevelOfs_[i], n = imageLevelOfs_[i + 1] - ofs;
                for (size_t j = 0; j < templFeatures_[i].size(); ++j)
                {
                    Feature templF = templFeatures_[i][j];

                    templF.p1.theta += angle;

                    templF.r1 *= scale;
                    templF.r2 *= scale;

                    templF.r1 = Point2d(cosVal * templF.r1.x - sinVal * templF.r1.y, sinVal * templF.r1.x + cosVal * templF.r1.y);
                    templF.r2 = Point2d(cosVal * templF.r2.x - sinVal * templF.r2.y, sinVal * templF.r2.x + cosVal * templF.r2.y);

                    params.tTheta = templF.p1.theta;
                    params.r1 = templF.r1;
                    params.r2 = templF.r2;

                    (atomic ? votePosition<true> : votePosition<false>)(
                        imageTheta_.data() + ofs, imageP1x_.data() + ofs, imageP1y_.data() + ofs,
                        imageP2x_.data() + ofs, imageP2y_.data() + ofs, n, params);
                }
            }
        };

        if (useParallel_)
            parallel_for_(Range(0, levels_ + 1), [&](const Range& r) { voteLevels(r, true); });
        else
            voteLevels(Range(0, levels_ + 1), false);

        for(int y = 0; y < histRows; ++y)
        {
            const int* prevRow = DHist.ptr<int>(y);
            const int* curRow = DHist.ptr<int>(y + 1);
            const int* nextRow = DHist.ptr<int>(y + 2);

            for(int x = 0; x < histCols; ++x)
            {
                const int votes = curRow[x + 1];

                if (votes > posThresh_ && votes > curRow[x] && votes >= curRow[x + 2] && votes > prevRow[x + 1] && votes >= nextRow[x + 1])
                {
                    posOutBuf_.push_back(Vec4f(static_cast<float>(x * dp_), static_cast<float>(y * dp_), static_cast<float>(scale), static_cast<float>(angle)));
                    voteOutBuf_.push_back(Vec3i(votes, scaleVotes, angleVotes));
                }
            }
        }
    }
}

Ptr<GeneralizedHoughGuil> cv::createGeneralizedHoughGuil()
{
    return makePtr<GeneralizedHoughGuilImpl>();
}
