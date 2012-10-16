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
#include <functional>

using namespace std;
using namespace cv;

namespace
{
    /////////////////////////////////////
    // Common

    template <typename T, class A> void releaseVector(vector<T, A>& v)
    {
        vector<T, A> empty;
        empty.swap(v);
    }

    double toRad(double a)
    {
        return a * CV_PI / 180.0;
    }

    bool notNull(float v)
    {
        return fabs(v) > numeric_limits<float>::epsilon();
    }

    class GHT_Pos : public GeneralizedHough
    {
    public:
        GHT_Pos();

    protected:
        void setTemplateImpl(const Mat& edges, const Mat& dx, const Mat& dy, Point templCenter);
        void detectImpl(const Mat& edges, const Mat& dx, const Mat& dy, OutputArray positions, OutputArray votes);
        void releaseImpl();

        virtual void processTempl() = 0;
        virtual void processImage() = 0;

        void filterMinDist();
        void convertTo(OutputArray positions, OutputArray votes);

        double minDist;

        Size templSize;
        Point templCenter;
        Mat templEdges;
        Mat templDx;
        Mat templDy;

        Size imageSize;
        Mat imageEdges;
        Mat imageDx;
        Mat imageDy;

        vector<Vec4f> posOutBuf;
        vector<Vec3i> voteOutBuf;
    };

    GHT_Pos::GHT_Pos()
    {
        minDist = 1.0;
    }

    void GHT_Pos::setTemplateImpl(const Mat& edges, const Mat& dx, const Mat& dy, Point templCenter_)
    {
        templSize = edges.size();
        templCenter = templCenter_;
        edges.copyTo(templEdges);
        dx.copyTo(templDx);
        dy.copyTo(templDy);

        processTempl();
    }

    void GHT_Pos::detectImpl(const Mat& edges, const Mat& dx, const Mat& dy, OutputArray positions, OutputArray votes)
    {
        imageSize = edges.size();
        edges.copyTo(imageEdges);
        dx.copyTo(imageDx);
        dy.copyTo(imageDy);

        posOutBuf.clear();
        voteOutBuf.clear();

        processImage();

        if (!posOutBuf.empty())
        {
            if (minDist > 1)
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

    void GHT_Pos::releaseImpl()
    {
        templSize = Size();
        templCenter = Point(-1, -1);
        templEdges.release();
        templDx.release();
        templDy.release();

        imageSize = Size();
        imageEdges.release();
        imageDx.release();
        imageDy.release();

        releaseVector(posOutBuf);
        releaseVector(voteOutBuf);
    }

    #define votes_cmp_gt(l1, l2) (aux[l1][0] > aux[l2][0])
    static CV_IMPLEMENT_QSORT_EX( sortIndexies, size_t, votes_cmp_gt, const Vec3i* )

    void GHT_Pos::filterMinDist()
    {
        size_t oldSize = posOutBuf.size();
        const bool hasVotes = !voteOutBuf.empty();

        CV_Assert(!hasVotes || voteOutBuf.size() == oldSize);

        vector<Vec4f> oldPosBuf(posOutBuf);
        vector<Vec3i> oldVoteBuf(voteOutBuf);

        vector<size_t> indexies(oldSize);
        for (size_t i = 0; i < oldSize; ++i)
            indexies[i] = i;
        sortIndexies(&indexies[0], oldSize, &oldVoteBuf[0]);

        posOutBuf.clear();
        voteOutBuf.clear();

        const int cellSize = cvRound(minDist);
        const int gridWidth = (imageSize.width + cellSize - 1) / cellSize;
        const int gridHeight = (imageSize.height + cellSize - 1) / cellSize;

        vector< vector<Point2f> > grid(gridWidth * gridHeight);

        const double minDist2 = minDist * minDist;

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
                    const vector<Point2f>& m = grid[yy * gridWidth + xx];

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

                posOutBuf.push_back(oldPosBuf[ind]);
                if (hasVotes)
                    voteOutBuf.push_back(oldVoteBuf[ind]);
            }
        }
    }

    void GHT_Pos::convertTo(OutputArray _positions, OutputArray _votes)
    {
        const int total = static_cast<int>(posOutBuf.size());
        const bool hasVotes = !voteOutBuf.empty();

        CV_Assert(!hasVotes || voteOutBuf.size() == posOutBuf.size());

        _positions.create(1, total, CV_32FC4);
        Mat positions = _positions.getMat();
        Mat(1, total, CV_32FC4, &posOutBuf[0]).copyTo(positions);

        if (_votes.needed())
        {
            if (!hasVotes)
                _votes.release();
            else
            {
                _votes.create(1, total, CV_32SC3);
                Mat votes = _votes.getMat();
                Mat(1, total, CV_32SC3, &voteOutBuf[0]).copyTo(votes);
            }
        }
    }

    /////////////////////////////////////
    // POSITION Ballard

    class GHT_Ballard_Pos : public GHT_Pos
    {
    public:
        AlgorithmInfo* info() const;

        GHT_Ballard_Pos();

    protected:
        void releaseImpl();

        void processTempl();
        void processImage();

        virtual void calcHist();
        virtual void findPosInHist();

        int levels;
        int votesThreshold;
        double dp;

        vector< vector<Point> > r_table;
        Mat hist;
    };

    CV_INIT_ALGORITHM(GHT_Ballard_Pos, "GeneralizedHough.POSITION",
                      obj.info()->addParam(obj, "minDist", obj.minDist, false, 0, 0,
                                           "Minimum distance between the centers of the detected objects.");
                      obj.info()->addParam(obj, "levels", obj.levels, false, 0, 0,
                                           "R-Table levels.");
                      obj.info()->addParam(obj, "votesThreshold", obj.votesThreshold, false, 0, 0,
                                           "The accumulator threshold for the template centers at the detection stage. The smaller it is, the more false positions may be detected.");
                      obj.info()->addParam(obj, "dp", obj.dp, false, 0, 0,
                                           "Inverse ratio of the accumulator resolution to the image resolution."));

    GHT_Ballard_Pos::GHT_Ballard_Pos()
    {
        levels = 360;
        votesThreshold = 100;
        dp = 1.0;
    }

    void GHT_Ballard_Pos::releaseImpl()
    {
        GHT_Pos::releaseImpl();

        releaseVector(r_table);
        hist.release();
    }

    void GHT_Ballard_Pos::processTempl()
    {
        CV_Assert(templEdges.type() == CV_8UC1);
        CV_Assert(templDx.type() == CV_32FC1 && templDx.size() == templSize);
        CV_Assert(templDy.type() == templDx.type() && templDy.size() == templSize);
        CV_Assert(levels > 0);

        const double thetaScale = levels / 360.0;

        r_table.resize(levels + 1);
        for_each(r_table.begin(), r_table.end(), mem_fun_ref(&vector<Point>::clear));

        for (int y = 0; y < templSize.height; ++y)
        {
            const uchar* edgesRow = templEdges.ptr(y);
            const float* dxRow = templDx.ptr<float>(y);
            const float* dyRow = templDy.ptr<float>(y);

            for (int x = 0; x < templSize.width; ++x)
            {
                const Point p(x, y);

                if (edgesRow[x] && (notNull(dyRow[x]) || notNull(dxRow[x])))
                {
                    const float theta = fastAtan2(dyRow[x], dxRow[x]);
                    const int n = cvRound(theta * thetaScale);
                    r_table[n].push_back(p - templCenter);
                }
            }
        }
    }

    void GHT_Ballard_Pos::processImage()
    {
        calcHist();
        findPosInHist();
    }

    void GHT_Ballard_Pos::calcHist()
    {
        CV_Assert(imageEdges.type() == CV_8UC1);
        CV_Assert(imageDx.type() == CV_32FC1 && imageDx.size() == imageSize);
        CV_Assert(imageDy.type() == imageDx.type() && imageDy.size() == imageSize);
        CV_Assert(levels > 0 && r_table.size() == static_cast<size_t>(levels + 1));
        CV_Assert(dp > 0.0);

        const double thetaScale = levels / 360.0;
        const double idp = 1.0 / dp;

        hist.create(cvCeil(imageSize.height * idp) + 2, cvCeil(imageSize.width * idp) + 2, CV_32SC1);
        hist.setTo(0);

        const int rows = hist.rows - 2;
        const int cols = hist.cols - 2;

        for (int y = 0; y < imageSize.height; ++y)
        {
            const uchar* edgesRow = imageEdges.ptr(y);
            const float* dxRow = imageDx.ptr<float>(y);
            const float* dyRow = imageDy.ptr<float>(y);

            for (int x = 0; x < imageSize.width; ++x)
            {
                const Point p(x, y);

                if (edgesRow[x] && (notNull(dyRow[x]) || notNull(dxRow[x])))
                {
                    const float theta = fastAtan2(dyRow[x], dxRow[x]);
                    const int n = cvRound(theta * thetaScale);

                    const vector<Point>& r_row = r_table[n];

                    for (size_t j = 0; j < r_row.size(); ++j)
                    {
                        Point c = p - r_row[j];

                        c.x = cvRound(c.x * idp);
                        c.y = cvRound(c.y * idp);

                        if (c.x >= 0 && c.x < cols && c.y >= 0 && c.y < rows)
                            ++hist.at<int>(c.y + 1, c.x + 1);
                    }
                }
            }
        }
    }

    void GHT_Ballard_Pos::findPosInHist()
    {
        CV_Assert(votesThreshold > 0);

        const int histRows = hist.rows - 2;
        const int histCols = hist.cols - 2;

        for(int y = 0; y < histRows; ++y)
        {
            const int* prevRow = hist.ptr<int>(y);
            const int* curRow = hist.ptr<int>(y + 1);
            const int* nextRow = hist.ptr<int>(y + 2);

            for(int x = 0; x < histCols; ++x)
            {
                const int votes = curRow[x + 1];

                if (votes > votesThreshold && votes > curRow[x] && votes >= curRow[x + 2] && votes > prevRow[x + 1] && votes >= nextRow[x + 1])
                {
                    posOutBuf.push_back(Vec4f(static_cast<float>(x * dp), static_cast<float>(y * dp), 1.0f, 0.0f));
                    voteOutBuf.push_back(Vec3i(votes, 0, 0));
                }
            }
        }
    }

    /////////////////////////////////////
    // POSITION & SCALE

    class GHT_Ballard_PosScale : public GHT_Ballard_Pos
    {
    public:
        AlgorithmInfo* info() const;

        GHT_Ballard_PosScale();

    protected:
        void calcHist();
        void findPosInHist();

        double minScale;
        double maxScale;
        double scaleStep;

        class Worker;
        friend class Worker;
    };

    CV_INIT_ALGORITHM(GHT_Ballard_PosScale, "GeneralizedHough.POSITION_SCALE",
                      obj.info()->addParam(obj, "minDist", obj.minDist, false, 0, 0,
                                           "Minimum distance between the centers of the detected objects.");
                      obj.info()->addParam(obj, "levels", obj.levels, false, 0, 0,
                                           "R-Table levels.");
                      obj.info()->addParam(obj, "votesThreshold", obj.votesThreshold, false, 0, 0,
                                           "The accumulator threshold for the template centers at the detection stage. The smaller it is, the more false positions may be detected.");
                      obj.info()->addParam(obj, "dp", obj.dp, false, 0, 0,
                                           "Inverse ratio of the accumulator resolution to the image resolution.");
                      obj.info()->addParam(obj, "minScale", obj.minScale, false, 0, 0,
                                           "Minimal scale to detect.");
                      obj.info()->addParam(obj, "maxScale", obj.maxScale, false, 0, 0,
                                           "Maximal scale to detect.");
                      obj.info()->addParam(obj, "scaleStep", obj.scaleStep, false, 0, 0,
                                           "Scale step."));

    GHT_Ballard_PosScale::GHT_Ballard_PosScale()
    {
        minScale = 0.5;
        maxScale = 2.0;
        scaleStep = 0.05;
    }

    class GHT_Ballard_PosScale::Worker : public ParallelLoopBody
    {
    public:
        explicit Worker(GHT_Ballard_PosScale* base_) : base(base_) {}

        void operator ()(const Range& range) const;

    private:
        GHT_Ballard_PosScale* base;
    };

    void GHT_Ballard_PosScale::Worker::operator ()(const Range& range) const
    {
        const double thetaScale = base->levels / 360.0;
        const double idp = 1.0 / base->dp;

        for (int s = range.start; s < range.end; ++s)
        {
            const double scale = base->minScale + s * base->scaleStep;

            Mat curHist(base->hist.size[1], base->hist.size[2], CV_32SC1, base->hist.ptr(s + 1), base->hist.step[1]);

            for (int y = 0; y < base->imageSize.height; ++y)
            {
                const uchar* edgesRow = base->imageEdges.ptr(y);
                const float* dxRow = base->imageDx.ptr<float>(y);
                const float* dyRow = base->imageDy.ptr<float>(y);

                for (int x = 0; x < base->imageSize.width; ++x)
                {
                    const Point2d p(x, y);

                    if (edgesRow[x] && (notNull(dyRow[x]) || notNull(dxRow[x])))
                    {
                        const float theta = fastAtan2(dyRow[x], dxRow[x]);
                        const int n = cvRound(theta * thetaScale);

                        const vector<Point>& r_row = base->r_table[n];

                        for (size_t j = 0; j < r_row.size(); ++j)
                        {
                            Point2d d = r_row[j];
                            Point2d c = p - d * scale;

                            c.x *= idp;
                            c.y *= idp;

                            if (c.x >= 0 && c.x < base->hist.size[2] - 2 && c.y >= 0 && c.y < base->hist.size[1] - 2)
                                ++curHist.at<int>(cvRound(c.y + 1), cvRound(c.x + 1));
                        }
                    }
                }
            }
        }
    }

    void GHT_Ballard_PosScale::calcHist()
    {
        CV_Assert(imageEdges.type() == CV_8UC1);
        CV_Assert(imageDx.type() == CV_32FC1 && imageDx.size() == imageSize);
        CV_Assert(imageDy.type() == imageDx.type() && imageDy.size() == imageSize);
        CV_Assert(levels > 0 && r_table.size() == static_cast<size_t>(levels + 1));
        CV_Assert(dp > 0.0);
        CV_Assert(minScale > 0.0 && minScale < maxScale);
        CV_Assert(scaleStep > 0.0);

        const double idp = 1.0 / dp;
        const int scaleRange = cvCeil((maxScale - minScale) / scaleStep);

        const int sizes[] = {scaleRange + 2, cvCeil(imageSize.height * idp) + 2, cvCeil(imageSize.width * idp) + 2};
        hist.create(3, sizes, CV_32SC1);
        hist.setTo(0);

        parallel_for_(Range(0, scaleRange), Worker(this));
    }

    void GHT_Ballard_PosScale::findPosInHist()
    {
        CV_Assert(votesThreshold > 0);

        const int scaleRange = hist.size[0] - 2;
        const int histRows = hist.size[1] - 2;
        const int histCols = hist.size[2] - 2;

        for (int s = 0; s < scaleRange; ++s)
        {
            const float scale = static_cast<float>(minScale + s * scaleStep);

            const Mat prevHist(histRows + 2, histCols + 2, CV_32SC1, hist.ptr(s), hist.step[1]);
            const Mat curHist(histRows + 2, histCols + 2, CV_32SC1, hist.ptr(s + 1), hist.step[1]);
            const Mat nextHist(histRows + 2, histCols + 2, CV_32SC1, hist.ptr(s + 2), hist.step[1]);

            for(int y = 0; y < histRows; ++y)
            {
                const int* prevHistRow = prevHist.ptr<int>(y + 1);
                const int* prevRow = curHist.ptr<int>(y);
                const int* curRow = curHist.ptr<int>(y + 1);
                const int* nextRow = curHist.ptr<int>(y + 2);
                const int* nextHistRow = nextHist.ptr<int>(y + 1);

                for(int x = 0; x < histCols; ++x)
                {
                    const int votes = curRow[x + 1];

                    if (votes > votesThreshold &&
                        votes > curRow[x] &&
                        votes >= curRow[x + 2] &&
                        votes > prevRow[x + 1] &&
                        votes >= nextRow[x + 1] &&
                        votes > prevHistRow[x + 1] &&
                        votes >= nextHistRow[x + 1])
                    {
                        posOutBuf.push_back(Vec4f(static_cast<float>(x * dp), static_cast<float>(y * dp), scale, 0.0f));
                        voteOutBuf.push_back(Vec3i(votes, votes, 0));
                    }
                }
            }
        }
    }

    /////////////////////////////////////
    // POSITION & ROTATION

    class GHT_Ballard_PosRotation : public GHT_Ballard_Pos
    {
    public:
        AlgorithmInfo* info() const;

        GHT_Ballard_PosRotation();

    protected:
        void calcHist();
        void findPosInHist();

        double minAngle;
        double maxAngle;
        double angleStep;

        class Worker;
        friend class Worker;
    };

    CV_INIT_ALGORITHM(GHT_Ballard_PosRotation, "GeneralizedHough.POSITION_ROTATION",
                      obj.info()->addParam(obj, "minDist", obj.minDist, false, 0, 0,
                                           "Minimum distance between the centers of the detected objects.");
                      obj.info()->addParam(obj, "levels", obj.levels, false, 0, 0,
                                           "R-Table levels.");
                      obj.info()->addParam(obj, "votesThreshold", obj.votesThreshold, false, 0, 0,
                                           "The accumulator threshold for the template centers at the detection stage. The smaller it is, the more false positions may be detected.");
                      obj.info()->addParam(obj, "dp", obj.dp, false, 0, 0,
                                           "Inverse ratio of the accumulator resolution to the image resolution.");
                      obj.info()->addParam(obj, "minAngle", obj.minAngle, false, 0, 0,
                                           "Minimal rotation angle to detect in degrees.");
                      obj.info()->addParam(obj, "maxAngle", obj.maxAngle, false, 0, 0,
                                           "Maximal rotation angle to detect in degrees.");
                      obj.info()->addParam(obj, "angleStep", obj.angleStep, false, 0, 0,
                                           "Angle step in degrees."));

    GHT_Ballard_PosRotation::GHT_Ballard_PosRotation()
    {
        minAngle = 0.0;
        maxAngle = 360.0;
        angleStep = 1.0;
    }

    class GHT_Ballard_PosRotation::Worker : public ParallelLoopBody
    {
    public:
        explicit Worker(GHT_Ballard_PosRotation* base_) : base(base_) {}

        void operator ()(const Range& range) const;

    private:
        GHT_Ballard_PosRotation* base;
    };

    void GHT_Ballard_PosRotation::Worker::operator ()(const Range& range) const
    {
        const double thetaScale = base->levels / 360.0;
        const double idp = 1.0 / base->dp;

        for (int a = range.start; a < range.end; ++a)
        {
            const double angle = base->minAngle + a * base->angleStep;

            const double sinA = ::sin(toRad(angle));
            const double cosA = ::cos(toRad(angle));

            Mat curHist(base->hist.size[1], base->hist.size[2], CV_32SC1, base->hist.ptr(a + 1), base->hist.step[1]);

            for (int y = 0; y < base->imageSize.height; ++y)
            {
                const uchar* edgesRow = base->imageEdges.ptr(y);
                const float* dxRow = base->imageDx.ptr<float>(y);
                const float* dyRow = base->imageDy.ptr<float>(y);

                for (int x = 0; x < base->imageSize.width; ++x)
                {
                    const Point2d p(x, y);

                    if (edgesRow[x] && (notNull(dyRow[x]) || notNull(dxRow[x])))
                    {
                        double theta = fastAtan2(dyRow[x], dxRow[x]) - angle;
                        if (theta < 0)
                            theta += 360.0;
                        const int n = cvRound(theta * thetaScale);

                        const vector<Point>& r_row = base->r_table[n];

                        for (size_t j = 0; j < r_row.size(); ++j)
                        {
                            Point2d d = r_row[j];
                            Point2d c = p - Point2d(d.x * cosA - d.y * sinA, d.x * sinA + d.y * cosA);

                            c.x *= idp;
                            c.y *= idp;

                            if (c.x >= 0 && c.x < base->hist.size[2] - 2 && c.y >= 0 && c.y < base->hist.size[1] - 2)
                                ++curHist.at<int>(cvRound(c.y + 1), cvRound(c.x + 1));
                        }
                    }
                }
            }
        }
    }

    void GHT_Ballard_PosRotation::calcHist()
    {
        CV_Assert(imageEdges.type() == CV_8UC1);
        CV_Assert(imageDx.type() == CV_32FC1 && imageDx.size() == imageSize);
        CV_Assert(imageDy.type() == imageDx.type() && imageDy.size() == imageSize);
        CV_Assert(levels > 0 && r_table.size() == static_cast<size_t>(levels + 1));
        CV_Assert(dp > 0.0);
        CV_Assert(minAngle >= 0.0 && minAngle < maxAngle && maxAngle <= 360.0);
        CV_Assert(angleStep > 0.0 && angleStep < 360.0);

        const double idp = 1.0 / dp;
        const int angleRange = cvCeil((maxAngle - minAngle) / angleStep);

        const int sizes[] = {angleRange + 2, cvCeil(imageSize.height * idp) + 2, cvCeil(imageSize.width * idp) + 2};
        hist.create(3, sizes, CV_32SC1);
        hist.setTo(0);

        parallel_for_(Range(0, angleRange), Worker(this));
    }

    void GHT_Ballard_PosRotation::findPosInHist()
    {
        CV_Assert(votesThreshold > 0);

        const int angleRange = hist.size[0] - 2;
        const int histRows = hist.size[1] - 2;
        const int histCols = hist.size[2] - 2;

        for (int a = 0; a < angleRange; ++a)
        {
            const float angle = static_cast<float>(minAngle + a * angleStep);

            const Mat prevHist(histRows + 2, histCols + 2, CV_32SC1, hist.ptr(a), hist.step[1]);
            const Mat curHist(histRows + 2, histCols + 2, CV_32SC1, hist.ptr(a + 1), hist.step[1]);
            const Mat nextHist(histRows + 2, histCols + 2, CV_32SC1, hist.ptr(a + 2), hist.step[1]);

            for(int y = 0; y < histRows; ++y)
            {
                const int* prevHistRow = prevHist.ptr<int>(y + 1);
                const int* prevRow = curHist.ptr<int>(y);
                const int* curRow = curHist.ptr<int>(y + 1);
                const int* nextRow = curHist.ptr<int>(y + 2);
                const int* nextHistRow = nextHist.ptr<int>(y + 1);

                for(int x = 0; x < histCols; ++x)
                {
                    const int votes = curRow[x + 1];

                    if (votes > votesThreshold &&
                        votes > curRow[x] &&
                        votes >= curRow[x + 2] &&
                        votes > prevRow[x + 1] &&
                        votes >= nextRow[x + 1] &&
                        votes > prevHistRow[x + 1] &&
                        votes >= nextHistRow[x + 1])
                    {
                        posOutBuf.push_back(Vec4f(static_cast<float>(x * dp), static_cast<float>(y * dp), 1.0f, angle));
                        voteOutBuf.push_back(Vec3i(votes, 0, votes));
                    }
                }
            }
        }
    }

    /////////////////////////////////////////
    // POSITION & SCALE & ROTATION

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

    class GHT_Guil_Full : public GHT_Pos
    {
    public:
        AlgorithmInfo* info() const;

        GHT_Guil_Full();

    protected:
        void releaseImpl();

        void processTempl();
        void processImage();

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

        void buildFeatureList(const Mat& edges, const Mat& dx, const Mat& dy, vector< vector<Feature> >& features, Point2d center = Point2d());
        void getContourPoints(const Mat& edges, const Mat& dx, const Mat& dy, vector<ContourPoint>& points);

        void calcOrientation();
        void calcScale(double angle);
        void calcPosition(double angle, int angleVotes, double scale, int scaleVotes);

        int maxSize;
        double xi;
        int levels;
        double angleEpsilon;

        double minAngle;
        double maxAngle;
        double angleStep;
        int angleThresh;

        double minScale;
        double maxScale;
        double scaleStep;
        int scaleThresh;

        double dp;
        int posThresh;

        vector< vector<Feature> > templFeatures;
        vector< vector<Feature> > imageFeatures;

        vector< pair<double, int> > angles;
        vector< pair<double, int> > scales;
    };

    CV_INIT_ALGORITHM(GHT_Guil_Full, "GeneralizedHough.POSITION_SCALE_ROTATION",
                      obj.info()->addParam(obj, "minDist", obj.minDist, false, 0, 0,
                                           "Minimum distance between the centers of the detected objects.");
                      obj.info()->addParam(obj, "maxSize", obj.maxSize, false, 0, 0,
                                           "Maximal size of inner buffers.");
                      obj.info()->addParam(obj, "xi", obj.xi, false, 0, 0,
                                           "Angle difference in degrees between two points in feature.");
                      obj.info()->addParam(obj, "levels", obj.levels, false, 0, 0,
                                           "Feature table levels.");
                      obj.info()->addParam(obj, "angleEpsilon", obj.angleEpsilon, false, 0, 0,
                                           "Maximal difference between angles that treated as equal.");
                      obj.info()->addParam(obj, "minAngle", obj.minAngle, false, 0, 0,
                                           "Minimal rotation angle to detect in degrees.");
                      obj.info()->addParam(obj, "maxAngle", obj.maxAngle, false, 0, 0,
                                           "Maximal rotation angle to detect in degrees.");
                      obj.info()->addParam(obj, "angleStep", obj.angleStep, false, 0, 0,
                                           "Angle step in degrees.");
                      obj.info()->addParam(obj, "angleThresh", obj.angleThresh, false, 0, 0,
                                           "Angle threshold.");
                      obj.info()->addParam(obj, "minScale", obj.minScale, false, 0, 0,
                                           "Minimal scale to detect.");
                      obj.info()->addParam(obj, "maxScale", obj.maxScale, false, 0, 0,
                                           "Maximal scale to detect.");
                      obj.info()->addParam(obj, "scaleStep", obj.scaleStep, false, 0, 0,
                                           "Scale step.");
                      obj.info()->addParam(obj, "scaleThresh", obj.scaleThresh, false, 0, 0,
                                           "Scale threshold.");
                      obj.info()->addParam(obj, "dp", obj.dp, false, 0, 0,
                                           "Inverse ratio of the accumulator resolution to the image resolution.");
                      obj.info()->addParam(obj, "posThresh", obj.posThresh, false, 0, 0,
                                           "Position threshold."));

    GHT_Guil_Full::GHT_Guil_Full()
    {
        maxSize = 1000;
        xi = 90.0;
        levels = 360;
        angleEpsilon = 1.0;

        minAngle = 0.0;
        maxAngle = 360.0;
        angleStep = 1.0;
        angleThresh = 15000;

        minScale = 0.5;
        maxScale = 2.0;
        scaleStep = 0.05;
        scaleThresh = 1000;

        dp = 1.0;
        posThresh = 100;
    }

    void GHT_Guil_Full::releaseImpl()
    {
        GHT_Pos::releaseImpl();

        releaseVector(templFeatures);
        releaseVector(imageFeatures);

        releaseVector(angles);
        releaseVector(scales);
    }

    void GHT_Guil_Full::processTempl()
    {
        buildFeatureList(templEdges, templDx, templDy, templFeatures, templCenter);
    }

    void GHT_Guil_Full::processImage()
    {
        buildFeatureList(imageEdges, imageDx, imageDy, imageFeatures);

        calcOrientation();

        for (size_t i = 0; i < angles.size(); ++i)
        {
            const double angle = angles[i].first;
            const int angleVotes = angles[i].second;

            calcScale(angle);

            for (size_t j = 0; j < scales.size(); ++j)
            {
                const double scale = scales[j].first;
                const int scaleVotes = scales[j].second;

                calcPosition(angle, angleVotes, scale, scaleVotes);
            }
        }
    }

    void GHT_Guil_Full::buildFeatureList(const Mat& edges, const Mat& dx, const Mat& dy, vector< vector<Feature> >& features, Point2d center)
    {
        CV_Assert(levels > 0);

        const double maxDist = sqrt((double) templSize.width * templSize.width + templSize.height * templSize.height) * maxScale;

        const double alphaScale = levels / 360.0;

        vector<ContourPoint> points;
        getContourPoints(edges, dx, dy, points);

        features.resize(levels + 1);
        for_each(features.begin(), features.end(), mem_fun_ref(&vector<Feature>::clear));
        for_each(features.begin(), features.end(), bind2nd(mem_fun_ref(&vector<Feature>::reserve), maxSize));

        for (size_t i = 0; i < points.size(); ++i)
        {
            ContourPoint p1 = points[i];

            for (size_t j = 0; j < points.size(); ++j)
            {
                ContourPoint p2 = points[j];

                if (angleEq(p1.theta - p2.theta, xi, angleEpsilon))
                {
                    const Point2d d = p1.pos - p2.pos;

                    Feature f;

                    f.p1 = p1;
                    f.p2 = p2;

                    f.alpha12 = clampAngle(fastAtan2((float)d.y, (float)d.x) - p1.theta);
                    f.d12 = norm(d);

                    if (f.d12 > maxDist)
                        continue;

                    f.r1 = p1.pos - center;
                    f.r2 = p2.pos - center;

                    const int n = cvRound(f.alpha12 * alphaScale);

                    if (features[n].size() < static_cast<size_t>(maxSize))
                        features[n].push_back(f);
                }
            }
        }
    }

    void GHT_Guil_Full::getContourPoints(const Mat& edges, const Mat& dx, const Mat& dy, vector<ContourPoint>& points)
    {
        CV_Assert(edges.type() == CV_8UC1);
        CV_Assert(dx.type() == CV_32FC1 && dx.size == edges.size);
        CV_Assert(dy.type() == dx.type() && dy.size == edges.size);

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

    void GHT_Guil_Full::calcOrientation()
    {
        CV_Assert(levels > 0);
        CV_Assert(templFeatures.size() == static_cast<size_t>(levels + 1));
        CV_Assert(imageFeatures.size() == templFeatures.size());
        CV_Assert(minAngle >= 0.0 && minAngle < maxAngle && maxAngle <= 360.0);
        CV_Assert(angleStep > 0.0 && angleStep < 360.0);
        CV_Assert(angleThresh > 0);

        const double iAngleStep = 1.0 / angleStep;
        const int angleRange = cvCeil((maxAngle - minAngle) * iAngleStep);

        vector<int> OHist(angleRange + 1, 0);
        for (int i = 0; i <= levels; ++i)
        {
            const vector<Feature>& templRow = templFeatures[i];
            const vector<Feature>& imageRow = imageFeatures[i];

            for (size_t j = 0; j < templRow.size(); ++j)
            {
                Feature templF = templRow[j];

                for (size_t k = 0; k < imageRow.size(); ++k)
                {
                    Feature imF = imageRow[k];

                    const double angle = clampAngle(imF.p1.theta - templF.p1.theta);
                    if (angle >= minAngle && angle <= maxAngle)
                    {
                        const int n = cvRound((angle - minAngle) * iAngleStep);
                        ++OHist[n];
                    }
                }
            }
        }

        angles.clear();

        for (int n = 0; n < angleRange; ++n)
        {
            if (OHist[n] >= angleThresh)
            {
                const double angle = minAngle + n * angleStep;
                angles.push_back(make_pair(angle, OHist[n]));
            }
        }
    }

    void GHT_Guil_Full::calcScale(double angle)
    {
        CV_Assert(levels > 0);
        CV_Assert(templFeatures.size() == static_cast<size_t>(levels + 1));
        CV_Assert(imageFeatures.size() == templFeatures.size());
        CV_Assert(minScale > 0.0 && minScale < maxScale);
        CV_Assert(scaleStep > 0.0);
        CV_Assert(scaleThresh > 0);

        const double iScaleStep = 1.0 / scaleStep;
        const int scaleRange = cvCeil((maxScale - minScale) * iScaleStep);

        vector<int> SHist(scaleRange + 1, 0);

        for (int i = 0; i <= levels; ++i)
        {
            const vector<Feature>& templRow = templFeatures[i];
            const vector<Feature>& imageRow = imageFeatures[i];

            for (size_t j = 0; j < templRow.size(); ++j)
            {
                Feature templF = templRow[j];

                templF.p1.theta += angle;

                for (size_t k = 0; k < imageRow.size(); ++k)
                {
                    Feature imF = imageRow[k];

                    if (angleEq(imF.p1.theta, templF.p1.theta, angleEpsilon))
                    {
                        const double scale = imF.d12 / templF.d12;
                        if (scale >= minScale && scale <= maxScale)
                        {
                            const int s = cvRound((scale - minScale) * iScaleStep);
                            ++SHist[s];
                        }
                    }
                }
            }
        }

        scales.clear();

        for (int s = 0; s < scaleRange; ++s)
        {
            if (SHist[s] >= scaleThresh)
            {
                const double scale = minScale + s * scaleStep;
                scales.push_back(make_pair(scale, SHist[s]));
            }
        }
    }

    void GHT_Guil_Full::calcPosition(double angle, int angleVotes, double scale, int scaleVotes)
    {
        CV_Assert(levels > 0);
        CV_Assert(templFeatures.size() == static_cast<size_t>(levels + 1));
        CV_Assert(imageFeatures.size() == templFeatures.size());
        CV_Assert(dp > 0.0);
        CV_Assert(posThresh > 0);

        const double sinVal = sin(toRad(angle));
        const double cosVal = cos(toRad(angle));
        const double idp = 1.0 / dp;

        const int histRows = cvCeil(imageSize.height * idp);
        const int histCols = cvCeil(imageSize.width * idp);

        Mat DHist(histRows + 2, histCols + 2, CV_32SC1, Scalar::all(0));

        for (int i = 0; i <= levels; ++i)
        {
            const vector<Feature>& templRow = templFeatures[i];
            const vector<Feature>& imageRow = imageFeatures[i];

            for (size_t j = 0; j < templRow.size(); ++j)
            {
                Feature templF = templRow[j];

                templF.p1.theta += angle;

                templF.r1 *= scale;
                templF.r2 *= scale;

                templF.r1 = Point2d(cosVal * templF.r1.x - sinVal * templF.r1.y, sinVal * templF.r1.x + cosVal * templF.r1.y);
                templF.r2 = Point2d(cosVal * templF.r2.x - sinVal * templF.r2.y, sinVal * templF.r2.x + cosVal * templF.r2.y);

                for (size_t k = 0; k < imageRow.size(); ++k)
                {
                    Feature imF = imageRow[k];

                    if (angleEq(imF.p1.theta, templF.p1.theta, angleEpsilon))
                    {
                        Point2d c1, c2;

                        c1 = imF.p1.pos - templF.r1;
                        c1 *= idp;

                        c2 = imF.p2.pos - templF.r2;
                        c2 *= idp;

                        if (fabs(c1.x - c2.x) > 1 || fabs(c1.y - c2.y) > 1)
                            continue;

                        if (c1.y >= 0 && c1.y < histRows && c1.x >= 0 && c1.x < histCols)
                            ++DHist.at<int>(cvRound(c1.y) + 1, cvRound(c1.x) + 1);
                    }
                }
            }
        }

        for(int y = 0; y < histRows; ++y)
        {
            const int* prevRow = DHist.ptr<int>(y);
            const int* curRow = DHist.ptr<int>(y + 1);
            const int* nextRow = DHist.ptr<int>(y + 2);

            for(int x = 0; x < histCols; ++x)
            {
                const int votes = curRow[x + 1];

                if (votes > posThresh && votes > curRow[x] && votes >= curRow[x + 2] && votes > prevRow[x + 1] && votes >= nextRow[x + 1])
                {
                    posOutBuf.push_back(Vec4f(static_cast<float>(x * dp), static_cast<float>(y * dp), static_cast<float>(scale), static_cast<float>(angle)));
                    voteOutBuf.push_back(Vec3i(votes, scaleVotes, angleVotes));
                }
            }
        }
    }
}

Ptr<GeneralizedHough> cv::GeneralizedHough::create(int method)
{
    switch (method)
    {
    case GHT_POSITION:
        CV_Assert( !GHT_Ballard_Pos_info_auto.name().empty() );
        return new GHT_Ballard_Pos();

    case (GHT_POSITION | GHT_SCALE):
        CV_Assert( !GHT_Ballard_PosScale_info_auto.name().empty() );
        return new GHT_Ballard_PosScale();

    case (GHT_POSITION | GHT_ROTATION):
        CV_Assert( !GHT_Ballard_PosRotation_info_auto.name().empty() );
        return new GHT_Ballard_PosRotation();

    case (GHT_POSITION | GHT_SCALE | GHT_ROTATION):
        CV_Assert( !GHT_Guil_Full_info_auto.name().empty() );
        return new GHT_Guil_Full();
    }

    CV_Error(CV_StsBadArg, "Unsupported method");
    return Ptr<GeneralizedHough>();
}

cv::GeneralizedHough::~GeneralizedHough()
{
}

void cv::GeneralizedHough::setTemplate(InputArray _templ, int cannyThreshold, Point templCenter)
{
    Mat templ = _templ.getMat();

    CV_Assert(templ.type() == CV_8UC1);
    CV_Assert(cannyThreshold > 0);

    Canny(templ, edges_, cannyThreshold / 2, cannyThreshold);
    Sobel(templ, dx_, CV_32F, 1, 0);
    Sobel(templ, dy_, CV_32F, 0, 1);

    if (templCenter == Point(-1, -1))
        templCenter = Point(templ.cols / 2, templ.rows / 2);

    setTemplateImpl(edges_, dx_, dy_, templCenter);
}

void cv::GeneralizedHough::setTemplate(InputArray _edges, InputArray _dx, InputArray _dy, Point templCenter)
{
    Mat edges = _edges.getMat();
    Mat dx = _dx.getMat();
    Mat dy = _dy.getMat();

    if (templCenter == Point(-1, -1))
        templCenter = Point(edges.cols / 2, edges.rows / 2);

    setTemplateImpl(edges, dx, dy, templCenter);
}

void cv::GeneralizedHough::detect(InputArray _image, OutputArray positions, OutputArray votes, int cannyThreshold)
{
    Mat image = _image.getMat();

    CV_Assert(image.type() == CV_8UC1);
    CV_Assert(cannyThreshold > 0);

    Canny(image, edges_, cannyThreshold / 2, cannyThreshold);
    Sobel(image, dx_, CV_32F, 1, 0);
    Sobel(image, dy_, CV_32F, 0, 1);

    detectImpl(edges_, dx_, dy_, positions, votes);
}

void cv::GeneralizedHough::detect(InputArray _edges, InputArray _dx, InputArray _dy, OutputArray positions, OutputArray votes)
{
    cv::Mat edges = _edges.getMat();
    cv::Mat dx = _dx.getMat();
    cv::Mat dy = _dy.getMat();

    detectImpl(edges, dx, dy, positions, votes);
}

void cv::GeneralizedHough::release()
{
    edges_.release();
    dx_.release();
    dy_.release();
    releaseImpl();
}
