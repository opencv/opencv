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

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER) || !defined(HAVE_OPENCV_GPUARITHM)

Ptr<gpu::GeneralizedHough> cv::gpu::GeneralizedHough::create(int) { throw_no_cuda(); return Ptr<GeneralizedHough>(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace gpu { namespace cudev
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

        void Ballard_PosScale_calcHist_gpu(const unsigned int* coordList, const float* thetaList, int pointsCount,
                                           PtrStepSz<short2> r_table, const int* r_sizes,
                                           PtrStepi hist, int rows, int cols,
                                           float minScale, float scaleStep, int scaleRange,
                                           float dp, int levels);
        int Ballard_PosScale_findPosInHist_gpu(PtrStepi hist, int rows, int cols, int scaleRange, float4* out, int3* votes, int maxSize,
                                               float minScale, float scaleStep, float dp, int threshold);

        void Ballard_PosRotation_calcHist_gpu(const unsigned int* coordList, const float* thetaList, int pointsCount,
                                              PtrStepSz<short2> r_table, const int* r_sizes,
                                              PtrStepi hist, int rows, int cols,
                                              float minAngle, float angleStep, int angleRange,
                                              float dp, int levels);
        int Ballard_PosRotation_findPosInHist_gpu(PtrStepi hist, int rows, int cols, int angleRange, float4* out, int3* votes, int maxSize,
                                                  float minAngle, float angleStep, float dp, int threshold);

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

namespace
{
    /////////////////////////////////////
    // GeneralizedHoughBase

    class GeneralizedHoughBase : public gpu::GeneralizedHough
    {
    public:
        GeneralizedHoughBase();

        void setTemplate(InputArray templ, int cannyThreshold = 100, Point templCenter = Point(-1, -1));
        void setTemplate(InputArray edges, InputArray dx, InputArray dy, Point templCenter = Point(-1, -1));

        void detect(InputArray image, OutputArray positions, int cannyThreshold = 100);
        void detect(InputArray edges, InputArray dx, InputArray dy, OutputArray positions);

        void downloadResults(InputArray d_positions, OutputArray h_positions, OutputArray h_votes = noArray());

    protected:
        virtual void setTemplateImpl(const GpuMat& edges, const GpuMat& dx, const GpuMat& dy, Point templCenter) = 0;
        virtual void detectImpl(const GpuMat& edges, const GpuMat& dx, const GpuMat& dy, OutputArray positions) = 0;

    private:
#ifdef HAVE_OPENCV_GPUFILTERS
        GpuMat dx_, dy_;
        GpuMat edges_;
        Ptr<gpu::CannyEdgeDetector> canny_;
        Ptr<gpu::Filter> filterDx_;
        Ptr<gpu::Filter> filterDy_;
#endif
    };

    GeneralizedHoughBase::GeneralizedHoughBase()
    {
#ifdef HAVE_OPENCV_GPUFILTERS
        canny_ = gpu::createCannyEdgeDetector(50, 100);
        filterDx_ = gpu::createSobelFilter(CV_8UC1, CV_32S, 1, 0);
        filterDy_ = gpu::createSobelFilter(CV_8UC1, CV_32S, 0, 1);
#endif
    }

    void GeneralizedHoughBase::setTemplate(InputArray _templ, int cannyThreshold, Point templCenter)
    {
#ifndef HAVE_OPENCV_GPUFILTERS
        (void) _templ;
        (void) cannyThreshold;
        (void) templCenter;
        throw_no_cuda();
#else
        GpuMat templ = _templ.getGpuMat();

        CV_Assert( templ.type() == CV_8UC1 );
        CV_Assert( cannyThreshold > 0 );

        ensureSizeIsEnough(templ.size(), CV_32SC1, dx_);
        ensureSizeIsEnough(templ.size(), CV_32SC1, dy_);

        filterDx_->apply(templ, dx_);
        filterDy_->apply(templ, dy_);

        ensureSizeIsEnough(templ.size(), CV_8UC1, edges_);

        canny_->setLowThreshold(cannyThreshold / 2);
        canny_->setHighThreshold(cannyThreshold);
        canny_->detect(dx_, dy_, edges_);

        if (templCenter == Point(-1, -1))
            templCenter = Point(templ.cols / 2, templ.rows / 2);

        setTemplateImpl(edges_, dx_, dy_, templCenter);
#endif
    }

    void GeneralizedHoughBase::setTemplate(InputArray _edges, InputArray _dx, InputArray _dy, Point templCenter)
    {
        GpuMat edges = _edges.getGpuMat();
        GpuMat dx = _dx.getGpuMat();
        GpuMat dy = _dy.getGpuMat();

        if (templCenter == Point(-1, -1))
            templCenter = Point(edges.cols / 2, edges.rows / 2);

        setTemplateImpl(edges, dx, dy, templCenter);
    }

    void GeneralizedHoughBase::detect(InputArray _image, OutputArray positions, int cannyThreshold)
    {
#ifndef HAVE_OPENCV_GPUFILTERS
        (void) _image;
        (void) positions;
        (void) cannyThreshold;
        throw_no_cuda();
#else
        GpuMat image = _image.getGpuMat();

        CV_Assert( image.type() == CV_8UC1 );
        CV_Assert( cannyThreshold > 0 );

        ensureSizeIsEnough(image.size(), CV_32SC1, dx_);
        ensureSizeIsEnough(image.size(), CV_32SC1, dy_);

        filterDx_->apply(image, dx_);
        filterDy_->apply(image, dy_);

        ensureSizeIsEnough(image.size(), CV_8UC1, edges_);

        canny_->setLowThreshold(cannyThreshold / 2);
        canny_->setHighThreshold(cannyThreshold);
        canny_->detect(dx_, dy_, edges_);

        detectImpl(edges_, dx_, dy_, positions);
#endif
    }

    void GeneralizedHoughBase::detect(InputArray _edges, InputArray _dx, InputArray _dy, OutputArray positions)
    {
        GpuMat edges = _edges.getGpuMat();
        GpuMat dx = _dx.getGpuMat();
        GpuMat dy = _dy.getGpuMat();

        detectImpl(edges, dx, dy, positions);
    }

    void GeneralizedHoughBase::downloadResults(InputArray _d_positions, OutputArray h_positions, OutputArray h_votes)
    {
        GpuMat d_positions = _d_positions.getGpuMat();

        if (d_positions.empty())
        {
            h_positions.release();
            if (h_votes.needed())
                h_votes.release();
            return;
        }

        CV_Assert( d_positions.rows == 2 && d_positions.type() == CV_32FC4 );

        d_positions.row(0).download(h_positions);

        if (h_votes.needed())
        {
            GpuMat d_votes(1, d_positions.cols, CV_32SC3, d_positions.ptr<int3>(1));
            d_votes.download(h_votes);
        }
    }

    /////////////////////////////////////
    // GHT_Pos

    template <typename T, class A> void releaseVector(std::vector<T, A>& v)
    {
        std::vector<T, A> empty;
        empty.swap(v);
    }

    class GHT_Pos : public GeneralizedHoughBase
    {
    public:
        GHT_Pos();

    protected:
        void setTemplateImpl(const GpuMat& edges, const GpuMat& dx, const GpuMat& dy, Point templCenter);
        void detectImpl(const GpuMat& edges, const GpuMat& dx, const GpuMat& dy, OutputArray positions);
        void releaseImpl();

        virtual void processTempl() = 0;
        virtual void processImage() = 0;

        void buildEdgePointList(const GpuMat& edges, const GpuMat& dx, const GpuMat& dy);
        void filterMinDist();
        void convertTo(OutputArray positions);

        int maxSize;
        double minDist;

        Size templSize;
        Point templCenter;
        GpuMat templEdges;
        GpuMat templDx;
        GpuMat templDy;

        Size imageSize;
        GpuMat imageEdges;
        GpuMat imageDx;
        GpuMat imageDy;

        GpuMat edgePointList;

        GpuMat outBuf;
        int posCount;

        std::vector<float4> oldPosBuf;
        std::vector<int3> oldVoteBuf;
        std::vector<float4> newPosBuf;
        std::vector<int3> newVoteBuf;
        std::vector<int> indexies;
    };

    GHT_Pos::GHT_Pos()
    {
        maxSize = 10000;
        minDist = 1.0;
    }

    void GHT_Pos::setTemplateImpl(const GpuMat& edges, const GpuMat& dx, const GpuMat& dy, Point templCenter_)
    {
        templSize = edges.size();
        templCenter = templCenter_;

        ensureSizeIsEnough(templSize, edges.type(), templEdges);
        ensureSizeIsEnough(templSize, dx.type(), templDx);
        ensureSizeIsEnough(templSize, dy.type(), templDy);

        edges.copyTo(templEdges);
        dx.copyTo(templDx);
        dy.copyTo(templDy);

        processTempl();
    }

    void GHT_Pos::detectImpl(const GpuMat& edges, const GpuMat& dx, const GpuMat& dy, OutputArray positions)
    {
        imageSize = edges.size();

        ensureSizeIsEnough(imageSize, edges.type(), imageEdges);
        ensureSizeIsEnough(imageSize, dx.type(), imageDx);
        ensureSizeIsEnough(imageSize, dy.type(), imageDy);

        edges.copyTo(imageEdges);
        dx.copyTo(imageDx);
        dy.copyTo(imageDy);

        posCount = 0;

        processImage();

        if (posCount == 0)
            positions.release();
        else
        {
            if (minDist > 1)
                filterMinDist();
            convertTo(positions);
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

        edgePointList.release();

        outBuf.release();
        posCount = 0;

        releaseVector(oldPosBuf);
        releaseVector(oldVoteBuf);
        releaseVector(newPosBuf);
        releaseVector(newVoteBuf);
        releaseVector(indexies);
    }

    void GHT_Pos::buildEdgePointList(const GpuMat& edges, const GpuMat& dx, const GpuMat& dy)
    {
        using namespace cv::gpu::cudev::ght;

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

        CV_Assert(edges.type() == CV_8UC1);
        CV_Assert(dx.size() == edges.size());
        CV_Assert(dy.type() == dx.type() && dy.size() == edges.size());

        const func_t func = funcs[dx.depth()];
        CV_Assert(func != 0);

        edgePointList.cols = (int) (edgePointList.step / sizeof(int));
        ensureSizeIsEnough(2, edges.size().area(), CV_32SC1, edgePointList);

        edgePointList.cols = func(edges, dx, dy, edgePointList.ptr<unsigned int>(0), edgePointList.ptr<float>(1));
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

    void GHT_Pos::filterMinDist()
    {
        oldPosBuf.resize(posCount);
        oldVoteBuf.resize(posCount);

        cudaSafeCall( cudaMemcpy(&oldPosBuf[0], outBuf.ptr(0), posCount * sizeof(float4), cudaMemcpyDeviceToHost) );
        cudaSafeCall( cudaMemcpy(&oldVoteBuf[0], outBuf.ptr(1), posCount * sizeof(int3), cudaMemcpyDeviceToHost) );

        indexies.resize(posCount);
        for (int i = 0; i < posCount; ++i)
            indexies[i] = i;
        std::sort(indexies.begin(), indexies.end(), IndexCmp(&oldVoteBuf[0]));

        newPosBuf.clear();
        newVoteBuf.clear();
        newPosBuf.reserve(posCount);
        newVoteBuf.reserve(posCount);

        const int cellSize = cvRound(minDist);
        const int gridWidth = (imageSize.width + cellSize - 1) / cellSize;
        const int gridHeight = (imageSize.height + cellSize - 1) / cellSize;

        std::vector< std::vector<Point2f> > grid(gridWidth * gridHeight);

        const double minDist2 = minDist * minDist;

        for (int i = 0; i < posCount; ++i)
        {
            const int ind = indexies[i];

            Point2f p(oldPosBuf[ind].x, oldPosBuf[ind].y);

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

                newPosBuf.push_back(oldPosBuf[ind]);
                newVoteBuf.push_back(oldVoteBuf[ind]);
            }
        }

        posCount = static_cast<int>(newPosBuf.size());
        cudaSafeCall( cudaMemcpy(outBuf.ptr(0), &newPosBuf[0], posCount * sizeof(float4), cudaMemcpyHostToDevice) );
        cudaSafeCall( cudaMemcpy(outBuf.ptr(1), &newVoteBuf[0], posCount * sizeof(int3), cudaMemcpyHostToDevice) );
    }

    void GHT_Pos::convertTo(OutputArray positions)
    {
        ensureSizeIsEnough(2, posCount, CV_32FC4, positions);
        GpuMat(2, posCount, CV_32FC4, outBuf.data, outBuf.step).copyTo(positions);
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

        GpuMat r_table;
        GpuMat r_sizes;

        GpuMat hist;
    };

    CV_INIT_ALGORITHM(GHT_Ballard_Pos, "GeneralizedHough_GPU.POSITION",
                      obj.info()->addParam(obj, "maxSize", obj.maxSize, false, 0, 0,
                                           "Maximal size of inner buffers.");
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

        r_table.release();
        r_sizes.release();

        hist.release();
    }

    void GHT_Ballard_Pos::processTempl()
    {
        using namespace cv::gpu::cudev::ght;

        CV_Assert(levels > 0);

        buildEdgePointList(templEdges, templDx, templDy);

        ensureSizeIsEnough(levels + 1, maxSize, CV_16SC2, r_table);
        ensureSizeIsEnough(1, levels + 1, CV_32SC1, r_sizes);
        r_sizes.setTo(Scalar::all(0));

        if (edgePointList.cols > 0)
        {
            buildRTable_gpu(edgePointList.ptr<unsigned int>(0), edgePointList.ptr<float>(1), edgePointList.cols,
                            r_table, r_sizes.ptr<int>(), make_short2(templCenter.x, templCenter.y), levels);
            gpu::min(r_sizes, maxSize, r_sizes);
        }
    }

    void GHT_Ballard_Pos::processImage()
    {
        calcHist();
        findPosInHist();
    }

    void GHT_Ballard_Pos::calcHist()
    {
        using namespace cv::gpu::cudev::ght;

        CV_Assert(levels > 0 && r_table.rows == (levels + 1) && r_sizes.cols == (levels + 1));
        CV_Assert(dp > 0.0);

        const double idp = 1.0 / dp;

        buildEdgePointList(imageEdges, imageDx, imageDy);

        ensureSizeIsEnough(cvCeil(imageSize.height * idp) + 2, cvCeil(imageSize.width * idp) + 2, CV_32SC1, hist);
        hist.setTo(Scalar::all(0));

        if (edgePointList.cols > 0)
        {
            Ballard_Pos_calcHist_gpu(edgePointList.ptr<unsigned int>(0), edgePointList.ptr<float>(1), edgePointList.cols,
                                     r_table, r_sizes.ptr<int>(),
                                     hist,
                                     (float)dp, levels);
        }
    }

    void GHT_Ballard_Pos::findPosInHist()
    {
        using namespace cv::gpu::cudev::ght;

        CV_Assert(votesThreshold > 0);

        ensureSizeIsEnough(2, maxSize, CV_32FC4, outBuf);

        posCount = Ballard_Pos_findPosInHist_gpu(hist, outBuf.ptr<float4>(0), outBuf.ptr<int3>(1), maxSize, (float)dp, votesThreshold);
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
    };

    CV_INIT_ALGORITHM(GHT_Ballard_PosScale, "GeneralizedHough_GPU.POSITION_SCALE",
                      obj.info()->addParam(obj, "maxSize", obj.maxSize, false, 0, 0,
                                           "Maximal size of inner buffers.");
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

    void GHT_Ballard_PosScale::calcHist()
    {
        using namespace cv::gpu::cudev::ght;

        CV_Assert(levels > 0 && r_table.rows == (levels + 1) && r_sizes.cols == (levels + 1));
        CV_Assert(dp > 0.0);
        CV_Assert(minScale > 0.0 && minScale < maxScale);
        CV_Assert(scaleStep > 0.0);

        const double idp = 1.0 / dp;
        const int scaleRange = cvCeil((maxScale - minScale) / scaleStep);
        const int rows = cvCeil(imageSize.height * idp);
        const int cols = cvCeil(imageSize.width * idp);

        buildEdgePointList(imageEdges, imageDx, imageDy);

        ensureSizeIsEnough((scaleRange + 2) * (rows + 2), cols + 2, CV_32SC1, hist);
        hist.setTo(Scalar::all(0));

        if (edgePointList.cols > 0)
        {
            Ballard_PosScale_calcHist_gpu(edgePointList.ptr<unsigned int>(0), edgePointList.ptr<float>(1), edgePointList.cols,
                                          r_table, r_sizes.ptr<int>(),
                                          hist, rows, cols,
                                          (float)minScale, (float)scaleStep, scaleRange, (float)dp, levels);
        }
    }

    void GHT_Ballard_PosScale::findPosInHist()
    {
        using namespace cv::gpu::cudev::ght;

        CV_Assert(votesThreshold > 0);

        const double idp = 1.0 / dp;
        const int scaleRange = cvCeil((maxScale - minScale) / scaleStep);
        const int rows = cvCeil(imageSize.height * idp);
        const int cols = cvCeil(imageSize.width * idp);

        ensureSizeIsEnough(2, maxSize, CV_32FC4, outBuf);

        posCount =  Ballard_PosScale_findPosInHist_gpu(hist, rows, cols, scaleRange, outBuf.ptr<float4>(0), outBuf.ptr<int3>(1), maxSize, (float)minScale, (float)scaleStep, (float)dp, votesThreshold);
    }

    /////////////////////////////////////
    // POSITION & Rotation

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
    };

    CV_INIT_ALGORITHM(GHT_Ballard_PosRotation, "GeneralizedHough_GPU.POSITION_ROTATION",
                      obj.info()->addParam(obj, "maxSize", obj.maxSize, false, 0, 0,
                                           "Maximal size of inner buffers.");
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

    void GHT_Ballard_PosRotation::calcHist()
    {
        using namespace cv::gpu::cudev::ght;

        CV_Assert(levels > 0 && r_table.rows == (levels + 1) && r_sizes.cols == (levels + 1));
        CV_Assert(dp > 0.0);
        CV_Assert(minAngle >= 0.0 && minAngle < maxAngle && maxAngle <= 360.0);
        CV_Assert(angleStep > 0.0 && angleStep < 360.0);

        const double idp = 1.0 / dp;
        const int angleRange = cvCeil((maxAngle - minAngle) / angleStep);
        const int rows = cvCeil(imageSize.height * idp);
        const int cols = cvCeil(imageSize.width * idp);

        buildEdgePointList(imageEdges, imageDx, imageDy);

        ensureSizeIsEnough((angleRange + 2) * (rows + 2), cols + 2, CV_32SC1, hist);
        hist.setTo(Scalar::all(0));

        if (edgePointList.cols > 0)
        {
            Ballard_PosRotation_calcHist_gpu(edgePointList.ptr<unsigned int>(0), edgePointList.ptr<float>(1), edgePointList.cols,
                                             r_table, r_sizes.ptr<int>(),
                                             hist, rows, cols,
                                             (float)minAngle, (float)angleStep, angleRange, (float)dp, levels);
        }
    }

    void GHT_Ballard_PosRotation::findPosInHist()
    {
        using namespace cv::gpu::cudev::ght;

        CV_Assert(votesThreshold > 0);

        const double idp = 1.0 / dp;
        const int angleRange = cvCeil((maxAngle - minAngle) / angleStep);
        const int rows = cvCeil(imageSize.height * idp);
        const int cols = cvCeil(imageSize.width * idp);

        ensureSizeIsEnough(2, maxSize, CV_32FC4, outBuf);

        posCount = Ballard_PosRotation_findPosInHist_gpu(hist, rows, cols, angleRange, outBuf.ptr<float4>(0), outBuf.ptr<int3>(1), maxSize, (float)minAngle, (float)angleStep, (float)dp, votesThreshold);
    }

    /////////////////////////////////////////
    // POSITION & SCALE & ROTATION

    double toRad(double a)
    {
        return a * CV_PI / 180.0;
    }

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
            void release();
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

        Feature templFeatures;
        Feature imageFeatures;

        std::vector< std::pair<double, int> > angles;
        std::vector< std::pair<double, int> > scales;

        GpuMat hist;
        std::vector<int> h_buf;
    };

    CV_INIT_ALGORITHM(GHT_Guil_Full, "GeneralizedHough_GPU.POSITION_SCALE_ROTATION",
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

        templFeatures.release();
        imageFeatures.release();

        releaseVector(angles);
        releaseVector(scales);

        hist.release();
        releaseVector(h_buf);
    }

    void GHT_Guil_Full::processTempl()
    {
        using namespace cv::gpu::cudev::ght;

        buildFeatureList(templEdges, templDx, templDy, templFeatures,
            Guil_Full_setTemplFeatures, Guil_Full_buildTemplFeatureList_gpu,
            true, templCenter);

        h_buf.resize(templFeatures.sizes.cols);
        cudaSafeCall( cudaMemcpy(&h_buf[0], templFeatures.sizes.data, h_buf.size() * sizeof(int), cudaMemcpyDeviceToHost) );
        templFeatures.maxSize = *max_element(h_buf.begin(), h_buf.end());
    }

    void GHT_Guil_Full::processImage()
    {
        using namespace cv::gpu::cudev::ght;

        CV_Assert(levels > 0);
        CV_Assert(templFeatures.sizes.cols == levels + 1);
        CV_Assert(minAngle >= 0.0 && minAngle < maxAngle && maxAngle <= 360.0);
        CV_Assert(angleStep > 0.0 && angleStep < 360.0);
        CV_Assert(angleThresh > 0);
        CV_Assert(minScale > 0.0 && minScale < maxScale);
        CV_Assert(scaleStep > 0.0);
        CV_Assert(scaleThresh > 0);
        CV_Assert(dp > 0.0);
        CV_Assert(posThresh > 0);

        const double iAngleStep = 1.0 / angleStep;
        const int angleRange = cvCeil((maxAngle - minAngle) * iAngleStep);

        const double iScaleStep = 1.0 / scaleStep;
        const int scaleRange = cvCeil((maxScale - minScale) * iScaleStep);

        const double idp = 1.0 / dp;
        const int histRows = cvCeil(imageSize.height * idp);
        const int histCols = cvCeil(imageSize.width * idp);

        ensureSizeIsEnough(histRows + 2, std::max(angleRange + 1, std::max(scaleRange + 1, histCols + 2)), CV_32SC1, hist);
        h_buf.resize(std::max(angleRange + 1, scaleRange + 1));

        ensureSizeIsEnough(2, maxSize, CV_32FC4, outBuf);

        buildFeatureList(imageEdges, imageDx, imageDy, imageFeatures,
            Guil_Full_setImageFeatures, Guil_Full_buildImageFeatureList_gpu,
            false);

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

    void GHT_Guil_Full::Feature::create(int levels, int maxCapacity, bool isTempl)
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

    void GHT_Guil_Full::Feature::release()
    {
        p1_pos.release();
        p1_theta.release();
        p2_pos.release();

        d12.release();

        r1.release();
        r2.release();

        sizes.release();

        maxSize = 0;
    }

    void GHT_Guil_Full::buildFeatureList(const GpuMat& edges, const GpuMat& dx, const GpuMat& dy, Feature& features,
                                         set_func_t set_func, build_func_t build_func, bool isTempl, Point2d center)
    {
        CV_Assert(levels > 0);

        const double maxDist = sqrt((double) templSize.width * templSize.width + templSize.height * templSize.height) * maxScale;

        features.create(levels, maxSize, isTempl);
        set_func(features.p1_pos, features.p1_theta, features.p2_pos, features.d12, features.r1, features.r2);

        buildEdgePointList(edges, dx, dy);

        if (edgePointList.cols > 0)
        {
            build_func(edgePointList.ptr<unsigned int>(0), edgePointList.ptr<float>(1), edgePointList.cols,
                features.sizes.ptr<int>(), maxSize, (float)xi, (float)angleEpsilon, levels, make_float2((float)center.x, (float)center.y), (float)maxDist);
        }
    }

    void GHT_Guil_Full::calcOrientation()
    {
        using namespace cv::gpu::cudev::ght;

        const double iAngleStep = 1.0 / angleStep;
        const int angleRange = cvCeil((maxAngle - minAngle) * iAngleStep);

        hist.setTo(Scalar::all(0));
        Guil_Full_calcOHist_gpu(templFeatures.sizes.ptr<int>(), imageFeatures.sizes.ptr<int>(0), hist.ptr<int>(),
                                (float)minAngle, (float)maxAngle, (float)angleStep, angleRange, levels, templFeatures.maxSize);
        cudaSafeCall( cudaMemcpy(&h_buf[0], hist.data, h_buf.size() * sizeof(int), cudaMemcpyDeviceToHost) );

        angles.clear();

        for (int n = 0; n < angleRange; ++n)
        {
            if (h_buf[n] >= angleThresh)
            {
                const double angle = minAngle + n * angleStep;
                angles.push_back(std::make_pair(angle, h_buf[n]));
            }
        }
    }

    void GHT_Guil_Full::calcScale(double angle)
    {
        using namespace cv::gpu::cudev::ght;

        const double iScaleStep = 1.0 / scaleStep;
        const int scaleRange = cvCeil((maxScale - minScale) * iScaleStep);

        hist.setTo(Scalar::all(0));
        Guil_Full_calcSHist_gpu(templFeatures.sizes.ptr<int>(), imageFeatures.sizes.ptr<int>(0), hist.ptr<int>(),
                                (float)angle, (float)angleEpsilon, (float)minScale, (float)maxScale,
                                (float)iScaleStep, scaleRange, levels, templFeatures.maxSize);
        cudaSafeCall( cudaMemcpy(&h_buf[0], hist.data, h_buf.size() * sizeof(int), cudaMemcpyDeviceToHost) );

        scales.clear();

        for (int s = 0; s < scaleRange; ++s)
        {
            if (h_buf[s] >= scaleThresh)
            {
                const double scale = minScale + s * scaleStep;
                scales.push_back(std::make_pair(scale, h_buf[s]));
            }
        }
    }

    void GHT_Guil_Full::calcPosition(double angle, int angleVotes, double scale, int scaleVotes)
    {
        using namespace cv::gpu::cudev::ght;

        hist.setTo(Scalar::all(0));
        Guil_Full_calcPHist_gpu(templFeatures.sizes.ptr<int>(), imageFeatures.sizes.ptr<int>(0), hist,
                                (float)angle, (float)angleEpsilon, (float)scale, (float)dp, levels, templFeatures.maxSize);

        posCount = Guil_Full_findPosInHist_gpu(hist, outBuf.ptr<float4>(0), outBuf.ptr<int3>(1),
                                               posCount, maxSize, (float)angle, angleVotes,
                                               (float)scale, scaleVotes, (float)dp, posThresh);
    }
}

Ptr<gpu::GeneralizedHough> cv::gpu::GeneralizedHough::create(int method)
{
    switch (method)
    {
    case cv::GeneralizedHough::GHT_POSITION:
        CV_Assert( !GHT_Ballard_Pos_info_auto.name().empty() );
        return new GHT_Ballard_Pos();

    case (cv::GeneralizedHough::GHT_POSITION | cv::GeneralizedHough::GHT_SCALE):
        CV_Assert( !GHT_Ballard_PosScale_info_auto.name().empty() );
        return new GHT_Ballard_PosScale();

    case (cv::GeneralizedHough::GHT_POSITION | cv::GeneralizedHough::GHT_ROTATION):
        CV_Assert( !GHT_Ballard_PosRotation_info_auto.name().empty() );
        return new GHT_Ballard_PosRotation();

    case (cv::GeneralizedHough::GHT_POSITION | cv::GeneralizedHough::GHT_SCALE | cv::GeneralizedHough::GHT_ROTATION):
        CV_Assert( !GHT_Guil_Full_info_auto.name().empty() );
        return new GHT_Guil_Full();

    default:
        CV_Error(Error::StsBadArg, "Unsupported method");
        return Ptr<GeneralizedHough>();
    }
}

#endif /* !defined (HAVE_CUDA) */
