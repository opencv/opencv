// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

namespace cv {

// Read an input point cloud (CV_32FC3, or Nx3 / 3xN CV_32F) into a vector<Point3f>.
static void toPointVec(InputArray inputCloud, std::vector<Point3f>& points)
{
    points.clear();
    Mat m = inputCloud.getMat();
    if (m.empty())
        return;

    Mat mf;
    if (m.depth() != CV_32F)
        m.convertTo(mf, CV_32F);
    else
        mf = m;

    // getMat() may hand back a non-contiguous view (ROI / column-slice / transpose);
    // reshape() below requires contiguous data.
    if (!mf.isContinuous())
        mf = mf.clone();

    if (mf.channels() == 1)
    {
        // Accept Nx3 (or 3xN) single-channel layouts as well.
        CV_Assert(mf.cols == 3 || mf.rows == 3);
        if (mf.cols != 3 && mf.rows == 3)
            mf = mf.t();
        mf = mf.reshape(3);            // N x 1, CV_32FC3
    }
    CV_Assert(mf.channels() == 3);

    mf = mf.reshape(3, (int)mf.total());   // guarantee N x 1, CV_32FC3
    points.assign(mf.begin<Point3f>(), mf.end<Point3f>());
}

// Build an octree over the points with a resolution derived from the bounding box.
static Ptr<Octree> buildOctree(const std::vector<Point3f>& points)
{
    Mat p3 = Mat(points).reshape(1, (int)points.size());   // Nx3 CV_32F
    Mat lo, hi;
    reduce(p3, lo, 0, REDUCE_MIN);
    reduce(p3, hi, 0, REDUCE_MAX);
    double diag = norm(hi - lo);
    double resolution = std::max(diag / 256.0, 1e-6);
    return Octree::createWithResolution(resolution, points);
}

// Emit the kept subset into the outputs.
static void emit(const std::vector<Point3f>& points, const std::vector<int>& keptIdx,
                 OutputArray outputCloud, OutputArray keptIndices)
{
    std::vector<Point3f> kept;
    kept.reserve(keptIdx.size());
    for (int idx : keptIdx)
        kept.push_back(points[idx]);
    Mat(kept).copyTo(outputCloud);     // N x 1, CV_32FC3
    if (keptIndices.needed())
        Mat(keptIdx).copyTo(keptIndices);
}

void removeStatisticalOutliers(InputArray inputCloud, OutputArray outputCloud,
                               int meanK, double stddevMulThresh, OutputArray keptIndices)
{
    CV_TRACE_FUNCTION();
    CV_Assert(meanK > 0);

    std::vector<Point3f> points;
    toPointVec(inputCloud, points);
    const int N = (int)points.size();
    if (N == 0) { outputCloud.release(); if (keptIndices.needed()) keptIndices.release(); return; }

    Ptr<Octree> tree = buildOctree(points);

    // Pass 1: mean distance to the meanK nearest neighbors of each point (parallel;
    // the octree searches are const and touch no shared state, so this is safe).
    std::vector<double> meanDist(N, 0.0);
    parallel_for_(Range(0, N), [&](const Range& range)
    {
        std::vector<float> sqDists;   // reused across the chunk to avoid per-point allocation
        for (int i = range.start; i < range.end; i++)
        {
            tree->KNNSearch(points[i], meanK + 1, noArray(), sqDists);   // near->far, [0] is self
            double sum = 0.0; int cnt = 0;
            for (int j = 1; j < (int)sqDists.size(); j++)                // skip self
            {
                sum += std::sqrt((double)sqDists[j]);
                cnt++;
            }
            meanDist[i] = cnt ? sum / cnt : 0.0;
        }
    });

    // Pass 2: threshold at global_mean + stddevMulThresh * global_stddev.
    Scalar mu, sigma;
    meanStdDev(Mat(meanDist), mu, sigma);
    const double threshold = mu[0] + stddevMulThresh * sigma[0];

    std::vector<int> keptIdx;
    keptIdx.reserve(N);
    for (int i = 0; i < N; i++)
        if (meanDist[i] <= threshold)
            keptIdx.push_back(i);

    emit(points, keptIdx, outputCloud, keptIndices);
}

void removeRadiusOutliers(InputArray inputCloud, OutputArray outputCloud,
                          double radius, int minNeighbors, OutputArray keptIndices)
{
    CV_TRACE_FUNCTION();
    CV_Assert(radius > 0.0 && minNeighbors >= 0);

    std::vector<Point3f> points;
    toPointVec(inputCloud, points);
    const int N = (int)points.size();
    if (N == 0) { outputCloud.release(); if (keptIndices.needed()) keptIndices.release(); return; }

    Ptr<Octree> tree = buildOctree(points);

    // Count neighbors per point (parallel), then collect kept indices in order.
    std::vector<uchar> keep(N, 0);
    parallel_for_(Range(0, N), [&](const Range& range)
    {
        for (int i = range.start; i < range.end; i++)
        {
            // radiusNNSearch includes the query point itself, so subtract 1 for the neighbor count.
            int found = tree->radiusNNSearch(points[i], (float)radius, noArray());
            keep[i] = (std::max(found - 1, 0) >= minNeighbors) ? (uchar)1 : (uchar)0;
        }
    });

    std::vector<int> keptIdx;
    keptIdx.reserve(N);
    for (int i = 0; i < N; i++)
        if (keep[i])
            keptIdx.push_back(i);

    emit(points, keptIdx, outputCloud, keptIndices);
}

} // namespace cv
