// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "ptcloud_utils.hpp"   // toPointVec

namespace cv {

void getPointCloudBounds(InputArray inputCloud, OutputArray minBound, OutputArray maxBound)
{
    CV_TRACE_FUNCTION();

    std::vector<Point3f> pts;
    toPointVec(inputCloud, pts);
    if (pts.empty()) { minBound.release(); maxBound.release(); return; }

    Mat p3 = Mat(pts).reshape(1, (int)pts.size());   // N x 3, CV_32F
    Mat lo, hi;
    reduce(p3, lo, 0, REDUCE_MIN);                    // 1 x 3
    reduce(p3, hi, 0, REDUCE_MAX);                    // 1 x 3
    lo.reshape(1, 3).copyTo(minBound);               // 3 x 1, CV_32F
    hi.reshape(1, 3).copyTo(maxBound);
}

void getOrientedBoundingBox(InputArray inputCloud, OutputArray center, OutputArray axes,
                            OutputArray halfExtents)
{
    CV_TRACE_FUNCTION();

    std::vector<Point3f> pts;
    toPointVec(inputCloud, pts);
    const int N = (int)pts.size();
    if (N == 0) { center.release(); axes.release(); halfExtents.release(); return; }

    Mat p3 = Mat(pts).reshape(1, N);                  // N x 3, CV_32F

    if (N < 3)   // too few points for PCA: return an axis-aligned box (identity axes)
    {
        Mat lo, hi; reduce(p3, lo, 0, REDUCE_MIN); reduce(p3, hi, 0, REDUCE_MAX);
        Mat cc = (hi + lo) * 0.5, hh = (hi - lo) * 0.5;
        Mat eye3 = Mat::eye(3, 3, CV_32F);
        cc.reshape(1, 3).copyTo(center);
        eye3.copyTo(axes);
        hh.reshape(1, 3).copyTo(halfExtents);
        return;
    }

    // Principal axes of the cloud (each row of evec is a unit axis, descending variance).
    Mat mean, evec;
    PCACompute(p3, mean, evec);
    mean.convertTo(mean, CV_32F);     // PCACompute may return CV_64F; keep the math in CV_32F
    evec.convertTo(evec, CV_32F);

    // Project points onto the axes, then take the per-axis extent.
    Mat proj;
    PCAProject(p3, mean, evec, proj);                 // N x 3, column k = coord on axis k
    Mat mn, mx;
    reduce(proj, mn, 0, REDUCE_MIN);                  // 1 x 3
    reduce(proj, mx, 0, REDUCE_MAX);

    Mat he = (mx - mn) * 0.5;                          // half-extent per axis (1 x 3)
    Mat cp = (mn + mx) * 0.5;                          // box center in axis coords (1 x 3)
    Mat c3 = mean + cp * evec;                         // box center in world coords (1 x 3)

    c3.reshape(1, 3).copyTo(center);                  // 3 x 1
    evec.copyTo(axes);                                // 3 x 3, each row a unit axis
    he.reshape(1, 3).copyTo(halfExtents);             // 3 x 1
}

double getBoundingSphere(InputArray inputCloud, OutputArray center)
{
    CV_TRACE_FUNCTION();

    std::vector<Point3f> pts;
    toPointVec(inputCloud, pts);
    const int N = (int)pts.size();
    if (N == 0) { center.release(); return 0.0; }

    // Ritter's approximation: seed from a far-apart pair, then grow to include every point.
    auto sq = [](const Point3f& a, const Point3f& b)
    {
        float dx = a.x-b.x, dy = a.y-b.y, dz = a.z-b.z;
        return dx*dx + dy*dy + dz*dz;
    };
    auto farthest = [&](const Point3f& q)
    {
        int bi = 0; float bd = -1.f;
        for (int i = 0; i < N; i++) { float d = sq(pts[i], q); if (d > bd) { bd = d; bi = i; } }
        return bi;
    };

    Point3f y = pts[farthest(pts[0])];
    Point3f z = pts[farthest(y)];
    Point3f c((y.x+z.x)*0.5f, (y.y+z.y)*0.5f, (y.z+z.z)*0.5f);
    float r = std::sqrt(sq(y, z)) * 0.5f;

    for (int i = 0; i < N; i++)
    {
        float d = std::sqrt(sq(pts[i], c));
        if (d > r)
        {
            float nr = (r + d) * 0.5f;
            float t = (nr - r) / d;                    // move center toward the outlier
            c.x += (pts[i].x - c.x) * t;
            c.y += (pts[i].y - c.y) * t;
            c.z += (pts[i].z - c.z) * t;
            r = nr;
        }
    }

    Mat(Matx31f(c.x, c.y, c.z)).copyTo(center);        // 3 x 1, CV_32F
    return (double)r;
}

} // namespace cv
