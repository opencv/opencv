// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "ptcloud_utils.hpp"             // toPointVec
#include "opencv2/flann.hpp"             // cv::flann::Index (nearest-neighbor indices)
#include "opencv2/geometry/mst.hpp"      // cv::buildMST, cv::MSTEdge

#include <deque>

namespace cv {

// Normals come from cv::normalEstimate (geometry); these functions orient them.

void orientNormals(InputArray inputCloud, InputOutputArray normals, const Point3f& viewpoint)
{
    CV_TRACE_FUNCTION();

    std::vector<Point3f> points;
    toPointVec(inputCloud, points);
    const int N = (int)points.size();
    if (N == 0) return;

    Mat nm = normals.getMat();
    CV_Assert(!nm.empty() && nm.channels() * (int)nm.total() == 3 * N);
    // reshape() needs contiguous data; clone a non-contiguous view and copy back at the end.
    Mat work = nm.isContinuous() ? nm : nm.clone();
    Mat nmf = work.reshape(3, N);   // N x 1, CV_32FC3

    // Flip each normal, if needed, so it points towards the viewpoint (independent per point).
    parallel_for_(Range(0, N), [&](const Range& r)
    {
        for (int i = r.start; i < r.end; i++)
        {
            Vec3f n = nmf.at<Vec3f>(i);
            Vec3f toView(viewpoint.x - points[i].x, viewpoint.y - points[i].y, viewpoint.z - points[i].z);
            if (n.dot(toView) < 0.f)
                nmf.at<Vec3f>(i) = -n;
        }
    });

    if (work.data != nm.data)
        work.copyTo(nm);   // matching shape/type; copyTo handles a non-contiguous destination
}

void orientNormalsConsistent(InputArray inputCloud, InputOutputArray normals, int k)
{
    CV_TRACE_FUNCTION();
    CV_Assert(k >= 2);

    std::vector<Point3f> points;
    toPointVec(inputCloud, points);
    const int N = (int)points.size();
    if (N == 0) return;

    Mat nm = normals.getMat();
    CV_Assert(!nm.empty() && nm.channels() * (int)nm.total() == 3 * N);
    // reshape() needs contiguous data; clone a non-contiguous view and copy back at the end.
    Mat work = nm.isContinuous() ? nm : nm.clone();
    Mat nmf = work.reshape(3, N);

    std::vector<Vec3f> n(N);
    for (int i = 0; i < N; i++)
    {
        Vec3f v = nmf.at<Vec3f>(i);
        float len = (float)cv::norm(v);
        n[i] = (len > 1e-12f) ? v * (1.f / len) : Vec3f(0.f, 0.f, 1.f);   // guard degenerate normals
    }

    // Riemannian graph over kNN, weight 1 - |n_i . n_j| (Hoppe et al. 1992).
    Mat pts = Mat(points).reshape(1, N);
    const int kk = std::min(k + 1, N);
    flann::Index index(pts, flann::KDTreeIndexParams(4));
    Mat nnIdx, nnDist;
    index.knnSearch(pts, nnIdx, nnDist, kk);

    std::vector<MSTEdge> edges;
    edges.reserve((size_t)N * kk);
    for (int i = 0; i < N; i++)
        for (int c = 1; c < kk; c++)                  // skip self at column 0
        {
            int j = nnIdx.at<int>(i, c);
            if (j >= 0 && j != i)
                edges.push_back(MSTEdge{ i, j, 1.0 - std::fabs((double)n[i].dot(n[j])) });
        }

    // Seed the +Z-extreme point outward; orientation propagates along the MST.
    int seed = 0;
    for (int i = 1; i < N; i++)
        if (points[i].z > points[seed].z) seed = i;
    if (n[seed].dot(Vec3f(0.f, 0.f, 1.f)) < 0.f) n[seed] = -n[seed];

    std::vector<MSTEdge> mst;
    if (buildMST(N, edges, mst, MST_KRUSKAL, seed))
    {
        std::vector<std::vector<int>> adj(N);
        for (const MSTEdge& e : mst)
        {
            adj[e.source].push_back(e.target);
            adj[e.target].push_back(e.source);
        }
        std::vector<char> visited(N, 0);
        std::deque<int> q;
        q.push_back(seed); visited[seed] = 1;
        while (!q.empty())
        {
            int a = q.front(); q.pop_front();
            for (int b : adj[a])
                if (!visited[b])
                {
                    visited[b] = 1;
                    if (n[a].dot(n[b]) < 0.f) n[b] = -n[b];
                    q.push_back(b);
                }
        }
    }
    else
    {
        // Disconnected graph: fall back to outward-about-centroid (deterministic).
        Point3f c(0.f, 0.f, 0.f);
        for (const Point3f& p : points) c += p;
        c *= 1.0f / N;
        for (int i = 0; i < N; i++)
        {
            Vec3f out(points[i].x - c.x, points[i].y - c.y, points[i].z - c.z);
            if (n[i].dot(out) < 0.f) n[i] = -n[i];
        }
    }

    for (int i = 0; i < N; i++)
        nmf.at<Vec3f>(i) = n[i];

    if (work.data != nm.data)
        work.copyTo(nm);   // matching shape/type; copyTo handles a non-contiguous destination
}

} // namespace cv
