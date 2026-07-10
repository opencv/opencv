// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"
#include "opencv2/geometry/segment.hpp"   // cv::normalEstimate

namespace opencv_test { namespace {

using namespace perf;

static Mat makeSphere(int n)
{
    std::vector<Point3f> pts;
    pts.reserve(n);
    const float ga = (float)(CV_PI * (3.0 - std::sqrt(5.0)));
    for (int i = 0; i < n; i++)
    {
        float z = 1.f - 2.f * (i + 0.5f) / n;
        float r = std::sqrt(std::max(0.f, 1.f - z * z));
        float t = ga * i;
        pts.emplace_back(r * std::cos(t), r * std::sin(t), z);
    }
    return Mat(pts).clone();
}

typedef TestBaseWithParam<int> Ptcloud_BpaPerf;

PERF_TEST_P(Ptcloud_BpaPerf, createMeshBPA, testing::Values(20000, 60000))
{
    Mat cloud = makeSphere(GetParam());
    Mat normals, curv;
    normalEstimate(normals, curv, cloud, noArray(), 15);
    normals = normals.reshape(3, (int)cloud.total());
    orientNormalsConsistent(cloud, normals, 15);

    Mat vertices, triangles;
    TEST_CYCLE() createMeshBPA(cloud, normals, vertices, triangles);
    SANITY_CHECK_NOTHING();
}

}} // namespace
