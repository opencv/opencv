// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Demonstrates the point-cloud processing pipeline added in the ptcloud module:
// outlier removal, normal estimation/orientation, ball-pivoting meshing and
// bounding-box estimation, visualized with cv::viz3d.

#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/geometry.hpp>   // cv::normalEstimate
#include <opencv2/ptcloud.hpp>

using namespace cv;

// Evenly sampled unit sphere (Fibonacci lattice) with a few injected far outliers.
// Returned as N x 1, CV_32FC3 - the layout the ptcloud functions expect.
static Mat makeNoisySphere(int surfacePoints, int outliers)
{
    std::vector<Point3f> pts;
    pts.reserve(surfacePoints + outliers);

    const float ga = (float)(CV_PI * (3.0 - std::sqrt(5.0)));
    for (int i = 0; i < surfacePoints; ++i)
    {
        float z = 1.f - 2.f * (i + 0.5f) / surfacePoints;
        float r = std::sqrt(std::max(0.f, 1.f - z * z));
        float t = ga * i;
        pts.emplace_back(r * std::cos(t), r * std::sin(t), z);
    }

    RNG rng(12345);
    for (int i = 0; i < outliers; ++i)
        pts.emplace_back((float)rng.uniform(-3.0, 3.0),
                         (float)rng.uniform(-3.0, 3.0),
                         (float)rng.uniform(-3.0, 3.0));

    return Mat(pts).clone();  // N x 1, CV_32FC3
}

// Attach a uniform color so the cloud can be shown with viz3d::showPoints (expects [x y z r g b]).
static Mat withColor(const Mat& cloud, const Vec3f& color)
{
    Mat xyz = cloud.reshape(1, (int)cloud.total());                 // N x 3, CV_32F
    Mat rgb(xyz.rows, 1, CV_32FC3, Scalar(color[0], color[1], color[2]));
    Mat out;
    hconcat(xyz, rgb.reshape(1, xyz.rows), out);
    return out;                                                     // N x 6, CV_32F
}

int main()
{
    const int surfacePoints = 3000, outliers = 150;
    Mat cloud = makeNoisySphere(surfacePoints, outliers);
    std::cout << "input cloud: " << cloud.total() << " points (" << outliers << " outliers)" << std::endl;

    // 1) Statistical outlier removal.
    Mat cleaned;
    removeStatisticalOutliers(cloud, cleaned, 20, 2.0);
    std::cout << "after statistical outlier removal: " << cleaned.total()
              << " points (removed " << cloud.total() - cleaned.total() << ")" << std::endl;

    // 2) Mean spacing and normals (estimate, then orient consistently).
    std::cout << "median spacing: " << estimateMedianSpacing(cleaned) << std::endl;

    Mat normals, curvatures;
    normalEstimate(normals, curvatures, cleaned, noArray(), 12);
    normals = normals.reshape(3, (int)cleaned.total());
    orientNormalsConsistent(cleaned, normals, 12);

    // 3) Surface reconstruction via ball pivoting.
    Mat vertices, triangles;
    createMeshBPA(cleaned, normals, vertices, triangles);
    std::cout << "ball-pivoting mesh: " << vertices.total() << " vertices, "
              << triangles.total() << " triangles" << std::endl;

    // 4) Bounding volumes.
    Mat center, axes, halfExtents;
    orientedBoundingBox3D(cleaned, center, axes, halfExtents);
    Mat sphereCenter;
    double sphereRadius = approxEnclosingSphere3D(cleaned, sphereCenter);
    std::cout << "oriented bounding box half-extents: " << halfExtents.reshape(1, 1) << std::endl;
    std::cout << "bounding sphere radius: " << sphereRadius << std::endl;

    // 5) Visualize: noisy input (red), cleaned cloud (green), reconstructed mesh.
    viz3d::showPoints("processing", "input", withColor(cloud, {1.0f, 0.2f, 0.2f}));
    viz3d::setObjectPosition("processing", "input", {-3.0f, 0.0f, 0.0f});

    viz3d::showPoints("processing", "cleaned", withColor(cleaned, {0.2f, 1.0f, 0.2f}));

    Mat meshVerts = vertices.reshape(1, (int)vertices.total());     // N x 3, CV_32F
    viz3d::showMesh("processing", "mesh", meshVerts, triangles);    // triangles is already M x 3, CV_32S
    viz3d::setObjectPosition("processing", "mesh", {3.0f, 0.0f, 0.0f});

    viz3d::setGridVisible("processing", true);

    std::cout << "Press ESC in the window to exit." << std::endl;
    while (waitKey(16) != 27)
        ;

    return 0;
}
