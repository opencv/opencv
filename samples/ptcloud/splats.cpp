// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ptcloud.hpp>

#include <algorithm>
#include <cfloat>
#include <iostream>

using namespace cv;

int main(int argc, char** argv)
{
    const char* keys =
        "{ help h    |       | print this message }"
        "{ @splats   |       | path to a trained 3D Gaussian Splatting .ply or .splat file }"
        "{ points    | false | also show the splat centers as an opaque point cloud }"
        "{ raw       | false | keep the file's world coordinates instead of recentering }"
        "{ flip      | true  | rotate 180 deg about X, for the COLMAP Y-down convention }"
        "{ fit       | 8.0   | half-extent to scale the scene to when recentering }"
        "{ size      | 0.5   | splat sigma as a multiple of point spacing, for plain point clouds }";

    CommandLineParser parser(argc, argv, keys);
    parser.about("Renders a 3D Gaussian Splatting scene with cv::viz3d::showSplats.");

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    if (parser.has("help") || !parser.has("@splats"))
    {
        parser.printMessage();
        std::cout << "\nA trained 3DGS scene is not bundled with OpenCV; pass your own, e.g.\n"
                  << "  ./example_ptcloud_splats /path/to/point_cloud.ply\n"
                  << "  ./example_ptcloud_splats /path/to/scene.splat\n"
                  << "A plain colored point cloud also works, rendered as isotropic splats:\n"
                  << "  ./example_ptcloud_splats ../samples/data/anthidium-forcipatum.ply\n";
        return 0;
    }

    const String filename = parser.get<String>("@splats");

    Mat splats;
    loadGaussianSplats(filename, splats);
    const bool trained = !splats.empty();

    if (!trained)
    {
        Mat verts, rgb;
        loadPointCloud(filename, verts, noArray(), rgb);
        if (verts.empty())
        {
            std::cerr << "Failed to load '" << filename << "' as splats or as a point cloud.\n";
            return 1;
        }

        verts = verts.reshape(1, verts.rows);
        if (!rgb.empty())
            rgb = rgb.reshape(1, rgb.rows);

        Vec3f plo(FLT_MAX, FLT_MAX, FLT_MAX), phi(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        for (int i = 0; i < verts.rows; i++)
            for (int k = 0; k < 3; k++)
            {
                plo[k] = std::min(plo[k], verts.at<float>(i, k));
                phi[k] = std::max(phi[k], verts.at<float>(i, k));
            }

        // Scanned clouds sample a surface, not a volume, so spacing goes as sqrt(area/N).
        Vec3f ext;
        for (int k = 0; k < 3; k++)
            ext[k] = std::max(1e-6f, phi[k] - plo[k]);
        double area = 2.0 * (ext[0] * ext[1] + ext[1] * ext[2] + ext[2] * ext[0]);
        float sigma = parser.get<float>("size")
                    * (float)std::sqrt(area / std::max(1, verts.rows));

        splats = Mat::zeros(verts.rows, 13, CV_32F);
        for (int i = 0; i < verts.rows; i++)
        {
            float* d = splats.ptr<float>(i);
            for (int k = 0; k < 3; k++)
                d[k] = verts.at<float>(i, k);
            d[3] = d[6] = d[8] = sigma * sigma;
            for (int k = 0; k < 3; k++)
                d[9 + k] = rgb.empty() ? 0.8f : rgb.at<float>(i, k);
            d[12] = 1.0f;
        }

        std::cout << "'" << filename << "' is not a trained 3DGS file; loaded it as a point cloud "
                  << "and synthesized isotropic splats with sigma " << sigma
                  << " (tune with --size)" << std::endl;
    }

    std::cout << "Loaded " << splats.rows << " splats from " << filename << std::endl;

    if (trained && parser.get<bool>("flip"))
    {
        for (int i = 0; i < splats.rows; i++)
        {
            float* p = splats.ptr<float>(i);
            p[1] = -p[1];
            p[2] = -p[2];
            p[4] = -p[4];
            p[5] = -p[5];
        }
        std::cout << "rotated 180 degrees about X for the COLMAP Y-down convention"
                  << " (pass --flip=false to disable)" << std::endl;
    }

    Vec3f lo(FLT_MAX, FLT_MAX, FLT_MAX), hi(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (int i = 0; i < splats.rows; i++)
    {
        const float* p = splats.ptr<float>(i);
        for (int k = 0; k < 3; k++)
        {
            lo[k] = std::min(lo[k], p[k]);
            hi[k] = std::max(hi[k], p[k]);
        }
    }

    std::cout << "bounds  min [" << lo[0] << ", " << lo[1] << ", " << lo[2] << "]"
              << "  max [" << hi[0] << ", " << hi[1] << ", " << hi[2] << "]" << std::endl;

    if (!parser.get<bool>("raw"))
    {
        std::vector<float> axis(splats.rows);
        Vec3f med(0.f, 0.f, 0.f);
        for (int k = 0; k < 3; k++)
        {
            for (int i = 0; i < splats.rows; i++)
                axis[i] = splats.ptr<float>(i)[k];
            std::nth_element(axis.begin(), axis.begin() + axis.size() / 2, axis.end());
            med[k] = axis[axis.size() / 2];
        }

        std::vector<float> rad(splats.rows);
        for (int i = 0; i < splats.rows; i++)
        {
            const float* p = splats.ptr<float>(i);
            rad[i] = std::max(std::max(std::abs(p[0] - med[0]), std::abs(p[1] - med[1])),
                              std::abs(p[2] - med[2]));
        }
        size_t q = (size_t)(rad.size() * 0.9);
        std::nth_element(rad.begin(), rad.begin() + q, rad.end());
        float extent = rad[q];

        std::cout << "median [" << med[0] << ", " << med[1] << ", " << med[2]
                  << "]  p90 half-extent " << extent << std::endl;

        float scale = (extent > 0.0f) ? parser.get<float>("fit") / extent : 1.0f;

        for (int i = 0; i < splats.rows; i++)
        {
            float* p = splats.ptr<float>(i);
            for (int k = 0; k < 3; k++)
                p[k] = (p[k] - med[k]) * scale;
            for (int k = 0; k < 6; k++)
                p[3 + k] *= scale * scale;
        }

        std::cout << "recentered on the origin and scaled by " << scale
                  << " (pass --raw to disable)" << std::endl;
    }

    viz3d::setPerspective("splats", 1.3f, 0.01f, 1000.0f);
    viz3d::showSplats("splats", "scene", splats);

    if (parser.get<bool>("points"))
    {
        Mat centers(splats.rows, 6, CV_32F);
        splats.colRange(0, 3).copyTo(centers.colRange(0, 3));
        splats.colRange(9, 12).copyTo(centers.colRange(3, 6));
        viz3d::showPoints("splats", "centers", centers);
    }

    std::cout << "Drag with the left mouse button to orbit, right to pan, wheel to zoom. "
              << "Press Esc to quit." << std::endl;

    while (waitKey(16) != 27)
        ;

    return 0;
}
