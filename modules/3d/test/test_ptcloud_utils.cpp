// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021, Wanli Zhong <zhongwl2018@mail.sustech.edu.cn>

#include "test_ptcloud_utils.hpp"

namespace opencv_test {

void generatePlane(OutputArray plane_pts, const vector<float> &model, float thr, int num,
        const vector<float> &limit)
{
    if (plane_pts.channels() == 3 && plane_pts.isVector())
    {
        // std::vector<cv::Point3f>
        plane_pts.create(1, num, CV_32FC3);
    }
    else
    {
        // cv::Mat
        plane_pts.create(num, 3, CV_32F);
    }

    cv::RNG rng(0);
    auto *plane_pts_ptr = (float *) plane_pts.getMat().data;

    // Part of the points are generated for the specific model
    // The other part of the points are used to increase the thickness of the plane
    int std_num = (int) (num / 2);
    // Difference of maximum d between two parallel planes
    float d_thr = thr * sqrt(model[0] * model[0] + model[1] * model[1] + model[2] * model[2]);

    for (int i = 0; i < num; i++)
    {
        // Let d change then generate thickness
        float d = i < std_num ? model[3] : rng.uniform(model[3] - d_thr, model[3] + d_thr);
        float x, y, z;
        // c is 0 means the plane is vertical
        if (model[2] == 0)
        {
            z = rng.uniform(limit[4], limit[5]);
            if (model[0] == 0)
            {
                x = rng.uniform(limit[0], limit[1]);
                y = -d / model[1];
            }
            else if (model[1] == 0)
            {
                x = -d / model[0];
                y = rng.uniform(limit[2], limit[3]);
            }
            else
            {
                x = rng.uniform(limit[0], limit[1]);
                y = -(model[0] * x + d) / model[1];
            }
        }
            // c is not 0
        else
        {
            x = rng.uniform(limit[0], limit[1]);
            y = rng.uniform(limit[2], limit[3]);
            z = -(model[0] * x + model[1] * y + d) / model[2];
        }

        plane_pts_ptr[3 * i] = x;
        plane_pts_ptr[3 * i + 1] = y;
        plane_pts_ptr[3 * i + 2] = z;
    }
}

void generateSphere(OutputArray sphere_pts, const vector<float> &model, float thr, int num,
        const vector<float> &limit)
{
    if (sphere_pts.channels() == 3 && sphere_pts.isVector())
    {
        // std::vector<cv::Point3f>
        sphere_pts.create(1, num, CV_32FC3);
    }
    else
    {
        // cv::Mat
        sphere_pts.create(num, 3, CV_32F);
    }
    cv::RNG rng(0);
    auto *sphere_pts_ptr = (float *) sphere_pts.getMat().data;

    // Part of the points are generated for the specific model
    // The other part of the points are used to increase the thickness of the sphere
    int sphere_num = (int) (num / 1.5);
    for (int i = 0; i < num; i++)
    {
        // Let r change then generate thickness
        float r = i < sphere_num ? model[3] : rng.uniform(model[3] - thr, model[3] + thr);
        // Generate a random vector and normalize it.
        // Note: these vectors are not spread uniformly across the sphere
        Vec3f vec(rng.uniform(limit[0], limit[1]), rng.uniform(limit[2], limit[3]),
                  rng.uniform(limit[4], limit[5]));
        float l = sqrt(vec.dot(vec));
        // Normalizes it to have a magnitude of r
        vec /= l / r;

        sphere_pts_ptr[3 * i] = model[0] + vec[0];
        sphere_pts_ptr[3 * i + 1] = model[1] + vec[1];
        sphere_pts_ptr[3 * i + 2] = model[2] + vec[2];
    }
}

} // opencv_test
