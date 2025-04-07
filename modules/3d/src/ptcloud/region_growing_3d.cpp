// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022, Wanli Zhong <zhongwl2018@mail.sustech.edu.cn>

#include "../precomp.hpp"
#include "region_growing_3d.hpp"

#include <queue>
#include <numeric>

namespace cv {
//    namespace _3d {

/////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////  RegionGrowing3DImpl  //////////////////////////////////////

int
RegionGrowing3DImpl::segment(OutputArrayOfArrays regions_idx_, OutputArray labels_,
        InputArray input_pts_, InputArray normals_, InputArrayOfArrays nn_idx_)
{
    Mat ori_pts, normals;
    getPointsMatFromInputArray(input_pts_, ori_pts, 0);
    getPointsMatFromInputArray(normals_, normals, 0);

    std::vector<Mat> nn_idx;
    nn_idx_.getMatVector(nn_idx);

    int pts_size = ori_pts.rows;
    bool has_curvatures = !curvatures.empty();
    bool has_seeds = !seeds.empty();

    CV_CheckGE(max_size, min_size,
            "The maximum size should be grater than or equal to the minimum size.");
    CV_CheckEQ(pts_size, (int) nn_idx.size(),
            "The point number of NN search result should be equal to the size of the point cloud.");
    CV_CheckEQ(pts_size, normals.rows,
            "The number of points in the normals should be equal to that in the point cloud.");
    if (has_curvatures)
    {
        CV_CheckEQ(pts_size, curvatures.cols,
                "The number of points in the curvatures should be equal to that in the point cloud.");
    }
    if (has_seeds)
    {
        CV_CheckGE(pts_size, seeds.cols,
                "The number of seed should be less than or equal to the number of points in the point cloud.");
    }

    if (!has_seeds && !has_curvatures)
    {
        // If the user does not set the seeds and curvatures, use the natural order of the points
        std::vector<int> *natural_order_seeds = new std::vector<int>(pts_size);
        std::iota(natural_order_seeds->begin(), natural_order_seeds->end(), 0);
        seeds = Mat(*natural_order_seeds);
    }
    else if (!has_seeds && has_curvatures)
    {
        // If the user sets the curvatures without setting the seeds, seeds will be sorted in
        // ascending order of curvatures
        sortIdx(curvatures, seeds, SORT_EVERY_ROW + SORT_ASCENDING);
    }

    const int *seeds_ptr = (int *) seeds.data;
    const float *curvatures_ptr = (float *) curvatures.data;

    int seeds_num = seeds.cols, flag = 1;
    double cos_smoothness_thr = std::cos(smoothness_thr);
    double cos_smoothness_thr_square = cos_smoothness_thr * cos_smoothness_thr;
    // Used to determine if the point can be grown
    std::vector<bool> has_grown(pts_size, false);
    // Store the indexes of the points in each region
    std::vector<std::vector<int>> regions_idx;
    for (int i = 0; i < seeds_num; ++i)
    {
        int cur_seed = seeds_ptr[i];

        // 1. If the number of regions is satisfied then stop running.
        // 2. Filter out seeds with curvature greater than the threshold.
        //    If no seed points have been set, it will use seeds sorted in ascending order of curvatures.
        //    When the curvature doesn't satisfy the threshold, the seed points that follow will not satisfy the threshold either.
        if (flag > region_num ||
            (!has_seeds && has_curvatures && curvatures_ptr[cur_seed] > curvature_thr))
        {
            break;
        }
        // 1. If current seed has been grown then grow the next one.
        // 2. Filter out seeds with curvature greater than the threshold.
        if (has_grown[cur_seed] ||
            (has_seeds && has_curvatures && curvatures_ptr[cur_seed] > curvature_thr))
        {
            continue;
        }

        Mat base_normal;
        if (!smooth_mode) base_normal = normals.row(cur_seed);

        has_grown[cur_seed] = true;
        std::vector<int> region = {cur_seed};
        std::queue<int> grow_list;
        grow_list.push(cur_seed);
        while (!grow_list.empty())
        {
            int cur_idx = grow_list.front();
            grow_list.pop();
            if (smooth_mode) base_normal = normals.row(cur_idx);

            // The maximum size that may be used for this row
            int bound = max_neighbor_num > nn_idx[cur_idx].cols ? nn_idx[cur_idx].cols
                                                                : max_neighbor_num;
            const int *nn_idx_ptr_base = (int *) nn_idx[cur_idx].data;
            // Start from index 1 because the first one of knn_idx is itself
            for (int j = 1; j < bound; ++j)
            {
                int cur_nei_idx = nn_idx_ptr_base[j];
                // If the index is less than 0,
                // the nn_idx of this row will no longer have any information.
                if (cur_nei_idx < 0) break;
                // 1. If current point has been grown then grow the next one
                // 2. Filter out points with curvature greater than the threshold
                if (has_grown[cur_nei_idx] ||
                    (has_curvatures && curvatures_ptr[cur_nei_idx] > curvature_thr))
                {
                    continue;
                }

                Mat cur_normal = normals.row(cur_nei_idx);
                double n12 = base_normal.dot(cur_normal);
                double n1m = base_normal.dot(base_normal);
                double n2m = cur_normal.dot(cur_normal);
                // If the smoothness threshold is satisfied, this neighbor will be pushed to the growth list
                if (n12 * n12 / (n1m * n2m) >= cos_smoothness_thr_square)
                {
                    has_grown[cur_nei_idx] = true;
                    region.emplace_back(cur_nei_idx);
                    grow_list.push(cur_nei_idx);
                }
            }
        }

        int region_size = (int) region.size();
        if (min_size <= region_size && region_size <= max_size)
        {
            regions_idx.emplace_back(Mat(region));
            flag++;
        }
        else if (region_size > max_size)
        {
            break;
        }
    }

    if (need_sort)
    {
        // Compare the size of two regions in descending order
        auto compareRegionSize = [](const std::vector<int> &r1,
                const std::vector<int> &r2) -> bool
        {
            return r1.size() > r2.size();
        };
        sort(regions_idx.begin(), regions_idx.end(), compareRegionSize);
    }

    std::vector<std::vector<int>> &_regions_idx = *regions_idx_.getObj<std::vector<std::vector<int>>>();
    _regions_idx.resize(regions_idx.size());
    Mat labels = Mat::zeros(pts_size, 1, CV_32S);
    int *labels_ptr = (int *) labels.data;
    for (int i = 0; i < (int) regions_idx.size(); ++i)
    {
        Mat(1, (int) regions_idx[i].size(), CV_32S, regions_idx[i].data()).copyTo(_regions_idx[i]);
        for (int j: regions_idx[i])
        {
            labels_ptr[j] = i + 1;
        }
    }
    labels.copyTo(labels_);

    return flag - 1;
}

//    } // _3d::
}  // cv::
