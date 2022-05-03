// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022, Wanli Zhong <zhongwl2018@mail.sustech.edu.cn>

#include "../precomp.hpp"
#include "region_growing_3d.hpp"
#include "opencv2/3d/ptcloud.hpp"

#include <queue>
#include <numeric>

namespace cv {
//    namespace _3d {

/////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////  RegionGrowing3DImpl  //////////////////////////////////////

int
RegionGrowing3DImpl::segment(OutputArray labels, InputArray input_pts_, InputArray normals_,
        InputArray knn_idx_)
{
    Mat ori_pts, normals, knn_idx;
    getPointsMatFromInputArray(input_pts_, ori_pts, 0);
    getPointsMatFromInputArray(normals_, normals, 0);
    if (knn_idx_.rows() < knn_idx_.cols())
    {
        // Convert layout of data in memory space to NxK
        transpose(knn_idx_.getMat(), knn_idx);
    }
    else
    {
        knn_idx = knn_idx_.getMat();
    }

    if (k == 0)
    {
        k = knn_idx.cols;
    }

    int pts_size = ori_pts.rows;
    bool has_curvatures = !curvatures.empty();
    bool has_seeds = !seeds.empty();

    CV_CheckGE(max_size, min_size,
            "The maximum size should be grater than or equal to the minimum size.");
    CV_CheckGE(knn_idx.cols, k,
            "The number of neighbors of KNN search result should be grater than or equal to that of region growing.");
    CV_CheckEQ(pts_size, knn_idx.rows,
            "The number of points in the KNN search result should be equal to that in the point cloud.");
    CV_CheckEQ(pts_size, max(normals.rows, normals.cols),
            "The number of points in the normals should be equal to that in the point cloud.");
    if (has_curvatures)
    {
        CV_CheckEQ(pts_size, max(curvatures.rows, curvatures.cols),
                "The number of points in the curvatures should be equal to that in the point cloud.");
    }
    if (has_seeds)
    {
        CV_CheckGE(pts_size, max(seeds.rows, seeds.cols),
                "The number of seed should be less than or equal to the number of points in the point cloud.");
    }

    std::vector<int> natural_order_seeds;
    if (!has_seeds && !has_curvatures)
    {
        // If the user does not set the seeds and curvatures, use the natural order of the points
        natural_order_seeds = std::vector<int>(pts_size);
        std::iota(natural_order_seeds.begin(), natural_order_seeds.end(), 0);
        seeds = Mat(natural_order_seeds);
    }
    else if (!has_seeds && has_curvatures)
    {
        // If the user sets the curvatures without setting the seeds, seeds will be sorted in
        // ascending order of curvatures
        sortIdx(curvatures, seeds, SORT_EVERY_COLUMN + SORT_ASCENDING);
    }

    const int *seeds_ptr = (int *) seeds.data;
    const float *curvatures_ptr = (float *) curvatures.data;

    int seeds_num = max(seeds.rows, seeds.cols);
    int flag = 1;
    double cos_smoothness_thr = std::cos(smoothness_thr);
    double cos_smoothness_thr_square = cos_smoothness_thr * cos_smoothness_thr;
    // Initialize labels to zero
    Mat _labels = Mat::zeros(pts_size, 1, CV_32S);
    for (int i = 0; i < seeds_num; i++)
    {
        int *_labels_ptr = (int *) _labels.data;
        int cur_seed = seeds_ptr[i];
        int region_size = 1;

        // If the number of regions is satisfied then stop running
        if (flag > region_num)
        {
            break;
        }
        // If current seed has been grown then grow the next one
        if (_labels_ptr[cur_seed] != 0)
        {
            continue;
        }
        // Filter out seeds with curvature greater than the threshold.
        if (has_seeds && has_curvatures && curvatures_ptr[cur_seed] > curvature_thr)
        {
            continue;
        }
        // Filter out seeds with curvature greater than the threshold
        // If no seed points have been set, it will use seeds sorted in ascending order of curvatures.
        // When the curvature doesn't satisfy the threshold, the seed points that follow will not satisfy the threshold either.
        if (!has_seeds && has_curvatures && curvatures_ptr[cur_seed] > curvature_thr)
        {
            break;
        }

        Mat base_normal;
        if (!smooth_mode)
        {
            base_normal = normals.row(cur_seed);
        }

        Mat labels_tmp;
        _labels.copyTo(labels_tmp);
        int *labels_tmp_ptr = (int *) labels_tmp.data;
        labels_tmp_ptr[cur_seed] = flag;

        std::queue<int> grow_list;
        grow_list.push(cur_seed);
        while (!grow_list.empty())
        {
            int cur_idx = grow_list.front();
            grow_list.pop();
            if (smooth_mode)
            {
                base_normal = normals.row(cur_idx);
            }

            const int *pt_knn_ptr = (int *) knn_idx.row(cur_idx).data;
            // Start from index 1 because the first one of knn_idx is itself
            for (int j = 1; j < k; j++)
            {
                int cur_neighbor_idx = pt_knn_ptr[j];
                // If current point has been grown then grow the next one
                if (labels_tmp_ptr[cur_neighbor_idx] != 0)
                {
                    continue;
                }
                // Filter out points with curvature greater than the threshold
                if (has_curvatures && curvatures_ptr[cur_neighbor_idx] > curvature_thr)
                {
                    continue;
                }

                Mat cur_normal = normals.row(cur_neighbor_idx);
                double n12 = base_normal.dot(cur_normal);
                double n1m = base_normal.dot(base_normal);
                double n2m = cur_normal.dot(cur_normal);
                // If the smoothness threshold is satisfied, this neighbor will be pushed to the growth list
                if (n12 * n12 / (n1m * n2m) > cos_smoothness_thr_square)
                {
                    labels_tmp_ptr[cur_neighbor_idx] = flag;
                    grow_list.push(cur_neighbor_idx);
                    region_size++;
                }
            }
            // Check if the current region size are less than the maximum size
            if (region_size > max_size)
            {
                break;
            }
        }

        // Check if the current region size are within the range
        if (min_size <= region_size && region_size <= max_size)
        {
            swap(_labels, labels_tmp);
            flag++;
        }
    }

    _labels.copyTo(labels);
    return flag - 1;
}

//    } // _3d::
}  // cv::