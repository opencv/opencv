// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022, Wanli Zhong <zhongwl2018@mail.sustech.edu.cn>

#include "../precomp.hpp"
#include "ptcloud_utils.hpp"

namespace cv {

//! @addtogroup _3d
//! @{

void
normalEstimate(OutputArray normals, OutputArray curvatures, InputArray input_pts_,
        InputArrayOfArrays nn_idx_, int max_neighbor_num_)
{
    Mat ori_pts;
    getPointsMatFromInputArray(input_pts_, ori_pts, 0);
    int pts_size = ori_pts.rows;

    std::vector<Mat> nn_idx;
    nn_idx_.getMatVector(nn_idx);

    CV_CheckEQ((int) nn_idx.size(), pts_size,
            "The point number of NN search result should be equal to the size of the point cloud.");

    if (normals.channels() == 3 && normals.isVector())
    {
        // std::vector<cv::Point3f>
        normals.create(1, pts_size, CV_32FC3);
    }
    else
    {
        // cv::Mat
        normals.create(pts_size, 3, CV_32F);
    }

    curvatures.create(pts_size, 1, CV_32F);

    int max_neighbor_num = max_neighbor_num_ <= 0 ? INT_MAX : max_neighbor_num_;

    float *normals_ptr = (float *) normals.getMat().data;
    float *curvatures_ptr = (float *) curvatures.getMat().data;

    parallel_for_(Range(0, pts_size), [&](const Range &range) {
        // Index of current nearest neighbor point
        int cur_nei_idx;
        for (int i = range.start; i < range.end; ++i)
        {
            // The maximum size that may be used for this row
            int bound = max_neighbor_num > nn_idx[i].cols ? nn_idx[i].cols : max_neighbor_num;
            const int *nn_idx_ptr_base = (int *) nn_idx[i].data;
            // The first point should be itself
            Mat pt_set(ori_pts.row(i));
            // Push the nearest neighbor points to pt_set
            for (int j = 1; j < bound; ++j)
            {
                cur_nei_idx = nn_idx_ptr_base[j];
                // If the index is less than 0,
                // the nn_idx of this row will no longer have any information.
                if (cur_nei_idx < 0) break;
                pt_set.push_back(ori_pts.row(cur_nei_idx));
            }

            Mat mean;
            // Calculate the mean of point set, use "reduce()" is faster than default method in PCA
            reduce(pt_set, mean, 0, REDUCE_AVG);
            // Use PCA to get eigenvalues and eigenvectors of pt_set
            PCA pca(pt_set, mean, PCA::DATA_AS_ROW);

            const float *eigenvectors_ptr = (float *) pca.eigenvectors.data;
            float *normals_ptr_base = normals_ptr + 3 * i;
            normals_ptr_base[0] = eigenvectors_ptr[6];
            normals_ptr_base[1] = eigenvectors_ptr[7];
            normals_ptr_base[2] = eigenvectors_ptr[8];

            const float *eigenvalues_ptr = (float *) pca.eigenvalues.data;
            curvatures_ptr[i] = eigenvalues_ptr[2] /
                                (eigenvalues_ptr[0] + eigenvalues_ptr[1] + eigenvalues_ptr[2]);
        }
    });
}

//! @} _3d
} //end namespace cv