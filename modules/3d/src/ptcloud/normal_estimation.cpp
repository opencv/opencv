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
        InputArray knn_idx_, int k)
{
    Mat ori_pts;
    getPointsMatFromInputArray(input_pts_, ori_pts, 0);
    int pts_size = ori_pts.rows;

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

    Mat knn_idx;
    if (knn_idx_.rows() < knn_idx_.cols())
    {
        // Convert layout of data in memory space to NxK
        transpose(knn_idx_.getMat(), knn_idx);
    }
    else
    {
        knn_idx = knn_idx_.getMat();
    }

    if (k == 0) k = knn_idx.cols;

    CV_CheckGE(k, 2, "The number of neighbors should be greater than 2.");
    CV_CheckGE(knn_idx.cols, k,
            "The number of neighbors of KNN search result should be grater than or equal to that of normal estimation.");
    CV_CheckEQ(pts_size, knn_idx.rows,
            "The number of points in the KNN search result should be equal to that in the point cloud.");

    const float *ori_pts_ptr = (float *) ori_pts.data;
    const int *knn_idx_ptr = (int *) knn_idx.data;
    float *normals_ptr = (float *) normals.getMat().data;
    float *curvatures_ptr = (float *) curvatures.getMat().data;

    parallel_for_(Range(0, pts_size), [&](const Range &range) {
        int i = range.start;
        Mat pt_set(k, 3, CV_32F);
        // Copy the nearest k points to pt_set
        float *pt_set_ptr = (float *) pt_set.data;
        long ik = (long) i * k;
        for (int j = 0; j < k; ++j)
        {
            int idx = knn_idx_ptr[ik + j];
            const float *ori_pts_ptr_base = ori_pts_ptr + 3 * idx;
            float *pt_set_ptr_base = pt_set_ptr + 3 * j;
            pt_set_ptr_base[0] = ori_pts_ptr_base[0];
            pt_set_ptr_base[1] = ori_pts_ptr_base[1];
            pt_set_ptr_base[2] = ori_pts_ptr_base[2];
        }

        Mat mean;
        // Calculate the mean of point set
        reduce(pt_set, mean, 0, REDUCE_AVG);
        // Use PCA to get eigenvalues and eigenvectors of pt_set
        PCA pca(pt_set, mean, PCA::DATA_AS_ROW);

        const float *eigenvectors_ptr = (float *) pca.eigenvectors.data;
        float *normals_ptr_base = normals_ptr + 3 * i;
        normals_ptr_base[0] = eigenvectors_ptr[6];
        normals_ptr_base[1] = eigenvectors_ptr[7];
        normals_ptr_base[2] = eigenvectors_ptr[8];

        const float *eigenvalues_ptr = (float *) pca.eigenvalues.data;
        curvatures_ptr[i] =
                eigenvalues_ptr[2] / (eigenvalues_ptr[0] + eigenvalues_ptr[1] + eigenvalues_ptr[2]);
    });
}

void
getKNNSearchResultsByKDTree(OutputArray knn_idx, OutputArray knn_dist, InputArray input_pts_, int k,
        flann::KDTreeIndexParams *kdtree_params_, flann::SearchParams *search_params_)
{
    Mat ori_pts;
    getPointsMatFromInputArray(input_pts_, ori_pts, 0);
    int pts_size = ori_pts.rows;

    CV_CheckGE(k, 2, "The number of neighbors should be greater than 2.");
    CV_CheckGE(pts_size, k,
            "The point cloud size should be greater than or equal to the number of neighbors.");

    // Build KDTree for knn search
    flann::KDTreeIndexParams *kdtree_params = kdtree_params_ ? kdtree_params_
                                                             : new flann::KDTreeIndexParams();
    flann::SearchParams *search_params = search_params_ ? search_params_
                                                        : new flann::SearchParams();
    flann::Index tree(ori_pts, *kdtree_params);

    bool need_knn_idx = &knn_idx != &noArray();
    bool need_knn_dist = &knn_dist != &noArray();
    // Create knn_idx or knn_dist to save the results of KNN search
    if (need_knn_idx) knn_idx.create(pts_size, k, CV_32S);
    if (need_knn_dist) knn_dist.create(pts_size, k, CV_32F);

    int *knn_idx_ptr = (int *) knn_idx.getMat().data;
    float *knn_dist_ptr = (float *) knn_dist.getMat().data;

    parallel_for_(Range(0, pts_size), [&](const Range &range) {
        int i = range.start;
        Mat idx_set(1, k, CV_32S), dist_set(1, k, CV_32F);
        tree.knnSearch(ori_pts.row(i), idx_set, dist_set, k, *search_params);

        const int *idx_set_ptr = (int *) idx_set.data;
        const float *dist_set_ptr = (float *) dist_set.data;
        long ik = (long) i * k;

        if (need_knn_idx)
            // Copy result of knn search to _knn_idx
            for (int j = 0; j < k; ++j) knn_idx_ptr[ik + j] = idx_set_ptr[j];
        if (need_knn_dist)
            // Copy result of knn search to _knn_dist
            for (int j = 0; j < k; ++j) knn_dist_ptr[ik + j] = dist_set_ptr[j];
    });
}

//! @} _3d
} //end namespace cv