#include "../precomp.hpp"
#include "region_growing_3d.hpp"
#include "opencv2/3d/ptcloud.hpp"

#include "queue"

namespace cv {
//    namespace _3d {

/////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////  RegionGrowing3DImpl  //////////////////////////////////////

int
RegionGrowing3DImpl::segment(OutputArray labels)
{
    int pts_size = input_pts.rows;

    CV_CheckGE(max_size, min_size,
            "The maximum size should be grater than or equal to the minimum size.");
    CV_CheckGE(knn_idx.cols, k,
            "The number of neighbors of KNN search result should be grater than or equal to that of region growing.");
    CV_CheckGE(pts_size, max(seeds.rows, seeds.cols),
            "The number of seed should be less than or equal to the number of points in the point cloud.");
    CV_CheckEQ(pts_size, knn_idx.rows,
            "The number of points in the KNN search result should be equal to that in the point cloud.");
    CV_CheckEQ(pts_size, max(normals.rows, normals.cols),
            "The number of points in the normals should be equal to that in the point cloud.");
    CV_CheckEQ(pts_size, max(curvatures.rows, curvatures.cols),
            "The number of points in the curvatures should be equal to that in the point cloud.");

    if (k == 0) k = knn_idx.cols;

    // Set labels to zero
    Mat _labels = Mat::zeros(pts_size, 1, CV_32S);
    // If no seed point is set, curvature sorting is used by default
    if (seeds.empty()) sortIdx(curvatures, seeds, SORT_EVERY_COLUMN + SORT_ASCENDING);

    const int *seeds_ptr = (int *) seeds.data;
    const float *curvatures_ptr = (float *) curvatures.data;

    int flag = 1;
    for (int i = 0; i < pts_size; i++)
    {
        int *labels_ptr = (int *) _labels.data;
        int cur_seed = seeds_ptr[i];
        int region_size = 1;

        // If the number of regions is satisfied then stop running
        if (flag > region_num) break;
        // If current seed has been grown then grow the next one
        if (labels_ptr[cur_seed] != 0) continue;
        // Filter out seeds with curvature greater than the threshold
        if (curvatures_ptr[cur_seed] > curvature_thr) break;

        Mat base_normal;
        if (!smooth_mode) base_normal = normals.row(cur_seed);

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
            if (smooth_mode) base_normal = normals.row(cur_idx);

            const int *pt_knn_ptr = (int *) knn_idx.row(cur_idx).data;
            for (int j = 0; j < k; j++)
            {
                int cur_neighbor_idx = pt_knn_ptr[j];
                // If current point has been grown then grow the next one
                if (labels_tmp_ptr[cur_neighbor_idx] != 0) continue;
                // Filter out points with curvature greater than the threshold
                if (curvatures_ptr[cur_neighbor_idx] > curvature_thr) continue;

                Mat cur_normal = normals.row(cur_neighbor_idx);
                float n12 = (float) base_normal.dot(cur_normal);
                float n1m = (float) base_normal.dot(base_normal);
                float n2m = (float) cur_normal.dot(cur_normal);
                float cos_smoothness_thr = std::cos(smoothness_thr);
                // If the smoothness threshold is satisfied, this neighbor will be pushed to the growth list
                if (n12 * n12 / (n1m * n2m) > cos_smoothness_thr * cos_smoothness_thr)
                {
                    labels_tmp_ptr[cur_neighbor_idx] = flag;
                    grow_list.push(cur_neighbor_idx);
                    region_size++;
                }
            }
            // Check if the current region size are less than the maximum size
            if (region_size > max_size) break;
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