// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021, Yechun Ruan <ruanyc@mail.sustech.edu.cn>
// Acknowledgement of the support from Huawei Technologies Co., Ltd.


#include "../precomp.hpp"
#include "opencv2/3d/ptcloud.hpp"
#include "ptcloud_utils.hpp"
#include <unordered_map>

namespace cv {

//! @addtogroup _3d
//! @{

template<typename Tp>
static inline void _swap(Tp &n, Tp &m)
{
    Tp tmp = n;
    n = m;
    m = tmp;
}

int voxelGridSampling(OutputArray sampled_point_flags, InputArray input_pts,
        const float length, const float width, const float height)
{
    CV_CheckGT(length, 0.0f, "Invalid length of grid");
    CV_CheckGT(width, 0.0f, "Invalid width of grid");
    CV_CheckGT(height, 0.0f, "Invalid height of grid");

    // Get input point cloud data
    Mat ori_pts;
    getPointsMatFromInputArray(input_pts, ori_pts, 0);

    const int ori_pts_size = ori_pts.rows;

    float *const ori_pts_ptr = (float *) ori_pts.data;

    // Compute the minimum and maximum bounding box values
    float x_min, x_max, y_min, y_max, z_min, z_max;
    x_max = x_min = ori_pts_ptr[0];
    y_max = y_min = ori_pts_ptr[1];
    z_max = z_min = ori_pts_ptr[2];

    for (int i = 1; i < ori_pts_size; ++i)
    {
        float *const ptr_base = ori_pts_ptr + 3 * i;
        float x = ptr_base[0], y = ptr_base[1], z = ptr_base[2];

        x_min = std::min(x_min, x);
        x_max = std::max(x_max, x);

        y_min = std::min(y_min, y);
        y_max = std::max(y_max, y);

        z_min = std::min(z_min, z);
        z_max = std::max(z_max, z);
    }

    // Up to 2^64 grids for key type int64_t
    // For larger point clouds or smaller grid sizes, use string or hierarchical map
    typedef int64_t keyType;
    std::unordered_map<keyType, std::vector<int>> grids;

//    int init_size = ori_pts_size * 0.02;
//    grids.reserve(init_size);

    // Divide points into different grids

    keyType offset_y = static_cast<keyType>(cvCeil((x_max - x_min) / length));
    keyType offset_z = offset_y * static_cast<keyType>(cvCeil((y_max - y_min) / width));

    for (int i = 0; i < ori_pts_size; ++i)
    {
        int ii = 3 * i;
        keyType hx = static_cast<keyType>((ori_pts_ptr[ii] - x_min) / length);
        keyType hy = static_cast<keyType>((ori_pts_ptr[ii + 1] - y_min) / width);
        keyType hz = static_cast<keyType>((ori_pts_ptr[ii + 2] - z_min) / height);
        // Convert three-dimensional coordinates to one-dimensional coordinates key
        // Place the stacked three-dimensional grids(boxes) into one dimension (placed in a straight line)
        keyType key = hx + hy * offset_y + hz * offset_z;

        std::unordered_map<keyType, std::vector<int>>::iterator iter = grids.find(key);
        if (iter == grids.end())
            grids[key] = {i};
        else
            iter->second.push_back(i);
    }

    const int pts_new_size = static_cast<int>(grids.size());
    std::vector<char> _sampled_point_flags(ori_pts_size, 0);

    // Take out the points in the grid and calculate the point closest to the centroid
    std::unordered_map<keyType, std::vector<int>>::iterator grid_iter = grids.begin();

    for (int label_id = 0; label_id < pts_new_size; ++label_id, ++grid_iter)
    {
        std::vector<int> grid_pts = grid_iter->second;
        int grid_pts_cnt = static_cast<int>(grid_iter->second.size());
        int sampled_point_idx = grid_pts[0];
        // 1. one point in the grid, select it directly, no need to calculate.
        // 2. two points in the grid, the distance from these two points to their centroid is the same,
        //        can directly select one of them, also do not need to calculate
        if (grid_pts_cnt > 2)
        {
            // Calculate the centroid position
            float sum_x = 0, sum_y = 0, sum_z = 0;
            for (const int &item: grid_pts)
            {
                float *const ptr_base = ori_pts_ptr + 3 * item;
                sum_x += ptr_base[0];
                sum_y += ptr_base[1];
                sum_z += ptr_base[2];
            }

            float centroid_x = sum_x / grid_pts_cnt, centroid_y = sum_y / grid_pts_cnt, centroid_z =
                    sum_z / grid_pts_cnt;

            // Find the point closest to the centroid
            float min_dist_square = FLT_MAX;
            for (const int &item: grid_pts)
            {
                float *const ptr_base = ori_pts_ptr + item * 3;
                float x = ptr_base[0], y = ptr_base[1], z = ptr_base[2];

                float dist_square = (x - centroid_x) * (x - centroid_x) +
                                    (y - centroid_y) * (y - centroid_y) +
                                    (z - centroid_z) * (z - centroid_z);
                if (dist_square < min_dist_square)
                {
                    min_dist_square = dist_square;
                    sampled_point_idx = item;
                }
            }
        }

        _sampled_point_flags[sampled_point_idx] = 1;
    }
    Mat(_sampled_point_flags).copyTo(sampled_point_flags);

    return pts_new_size;
} // voxelGrid()

void
randomSampling(OutputArray sampled_pts, InputArray input_pts, const int sampled_pts_size, RNG *rng)
{
    CV_CheckGT(sampled_pts_size, 0, "The point cloud size after sampling must be greater than 0.");
    CV_CheckDepth(sampled_pts.depth(), sampled_pts.isMat() || sampled_pts.depth() == CV_32F,
                  "The output data type only supports Mat, vector<Point3f>.");

    // Get input point cloud data
    Mat ori_pts;
    getPointsMatFromInputArray(input_pts, ori_pts, 0);

    const int ori_pts_size = ori_pts.rows;
    CV_CheckLT(sampled_pts_size, ori_pts_size,
               "The sampled point cloud size must be smaller than the original point cloud size.");

    std::vector<int> pts_idxs(ori_pts_size);
    for (int i = 0; i < ori_pts_size; ++i) pts_idxs[i] = i;
    randShuffle(pts_idxs, 1, rng);

    int channels = sampled_pts.channels();
    if (channels == 3 && sampled_pts.isVector())
    {
        // std::vector<cv::Point3f>
        sampled_pts.create(1, sampled_pts_size, CV_32FC3);
    }
    else
    {
        // std::vector<float> or cv::Mat
        sampled_pts.create(sampled_pts_size, 3, CV_32F);
    }

    Mat out = sampled_pts.getMat();

    float *const ori_pts_ptr = (float *) ori_pts.data;
    float *const sampled_pts_ptr = (float *) out.data;
    for (int i = 0; i < sampled_pts_size; ++i)
    {
        float *const ori_pts_ptr_base = ori_pts_ptr + pts_idxs[i] * 3;
        float *const sampled_pts_ptr_base = sampled_pts_ptr + i * 3;
        sampled_pts_ptr_base[0] = ori_pts_ptr_base[0];
        sampled_pts_ptr_base[1] = ori_pts_ptr_base[1];
        sampled_pts_ptr_base[2] = ori_pts_ptr_base[2];
    }

} // randomSampling()

void
randomSampling(OutputArray sampled_pts, InputArray input_pts, const float sampled_scale, RNG *rng)
{
    CV_CheckGT(sampled_scale, 0.0f, "The point cloud sampled scale must greater than 0.");
    CV_CheckLT(sampled_scale, 1.0f, "The point cloud sampled scale must less than 1.");
    Mat ori_pts;
    getPointsMatFromInputArray(input_pts, ori_pts, 0);
    randomSampling(sampled_pts, input_pts, cvCeil(sampled_scale * ori_pts.rows), rng);
} // randomSampling()

/**
 * FPS Algorithm:\n
 *   Input: Point cloud *C*, *sampled_pts_size*, *dist_lower_limit* \n
 *   Initialize: Set sampled point cloud S to the empty set \n
 *   Step: \n
 *     1. Randomly take a seed point from C and take it from C to S; \n
 *     2. Find a point in C that is the farthest away from S and take it from C to S; \n
 *       (The distance from point to set S is the smallest distance from point to all points in S) \n
 *     3. Repeat *step 2* until the farthest distance of the point in C from S \n
 *       is less than *dist_lower_limit*, or the size of S is equal to *sampled_pts_size*. \n
 *   Output: Sampled point cloud S \n
 */
int farthestPointSampling(OutputArray sampled_point_flags, InputArray input_pts,
        const int sampled_pts_size, const float dist_lower_limit, RNG *rng)
{
    CV_CheckGT(sampled_pts_size, 0, "The point cloud size after sampling must be greater than 0.");
    CV_CheckGE(dist_lower_limit, 0.0f,
               "The distance lower bound must be greater than or equal to 0.");

    // Get input point cloud data
    Mat ori_pts;
    // In order to keep the points continuous in memory (which allows better support for SIMD),
    // the position of the points may be changed, data copying is mandatory
    getPointsMatFromInputArray(input_pts, ori_pts, 1, true);

    const int ori_pts_size = ori_pts.rows * ori_pts.cols / 3;
    CV_CheckLT(sampled_pts_size, ori_pts_size,
               "The sampled point cloud size must be smaller than the original point cloud size.");


    // idx arr [ . . . . . . . . . ]  --- sampling ---> [ . . . . . . . . . ]
    //                  C                                   S    |    C
    // _idxs records the original location/id of the point
    AutoBuffer<int> _idxs(ori_pts_size);
    // _dist_square records the distance from point(in C) to S
    AutoBuffer<float> _dist_square(ori_pts_size);
    int *idxs = _idxs.data();
    float *dist_square = _dist_square.data();
    for (int i = 0; i < ori_pts_size; ++i)
    {
        idxs[i] = i;
        dist_square[i] = FLT_MAX;
    }

    // Randomly take a seed point from C and take it from C to S
    int seed = (int) ((rng ? rng->next() : theRNG().next()) % ori_pts_size);
    idxs[0] = seed;
    idxs[seed] = 0;

    // Pointer (base address) of access point data x,y,z
    float *const ori_pts_ptr_x = (float *) ori_pts.data;
    float *const ori_pts_ptr_y = ori_pts_ptr_x + ori_pts_size;
    float *const ori_pts_ptr_z = ori_pts_ptr_y + ori_pts_size;

    // Ensure that the point(in C) data x,y,z is continuous in the memory respectively
    _swap(ori_pts_ptr_x[seed], ori_pts_ptr_x[0]);
    _swap(ori_pts_ptr_y[seed], ori_pts_ptr_y[0]);
    _swap(ori_pts_ptr_z[seed], ori_pts_ptr_z[0]);

    int sampled_cnt = 1;
    const float dist_lower_limit_square = dist_lower_limit * dist_lower_limit;
    while (sampled_cnt < sampled_pts_size)
    {
        int last_pt = sampled_cnt - 1;
        float last_pt_x = ori_pts_ptr_x[last_pt];
        float last_pt_y = ori_pts_ptr_y[last_pt];
        float last_pt_z = ori_pts_ptr_z[last_pt];

        // Calculate the distance from point in C to set S
        float max_dist_square = 0;
        int next_pt = sampled_cnt;
        int i = sampled_cnt;
#if CV_SIMD
        v_float32 v_last_p_x = vx_setall_f32(last_pt_x);
        v_float32 v_last_p_y = vx_setall_f32(last_pt_y);
        v_float32 v_last_p_z = vx_setall_f32(last_pt_z);

        for (; i <= ori_pts_size - VTraits<v_float32>::vlanes(); i += VTraits<v_float32>::vlanes())
        {
            v_float32 vx_diff = v_sub(v_last_p_x, vx_load(ori_pts_ptr_x + i));
            v_float32 vy_diff = v_sub(v_last_p_y, vx_load(ori_pts_ptr_y + i));
            v_float32 vz_diff = v_sub(v_last_p_z, vx_load(ori_pts_ptr_z + i));

            v_float32 v_next_dist_square =
                    v_add(v_add(v_mul(vx_diff, vx_diff), v_mul(vy_diff, vy_diff)), v_mul(vz_diff, vz_diff));

            // Update the distance from the points(in C) to S
            float *dist_square_ptr = dist_square + i;
            v_float32 v_dist_square = vx_load(dist_square_ptr);
            v_dist_square = v_min(v_dist_square, v_next_dist_square);
            vx_store(dist_square_ptr, v_dist_square);

            // Find a point in C that is the farthest away from S and take it from C to S
            if (v_check_any(v_gt(v_dist_square, vx_setall_f32(max_dist_square))))
            {
                for (int m = 0; m < VTraits<v_float32>::vlanes(); ++m)
                {
                    if (dist_square_ptr[m] > max_dist_square)
                    {
                        next_pt = i + m;
                        max_dist_square = dist_square_ptr[m];
                    }
                }
            }

        }

#endif
        for (; i < ori_pts_size; ++i)
        {
            float x_diff = (last_pt_x - ori_pts_ptr_x[i]);
            float y_diff = (last_pt_y - ori_pts_ptr_y[i]);
            float z_diff = (last_pt_z - ori_pts_ptr_z[i]);
            float next_dist_square = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff;
            // Update the distance from the points(in C) to S
            dist_square[i] = std::min(dist_square[i], next_dist_square);
            // Find a point in C that is the farthest away from S and take it from C to S
            if (dist_square[i] > max_dist_square)
            {
                next_pt = i;
                max_dist_square = dist_square[i];
            }
        }


        if (max_dist_square < dist_lower_limit_square)
            break;

        // Take point next_pt from C to S
        _swap(idxs[next_pt], idxs[sampled_cnt]);
        _swap(dist_square[next_pt], dist_square[sampled_cnt]);

        // Ensure that the point(in C) data x,y,z is continuous in the memory respectively
        _swap(ori_pts_ptr_x[next_pt], ori_pts_ptr_x[sampled_cnt]);
        _swap(ori_pts_ptr_y[next_pt], ori_pts_ptr_y[sampled_cnt]);
        _swap(ori_pts_ptr_z[next_pt], ori_pts_ptr_z[sampled_cnt]);

        ++sampled_cnt;
    }

    std::vector<char> _sampled_point_flags(ori_pts_size, 0);
    for (int j = 0; j < sampled_cnt; ++j)
    {
        _sampled_point_flags[idxs[j]] = 1;
    }

    Mat(_sampled_point_flags).copyTo(sampled_point_flags);

    return sampled_cnt;
} // farthestPointSampling()

int farthestPointSampling(OutputArray sampled_point_flags, InputArray input_pts,
        const float sampled_scale, const float dist_lower_limit, RNG *rng)
{
    CV_CheckGT(sampled_scale, 0.0f, "The point cloud sampled scale must greater than 0.");
    CV_CheckLT(sampled_scale, 1.0f, "The point cloud sampled scale must less than 1.");
    CV_CheckGE(dist_lower_limit, 0.0f,
               "The distance lower bound must be greater than or equal to 0.");
    Mat ori_pts;
    getPointsMatFromInputArray(input_pts, ori_pts, 1);
    return farthestPointSampling(sampled_point_flags, input_pts,
                                 cvCeil(sampled_scale * ori_pts.cols), dist_lower_limit, rng);
} // farthestPointSampling()

//! @} _3d
} //end namespace cv
