// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#include "../precomp.hpp"
#include "opencv2/3d/ptcloud.hpp"
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

static inline void _getMatFromInputArray(cv::InputArray input_pts, cv::Mat &mat)
{
    cv::_InputArray::KindFlag kind = input_pts.kind();

    CV_CheckType(kind, kind == _InputArray::STD_VECTOR || kind == _InputArray::MAT
                 || kind == _InputArray::MATX || kind == _InputArray::UMAT,
                 "Point cloud data storage type should be vector of Point3 or Mat of size Nx3");

    mat = input_pts.getMat();

    if (kind == _InputArray::STD_VECTOR)
    {
        int new_rows = mat.rows * mat.cols * mat.channels() / 3;
        mat = cv::Mat(new_rows, 3, CV_32F, mat.data);
    }
    else
    {
        if (mat.channels() != 1)
            mat = mat.reshape(1, static_cast<int>(mat.total())); // Convert to single channel
        if (mat.cols != 3 && mat.rows == 3)
            cv::transpose(mat, mat);

        CV_CheckEQ(mat.cols, 3, "Invalid dimension of point cloud");
        if (mat.type() != CV_32F)
            mat.convertTo(mat, CV_32F);// Use float to store data
    }
}

void voxelGridSampling(std::vector<bool> &sampled_point_flags, cv::InputArray input_pts,
                       const float length, const float width, const float height)
{
    CV_CheckGT(length, 0.0f, "Invalid length of grid");
    CV_CheckGT(width, 0.0f, "Invalid width of grid");
    CV_CheckGT(height, 0.0f, "Invalid height of grid");

    // Get input point cloud data
    cv::Mat ori_pts;
    _getMatFromInputArray(input_pts, ori_pts);

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
        float x = *(ptr_base), y = *(ptr_base + 1), z = *(ptr_base + 2);

        if (x_min > x) x_min = x;
        if (x_max < x) x_max = x;

        if (y_min > y) y_min = y;
        if (y_max < y) y_max = y;

        if (z_min > z) z_min = z;
        if (z_max < z) z_max = z;
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
    sampled_point_flags.resize(ori_pts_size);
    std::fill(sampled_point_flags.begin(), sampled_point_flags.end(), false);

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
                sum_x += *(ptr_base);
                sum_y += *(ptr_base + 1);
                sum_z += *(ptr_base + 2);
            }

            float centroid_x = sum_x / grid_pts_cnt, centroid_y = sum_y / grid_pts_cnt, centroid_z =
                    sum_z / grid_pts_cnt;

            // Find the point closest to the centroid
            float min_dist_square = FLT_MAX;
            for (const int &item: grid_pts)
            {
                float *const ptr_base = ori_pts_ptr + item * 3;
                float x = *(ptr_base), y = *(ptr_base + 1), z = *(ptr_base + 2);

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

        sampled_point_flags[sampled_point_idx] = true;
    }

} // voxelGrid()

void randomSampling(cv::OutputArray sampled_pts, cv::InputArray input_pts, const int sampled_pts_size, cv::RNG *rng)
{
    CV_CheckGT(sampled_pts_size, 0, "The point cloud size after sampling must be greater than 0.");

    // Get input point cloud data
    cv::Mat ori_pts;
    _getMatFromInputArray(input_pts, ori_pts);

    const int ori_pts_size = ori_pts.rows;
    CV_CheckLT(sampled_pts_size, ori_pts_size,
               "The sampled point cloud size must be smaller than the original point cloud size.");

    std::vector<int> pts_idxs(ori_pts_size);
    for (int i = 0; i < ori_pts_size; ++i) pts_idxs[i] = i;
    cv::randShuffle(pts_idxs, 1, rng);

    sampled_pts.create(sampled_pts_size, 3, CV_32F);

    float *const ori_pts_ptr = (float *) ori_pts.data;
    float *const sampled_pts_ptr = (float *) sampled_pts.getMat().data;
    for (int i = 0; i < sampled_pts_size; ++i)
    {
        float *const ori_pts_ptr_base = ori_pts_ptr + pts_idxs[i] * 3;
        float *const sampled_pts_ptr_base = sampled_pts_ptr + i * 3;
        *(sampled_pts_ptr_base) = *(ori_pts_ptr_base);
        *(sampled_pts_ptr_base + 1) = *(ori_pts_ptr_base + 1);
        *(sampled_pts_ptr_base + 2) = *(ori_pts_ptr_base + 2);
    }

} // randomSampling()

void randomSampling(cv::OutputArray sampled_pts, cv::InputArray input_pts, const float sampled_scale, cv::RNG *rng)
{
    CV_CheckGT(sampled_scale, 0.0f, "The point cloud sampled scale must greater than 0.");
    CV_CheckLT(sampled_scale, 1.0f, "The point cloud sampled scale must less than 1.");
    cv::Mat ori_pts;
    _getMatFromInputArray(input_pts, ori_pts);
    randomSampling(sampled_pts, input_pts, cvCeil(sampled_scale * ori_pts.rows), rng);
} // randomSampling()

/**
 * Input point cloud C, sampled point cloud S, S initially has a size of 0
 * 1. Randomly take a seed point from C and put it into S
 * 2. Find a point in C that is the farthest away from S and put it into S
 * The distance from point to S set is the smallest distance from point to all points in S
 */
void farthestPointSampling(std::vector<bool> &sampled_point_flags, cv::InputArray input_pts,
                           const int sampled_pts_size, const float dist_lower_limit, cv::RNG *rng)
{
    CV_CheckGT(sampled_pts_size, 0, "The point cloud size after sampling must be greater than 0.");
    CV_CheckGE(dist_lower_limit, 0.0f, "The distance lower bound must be greater than or equal to 0.");

    // Get input point cloud data
    cv::Mat ori_pts;
    _getMatFromInputArray(input_pts, ori_pts);

    const int ori_pts_size = ori_pts.rows;
    CV_CheckLT(sampled_pts_size, ori_pts_size,
               "The sampled point cloud size must be smaller than the original point cloud size.");


    // idx arr [ . . . . . . . . . ]  --- sampling ---> [ . . . . . . . . . ]
    //                  C                                   S    |    C
    cv::AutoBuffer<int> _idxs(ori_pts_size);
    cv::AutoBuffer<float> _dist_square(ori_pts_size);
    int *idxs = _idxs.data();
    float *dist_square = _dist_square.data();
    for (int i = 0; i < ori_pts_size; ++i)
    {
        idxs[i] = i;
        dist_square[i] = FLT_MAX;
    }

    // Randomly take a seed point from C and put it into S
    int seed = (int)((rng? rng->next(): cv::theRNG().next()) % ori_pts_size);
    idxs[0] = seed;
    idxs[seed] = 0;

    sampled_point_flags.resize(ori_pts_size);
    std::fill(sampled_point_flags.begin(), sampled_point_flags.end(), false);
    sampled_point_flags[seed] = true;

    float *const ori_pts_ptr = (float *) ori_pts.data;
    int sampled_cnt = 1;
    const float dist_lower_limit_square = dist_lower_limit * dist_lower_limit;
    while (sampled_cnt < sampled_pts_size)
    {
        int last_pt = sampled_cnt - 1;
        float *const last_pt_ptr_base = ori_pts_ptr + 3 * idxs[last_pt];
        float last_pt_x = *last_pt_ptr_base, last_pt_y = *(last_pt_ptr_base + 1), last_pt_z = *(last_pt_ptr_base + 2);

        // Calculate the distance from point in C to set S
        float max_dist_square = 0;
        for (int i = sampled_cnt; i < ori_pts_size; ++i)
        {
            float *const ori_pts_ptr_base = ori_pts_ptr + 3 * idxs[i];
            float x_diff = (last_pt_x - *ori_pts_ptr_base);
            float y_diff = (last_pt_y - *(ori_pts_ptr_base + 1));
            float z_diff = (last_pt_z - *(ori_pts_ptr_base + 2));
            float next_dist_square = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff;
            if (next_dist_square < dist_square[i])
            {
                dist_square[i] = next_dist_square;
            }
            if (dist_square[i] > max_dist_square)
            {
                last_pt = i;
                max_dist_square = dist_square[i];
            }
        }

        sampled_point_flags[idxs[last_pt]] = true;

        if (max_dist_square < dist_lower_limit_square) break;

        _swap(idxs[sampled_cnt], idxs[last_pt]);
        _swap(dist_square[sampled_cnt], dist_square[last_pt]);
        ++sampled_cnt;
    }

} // farthestPointSampling()

void farthestPointSampling(std::vector<bool> &sampled_point_flags, cv::InputArray input_pts,
                           const float sampled_scale, const float dist_lower_limit, cv::RNG *rng)
{
    CV_CheckGT(sampled_scale, 0.0f, "The point cloud sampled scale must greater than 0.");
    CV_CheckLT(sampled_scale, 1.0f, "The point cloud sampled scale must less than 1.");
    CV_CheckGE(dist_lower_limit, 0.0f, "The distance lower bound must be greater than or equal to 0.");
    cv::Mat ori_pts;
    _getMatFromInputArray(input_pts, ori_pts);
    farthestPointSampling(sampled_point_flags, input_pts,
                          cvCeil(sampled_scale * ori_pts.rows), dist_lower_limit, rng);
} // farthestPointSampling()

//! @} _3d
} //end namespace cv