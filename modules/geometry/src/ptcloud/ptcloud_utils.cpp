// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021, Yechun Ruan <ruanyc@mail.sustech.edu.cn>

#include "../precomp.hpp"
#include "ptcloud_utils.hpp"


namespace cv {

void
getPointsMatFromInputArray(InputArray input_pts, Mat &mat, int arrangement_of_points, bool clone_data)
{
    CV_Check(input_pts.dims(), input_pts.dims() < 3,
            "Only support data with dimension less than 3.");

    // Guaranteed data can construct N×3 point clouds
    int rows = input_pts.rows(), cols = input_pts.cols(), channels = input_pts.channels();
    size_t total = rows * cols * channels;
    CV_Check(total, total % 3 == 0,
            "total = input_pts.rows() * input_pts.cols() * input_pts.channels() must be an integer multiple of 3");

    /**
     Layout of point cloud data in memory space.
     arrangement 0 : x1, y1, z1, ..., xn, yn, zn
                    For example, the input is std::vector<Point3d>, or std::vector<int>,
                    or cv::Mat with type N×1 CV_32FC3
     arrangement 1 : x1, ..., xn, y1, ..., yn, z1, ..., zn
                    For example, the input is cv::Mat with type 3×N CV_32FC1
     */
    int ori_arrangement = (channels == 1 && rows == 3 && cols != 3) ? 1 : 0;

    // Convert to single channel without copying the data.
    mat = ori_arrangement == 0 ? input_pts.getMat().reshape(1, (int) (total / 3))
                               : input_pts.getMat();

    if (ori_arrangement != arrangement_of_points)
    {
        Mat tmp;
        transpose(mat, tmp);
        swap(mat, tmp);
    }

    if (mat.type() != CV_32F)
    {
        Mat tmp;
        mat.convertTo(tmp, CV_32F); // Use float to store data
        swap(mat, tmp);
    }

    if (clone_data || (!mat.isContinuous()))
    {
        mat = mat.clone();
    }

}

void
copyPointDataByIdxs(const Mat &src, Mat &dst, const std::vector<int> &idxs, int dst_size,
        int arrangement_of_points, bool is_parallel)
{
    CV_CheckDepthEQ(src.depth(), CV_32F, "Data with only depth CV_32F are supported");
    CV_CheckChannelsEQ(src.channels(), 1, "Data with only one channel are supported");

    const int idxs_size = (int) idxs.size();
    dst_size = (dst_size < 0 || dst_size > idxs_size) ? idxs_size : dst_size;

    if (arrangement_of_points == 1)
    {
        dst = Mat(3, dst_size, CV_32F);
        const int src_size = src.rows * src.cols / 3;
        const float *const src_ptr_x = (float *) src.data;
        const float *const src_ptr_y = src_ptr_x + src_size;
        const float *const src_ptr_z = src_ptr_y + src_size;

        float *const dst_ptr_x = (float *) dst.data;
        float *const dst_ptr_y = dst_ptr_x + dst_size;
        float *const dst_ptr_z = dst_ptr_y + dst_size;

        int i = 0;
        if (is_parallel) {
            int number_of_threads = std::max(getNumThreads(), 1);
            int block_size = dst_size / number_of_threads;
            parallel_for_(Range(0, number_of_threads), [&](const Range &range) {
                int _start = range.start * block_size;
                int _end = _start + block_size;
                for (int _i = _start; _i < _end; ++_i) {
                    int _src_idx = idxs[_i];
                    dst_ptr_x[_i] = src_ptr_x[_src_idx];
                    dst_ptr_y[_i] = src_ptr_y[_src_idx];
                    dst_ptr_z[_i] = src_ptr_z[_src_idx];
                }
            });

            i = block_size * number_of_threads;
        }

        for (; i < dst_size; ++i)
        {
            int src_idx = idxs[i];
            dst_ptr_x[i] = src_ptr_x[src_idx];
            dst_ptr_y[i] = src_ptr_y[src_idx];
            dst_ptr_z[i] = src_ptr_z[src_idx];
        }

    }
    else if (arrangement_of_points == 0)
    {
        dst = Mat(dst_size, 3, CV_32F);

        const float *const src_ptr = (float *) src.data;
        float *const dst_ptr = (float *) dst.data;

        int i = 0;
        if (is_parallel) {
            int number_of_threads = std::max(getNumThreads(), 1);
            int block_size = dst_size / number_of_threads;
            parallel_for_(Range(0, number_of_threads), [&](const Range &range) {
                int _start = range.start * block_size;
                int _end = _start + block_size;
                for (int _i = _start; _i < _end; ++_i) {
                    const float *_src_ptr_base = src_ptr + 3 * idxs[_i];
                    float *_dst_ptr_base = dst_ptr + 3 * _i;
                    _dst_ptr_base[0] = _src_ptr_base[0];
                    _dst_ptr_base[1] = _src_ptr_base[1];
                    _dst_ptr_base[2] = _src_ptr_base[2];
                }
            });

            i = block_size * number_of_threads;
        }

        for (; i < dst_size; ++i)
        {
            const float *src_ptr_base = src_ptr + 3 * idxs[i];
            float *dst_ptr_base = dst_ptr + 3 * i;
            dst_ptr_base[0] = src_ptr_base[0];
            dst_ptr_base[1] = src_ptr_base[1];
            dst_ptr_base[2] = src_ptr_base[2];
        }
    }

}


void copyPointDataByFlags(const Mat &src, Mat &dst, const std::vector<bool> &flags,
        int arrangement_of_points, bool is_parallel)
{
    int pt_size = (int) flags.size();
    std::vector<int> idxs;
    for (int i = 0; i < pt_size; ++i)
    {
        if (flags[i])
        {
            idxs.emplace_back(i);
        }
    }
    copyPointDataByIdxs(src, dst, idxs, -1, arrangement_of_points, is_parallel);
}

}