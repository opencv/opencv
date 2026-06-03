// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021, Yechun Ruan <ruanyc@mail.sustech.edu.cn>


#ifndef OPENCV_3D_PTCLOUD_UTILS_HPP
#define OPENCV_3D_PTCLOUD_UTILS_HPP

namespace cv {

/**
 * @brief Get cv::Mat with Nx3 or 3xN type CV_32FC1 from cv::InputArray.
 *
 * @param input_pts Point cloud xyz data.
 * @param[out] mat Point cloud xyz data in cv::Mat with Nx3 or 3xN type CV_32FC1.
 * @param arrangement_of_points The arrangement of point data in the matrix,
 *                              0 by row (Nx3, [x1, y1, z1, ..., xn, yn, zn]),
 *                              1 by column (3xN, [x1, ..., xn, y1, ..., yn, z1, ..., zn]).
 * @param clone_data Flag to specify whether data cloning is mandatory.
 *
 * @note The following cases will clone data even if flag clone_data is false:
 *       1. Data is discontinuous in memory
 *       2. Data type is not float
 *       3. The original arrangement of data is not the same as the expected new arrangement.
 *          For example, transforming from
 *          Nx3(x1, y1, z1, ..., xn, yn, zn) to 3xN(x1, ..., xn, y1, ..., yn, z1, ..., zn)
 *
 */
void getPointsMatFromInputArray(InputArray input_pts, Mat &mat,
        int arrangement_of_points = 1, bool clone_data = false);

/** @brief Copy the data xyz of the point by specifying the indexs.
 *
 * @param src CV_32F Mat with size Nx3/3xN.
 * @param[out] dst CV_32F Mat with size Mx3/3xM.
 * @param idxs The index of the point copied from src to dst.
 * @param dst_size The first dst_size of idxs is valid.
 *                 If it is less than 0 or greater than idxs.size(),
 *                 it will be automatically adjusted to idxs.size().
 * @param arrangement_of_points The arrangement of point data in the matrix, \n
 *                              0 by row (Nx3, [x1, y1, z1, ..., xn, yn, zn]),  \n
 *                              1 by column (3xN, [x1, ..., xn, y1, ..., yn, z1, ..., zn]).
 * @param is_parallel Whether to enable parallelism.
 *        The number of threads used is obtained through cv::getnumthreads().
 */
void copyPointDataByIdxs(const Mat &src, Mat &dst, const std::vector<int> &idxs, int dst_size = -1,
        int arrangement_of_points = 1, bool is_parallel = false);

/** @overload
 *
 * @param src CV_32F Mat with size Nx3/3xN.
 * @param[out] dst CV_32F Mat with size Mx3/3xM.
 * @param flags If flags[i] is true, the i-th point will be copied from src to dst.
 * @param arrangement_of_points The arrangement of point data in the matrix, \n
 *                              0 by row (Nx3, [x1, y1, z1, ..., xn, yn, zn]),  \n
 *                              1 by column (3xN, [x1, ..., xn, y1, ..., yn, z1, ..., zn]).
 * @param is_parallel Whether to enable parallelism.
 *        The number of threads used is obtained through cv::getnumthreads().
 */
void copyPointDataByFlags(const Mat &src, Mat &dst, const std::vector<bool> &flags,
        int arrangement_of_points = 1, bool is_parallel = false);

}

#endif //OPENCV_3D_PTCLOUD_UTILS_HPP
