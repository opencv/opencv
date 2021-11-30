// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


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
inline void _getMatFromInputArray(InputArray input_pts, Mat &mat,
        int arrangement_of_points = 1, bool clone_data = false)
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

}

#endif //OPENCV_3D_PTCLOUD_UTILS_HPP
