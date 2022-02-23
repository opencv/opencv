// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021, Yechun Ruan <ruanyc@mail.sustech.edu.cn>

#ifndef OPENCV_3D_PTCLOUD_WRAPPER_HPP
#define OPENCV_3D_PTCLOUD_WRAPPER_HPP

namespace cv {

/** @brief 3D Point Cloud Wrapper.

A wrapper that encapsulates pointers to access point cloud data,
making the construction of point cloud data access simpler.

@note The point cloud data XYZ matrix should be 3xN, single channel, CV_32F, continuous in memory.

 */
class PointCloudWrapper
{
protected:
    const Mat *points_mat;
    const int pts_cnt;
    const float *pts_ptr_x;
    const float *pts_ptr_y;
    const float *pts_ptr_z;

public:
    explicit PointCloudWrapper(const Mat &points_)
            : points_mat(&points_), pts_cnt(points_.rows * points_.cols / 3),
              pts_ptr_x((float *) points_.data), pts_ptr_y(pts_ptr_x + pts_cnt),
              pts_ptr_z(pts_ptr_y + pts_cnt)
    {
        CV_CheckDepthEQ(points_.depth(), CV_32F,
                "Data with only depth CV_32F are supported");
        CV_CheckChannelsEQ(points_.channels(), 1,
                "Data with only one channel are supported");
        CV_CheckEQ(points_.rows, 3,
                "Data with only Mat with 3xN are supported");
        CV_Assert(points_.isContinuous());
    }

};

}  // cv::

#endif //OPENCV_3D_PTCLOUD_WRAPPER_HPP
