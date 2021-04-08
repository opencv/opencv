// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_WillowGarage.md file found in this module's directory

#ifndef __OPENCV_RGBD_DEPTH_TO_3D_HPP__
#define __OPENCV_RGBD_DEPTH_TO_3D_HPP__

namespace cv
{
namespace rgbd
{

/**
 * @param depth the depth image, containing depth with the value T
 * @param the mask, containing CV_8UC1
 */
template <typename T>
size_t
convertDepthToFloat(const cv::Mat& depth, const cv::Mat& mask, float scale, cv::Mat_<float> &u_mat, cv::Mat_<float> &v_mat, cv::Mat_<float> &z_mat)
{
  CV_Assert (depth.size == mask.size);

  cv::Size depth_size = depth.size();

  cv::Mat_<uchar> uchar_mask = mask;

  if (mask.depth() != CV_8U)
    mask.convertTo(uchar_mask, CV_8U);

  u_mat = cv::Mat_<float>(depth_size.area(), 1);
  v_mat = cv::Mat_<float>(depth_size.area(), 1);
  z_mat = cv::Mat_<float>(depth_size.area(), 1);

  // Raw data from the Kinect has int
  size_t n_points = 0;

  for (int v = 0; v < depth_size.height; v++)
  {
    uchar* r = uchar_mask.ptr<uchar>(v, 0);

    for (int u = 0; u < depth_size.width; u++, ++r)
      if (*r)
      {
        u_mat((int)n_points, 0) = (float)u;
        v_mat((int)n_points, 0) = (float)v;
        T depth_i = depth.at<T>(v, u);

        if (cvIsNaN((float)depth_i) || (depth_i == std::numeric_limits<T>::min()) || (depth_i == std::numeric_limits<T>::max()))
          z_mat((int)n_points, 0) = std::numeric_limits<float>::quiet_NaN();
        else
          z_mat((int)n_points, 0) = depth_i * scale;

        ++n_points;
      }
  }

  return n_points;
}

/**
 * @param depth the depth image, containing depth with the value T
 * @param the mask, containing CV_8UC1
 */
template <typename T>
void
convertDepthToFloat(const cv::Mat& depth, float scale, const cv::Mat &uv_mat, cv::Mat_<float> &z_mat)
{
  z_mat = cv::Mat_<float>(uv_mat.size());

  // Raw data from the Kinect has int
  float* z_mat_iter = reinterpret_cast<float*>(z_mat.data);

  for (cv::Mat_<cv::Vec2f>::const_iterator uv_iter = uv_mat.begin<cv::Vec2f>(), uv_end = uv_mat.end<cv::Vec2f>();
      uv_iter != uv_end; ++uv_iter, ++z_mat_iter)
  {
    T depth_i = depth.at < T > ((int)(*uv_iter)[1], (int)(*uv_iter)[0]);

    if (cvIsNaN((float)depth_i) || (depth_i == std::numeric_limits < T > ::min())
        || (depth_i == std::numeric_limits < T > ::max()))
      *z_mat_iter = std::numeric_limits<float>::quiet_NaN();
    else
      *z_mat_iter = depth_i * scale;
  }
}

}
}

#endif

/* End of file. */
