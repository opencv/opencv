// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_PTCLOUD_HPP
#define OPENCV_PTCLOUD_HPP

namespace cv {

//! @addtogroup _3d
//! @{

/**
 * @brief Voxel Grid filter down sampling:
 *
 * Creates a 3D voxel grid (a set of tiny 3D boxes in space) over the input
 * point cloud data, in each voxel (i.e., 3D box), all the points present will be
 * approximated (i.e., downsampled) with the point closest to their centroid.
 *
 * @param sampled_pts  Point cloud after sampling
 * @param input_pts  Original point cloud, vector of Point3 or Mat of size Nx3/3xN
 * @param length Grid length
 * @param width  Grid width
 * @param height  Grid height
 */
CV_EXPORTS void voxelGridSampling(cv::OutputArray sampled_pts, cv::InputArray input_pts,
                                  float length, float width, float height);

/**
 * @brief Point cloud sampling by randomly select points
 *
 * Use cv::randShuffle to shuffle the point index, then take the first sampled_pts_size
 *
 * @param sampled_pts  Point cloud after sampling
 * @param input_pts  Original point cloud, vector of Point3 or Mat of size Nx3/3xN
 * @param sampled_pts_size The desired point cloud size after sampling
 * @param rng  Optional random number generator used for cv::randShuffle;
 *                      if it is nullptr, theRNG () is used instead.
 */
CV_EXPORTS void randomSampling(cv::OutputArray sampled_pts, cv::InputArray input_pts,
                               int sampled_pts_size, cv::RNG *rng = nullptr);

/**
 * @brief Point cloud sampling by randomly select points
 *
 * Use cv::randShuffle to shuffle the point index, then take the first input_pts.size*sampled_scale
 *
 * @param sampled_pts  Point cloud after sampling
 * @param input_pts  Original point cloud, vector of Point3 or Mat of size Nx3/3xN
 * @param sampled_scale The percentage of the sampled point cloud to the original size,
 *                      that is, sampled size = original size * sampled_scale, range (0, 1)
 * @param rng  Optional random number generator used for cv::randShuffle;
 *                      if it is nullptr, theRNG () is used instead.
 */
CV_EXPORTS void randomSampling(cv::OutputArray sampled_pts, cv::InputArray input_pts,
                               float sampled_scale, cv::RNG *rng = nullptr);

/**
 * @brief Farthest Point Sampling(FPS):
 *
 * Input point cloud C, sampled point cloud S, S initially has a size of 0
 * 1. Randomly take a seed point from C and put it into S
 * 2. Find a point in C that is the farthest away from S and put it into S
 * The distance from point to S set is the smallest distance from point to all points in S
 *
 * @param sampled_pts  Point cloud after sampling
 * @param input_pts  Original point cloud, vector of Point3 or Mat of size Nx3/3xN
 * @param sampled_pts_size The desired point cloud size after sampling
 * @param dist_lower_limit Sampling is terminated early if the distance from
 *                  the farthest point to S is less than dist_lower_limit, default 0
 * @param rng Optional random number generator used for selecting seed point for FPS;
 *                  if it is nullptr, theRNG () is used instead.
 */
CV_EXPORTS void farthestPointSampling(cv::OutputArray sampled_pts, cv::InputArray input_pts,
                                      int sampled_pts_size, float dist_lower_limit = 0, cv::RNG *rng = nullptr);

/**
 * @brief Details in farthestPointSampling(cv::InputArray, int, cv::OutputArray, float)
 *
 * @param sampled_pts  Point cloud after sampling
 * @param input_pts  Original point cloud, vector of Point3 or Mat of size Nx3/3xN
 * @param sampled_scale The percentage of the sampled point cloud to the original size,
 *                      that is, sampled size = original size * sampled_scale, range (0, 1)
 * @param dist_lower_limit Sampling is terminated early if the distance from
 *                  the farthest point to S is less than dist_lower_limit, default 0
 * @param rng Optional random number generator used for selecting seed point for FPS;
 *                  if it is nullptr, theRNG () is used instead.
 */
CV_EXPORTS void farthestPointSampling(cv::OutputArray sampled_pts, cv::InputArray input_pts,
                                      float sampled_scale, float dist_lower_limit = 0, cv::RNG *rng = nullptr);

//! @} _3d
} //end namespace cv
#endif //OPENCV_PTCLOUD_HPP
