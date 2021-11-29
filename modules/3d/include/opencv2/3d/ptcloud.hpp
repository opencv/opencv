// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_3D_PTCLOUD_HPP
#define OPENCV_3D_PTCLOUD_HPP

namespace cv {

//! @addtogroup _3d
//! @{

/**
 * @brief Point cloud sampling by Voxel Grid filter downsampling.
 *
 * Creates a 3D voxel grid (a set of tiny 3D boxes in space) over the input
 * point cloud data, in each voxel (i.e., 3D box), all the points present will be
 * approximated (i.e., downsampled) with the point closest to their centroid.
 *
 * @param[out] sampled_point_flags  Flags of the sampled point, (pass in std::vector<int> or std::vector<char> etc.)
 *                     sampled_point_flags[i] is 1 means i-th point selected, 0 means it is not selected.
 * @param input_pts  Original point cloud, vector of Point3 or Mat of size Nx3/3xN.
 * @param length Grid length.
 * @param width  Grid width.
 * @param height  Grid height.
 * @return The number of points actually sampled.
 */
CV_EXPORTS int voxelGridSampling(OutputArray sampled_point_flags, InputArray input_pts,
                                  float length, float width, float height);

/**
 * @brief Point cloud sampling by randomly select points.
 *
 * Use cv::randShuffle to shuffle the point index list,
 * then take the points corresponding to the front part of the list.
 *
 * @param sampled_pts  Point cloud after sampling.
 *                     Support cv::Mat(sampled_pts_size, 3, CV_32F), std::vector<cv::Point3f>.
 * @param input_pts  Original point cloud, vector of Point3 or Mat of size Nx3/3xN.
 * @param sampled_pts_size The desired point cloud size after sampling.
 * @param rng  Optional random number generator used for cv::randShuffle;
 *                      if it is nullptr, theRNG () is used instead.
 */
CV_EXPORTS void randomSampling(OutputArray sampled_pts, InputArray input_pts,
                               int sampled_pts_size, RNG *rng = nullptr);

/**
 * @overload
 *
 * @param sampled_pts  Point cloud after sampling.
 *                     Support cv::Mat(size * sampled_scale, 3, CV_32F), std::vector<cv::Point3f>.
 * @param input_pts  Original point cloud, vector of Point3 or Mat of size Nx3/3xN.
 * @param sampled_scale Range (0, 1), the percentage of the sampled point cloud to the original size,
 *                      that is, sampled size = original size * sampled_scale.
 * @param rng  Optional random number generator used for cv::randShuffle;
 *                      if it is nullptr, theRNG () is used instead.
 */
CV_EXPORTS void randomSampling(OutputArray sampled_pts, InputArray input_pts,
                               float sampled_scale, RNG *rng = nullptr);

/**
 * @brief Point cloud sampling by Farthest Point Sampling(FPS).
 *
 * FPS Algorithm:
 * + Input: Point cloud *C*, *sampled_pts_size*, *dist_lower_limit*
 * + Initialize: Set sampled point cloud S to the empty set
 * + Step:
 *     1. Randomly take a seed point from C and take it from C to S;
 *     2. Find a point in C that is the farthest away from S and take it from C to S;
 *       (The distance from point to set S is the smallest distance from point to all points in S)
 *     3. Repeat *step 2* until the farthest distance of the point in C from S
 *       is less than *dist_lower_limit*, or the size of S is equal to *sampled_pts_size*.
 * + Output: Sampled point cloud S
 *
 * @param[out] sampled_point_flags  Flags of the sampled point, (pass in std::vector<int> or std::vector<char> etc.)
 *                     sampled_point_flags[i] is 1 means i-th point selected, 0 means it is not selected.
 * @param input_pts  Original point cloud, vector of Point3 or Mat of size Nx3/3xN.
 * @param sampled_pts_size The desired point cloud size after sampling.
 * @param dist_lower_limit Sampling is terminated early if the distance from
 *                  the farthest point to S is less than dist_lower_limit, default 0.
 * @param rng Optional random number generator used for selecting seed point for FPS;
 *                  if it is nullptr, theRNG () is used instead.
 * @return The number of points actually sampled.
 */
CV_EXPORTS int farthestPointSampling(OutputArray sampled_point_flags, InputArray input_pts,
                                      int sampled_pts_size, float dist_lower_limit = 0, RNG *rng = nullptr);

/**
 * @overload
 *
 * @param[out] sampled_point_flags  Flags of the sampled point, (pass in std::vector<int> or std::vector<char> etc.)
 *                     sampled_point_flags[i] is 1 means i-th point selected, 0 means it is not selected.
 * @param input_pts  Original point cloud, vector of Point3 or Mat of size Nx3/3xN.
 * @param sampled_scale Range (0, 1), the percentage of the sampled point cloud to the original size,
 *                      that is, sampled size = original size * sampled_scale.
 * @param dist_lower_limit Sampling is terminated early if the distance from
 *                  the farthest point to S is less than dist_lower_limit, default 0.
 * @param rng Optional random number generator used for selecting seed point for FPS;
 *                  if it is nullptr, theRNG () is used instead.
 * @return The number of points actually sampled.
 */
CV_EXPORTS int farthestPointSampling(OutputArray sampled_point_flags, InputArray input_pts,
                                      float sampled_scale, float dist_lower_limit = 0, RNG *rng = nullptr);

//! @} _3d
} //end namespace cv
#endif //OPENCV_3D_PTCLOUD_HPP
