// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021, Yechun Ruan <ruanyc@mail.sustech.edu.cn>


#ifndef OPENCV_3D_PTCLOUD_HPP
#define OPENCV_3D_PTCLOUD_HPP

namespace cv {

//! @addtogroup _3d
//! @{


//! type of the robust estimation algorithm
enum SacMethod
{
    /** The RANSAC algorithm described in @cite fischler1981random.
     */
    SAC_METHOD_RANSAC,
    //    SAC_METHOD_MAGSAC,
    //    SAC_METHOD_LMEDS,
    //    SAC_METHOD_MSAC,
    //    SAC_METHOD_RRANSAC,
    //    SAC_METHOD_RMSAC,
    //    SAC_METHOD_MLESAC,
    //    SAC_METHOD_PROSAC
};

enum SacModelType
{
    /** The 3D PLANE model coefficients in list **[a, b, c, d]**,
     corresponding to the coefficients of equation
     \f$ ax + by + cz + d = 0 \f$. */
    SAC_MODEL_PLANE,
    /** The 3D SPHERE model coefficients in list **[center_x, center_y, center_z, radius]**,
     corresponding to the coefficients of equation
     \f$ (x - center\_x)^2 + (y - center\_y)^2 + (z - center\_z)^2 = radius^2 \f$.*/
    SAC_MODEL_SPHERE,
    //    SAC_MODEL_CYLINDER,

};


/** @brief Sample Consensus algorithm segmentation of 3D point cloud model.

Example of segmenting plane from a 3D point cloud using the RANSAC algorithm:
@snippet snippets/3d_sac_segmentation.cpp planeSegmentationUsingRANSAC

@see
1. Supported algorithms: enum SacMethod in ptcloud.hpp.
2. Supported models: enum SacModelType in ptcloud.hpp.
 */
class CV_EXPORTS SACSegmentation
{
public:
    /** @brief Custom function that take the model coefficients and return whether the model is acceptable or not.

     Example of constructing SACSegmentation::ModelConstraintFunction:
     @snippet snippets/3d_sac_segmentation.cpp usageExampleSacModelConstraintFunction

     @note The content of model_coefficients depends on the model.
     Refer to the comments inside enumeration type SacModelType.
     */
    using ModelConstraintFunction =
    std::function<bool(const std::vector<double> &/*model_coefficients*/)>;

    //-------------------------- CREATE -----------------------

    static Ptr<SACSegmentation> create(SacModelType sac_model_type = SAC_MODEL_PLANE,
            SacMethod sac_method = SAC_METHOD_RANSAC,
            double threshold = 0.5, int max_iterations = 1000);

    // -------------------------- CONSTRUCTOR, DESTRUCTOR --------------------------

    SACSegmentation() = default;

    virtual ~SACSegmentation() = default;

    //-------------------------- SEGMENT -----------------------

    /**
     * @brief Execute segmentation using the sample consensus method.
     *
     * @param input_pts Original point cloud, vector of Point3 or Mat of size Nx3/3xN.
     * @param[out] labels The label corresponds to the model number, 0 means it
     * does not belong to any model, range [0, Number of final resultant models obtained].
     * @param[out] models_coefficients The resultant models coefficients.
     * Currently supports passing in cv::Mat. Models coefficients are placed in a matrix of NxK
     * with depth CV_64F (will automatically adjust if the passing one does not look like this),
     * where N is the number of models and K is the number of coefficients of one model.
     * The coefficients for each model refer to the comments inside enumeration type SacModelType.
     * @return Number of final resultant models obtained by segmentation.
     */
    virtual int
    segment(InputArray input_pts, OutputArray labels, OutputArray models_coefficients) = 0;

    //-------------------------- Getter and Setter -----------------------

    //! Set the type of sample consensus model to use.
    virtual void setSacModelType(SacModelType sac_model_type) = 0;

    //! Get the type of sample consensus model used.
    virtual SacModelType getSacModelType() const = 0;

    //! Set the type of sample consensus method to use.
    virtual void setSacMethodType(SacMethod sac_method) = 0;

    //! Get the type of sample consensus method used.
    virtual SacMethod getSacMethodType() const = 0;

    //! Set the distance to the model threshold.
    //! Considered as inlier point if distance to the model less than threshold.
    virtual void setDistanceThreshold(double threshold) = 0;

    //! Get the distance to the model threshold.
    virtual double getDistanceThreshold() const = 0;

    //! Set the minimum and maximum radius limits for the model.
    //! Only used for models whose model parameters include a radius.
    virtual void setRadiusLimits(double radius_min, double radius_max) = 0;

    //! Get the minimum and maximum radius limits for the model.
    virtual void getRadiusLimits(double &radius_min, double &radius_max) const = 0;

    //! Set the maximum number of iterations to attempt.
    virtual void setMaxIterations(int max_iterations) = 0;

    //! Get the maximum number of iterations to attempt.
    virtual int getMaxIterations() const = 0;

    //! Set the confidence that ensure at least one of selections is an error-free set of data points.
    virtual void setConfidence(double confidence) = 0;

    //! Get the confidence that ensure at least one of selections is an error-free set of data points.
    virtual double getConfidence() const = 0;

    //! Set the number of models expected.
    virtual void setNumberOfModelsExpected(int number_of_models_expected) = 0;

    //! Get the expected number of models.
    virtual int getNumberOfModelsExpected() const = 0;

    //! Set whether to use parallelism or not.
    //! The number of threads is set by cv::setNumThreads(int nthreads).
    virtual void setParallel(bool is_parallel) = 0;

    //! Get whether to use parallelism or not.
    virtual bool isParallel() const = 0;

    //! Set state used to initialize the RNG(Random Number Generator).
    virtual void setRandomGeneratorState(uint64 rng_state) = 0;

    //! Get state used to initialize the RNG(Random Number Generator).
    virtual uint64 getRandomGeneratorState() const = 0;

    //! Set custom model coefficient constraint function.
    //! A custom function that takes model coefficients and returns whether the model is acceptable or not.
    virtual void
    setCustomModelConstraints(const ModelConstraintFunction &custom_model_constraints) = 0;

    //! Get custom model coefficient constraint function.
    virtual const ModelConstraintFunction &getCustomModelConstraints() const = 0;

};


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

/**
 * @brief Estimate the normal and curvature of each point in point cloud from NN results.
 *
 * Normal estimation by PCA:
 * + Input: Nearest neighbor points of a specific point: \f$ pt\_set \f$
 * + Step:
 *     1. Calculate the \f$ mean(\bar{x},\bar{y},\bar{z}) \f$ of \f$ pt\_set \f$;
 *     2. A 3x3 covariance matrix \f$ cov \f$ is obtained by \f$ mean^T \cdot mean \f$;
 *     3. Calculate the eigenvalues(\f$ λ_2 \ge λ_1 \ge λ_0 \f$) and corresponding
 *        eigenvectors(\f$ v_2, v_1, v_0 \f$) of \f$ cov \f$;
 *     4. \f$ v0 \f$ is the normal of the specific point,
 *        \f$ \frac{λ_0}{λ_0 + λ_1 + λ_2} \f$ is the curvature of the specific point;
 * + Output: Normal and curvature of the specific point.
 *
 * @param[out] normals Normal of each point, support vector<Point3f> and Mat of size Nx3.
 * @param[out] curvatures Curvature of each point, support vector<float> and Mat.
 * @param input_pts Original point cloud, support vector<Point3f> and Mat of size Nx3/3xN.
 * @param nn_idx Index information of nearest neighbors of all points. The first nearest neighbor of
 *               each point is itself. Support vector<vector<int>>, vector<Mat> and Mat of size NxK.
 *               If the information in a row is [0, 2, 1, -5, -1, 4, 7 ... negative number], it will
 *               use only non-negative indexes until it meets a negative number or bound of this row
 *               i.e. [0, 2, 1].
 * @param max_neighbor_num The maximum number of neighbors want to use including itself. Setting to
 *               a non-positive number or default will use the information from nn_idx.
 */

CV_EXPORTS void normalEstimate(OutputArray normals, OutputArray curvatures, InputArray input_pts,
        InputArrayOfArrays nn_idx, int max_neighbor_num = 0);

/**
 * @brief Region Growing algorithm in 3D point cloud.
 *
 * The key idea of region growing is to merge the nearest neighbor points that satisfy a certain
 * angle threshold into the same region according to the normal between the two points, so as to
 * achieve the purpose of segmentation. For more details, please refer to @cite Rabbani2006SegmentationOP.
 */
class CV_EXPORTS RegionGrowing3D
{
public:
    //-------------------------- CREATE -----------------------

    static Ptr<RegionGrowing3D> create();

    // -------------------------- CONSTRUCTOR, DESTRUCTOR --------------------------

    RegionGrowing3D() = default;

    virtual ~RegionGrowing3D() = default;

    //-------------------------- SEGMENT -----------------------

    /**
     * @brief Execute segmentation using the Region Growing algorithm.
     *
     * @param[out] regions_idx Index information of all points in each region, support
     *               vector<vector<int>>, vector<Mat>.
     * @param[out] labels The label corresponds to the model number, 0 means it does not belong to
     *               any model, range [0, Number of final resultant models obtained]. Support
     *               vector<int> and Mat.
     * @param input_pts Original point cloud, support vector<Point3f> and Mat of size Nx3/3xN.
     * @param normals Normal of each point, support vector<Point3f> and Mat of size Nx3.
     * @param nn_idx Index information of nearest neighbors of all points. The first nearest
     *               neighbor of each point is itself. Support vector<vector<int>>, vector<Mat> and
     *               Mat of size NxK. If the information in a row is
     *               [0, 2, 1, -5, -1, 4, 7 ... negative number]
     *               it will use only non-negative indexes until it meets a negative number or bound
     *               of this row i.e. [0, 2, 1].
     * @return Number of final resultant regions obtained by segmentation.
     */
    virtual int
    segment(OutputArrayOfArrays regions_idx, OutputArray labels, InputArray input_pts,
            InputArray normals, InputArrayOfArrays nn_idx) = 0;

    //-------------------------- Getter and Setter -----------------------

    //! Set the minimum size of region.
    //！Setting to a non-positive number or default will be unlimited.
    virtual void setMinSize(int min_size) = 0;

    //! Get the minimum size of region.
    virtual int getMinSize() const = 0;

    //! Set the maximum size of region.
    //！Setting to a non-positive number or default will be unlimited.
    virtual void setMaxSize(int max_size) = 0;

    //! Get the maximum size of region.
    virtual int getMaxSize() const = 0;

    //! Set whether to use the smoothness mode. Default will be true.
    //! If true it will check the angle between the normal of the current point and the normal of its neighbor.
    //! Otherwise, it will check the angle between the normal of the seed point and the normal of current neighbor.
    virtual void setSmoothModeFlag(bool smooth_mode) = 0;

    //! Get whether to use the smoothness mode.
    virtual bool getSmoothModeFlag() const = 0;

    //! Set threshold value of the angle between normals, the input value is in radian.
    //！Default will be 30(degree)*PI/180.
    virtual void setSmoothnessThreshold(double smoothness_thr) = 0;

    //! Get threshold value of the angle between normals.
    virtual double getSmoothnessThreshold() const = 0;

    //! Set threshold value of curvature. Default will be 0.05.
    //! Only points with curvature less than the threshold will be considered to belong to the same region.
    //! If the curvature of each point is not set, this option will not work.
    virtual void setCurvatureThreshold(double curvature_thr) = 0;

    //! Get threshold value of curvature.
    virtual double getCurvatureThreshold() const = 0;

    //! Set the maximum number of neighbors want to use including itself.
    //! Setting to a non-positive number or default will use the information from nn_idx.
    virtual void setMaxNumberOfNeighbors(int max_neighbor_num) = 0;

    //! Get the maximum number of neighbors including itself.
    virtual int getMaxNumberOfNeighbors() const = 0;

    //! Set the maximum number of regions you want.
    //！Setting to a non-positive number or default will be unlimited.
    virtual void setNumberOfRegions(int region_num) = 0;

    //! Get the maximum number of regions you want.
    virtual int getNumberOfRegions() const = 0;

    //! Set whether the results need to be sorted in descending order by the number of points.
    virtual void setNeedSort(bool need_sort) = 0;

    //! Get whether the results need to be sorted you have set.
    virtual bool getNeedSort() const = 0;

    //! Set the seed points, it will grow according to the seeds.
    //! If noArray() is set, the default method will be used:
    //! 1. If the curvature of each point is set, the seeds will be sorted in ascending order of curvatures.
    //! 2. Otherwise, the natural order of the point cloud will be used.
    virtual void setSeeds(InputArray seeds) = 0;

    //! Get the seed points.
    virtual void getSeeds(OutputArray seeds) const = 0;

    //! Set the curvature of each point, support vector<float> and Mat. If not, you can set it to noArray().
    virtual void setCurvatures(InputArray curvatures) = 0;

    //! Get the curvature of each point if you have set.
    virtual void getCurvatures(OutputArray curvatures) const = 0;
};
//! @} _3d
} //end namespace cv
#endif //OPENCV_3D_PTCLOUD_HPP
