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

//! @} _3d
} //end namespace cv
#endif //OPENCV_3D_PTCLOUD_HPP
