// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_3D_PTCLOUD_HPP
#define OPENCV_3D_PTCLOUD_HPP

namespace cv {

//! @addtogroup _3d
//! @{


//! type of the robust estimation algorithm
enum SacMethod
{
    /** "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and
     * Automated Cartography", Martin A. Fischler and Robert C. Bolles, Comm. Of the ACM 24: 381–395, June 1981.
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
    /** The 3D PLANE model coefficients in list [a, b, c, d],
     corresponding to the coefficients of equation
     ax + by + cz + d = 0. */
    SAC_MODEL_PLANE,
    /** The 3D SPHERE model coefficients in list [center_x, center_y, center_z, radius],
     corresponding to the coefficients of equation
     (x - center_x)^2 + (y - center_y)^2 + (z - center_z)^2 = radius^2.*/
    SAC_MODEL_SPHERE,
    //    SAC_MODEL_CYLINDER,

};

/**
 *
 * @param model_type 3D Model
 * @return At least a few are needed to determine a model.
 */
inline int sacModelMinimumSampleSize(SacModelType model_type)
{
    switch (model_type)
    {
        case SAC_MODEL_PLANE:
            return 3;
        case SAC_MODEL_SPHERE:
            return 4;
        default:
            CV_Error(cv::Error::StsNotImplemented, "SacModel Minimum Sample Size not defined!");
    }
}

/** @brief Sample Consensus algorithm segmentation of 3D point cloud model.

The following example is to use the RANSAC algorithm
for plane segmentation of a 3D point cloud.

@code{.cpp}
using namespace cv;

int planeSegmentationUsingRANSAC(const Mat &pt_cloud,
        Mat &plane_coeffs, vector<char> &labels) {

    SACSegmentation sacSegmentation;
    sacSegmentation.setSacModelType(SAC_MODEL_PLANE);
    sacSegmentation.setSacMethodType(SAC_METHOD_RANSAC);
    sacSegmentation.setDistanceThreshold(0.21);
    // The maximum number of iterations to attempt.(default 1000)
    sacSegmentation.setMaxIterations(1500);
    sacSegmentation.setNumberOfModelsExpected(2);
    // Number of final resultant models obtained by segmentation.
    int model_cnt = sacSegmentation.segment(pt_cloud,
            labels, plane_coeffs);

    return model_cnt;
}
@endcode

@see
Supported algorithms: enum SacMethod in ptcloud.hpp. \n
Supported models: enum SacModelType in ptcloud.hpp.
 */
class CV_EXPORTS SACSegmentation : public Algorithm
{
public:
    /** @brief Custom function that take the model coefficients and return whether the model is acceptable or not.

     The following example shows how to construct SACSegmentation::ModelConstraintFunction.
     @code{.cpp}
     bool customFunc(const std::vector<double> &model_coefficients) {
        // check model_coefficients
        // The plane needs to pass through the origin, i.e. ax+by+cz+d=0 --> d==0
        return model_coefficients[3] == 0;
     } // end of function customFunc()

     void example() {
        SACSegmentation::ModelConstraintFunction func_example1 = customFunc;
        SACSegmentation::ModelConstraintFunction func_example2 =
            [](const std::vector<double> &model_coefficients) {
                // check model_coefficients
                // The plane needs to pass through the origin, i.e. ax+by+cz+d=0 --> d==0
                return model_coefficients[3] == 0;
            };

        float x0 = 0.0, y0 = 0.0, z0 = 0.0;
        SACSegmentation::ModelConstraintFunction func_example3 =
            [x0, y0, z0](const std::vector<double> &model_coeffs) -> bool {
                // check model_coefficients
                // The plane needs to pass through the point (x0, y0, z0), i.e. ax0+by0+cz0+d == 0
                return model_coeffs[0]*x0 + model_coeffs[1]*y0 + model_coeffs[2]*z0
                       + model_coeffs[3] == 0;
            };

         auto constraint_func = cv::makePtr<ModelConstraintFunction>(func_example3);
         // ......

     } // end of function example()

     @endcode

     @note The content of model_coefficients depends on the model.
     Refer to the comments inside enumeration type SacModelType.
     */
    using ModelConstraintFunction =
    std::function<bool(const std::vector<double> &/*model_coefficients*/)>;
    //using ModelConstraintFunctionPtr = bool (*)(const std::vector<double> &/*model_coefficients*/);

    //! No-argument constructor using default configuration
    SACSegmentation()
            : sac_model_type(SAC_MODEL_PLANE), sac_method(SAC_METHOD_RANSAC), threshold(0.5),
              radius_min(DBL_MIN), radius_max(DBL_MAX),
              max_iterations(1000), confidence(0.999), number_of_models_expected(1),
              number_of_threads(-1), rng_state(0),
              custom_model_constraints(nullptr)
    {
    }

    ~SACSegmentation() override = default;

    //-------------------------- Getter and Setter -----------------------

    //! Set the type of sample consensus model to use.
    inline void setSacModelType(SacModelType sac_model_type_)
    {
        sac_model_type = sac_model_type_;
    }

    //! Get the type of sample consensus model used.
    inline SacModelType getSacModelType() const
    {
        return sac_model_type;
    }

    //! Set the type of sample consensus method to use.
    inline void setSacMethodType(SacMethod sac_method_)
    {
        sac_method = sac_method_;
    }

    //! Get the type of sample consensus method used.
    inline SacMethod getSacMethodType() const
    {
        return sac_method;
    }

    //! Set the distance to the model threshold.
    inline void setDistanceThreshold(double threshold_)
    {
        threshold = threshold_;
    }

    //! Get the distance to the model threshold.
    inline double getDistanceThreshold() const
    {
        return threshold;
    }

    //! Set the minimum and maximum radius limits for the model.
    //! Only used for models whose model parameters include a radius.
    inline void setRadiusLimits(double radius_min_, double radius_max_)
    {
        radius_min = radius_min_;
        radius_max = radius_max_;
    }

    //! Get the minimum and maximum radius limits for the model.
    inline void getRadiusLimits(double &radius_min_, double &radius_max_) const
    {
        radius_min_ = radius_min;
        radius_max_ = radius_max;
    }

    //! Set the maximum number of iterations to attempt.
    inline void setMaxIterations(int max_iterations_)
    {
        max_iterations = max_iterations_;
    }

    //! Get the maximum number of iterations to attempt.
    inline int getMaxIterations() const
    {
        return max_iterations;
    }

    //! Set the confidence that ensure at least one of selections is an error-free set of data points.
    inline void setConfidence(double confidence_)
    {
        confidence = confidence_;
    }

    //! Get the confidence that ensure at least one of selections is an error-free set of data points.
    inline double getConfidence() const
    {
        return confidence;
    }

    //! Set the number of models expected.
    inline void setNumberOfModelsExpected(int number_of_models_expected_)
    {
        number_of_models_expected = number_of_models_expected_;
    }

    //! Get the expected number of models.
    inline int getNumberOfModelsExpected() const
    {
        return number_of_models_expected;
    }

    /**
     * @brief Set the number of threads to be used.
     *
     * @param number_of_threads_ The number of threads to be used.
     * (0 sets the value automatically, a negative number turns parallelization off)
     *
     * @note Not all SAC methods have a parallel implementation. Some will ignore this setting.
     */
    inline void setNumberOfThreads(int number_of_threads_)
    {
        number_of_threads = number_of_threads_;
    }

    // Get the number of threads to be used.
    inline int getNumberOfThreads() const
    {
        return number_of_threads;
    }

    //! Set state used to initialize the RNG(Random Number Generator).
    inline void setRandomGeneratorState(uint64 rng_state_)
    {
        rng_state = rng_state_;
    }

    //! Get state used to initialize the RNG(Random Number Generator).
    inline uint64 getRandomGeneratorState() const
    {
        return rng_state;
    }

    //! Set custom model coefficient constraint function
    inline void setCustomModelConstraints(Ptr<ModelConstraintFunction> &custom_model_constraints_)
    {
        custom_model_constraints = custom_model_constraints_;
    }

    //! Get custom model coefficient constraint function
    inline void getCustomModelConstraints(Ptr<ModelConstraintFunction> &custom_model_constraints_) const
    {
        custom_model_constraints_ = custom_model_constraints;
    }

    /**
     * @brief Execute segmentation using the sample consensus method.
     *
     * @param input_pts Original point cloud, vector of Point3 or Mat of size Nx3/3xN.
     * @param[out] labels The label corresponds to the model number, 0 means it
     * does not belong to any model, range [0, Number of final resultant models obtained].
     * @param[out] models_coefficients The resultant models coefficients.
     * Currently supports passing in cv::Mat. Models coefficients are placed in a matrix of NxK,
     * where N is the number of models and K is the number of coefficients of one model.
     * The coefficients for each model refer to the comments inside enumeration type SacModelType.
     * @return Number of final resultant models obtained by segmentation.
     */
    int
    segment(InputArray input_pts, OutputArray labels, OutputArray models_coefficients = noArray());

protected:

    //! The type of sample consensus model used.
    SacModelType sac_model_type;

    //! The type of sample consensus method used.
    SacMethod sac_method;

    //! Considered as inlier point if distance to the model less than threshold.
    double threshold;

    //! The minimum and maximum radius limits for the model.
    //! Only used for models whose model parameters include a radius.
    double radius_min, radius_max;

    //!  The maximum number of iterations to attempt.
    int max_iterations;

    //! Confidence that ensure at least one of selections is an error-free set of data points.
    double confidence;

    //! Expected number of models.
    int number_of_models_expected;

    //! The number of threads the scheduler should use, or a negative number if no parallelization is wanted.
    int number_of_threads;

    //! 64-bit value used to initialize the RNG(Random Number Generator).
    uint64 rng_state;

    //! A user defined function that takes model coefficients and returns whether the model is acceptable or not.
    Ptr<ModelConstraintFunction> custom_model_constraints;

    /**
     * @brief Execute segmentation of a single model using the sample consensus method.
     *
     * @param model_coeffs Point cloud data, it must be a 3xN CV_32F Mat.
     * @param label label[i] is 1 means point i is inlier point of model
     * @param model_coefficients The resultant model coefficients.
     * @return number of model inliers
     */
    int segmentSingle(Mat &model_coeffs, std::vector<bool> &label, Mat &model_coefficients);

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
