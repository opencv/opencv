// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021, Yechun Ruan <ruanyc@mail.sustech.edu.cn>

#ifndef OPENCV_3D_SAC_SEGMENTATION_HPP
#define OPENCV_3D_SAC_SEGMENTATION_HPP

#include "opencv2/3d/ptcloud.hpp"

namespace cv {

class SACSegmentationImpl : public SACSegmentation
{
private:
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

    bool is_parallel;

    //! 64-bit value used to initialize the RNG(Random Number Generator).
    uint64 rng_state;

    //! A user defined function that takes model coefficients and returns whether the model is acceptable or not.
    ModelConstraintFunction custom_model_constraints;

    /**
     * @brief Execute segmentation of a single model using the sample consensus method.
     *
     * @param model_coeffs Point cloud data, it must be a 3xN CV_32F Mat.
     * @param label label[i] is 1 means point i is inlier point of model
     * @param model_coefficients The resultant model coefficients.
     * @return number of model inliers
     */
    int segmentSingle(Mat &model_coeffs, std::vector<bool> &label, Mat &model_coefficients);

public:
    //! No-argument constructor using default configuration
    SACSegmentationImpl(SacModelType sac_model_type_, SacMethod sac_method_,
            double threshold_, int max_iterations_)
            : sac_model_type(sac_model_type_), sac_method(sac_method_), threshold(threshold_),
              radius_min(DBL_MIN), radius_max(DBL_MAX),
              max_iterations(max_iterations_), confidence(0.999), number_of_models_expected(1),
              is_parallel(false), rng_state(0),
              custom_model_constraints()
    {
    }


    int segment(InputArray input_pts, OutputArray labels, OutputArray models_coefficients) override;

    //-------------------------- Getter and Setter -----------------------

    void setSacModelType(SacModelType sac_model_type_) override
    { sac_model_type = sac_model_type_; }

    SacModelType getSacModelType() const override
    { return sac_model_type; }

    void setSacMethodType(SacMethod sac_method_) override
    { sac_method = sac_method_; }

    SacMethod getSacMethodType() const override
    { return sac_method; }

    void setDistanceThreshold(double threshold_) override
    { threshold = threshold_; }

    double getDistanceThreshold() const override
    { return threshold; }

    void setRadiusLimits(double radius_min_, double radius_max_) override
    { radius_min = radius_min_;
        radius_max = radius_max_; }

    void getRadiusLimits(double &radius_min_, double &radius_max_) const override
    { radius_min_ = radius_min;
        radius_max_ = radius_max; }

    void setMaxIterations(int max_iterations_) override
    { max_iterations = max_iterations_; }

    int getMaxIterations() const override
    { return max_iterations; }

    void setConfidence(double confidence_) override
    { confidence = confidence_; }

    double getConfidence() const override
    { return confidence; }

    void setNumberOfModelsExpected(int number_of_models_expected_) override
    { number_of_models_expected = number_of_models_expected_; }

    int getNumberOfModelsExpected() const override
    { return number_of_models_expected; }

    void setParallel(bool is_parallel_) override { is_parallel = is_parallel_; }

    bool isParallel() const override { return is_parallel; }

    void setRandomGeneratorState(uint64 rng_state_) override
    { rng_state = rng_state_; }

    uint64 getRandomGeneratorState() const override
    { return rng_state; }

    void
    setCustomModelConstraints(const ModelConstraintFunction &custom_model_constraints_) override
    { custom_model_constraints = custom_model_constraints_; }

    const ModelConstraintFunction &getCustomModelConstraints() const override
    { return custom_model_constraints; }

};

Ptr <SACSegmentation> SACSegmentation::create(SacModelType sac_model_type_, SacMethod sac_method_,
        double threshold_, int max_iterations_)
{
    return makePtr<SACSegmentationImpl>(sac_model_type_, sac_method_, threshold_, max_iterations_);
}

} //end namespace cv

#endif //OPENCV_3D_SAC_SEGMENTATION_HPP
