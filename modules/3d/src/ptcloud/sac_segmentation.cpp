// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021, Yechun Ruan <ruanyc@mail.sustech.edu.cn>


#include "../precomp.hpp"
#include "sac_segmentation.hpp"
#include "opencv2/3d/ptcloud.hpp"
#include "ptcloud_utils.hpp"
#include "../usac.hpp"

namespace cv {
//    namespace _3d {

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


/////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////  SACSegmentationImpl  //////////////////////////////////////

//-------------------------- segmentSingle -----------------------
int
SACSegmentationImpl::segmentSingle(Mat &points, std::vector<bool> &label, Mat &model_coefficients)
{
    CV_CheckDepthEQ(points.depth(), CV_32F, "Data with only depth CV_32F are supported");
    CV_CheckChannelsEQ(points.channels(), 1, "Data with only one channel are supported");
    CV_CheckEQ(points.rows, 3, "Data with only Mat with 3xN are supported");

    // Since error function output squared error distance, so make
    // threshold squared as well
    double _threshold = threshold * threshold;
    int state = (int) rng_state;
    const int points_size = points.rows * points.cols / 3;
    const double _radius_min = radius_min, _radius_max = radius_max;
    const ModelConstraintFunction &_custom_constraint = custom_model_constraints;
    ModelConstraintFunction &constraint_func = custom_model_constraints;

    // RANSAC
    using namespace usac;
    SamplingMethod _sampling_method = SamplingMethod::SAMPLING_UNIFORM;
    LocalOptimMethod _lo_method = LocalOptimMethod::LOCAL_OPTIM_INNER_LO;
    ScoreMethod _score_method = ScoreMethod::SCORE_METHOD_RANSAC;
    NeighborSearchMethod _neighbors_search_method = NeighborSearchMethod::NEIGH_GRID;

    // Local optimization
    int lo_sample_size = 16, lo_inner_iterations = 15, lo_iterative_iterations = 8,
            lo_thr_multiplier = 15, lo_iter_sample_size = 30;

    Ptr <Sampler> sampler;
    Ptr <Quality> quality;
    Ptr <ModelVerifier> verifier;
    Ptr <LocalOptimization> lo;
    Ptr <Degeneracy> degeneracy;
    Ptr <Termination> termination;
    Ptr <FinalModelPolisher> polisher;
    Ptr <MinimalSolver> min_solver;
    Ptr <NonMinimalSolver> non_min_solver;
    Ptr <Estimator> estimator;
    Ptr <usac::Error> error;
    EstimationMethod est_method;
    switch (sac_model_type)
    {
        case SAC_MODEL_PLANE:
            est_method = EstimationMethod::PLANE;
            min_solver = PlaneModelMinimalSolver::create(points);
            non_min_solver = PlaneModelNonMinimalSolver::create(points);
            error = PlaneModelError::create(points);
            break;
            //        case SAC_MODEL_CYLINDER:
            //            min_solver = CylinderModelMinimalSolver::create(points);
            //            non_min_solver = CylinderModelNonMinimalSolver::create(points);
            //            error = CylinderModelError::create(points);
            //            break;
        case SAC_MODEL_SPHERE:
            est_method = EstimationMethod::SPHERE;
            min_solver = SphereModelMinimalSolver::create(points);
            non_min_solver = SphereModelNonMinimalSolver::create(points);
            error = SphereModelError::create(points);
            constraint_func = [_radius_min, _radius_max, _custom_constraint]
                    (const std::vector<double> &model_coeffs) {
                double radius = model_coeffs[3];
                return radius >= _radius_min && radius <= _radius_max &&
                       (!_custom_constraint || _custom_constraint(model_coeffs));
            };
            break;
        default:
            CV_Error(cv::Error::StsNotImplemented, "SAC_MODEL type is not implemented!");
    }

    const int min_sample_size = min_solver->getSampleSize();

    if (points_size < min_sample_size)
    {
        return 0;
    }

    estimator = PointCloudModelEstimator::create(min_solver, non_min_solver, constraint_func);
    sampler = UniformSampler::create(state++, min_sample_size, points_size);
    quality = RansacQuality::create(points_size, _threshold, error);
    verifier = ModelVerifier::create(quality);


    Ptr <RandomGenerator> lo_sampler = UniformRandomGenerator::create(state++, points_size,
            lo_sample_size);

    lo = InnerIterativeLocalOptimization::create(estimator, quality, lo_sampler, points_size,
            _threshold, false, lo_iter_sample_size, lo_inner_iterations,
            lo_iterative_iterations, lo_thr_multiplier);

    degeneracy = makePtr<Degeneracy>();
    termination = StandardTerminationCriteria::create
            (confidence, points_size, min_sample_size, max_iterations);

    Ptr <SimpleUsacConfig> usacConfig = SimpleUsacConfig::create(est_method);
    usacConfig->setMaxIterations(max_iterations);
    usacConfig->setRandomGeneratorState(state);
    usacConfig->setParallel(is_parallel);
    usacConfig->setNeighborsSearchMethod(_neighbors_search_method);
    usacConfig->setSamplingMethod(_sampling_method);
    usacConfig->setScoreMethod(_score_method);
    usacConfig->setLoMethod(_lo_method);
    // The mask is needed to remove the points of the model that has been segmented
    usacConfig->maskRequired(true);


    UniversalRANSAC ransac(usacConfig, points_size, estimator, quality, sampler,
            termination, verifier, degeneracy, lo, polisher);
    Ptr <usac::RansacOutput> ransac_output;
    if (!ransac.run(ransac_output))
    {
        return 0;
    }

    model_coefficients = ransac_output->getModel();
    label = ransac_output->getInliersMask();
    return ransac_output->getNumberOfInliers();
}

//-------------------------- segment -----------------------
int
SACSegmentationImpl::segment(InputArray input_pts, OutputArray labels,
        OutputArray models_coefficients)
{
    Mat points;
    // Get Mat with 3xN CV_32F
    getPointsMatFromInputArray(input_pts, points, 1);
    int pts_size = points.rows * points.cols / 3;

    std::vector<int> _labels(pts_size, 0);
    std::vector<Mat> _models_coefficients;


    // Keep the index array of the point corresponding to the original point
    AutoBuffer<int> ori_pts_idx(pts_size);
    int *pts_idx_ptr = ori_pts_idx.data();
    for (int i = 0; i < pts_size; ++i)
        pts_idx_ptr[i] = i;

    int min_sample_size = sacModelMinimumSampleSize(sac_model_type);
    for (int model_num = 1;
         pts_size >= min_sample_size && model_num <= number_of_models_expected; ++model_num)
    {
        Mat model_coefficients;
        std::vector<bool> label;

        int best_inls = segmentSingle(points, label, model_coefficients);
        if (best_inls < min_sample_size)
            break;

        _models_coefficients.emplace_back(model_coefficients);

        // Move the outlier to the new point cloud and continue to segment the model
        if (model_num != number_of_models_expected)
        {
            cv::Mat tmp_pts(points);
            int next_pts_size = pts_size - best_inls;
            points = cv::Mat(3, next_pts_size, CV_32F);

            // Pointer (base address) of access point data x,y,z
            float *const tmp_pts_ptr_x = (float *) tmp_pts.data;
            float *const tmp_pts_ptr_y = tmp_pts_ptr_x + pts_size;
            float *const tmp_pts_ptr_z = tmp_pts_ptr_y + pts_size;

            float *const next_pts_ptr_x = (float *) points.data;
            float *const next_pts_ptr_y = next_pts_ptr_x + next_pts_size;
            float *const next_pts_ptr_z = next_pts_ptr_y + next_pts_size;

            for (int j = 0, k = 0; k < pts_size; ++k)
            {
                if (label[k])
                {
                    // mark a label on this point
                    _labels[pts_idx_ptr[k]] = model_num;
                }
                else
                {
                    // If it is not inlier of the known plane,
                    //   add the next iteration to find a new plane
                    pts_idx_ptr[j] = pts_idx_ptr[k];
                    next_pts_ptr_x[j] = tmp_pts_ptr_x[k];
                    next_pts_ptr_y[j] = tmp_pts_ptr_y[k];
                    next_pts_ptr_z[j] = tmp_pts_ptr_z[k];
                    ++j;
                }
            }
            pts_size = next_pts_size;
        }
        else
        {
            for (int k = 0; k < pts_size; ++k)
            {
                if (label[k])
                    _labels[pts_idx_ptr[k]] = model_num;
            }
        }
    }

    int number_of_models = (int) _models_coefficients.size();
    if (labels.needed())
    {
        if (number_of_models != 0)
        {
            Mat(_labels).copyTo(labels);
        }
        else
        {
            labels.clear();
        }
    }


    if (models_coefficients.needed())
    {
        if (number_of_models != 0)
        {
            Mat result;
            for (int i = 0; i < number_of_models; i++)
            {
                result.push_back(_models_coefficients[i]);
            }
            result.copyTo(models_coefficients);
        }
        else
        {
            models_coefficients.clear();
        }

    }

    return number_of_models;
}

//    } // _3d::
}  // cv::