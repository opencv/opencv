// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"
#include "opencv2/imgproc/detail/gcgraph.hpp"

namespace cv { namespace usac {
class GraphCutImpl : public GraphCut {
protected:
    const Ptr<NeighborhoodGraph> neighborhood_graph;
    const Ptr<Estimator> estimator;
    const Ptr<Quality> quality;
    const Ptr<RandomGenerator> lo_sampler;
    const Ptr<Error> error;

    int gc_sample_size, lo_inner_iterations, points_size;
    double spatial_coherence, sqr_trunc_thr, one_minus_lambda;

    std::vector<int> labeling_inliers;
    std::vector<double> energies, weights;
    std::set<int> used_edges;
    std::vector<Mat> gc_models;

    Ptr<Termination> termination;
    int num_lo_optimizations = 0, current_ransac_iter = 0;
public:
    void setCurrentRANSACiter (int ransac_iter) override { current_ransac_iter = ransac_iter; }

    // In lo_sampler_ the sample size should be set and be equal gc_sample_size_
    GraphCutImpl (const Ptr<Estimator> &estimator_, const Ptr<Quality> &quality_,
              const Ptr<NeighborhoodGraph> &neighborhood_graph_, const Ptr<RandomGenerator> &lo_sampler_,
              double threshold_, double spatial_coherence_term, int gc_inner_iteration_number_, Ptr<Termination> termination_) :
              neighborhood_graph (neighborhood_graph_), estimator (estimator_), quality (quality_),
              lo_sampler (lo_sampler_), error (quality_->getErrorFnc()), termination(termination_) {

        points_size = quality_->getPointsSize();
        spatial_coherence = spatial_coherence_term;
        sqr_trunc_thr =  threshold_ * 2.25; // threshold is already squared
        gc_sample_size = lo_sampler_->getSubsetSize();
        lo_inner_iterations = gc_inner_iteration_number_;
        one_minus_lambda = 1.0 - spatial_coherence;

        energies = std::vector<double>(points_size);
        labeling_inliers = std::vector<int>(points_size);
        used_edges = std::set<int>();
        gc_models = std::vector<Mat> (estimator->getMaxNumSolutionsNonMinimal());
    }

    bool refineModel (const Mat &best_model, const Score &best_model_score,
                      Mat &new_model, Score &new_model_score) override {
        if (best_model_score.inlier_number < estimator->getNonMinimalSampleSize())
            return false;

        // improve best model by non minimal estimation
        new_model_score = Score(); // set score to inf (worst case)
        best_model.copyTo(new_model);

        bool is_best_model_updated = true;
        while (is_best_model_updated) {
            is_best_model_updated = false;

            // Build graph problem. Apply graph cut to G
            int labeling_inliers_size = labeling(new_model);
            for (int iter = 0; iter < lo_inner_iterations; iter++) {
                // sample to generate min (|I_7m|, |I|)
                int num_of_estimated_models;
                if (labeling_inliers_size > gc_sample_size) {
                    // generate random subset in range <0; |I|>
                    num_of_estimated_models = estimator->estimateModelNonMinimalSample
                            (lo_sampler->generateUniqueRandomSubset(labeling_inliers,
                                   labeling_inliers_size), gc_sample_size, gc_models, weights);
                } else {
                    if (iter > 0) break; // break inliers are not updated
                    num_of_estimated_models = estimator->estimateModelNonMinimalSample
                            (labeling_inliers, labeling_inliers_size, gc_models, weights);
                }
                for (int model_idx = 0; model_idx < num_of_estimated_models; model_idx++) {
                    const Score gc_temp_score = quality->getScore(gc_models[model_idx]);
                    // store the best model from estimated models
                    if (gc_temp_score.isBetter(new_model_score)) {
                        is_best_model_updated = true;
                        new_model_score = gc_temp_score;
                        gc_models[model_idx].copyTo(new_model);
                    }
                }

                if (termination != nullptr && is_best_model_updated && current_ransac_iter > termination->update(best_model, best_model_score.inlier_number)) {
                    is_best_model_updated = false; // to break outer loop
                }

            } // end of inner GC local optimization
        } // end of while loop
        return true;
    }

private:
    // find inliers using graph cut algorithm.
    int labeling (const Mat& model) {
        const auto &errors = error->getErrors(model);
        detail::GCGraph<double> graph;

        for (int pt = 0; pt < points_size; pt++)
            graph.addVtx();

        // The distance and energy for each point
        double tmp_squared_distance, energy;

        // Estimate the vertex capacities
        for (int pt = 0; pt < points_size; pt++) {
            tmp_squared_distance = errors[pt];
            if (std::isnan(tmp_squared_distance))
                tmp_squared_distance = std::numeric_limits<float>::max();
            energy = tmp_squared_distance / sqr_trunc_thr; // Truncated quadratic cost

            if (tmp_squared_distance <= sqr_trunc_thr)
                graph.addTermWeights(pt, 0, one_minus_lambda * (1 - energy));
            else
                graph.addTermWeights(pt, one_minus_lambda * energy, 0);

            energies[pt] = energy > 1 ? 1 : energy;
        }

        used_edges.clear();

        bool has_edges = false;
        // Iterate through all points and set their edges
        for (int point_idx = 0; point_idx < points_size; ++point_idx) {
            energy = energies[point_idx];

            // Iterate through  all neighbors
            for (int actual_neighbor_idx : neighborhood_graph->getNeighbors(point_idx)) {
                if (actual_neighbor_idx == point_idx ||
                    used_edges.count(actual_neighbor_idx*points_size + point_idx) > 0 ||
                    used_edges.count(point_idx*points_size + actual_neighbor_idx) > 0)
                    continue;

                used_edges.insert(actual_neighbor_idx*points_size + point_idx);
                used_edges.insert(point_idx*points_size + actual_neighbor_idx);

                double a = (0.5 * (energy + energies[actual_neighbor_idx])) * spatial_coherence,
                       b = spatial_coherence, c = spatial_coherence, d = 0;
                graph.addTermWeights(point_idx, d, a);
                b -= a;
                if (b + c < 0)
                    continue; // invalid regularity
                if (b < 0) {
                    graph.addTermWeights(point_idx, 0, b);
                    graph.addTermWeights(actual_neighbor_idx, 0, -b);
                    graph.addEdges(point_idx, actual_neighbor_idx, 0, b + c);
                } else if (c < 0) {
                    graph.addTermWeights(point_idx, 0, -c);
                    graph.addTermWeights(actual_neighbor_idx, 0, c);
                    graph.addEdges(point_idx, actual_neighbor_idx, b + c, 0);
                } else
                    graph.addEdges(point_idx, actual_neighbor_idx, b, c);
                has_edges = true;
            }
        }
        if (! has_edges)
            return quality->getInliers(model, labeling_inliers);
        graph.maxFlow();

        int inlier_number = 0;
        for (int pt = 0; pt < points_size; pt++)
            if (! graph.inSourceSegment(pt)) // check for sink
                labeling_inliers[inlier_number++] = pt;
        return inlier_number;
    }
    int getNumLOoptimizations () const override { return num_lo_optimizations; }
};
Ptr<GraphCut> GraphCut::create(const Ptr<Estimator> &estimator_,
       const Ptr<Quality> &quality_, const Ptr<NeighborhoodGraph> &neighborhood_graph_,
       const Ptr<RandomGenerator> &lo_sampler_, double threshold_,
       double spatial_coherence_term, int gc_inner_iteration_number, Ptr<Termination> termination_) {
    return makePtr<GraphCutImpl>(estimator_, quality_, neighborhood_graph_, lo_sampler_,
        threshold_, spatial_coherence_term, gc_inner_iteration_number, termination_);
}

// http://cmp.felk.cvut.cz/~matas/papers/chum-dagm03.pdf
class InnerIterativeLocalOptimizationImpl : public InnerIterativeLocalOptimization {
private:
    const Ptr<Estimator> estimator;
    const Ptr<Quality> quality;
    const Ptr<RandomGenerator> lo_sampler;
    Ptr<RandomGenerator> lo_iter_sampler;

    std::vector<Mat> lo_models, lo_iter_models;

    std::vector<int> inliers_of_best_model, virtual_inliers;
    int lo_inner_max_iterations, lo_iter_max_iterations, lo_sample_size, lo_iter_sample_size;

    bool is_iterative;

    double threshold, new_threshold, threshold_step;
    std::vector<double> weights;
public:
    InnerIterativeLocalOptimizationImpl (const Ptr<Estimator> &estimator_, const Ptr<Quality> &quality_,
         const Ptr<RandomGenerator> &lo_sampler_, int pts_size,
         double threshold_, bool is_iterative_, int lo_iter_sample_size_,
         int lo_inner_iterations_=10, int lo_iter_max_iterations_=5,
         double threshold_multiplier_=4)
        : estimator (estimator_), quality (quality_), lo_sampler (lo_sampler_)
        , lo_iter_sample_size(0), new_threshold(0), threshold_step(0) {
        lo_inner_max_iterations = lo_inner_iterations_;
        lo_iter_max_iterations = lo_iter_max_iterations_;

        threshold = threshold_;
        lo_sample_size = lo_sampler->getSubsetSize();
        is_iterative = is_iterative_;
        if (is_iterative) {
            lo_iter_sample_size = lo_iter_sample_size_;
            lo_iter_sampler = UniformRandomGenerator::create(0/*state*/, pts_size, lo_iter_sample_size_);
            lo_iter_models = std::vector<Mat>(estimator->getMaxNumSolutionsNonMinimal());
            virtual_inliers = std::vector<int>(pts_size);
            new_threshold = threshold_multiplier_ * threshold;
            // reduce multiplier threshold K·θ by this number in each iteration.
            // In the last iteration there be original threshold θ.
            threshold_step = (new_threshold - threshold) / lo_iter_max_iterations_;
        }
        lo_models = std::vector<Mat>(estimator->getMaxNumSolutionsNonMinimal());

        // Allocate max memory to avoid reallocation
        inliers_of_best_model = std::vector<int>(pts_size);
    }

    /*
     * Implementation of Locally Optimized Ransac
     * Inner + Iterative
     */
    bool refineModel (const Mat &so_far_the_best_model, const Score &best_model_score,
                      Mat &new_model, Score &new_model_score) override {
        if (best_model_score.inlier_number < estimator->getNonMinimalSampleSize())
            return false;

        so_far_the_best_model.copyTo(new_model);
        new_model_score = best_model_score;
        // get inliers from so far the best model.
        int num_inliers_of_best_model = quality->getInliers(so_far_the_best_model,
                                                           inliers_of_best_model);

        // Inner Local Optimization Ransac.
        for (int iters = 0; iters < lo_inner_max_iterations; iters++) {
            int num_estimated_models;
            // Generate sample of lo_sample_size from inliers from the best model.
            if (num_inliers_of_best_model > lo_sample_size) {
                // if there are many inliers take limited number at random.
                num_estimated_models = estimator->estimateModelNonMinimalSample
                        (lo_sampler->generateUniqueRandomSubset(inliers_of_best_model,
                                num_inliers_of_best_model), lo_sample_size, lo_models, weights);
            } else {
                // if model was not updated in first iteration, so break.
                if (iters > 0) break;
                // if inliers are less than limited number of sample then take all for estimation
                // if it fails -> end Lo.
                num_estimated_models = estimator->estimateModelNonMinimalSample
                    (inliers_of_best_model, num_inliers_of_best_model, lo_models, weights);
            }

            //////// Choose the best lo_model from estimated lo_models.
            for (int model_idx = 0; model_idx < num_estimated_models; model_idx++) {
                const Score temp_score = quality->getScore(lo_models[model_idx]);
                if (temp_score.isBetter(new_model_score)) {
                    new_model_score = temp_score;
                    lo_models[model_idx].copyTo(new_model);
                }
            }

            if (is_iterative) {
                double lo_threshold = new_threshold;
                // get max virtual inliers. Note that they are nor real inliers,
                // because we got them with bigger threshold.
                int virtual_inliers_size = quality->getInliers
                        (new_model, virtual_inliers, lo_threshold);

                Mat lo_iter_model;
                Score lo_iter_score = Score(); // set worst case
                for (int iterations = 0; iterations < lo_iter_max_iterations; iterations++) {
                    lo_threshold -= threshold_step;

                    if (virtual_inliers_size > lo_iter_sample_size) {
                        // if there are more inliers than limit for sample size then generate at random
                        // sample from LO model.
                        num_estimated_models = estimator->estimateModelNonMinimalSample
                                (lo_iter_sampler->generateUniqueRandomSubset (virtual_inliers,
                            virtual_inliers_size), lo_iter_sample_size, lo_iter_models, weights);
                    } else {
                        // break if failed, very low probability that it will not fail in next iterations
                        // estimate model with all virtual inliers
                        num_estimated_models = estimator->estimateModelNonMinimalSample
                                (virtual_inliers, virtual_inliers_size, lo_iter_models, weights);
                    }
                    if (num_estimated_models == 0) break;

                    // Get score and update virtual inliers with current threshold
                    ////// Choose the best lo_iter_model from estimated lo_iter_models.
                    lo_iter_models[0].copyTo(lo_iter_model);
                    lo_iter_score = quality->getScore(lo_iter_model);
                    for (int model_idx = 1; model_idx < num_estimated_models; model_idx++) {
                        const Score temp_score = quality->getScore(lo_iter_models[model_idx]);
                        if (temp_score.isBetter(lo_iter_score)) {
                            lo_iter_score = temp_score;
                            lo_iter_models[model_idx].copyTo(lo_iter_model);
                        }
                    }

                    if (iterations != lo_iter_max_iterations-1)
                        virtual_inliers_size = quality->getInliers(lo_iter_model, virtual_inliers, lo_threshold);
                }

                if (lo_iter_score.isBetter(new_model_score)) {
                    new_model_score = lo_iter_score;
                    lo_iter_model.copyTo(new_model);
                }
            }

            if (num_inliers_of_best_model < new_model_score.inlier_number && iters != lo_inner_max_iterations-1)
                num_inliers_of_best_model = quality->getInliers (new_model, inliers_of_best_model);
        }
        return true;
    }
};
Ptr<InnerIterativeLocalOptimization> InnerIterativeLocalOptimization::create
(const Ptr<Estimator> &estimator_, const Ptr<Quality> &quality_,
       const Ptr<RandomGenerator> &lo_sampler_, int pts_size,
       double threshold_, bool is_iterative_, int lo_iter_sample_size_,
       int lo_inner_iterations_, int lo_iter_max_iterations_,
       double threshold_multiplier_) {
    return makePtr<InnerIterativeLocalOptimizationImpl>(estimator_, quality_, lo_sampler_,
            pts_size, threshold_, is_iterative_, lo_iter_sample_size_,
            lo_inner_iterations_, lo_iter_max_iterations_, threshold_multiplier_);
}

class SimpleLocalOptimizationImpl : public SimpleLocalOptimization {
private:
    const Ptr<Quality> quality;
    const Ptr<Error> error;
    const Ptr<NonMinimalSolver> estimator;
    const Ptr<Termination> termination;
    const Ptr<RandomGenerator> random_generator;
    const Ptr<WeightFunction> weight_fnc;
    // unlike to @random_generator which has fixed subset size
    // @random_generator_smaller_subset is used to draw smaller
    // amount of points which depends on current number of inliers
    Ptr<RandomGenerator> random_generator_smaller_subset;
    int points_size, max_lo_iters, non_min_sample_size, current_ransac_iter;
    std::vector<double> weights;
    std::vector<int> inliers;
    std::vector<cv::Mat> models;
    double inlier_threshold_sqr;
    int num_lo_optimizations = 0;
    bool updated_lo = false;
public:
    SimpleLocalOptimizationImpl (const Ptr<Quality> &quality_, const Ptr<NonMinimalSolver> &estimator_,
            const Ptr<Termination> termination_, const Ptr<RandomGenerator> &random_gen, Ptr<WeightFunction> weight_fnc_,
            int max_lo_iters_, double inlier_threshold_sqr_, bool update_lo_) :
            quality(quality_), error(quality_->getErrorFnc()), estimator(estimator_), termination(termination_),
            random_generator(random_gen), weight_fnc(weight_fnc_) {
        max_lo_iters = max_lo_iters_;
        non_min_sample_size = random_generator->getSubsetSize();
        current_ransac_iter = 0;
        inliers = std::vector<int>(quality_->getPointsSize());
        models = std::vector<cv::Mat>(estimator_->getMaxNumberOfSolutions());
        points_size = quality_->getPointsSize();
        inlier_threshold_sqr = inlier_threshold_sqr_;
        if (weight_fnc != nullptr) weights = std::vector<double>(points_size);
        random_generator_smaller_subset = nullptr;
        updated_lo = update_lo_;
    }
    void setCurrentRANSACiter (int ransac_iter) override { current_ransac_iter = ransac_iter; }
    int getNumLOoptimizations () const override { return num_lo_optimizations; }
    bool refineModel (const Mat &best_model, const Score &best_model_score, Mat &new_model, Score &new_model_score) override {
        new_model_score = best_model_score;
        best_model.copyTo(new_model);

        int num_inliers;
        if (weights.empty())
            num_inliers = Quality::getInliers(error, best_model, inliers, inlier_threshold_sqr);
        else num_inliers = weight_fnc->getInliersWeights(error->getErrors(best_model), inliers, weights);
        auto update_generator = [&] (int num_inls) {
            if (num_inls <= non_min_sample_size) {
                // add new random generator if number of inliers is fewer than non-minimal sample size
                const int new_sample_size = (int)(0.6*num_inls);
                if (new_sample_size <= estimator->getMinimumRequiredSampleSize())
                    return false;
                if (random_generator_smaller_subset == nullptr)
                    random_generator_smaller_subset = UniformRandomGenerator::create(num_inls/*state*/, quality->getPointsSize(), new_sample_size);
                else random_generator_smaller_subset->setSubsetSize(new_sample_size);
            }
            return true;
        };
        if (!update_generator(num_inliers))
            return false;

        int max_lo_iters_ = max_lo_iters, last_update = 0, last_inliers_update = 0;
        for (int iter = 0; iter < max_lo_iters_; iter++) {
            int num_models;
            if (num_inliers <= non_min_sample_size)
                 num_models = estimator->estimate(new_model, random_generator_smaller_subset->generateUniqueRandomSubset(inliers, num_inliers),
                        random_generator_smaller_subset->getSubsetSize(), models, weights);
            else num_models = estimator->estimate(new_model, random_generator->generateUniqueRandomSubset(inliers, num_inliers), non_min_sample_size, models, weights);
            for (int m = 0; m < num_models; m++) {
                const auto score = quality->getScore(models[m]);
                if (score.isBetter(new_model_score)) {
                    last_update = iter; last_inliers_update = new_model_score.inlier_number - score.inlier_number;
                    if (updated_lo) {
                        if (max_lo_iters_ < iter + 5 && last_inliers_update >= 1) {
                            max_lo_iters_ = iter + 5;
                        }
                    }
                    models[m].copyTo(new_model);
                    new_model_score = score;
                    if (termination != nullptr && current_ransac_iter > termination->update(new_model, new_model_score.inlier_number))
                        return true; // terminate LO if max number of iterations reached
                    if (score.inlier_number >= best_model_score.inlier_number ||
                        score.inlier_number > non_min_sample_size) {
                        // update inliers if new model has more than previous best model
                        if (weights.empty())
                            num_inliers = Quality::getInliers(error, best_model, inliers, inlier_threshold_sqr);
                        else num_inliers = weight_fnc->getInliersWeights(error->getErrors(best_model), inliers, weights);
                        if (!update_generator(num_inliers))
                            return true;
                    }

                }
            }
            if (updated_lo && iter - last_update >= 10) {
                break;
            }
        }
        return true;
    }
};
Ptr<SimpleLocalOptimization> SimpleLocalOptimization::create (const Ptr<Quality> &quality_,
        const Ptr<NonMinimalSolver> &estimator_, const Ptr<Termination> termination_, const Ptr<RandomGenerator> &random_gen,
        const Ptr<WeightFunction> weight_fnc, int max_lo_iters_, double inlier_thr_sqr, bool updated_lo) {
    return makePtr<SimpleLocalOptimizationImpl> (quality_, estimator_, termination_, random_gen, weight_fnc, max_lo_iters_, inlier_thr_sqr, updated_lo);
}

class MagsacWeightFunctionImpl : public MagsacWeightFunction {
private:
    const std::vector<double> &stored_gamma_values;
    double C, max_sigma, max_sigma_sqr, scale_of_stored_gammas, one_over_sigma, gamma_k, squared_sigma_max_2, rescale_err;
    int DoF;
    unsigned int stored_gamma_number_min1;
public:
    MagsacWeightFunctionImpl (const Ptr<GammaValues> &gamma_generator,
            int DoF_, double upper_incomplete_of_sigma_quantile, double C_, double max_sigma_) :
            stored_gamma_values (gamma_generator->getGammaValues()) {
        gamma_k = upper_incomplete_of_sigma_quantile;
        stored_gamma_number_min1 = static_cast<unsigned int>(gamma_generator->getTableSize()-1);
        scale_of_stored_gammas = gamma_generator->getScaleOfGammaValues();
        DoF = DoF_; C = C_;
        max_sigma = max_sigma_;
        squared_sigma_max_2 = max_sigma * max_sigma * 2.0;
        one_over_sigma = C * pow(2.0, (DoF - 1.0) * 0.5) / max_sigma;
        max_sigma_sqr = squared_sigma_max_2 * 0.5;
        rescale_err = scale_of_stored_gammas / squared_sigma_max_2;
    }
    int getInliersWeights (const std::vector<float> &errors, std::vector<int> &inliers, std::vector<double> &weights) const override {
        return getInliersWeights(errors, inliers, weights, one_over_sigma, rescale_err, max_sigma_sqr);
    }
    int getInliersWeights (const std::vector<float> &errors, std::vector<int> &inliers, std::vector<double> &weights, double thr_sqr) const override {
        const auto _max_sigma = thr_sqr;
        const auto _squared_sigma_max_2 = _max_sigma * _max_sigma * 2.0;
        const auto _one_over_sigma = C * pow(2.0, (DoF - 1.0) * 0.5) / _max_sigma;
        const auto _max_sigma_sqr = _squared_sigma_max_2 * 0.5;
        const auto _rescale_err = scale_of_stored_gammas / _squared_sigma_max_2;
        return getInliersWeights(errors, inliers, weights, _one_over_sigma, _rescale_err, _max_sigma_sqr);
    }
    double getThreshold () const override {
        return max_sigma_sqr;
    }
private:
    int getInliersWeights (const std::vector<float> &errors, std::vector<int> &inliers, std::vector<double> &weights,
            double _one_over_sigma, double _rescale_err, double _max_sigma_sqr) const {
        int num_inliers = 0, p = 0;
        for (const auto &e : errors) {
            // Calculate the residual of the current point
            if (e < _max_sigma_sqr) {
                // Get the position of the gamma value in the lookup table
                auto x = static_cast<unsigned int>(_rescale_err * e);
                if (x > stored_gamma_number_min1)
                    x = stored_gamma_number_min1;
                inliers[num_inliers] = p; // store index of point for LSQ
                weights[num_inliers++] = _one_over_sigma * (stored_gamma_values[x] - gamma_k);
            }
            p++;
        }
        return num_inliers;
    }
};
Ptr<MagsacWeightFunction> MagsacWeightFunction::create(const Ptr<GammaValues> &gamma_generator_,
            int DoF_, double upper_incomplete_of_sigma_quantile, double C_, double max_sigma_) {
    return makePtr<MagsacWeightFunctionImpl>(gamma_generator_, DoF_, upper_incomplete_of_sigma_quantile, C_, max_sigma_);
}

/////////////////////////////////////////// FINAL MODEL POLISHER ////////////////////////
class NonMinimalPolisherImpl : public NonMinimalPolisher {
private:
    const Ptr<Quality> quality;
    const Ptr<NonMinimalSolver> solver;
    const Ptr<Error> error_fnc;
    const Ptr<WeightFunction> weight_fnc;
    std::vector<bool> mask, mask_best;
    std::vector<Mat> models;
    std::vector<double> weights;
    std::vector<float> errors_best;
    std::vector<int> inliers;
    double threshold, iou_thr, max_thr;
    int max_iters, points_size;
    bool is_covariance, CHANGE_WEIGHTS = true;
public:
    NonMinimalPolisherImpl (const Ptr<Quality> &quality_, const Ptr<NonMinimalSolver> &solver_,
            Ptr<WeightFunction> weight_fnc_, int max_iters_, double iou_thr_) :
            quality(quality_), solver(solver_), error_fnc(quality_->getErrorFnc()), weight_fnc(weight_fnc_) {
        max_iters = max_iters_;
        points_size = quality_->getPointsSize();
        threshold = quality_->getThreshold();
        iou_thr = iou_thr_;
        is_covariance = dynamic_cast<const cv::usac::CovarianceSolver*>(solver_.get()) != nullptr;
        mask = std::vector<bool>(points_size);
        mask_best = std::vector<bool>(points_size);
        inliers = std::vector<int>(points_size);
        if (weight_fnc != nullptr) {
            weights = std::vector<double>(points_size);
            max_thr = weight_fnc->getThreshold();
            if (is_covariance)
                CV_Error(cv::Error::StsBadArg, "Covariance polisher cannot be combined with weights!");
        }
    }

    bool polishSoFarTheBestModel (const Mat &model, const Score &best_model_score,
                                  Mat &new_model, Score &new_model_score) override {
        int num_inliers = 0;
        if (weights.empty()) {
            quality->getInliers(model, mask_best);
            if (!is_covariance)
                for (int p = 0; p < points_size; p++)
                    if (mask_best[p]) inliers[num_inliers++] = p;
        } else {
            errors_best = error_fnc->getErrors(model);
            num_inliers = weight_fnc->getInliersWeights(errors_best, inliers, weights, max_thr);
        }
        new_model_score = best_model_score;
        model.copyTo(new_model);
        int last_update = -1;
        for (int iter = 0; iter < max_iters; iter++) {
            int num_sols;
            if (is_covariance) num_sols = solver->estimate(mask_best, models, weights);
            else num_sols = solver->estimate(new_model, inliers, num_inliers, models, weights);
            Score prev_score;
            for (int i = 0; i < num_sols; i++) {
                const auto &errors = error_fnc->getErrors(models[i]);
                const auto score = quality->getScore(errors);
                if (score.isBetter(new_model_score)) {
                    last_update = iter;
                    models[i].copyTo(new_model);
                    errors_best = errors;
                    prev_score = new_model_score;
                    new_model_score = score;
                }
            }
            if (weights.empty()) {
                if (iter > last_update)
                    break;
                else {
                    Quality::getInliers(errors_best, mask, threshold);
                    if (Utils::intersectionOverUnion(mask, mask_best) >= iou_thr)
                        return true;
                    mask_best = mask;
                    num_inliers = 0;
                    if (!is_covariance)
                        for (int p = 0; p < points_size; p++)
                            if (mask_best[p]) inliers[num_inliers++] = p;
                }
            } else {
                if (iter > last_update) {
                    // new model is worse
                    if (CHANGE_WEIGHTS) {
                        // if failed more than 5 times then break
                        if (iter - std::max(0, last_update) >= 5)
                            break;
                        // try to change weights by changing threshold
                        if (fabs(new_model_score.score - prev_score.score) < FLT_EPSILON) {
                            // increase threshold if new model score is the same as the same as previous
                            max_thr *= 1.05; // increase by 5%
                        } else if (iter > last_update) {
                            // decrease max threshold if model is worse
                            max_thr *= 0.9;  // decrease by 10%
                        }
                    } else break; // break if not changing weights
                }
                // generate new weights and inliers
                num_inliers = weight_fnc->getInliersWeights(errors_best, inliers, weights, max_thr);
            }
        }
        return last_update >= 0;
    }
};
Ptr<NonMinimalPolisher> NonMinimalPolisher::create(const Ptr<Quality> &quality_, const Ptr<NonMinimalSolver> &solver_,
            Ptr<WeightFunction> weight_fnc_, int max_iters_, double iou_thr_) {
    return makePtr<NonMinimalPolisherImpl>(quality_, solver_, weight_fnc_, max_iters_, iou_thr_);
}
}}
