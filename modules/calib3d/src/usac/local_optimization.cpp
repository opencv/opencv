// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"
#include "opencv2/imgproc/detail/gcgraph.hpp"
#include "gamma_values.hpp"

namespace cv { namespace usac {
class GraphCutImpl : public GraphCut {
protected:
    const Ptr<NeighborhoodGraph> neighborhood_graph;
    const Ptr<Estimator> estimator;
    const Ptr<Quality> quality;
    const Ptr<Sampler> lo_sampler;
    const Ptr<Error> error;

    Score gc_temp_score;
    int gc_sample_size, lo_inner_iterations, points_size;
    double spatial_coherence, sqr_trunc_thr, one_minus_lambda;

    std::vector<int> sample, labeling_inliers;
    std::vector<double> energies, weights;
    std::vector<bool> used_edges;
    std::vector<Mat> gc_models;
public:

    // In lo_sampler_ the sample size should be set and be equal gc_sample_size_
    GraphCutImpl (const Ptr<Estimator> &estimator_, const Ptr<Error> &error_, const Ptr<Quality> &quality_,
              const Ptr<NeighborhoodGraph> &neighborhood_graph_, const Ptr<Sampler> &lo_sampler_,
              double threshold_, double spatial_coherence_term, int gc_inner_iteration_number_) :
              neighborhood_graph (neighborhood_graph_), estimator (estimator_), quality (quality_),
              lo_sampler (lo_sampler_), error (error_) {

        points_size = quality_->getPointsSize();
        spatial_coherence = spatial_coherence_term;
        sqr_trunc_thr =  threshold_ * threshold_ * 2.25;
        gc_sample_size = lo_sampler_->getSampleSize();
        lo_inner_iterations = gc_inner_iteration_number_;
        one_minus_lambda = 1.0 - spatial_coherence;

        energies = std::vector<double>(points_size);
        labeling_inliers = std::vector<int>(points_size);
        sample = std::vector<int>(gc_sample_size);
        used_edges = std::vector<bool>(points_size*points_size);
        gc_models = std::vector<Mat> (estimator->getMaxNumSolutionsNonMinimal());
    }

    bool refineModel (const Mat &best_model, const Score &best_model_score,
                      Mat &new_model, Score &new_model_score) override {
        if (best_model_score.inlier_number < gc_sample_size)
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
                    lo_sampler->generateSample (sample, labeling_inliers_size);
                    // sample from inliers of labeling
                    for (int smpl = 0; smpl < gc_sample_size; smpl++)
                        sample[smpl] = labeling_inliers[sample[smpl]];
                    num_of_estimated_models = estimator->estimateModelNonMinimalSample
                            (sample, gc_sample_size, gc_models, weights);
                    if (num_of_estimated_models == 0)
                        break; // break
                } else {
                    if (iter > 0)
                        break; // break inliers are not updated
                    num_of_estimated_models = estimator->estimateModelNonMinimalSample
                            (labeling_inliers, labeling_inliers_size, gc_models, weights);
                    if (num_of_estimated_models == 0)
                        break;
                }

                bool zero_inliers = false;
                for (int model_idx = 0; model_idx < num_of_estimated_models; model_idx++) {
                    gc_temp_score = quality->getScore(gc_models[model_idx]);
                    if (gc_temp_score.inlier_number == 0){
                        zero_inliers = true; break;
                    }

                    if (best_model_score.isBetter(gc_temp_score))
                        continue;

                    // store the best model from estimated models
                    if (gc_temp_score.isBetter(new_model_score)) {
                        is_best_model_updated = true;
                        new_model_score = gc_temp_score;
                        gc_models[model_idx].copyTo(new_model);
                    }
                }

                if (zero_inliers)
                    break;
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
                continue;
            energy = tmp_squared_distance / sqr_trunc_thr; // Truncated quadratic cost

            if (tmp_squared_distance <= sqr_trunc_thr)
                graph.addTermWeights(pt, 0, one_minus_lambda * (1 - energy));
            else
                graph.addTermWeights(pt, one_minus_lambda * energy, 0);

            if (energy > 1) energy = 1;
            energies[pt] = energy;
        }

        std::fill(used_edges.begin(), used_edges.end(), false);

        // Iterate through all points and set their edges
        for (int point_idx = 0; point_idx < points_size; ++point_idx) {
            energy = energies[point_idx];

            // Iterate through  all neighbors
            for (int actual_neighbor_idx : neighborhood_graph->getNeighbors(point_idx)) {
                if (actual_neighbor_idx == point_idx ||
                    used_edges[actual_neighbor_idx*points_size + point_idx] ||
                    used_edges[point_idx*points_size + actual_neighbor_idx])
                    continue;

                used_edges[actual_neighbor_idx*points_size + point_idx] = true;
                used_edges[point_idx*points_size + actual_neighbor_idx] = true;

                double a = (0.5 * (energy + energies[actual_neighbor_idx])) * spatial_coherence,
                       b = spatial_coherence, c = spatial_coherence, d = 0;
                graph.addTermWeights(point_idx, d, a);
                b -= a;
                if (b + c >= 0)
                    // Non-submodular expansion term detected; smooth costs must be a metric for expansion
                     continue;
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
            }
        }

        graph.maxFlow();

        int inlier_number = 0;
        for (int pt = 0; pt < points_size; pt++)
            if (! graph.inSourceSegment(pt)) // check for sink
                labeling_inliers[inlier_number++] = pt;
        return inlier_number;
    }
    Ptr<LocalOptimization> clone(int state) const override {
        return makePtr<GraphCutImpl>(estimator->clone(), error->clone(), quality->clone(),
                neighborhood_graph,lo_sampler->clone(state), sqrt(sqr_trunc_thr / 2),
                spatial_coherence, lo_inner_iterations);
    }
};
Ptr<GraphCut> GraphCut::create(const Ptr<Estimator> &estimator_, const Ptr<Error> &error_,
       const Ptr<Quality> &quality_, const Ptr<NeighborhoodGraph> &neighborhood_graph_,
       const Ptr<Sampler> &lo_sampler_, double threshold_,
       double spatial_coherence_term, int gc_inner_iteration_number) {
    return makePtr<GraphCutImpl>(estimator_, error_, quality_, neighborhood_graph_, lo_sampler_,
        threshold_, spatial_coherence_term, gc_inner_iteration_number);
}

/*
* http://cmp.felk.cvut.cz/~matas/papers/chum-dagm03.pdf
*/
class InnerIterativeLocalOptimizationImpl : public InnerIterativeLocalOptimization {
private:
    const Ptr<Estimator> estimator;
    const Ptr<Quality> quality;
    const Ptr<Sampler> lo_sampler;
    Ptr<UniformSampler> lo_iter_sampler;

    Score lo_score, lo_iter_score;
    std::vector<Mat> lo_models, lo_iter_models;

    std::vector<int> inliers_of_best_model, lo_sample, lo_iter_sample, virtual_inliers;
    int lo_inner_max_iterations, lo_iter_max_iterations, lo_sample_size, lo_iter_sample_size;

    bool is_sample_limit;

    double threshold, new_threshold, threshold_step;
    std::vector<double> weights;
public:

    InnerIterativeLocalOptimizationImpl (const Ptr<Estimator> &estimator_, const Ptr<Quality> &quality_,
         const Ptr<Sampler> &lo_sampler_, int pts_size,
         double threshold_, bool is_sample_limit_, int lo_iter_sample_size_,
         int lo_inner_iterations_=10, int lo_iter_max_iterations_=5,
         double threshold_multiplier_=4) : estimator (estimator_), quality (quality_),
                           lo_sampler (lo_sampler_) {

        lo_inner_max_iterations = lo_inner_iterations_;
        lo_iter_max_iterations = lo_iter_max_iterations_;

        threshold = threshold_;
        new_threshold = threshold_multiplier_ * threshold;
        // reduce multiplier threshold K·θ by this number in each iteration.
        // In the last iteration there be original threshold θ.
        threshold_step = (new_threshold - threshold) / lo_iter_max_iterations_;

        lo_sample_size = lo_sampler->getSampleSize();

        is_sample_limit = is_sample_limit_;
        if (is_sample_limit_) {
            lo_iter_sample_size = lo_iter_sample_size_;
            lo_iter_sampler = UniformSampler::create(0/*state*/, lo_iter_sample_size_, pts_size);
            lo_iter_sample = std::vector<int>(lo_iter_sample_size_);
        }

        lo_models = std::vector<Mat>(estimator->getMaxNumSolutionsNonMinimal());
        lo_iter_models = std::vector<Mat>(estimator->getMaxNumSolutionsNonMinimal());

        // Allocate max memory to avoid reallocation
        inliers_of_best_model = std::vector<int>(pts_size);
        virtual_inliers = std::vector<int>(pts_size);
        lo_sample = std::vector<int>(lo_sample_size);
    }

    /*
     * Implementation of Locally Optimized Ransac
     * Inner + Iterative
     */
    bool refineModel (const Mat &so_far_the_best_model, const Score &best_model_score,
                      Mat &new_model, Score &new_model_score) override {
        if (best_model_score.inlier_number < lo_sample_size)
            return false;

        new_model_score = Score(); // set score to inf (worst case)

        // get inliers from so far the best model.
        int num_inliers_of_best_model = quality->getInliers(so_far_the_best_model,
                                                           inliers_of_best_model);

        // temp score used to compare estimated models
        Score temp_score;

        // Inner Local Optimization Ransac.
        for (int iters = 0; iters < lo_inner_max_iterations; iters++) {

            int num_estimated_models;
            // Generate sample of lo_sample_size from inliers from the best model.
            if (num_inliers_of_best_model > lo_sample_size) {
                // if there are many inliers take limited number at random.
                lo_sampler->generateSample (lo_sample, num_inliers_of_best_model);
                // get inliers from maximum inliers from lo
                for (int smpl = 0; smpl < lo_sample_size; smpl++)
                    lo_sample[smpl] = inliers_of_best_model[lo_sample[smpl]];

                num_estimated_models = estimator->estimateModelNonMinimalSample
                        (lo_sample, lo_sample_size, lo_models, weights);
                if (num_estimated_models == 0) continue;
            } else {
                // if model was not updated in first iteration, so break.
                if (iters > 0) break;
                // if inliers are less than limited number of sample then take all for estimation
                // if it fails -> end Lo.
                num_estimated_models = estimator->estimateModelNonMinimalSample
                    (inliers_of_best_model, num_inliers_of_best_model, lo_models, weights);
                if (num_estimated_models == 0) return false;
            }

            //////// Choose the best lo_model from estimated lo_models.
            Mat lo_model = lo_models[0];
            lo_score = quality->getScore(lo_model);
            for (int model_idx = 1; model_idx < num_estimated_models; model_idx++) {
                temp_score = quality->getScore(lo_models[model_idx]);
                if (temp_score.isBetter(lo_score)) {
                    lo_score = temp_score;
                    lo_model = lo_models[model_idx];
                }
            }
            ////////////////////

            double lo_threshold = new_threshold;
            // get max virtual inliers. Note that they are nor real inliers,
            // because we got them with bigger threshold.
            int virtual_inliers_size = quality->getInliers
                    (lo_model, virtual_inliers, lo_threshold);

            Mat lo_iter_model;
            lo_iter_score = Score(); // set worst case
            for (int iterations = 0; iterations < lo_iter_max_iterations; iterations++) {
                lo_threshold -= threshold_step;

                if (is_sample_limit && virtual_inliers_size > lo_iter_sample_size) {
                    // if there are more inliers than limit for sample size then generate at random
                    // sample from LO model.

                    lo_iter_sampler->generateSample (lo_iter_sample, virtual_inliers_size);
                    for (int smpl = 0; smpl < lo_iter_sample_size; smpl++)
                        lo_iter_sample[smpl] = virtual_inliers[lo_iter_sample[smpl]];

                    num_estimated_models = estimator->estimateModelNonMinimalSample
                            (lo_iter_sample, lo_iter_sample_size, lo_iter_models, weights);
                    if (num_estimated_models == 0) break;

                } else {
                    // break if failed, very low probability that it will not fail in next iterations
                    // estimate model with all virtual inliers
                    num_estimated_models = estimator->estimateModelNonMinimalSample
                            (virtual_inliers, virtual_inliers_size, lo_iter_models, weights);
                    if (num_estimated_models == 0) break;
                }

                // Get score and update virtual inliers with current threshold
                //////// Choose the best lo_iter_model from estimated lo_iter_models.
                lo_iter_models[0].copyTo(lo_iter_model);
                lo_iter_score = quality->getScore(lo_iter_model);
                for (int model_idx = 1; model_idx < num_estimated_models; model_idx++) {
                    temp_score = quality->getScore(lo_iter_models[model_idx]);
                    if (temp_score.isBetter(lo_iter_score)) {
                        lo_iter_score = temp_score;
                        lo_iter_models[model_idx].copyTo(lo_iter_model);
                    }
                }
                ////////////////////

                virtual_inliers_size = quality->getInliers(lo_iter_model, virtual_inliers, lo_threshold);

                // In case of unlimited sample:
                // break if the best score is bigger, because after decreasing
                // threshold lo score could not be bigger in next iterations.
                if (! is_sample_limit && new_model_score.isBetter(lo_iter_score)) break;
            }

            if (fabs (lo_threshold - threshold) < FLT_EPSILON) {
                // Success, threshold does not differ
                // last score correspond to user-defined threshold. Inliers are real.
                if (lo_iter_score.isBetter(lo_score)) {
                    lo_score = lo_iter_score;
                    lo_model = lo_iter_model;
                }
            }

            if (best_model_score.isBetter(lo_score))
                continue;

            if (lo_score.isBetter(new_model_score)) {
                new_model_score = lo_score;
                lo_model.copyTo(new_model);

                if (num_inliers_of_best_model < new_model_score.inlier_number)
                    num_inliers_of_best_model = quality->getInliers (new_model, inliers_of_best_model);
            }
        }
        return true;
    }
    Ptr<LocalOptimization> clone(int state) const override {
        return makePtr<InnerIterativeLocalOptimizationImpl>(estimator->clone(), quality->clone(),
            lo_sampler->clone(state),(int)inliers_of_best_model.size(), threshold, is_sample_limit,
            lo_iter_sample_size, lo_inner_max_iterations, lo_iter_max_iterations,
            new_threshold / threshold);
    }
};

Ptr<InnerIterativeLocalOptimization> InnerIterativeLocalOptimization::create
(const Ptr<Estimator> &estimator_, const Ptr<Quality> &quality_,
       const Ptr<Sampler> &lo_sampler_, int pts_size,
       double threshold_, bool is_sample_limit_, int lo_iter_sample_size_,
       int lo_inner_iterations_, int lo_iter_max_iterations_,
       double threshold_multiplier_) {
    return makePtr<InnerIterativeLocalOptimizationImpl>(estimator_, quality_, lo_sampler_,
            pts_size, threshold_, is_sample_limit_, lo_iter_sample_size_,
            lo_inner_iterations_, lo_iter_max_iterations_, threshold_multiplier_);
}

/*
* Reference:
* http://cmp.felk.cvut.cz/~matas/papers/chum-dagm03.pdf
*/
class InnerLocalOptimizationImpl : public InnerLocalOptimization {
private:
    const Ptr<Estimator> estimator;
    const Ptr<Quality> quality;
    const Ptr<Sampler> lo_sampler;

    Score lo_score;
    std::vector<Mat> lo_models;
    std::vector<int> inliers_of_best_model, lo_sample;
    std::vector<double> weights;
    int lo_inner_max_iterations, lo_sample_size;
public:

    InnerLocalOptimizationImpl (const Ptr<Estimator> &estimator_,
            const Ptr<Quality> &quality_, const Ptr<Sampler> &lo_sampler_,
            int lo_inner_iterations_)
            : estimator (estimator_), quality (quality_), lo_sampler (lo_sampler_) {

        lo_inner_max_iterations = lo_inner_iterations_;
        lo_sample_size = lo_sampler->getSampleSize();

        // Allocate max memory to avoid reallocation
        inliers_of_best_model = std::vector<int>(quality_->getPointsSize());
        lo_sample = std::vector<int>(lo_sample_size);
        lo_models = std::vector<Mat> (estimator->getMaxNumSolutionsNonMinimal());
    }

    // Implementation of Inner Locally Optimized Ransac
    bool refineModel (const Mat &so_far_the_best_model, const Score &best_model_score,
                      Mat &new_model, Score &new_model_score) override {
        if (best_model_score.inlier_number < lo_sample_size)
            return false;

        new_model_score = Score(); // set score to inf (worst case)

        // get inliers from so far the best model.
        int num_inliers_of_best_model = quality->getInliers(so_far_the_best_model,
                                                           inliers_of_best_model);

        // Inner Local Optimization Ransac.
        for (int iters = 0; iters < lo_inner_max_iterations; iters++) {
            // Generate sample of lo_sample_size from inliers from the best model.
            int num_estimated_models;
            if (num_inliers_of_best_model > lo_sample_size) {
                // if there are many inliers take limited number at random.
                lo_sampler->generateSample (lo_sample, num_inliers_of_best_model);
                // get inliers from maximum inliers from lo
                for (int smpl = 0; smpl < lo_sample_size; smpl++)
                    lo_sample[smpl] = inliers_of_best_model[lo_sample[smpl]];

                num_estimated_models = estimator->estimateModelNonMinimalSample
                        (lo_sample, lo_sample_size, lo_models, weights);
                if (num_estimated_models == 0) continue;
            } else {
                // if model was not updated in first iteration, so break.
                if (iters > 0) break;
                // if inliers are less than limited number of sample then take all of them for estimation
                // if it fails -> end Lo.
                num_estimated_models = estimator->estimateModelNonMinimalSample(
                        inliers_of_best_model, num_inliers_of_best_model, lo_models, weights);
                if (num_estimated_models == 0)
                    return false;
            }

            for (int model_idx = 0; model_idx < num_estimated_models; model_idx++) {
                // get score of new estimated model
                lo_score = quality->getScore(lo_models[model_idx]);

                if (best_model_score.isBetter(lo_score))
                    continue;

                if (lo_score.isBetter(new_model_score)) {
                    // update best model
                    lo_models[model_idx].copyTo(new_model);
                    new_model_score = lo_score;
                }
            }

            if (num_inliers_of_best_model < new_model_score.inlier_number)
                // update inliers of the best model.
                num_inliers_of_best_model = quality->getInliers(new_model,inliers_of_best_model);

        }
        return true;
    }
    Ptr<LocalOptimization> clone(int state) const override {
        return makePtr<InnerLocalOptimizationImpl>(estimator->clone(), quality->clone(),
           lo_sampler->clone(state), lo_inner_max_iterations);
    }
};
Ptr<InnerLocalOptimization> InnerLocalOptimization::create
(const Ptr<Estimator> &estimator_, const Ptr<Quality> &quality_,
       const Ptr<Sampler> &lo_sampler_, int lo_inner_iterations_) {
    return makePtr<InnerLocalOptimizationImpl>(estimator_, quality_, lo_sampler_,
          lo_inner_iterations_);
}

class SigmaConsensusImpl : public SigmaConsensus {
private:
    const Ptr<Estimator> estimator;
    const Ptr<Quality> quality;
    const Ptr<Error> error;
    const Ptr<ModelVerifier> verifier;

    // The degrees of freedom of the data from which the model is estimated.
    // E.g., for models coming from point correspondences (x1,y1,x2,y2), it is 4.
    const int degrees_of_freedom;
    // A 0.99 quantile of the Chi^2-distribution to convert sigma values to residuals
    const double k;
    // Calculating (DoF - 1) / 2 which will be used for the estimation and,
    // due to being constant, it is better to calculate it a priori.
    double dof_minus_one_per_two;
    const double C;
    // The size of a minimal sample used for the estimation
    const int sample_size;
    // Calculating 2^(DoF - 1) which will be used for the estimation and,
    // due to being constant, it is better to calculate it a priori.
    double two_ad_dof;
    // Calculating C * 2^(DoF - 1) which will be used for the estimation and,
    // due to being constant, it is better to calculate it a priori.
    double C_times_two_ad_dof;
    // Calculating the gamma value of (DoF - 1) / 2 which will be used for the estimation and,
    // due to being constant, it is better to calculate it a priori.
    double gamma_value, squared_sigma_max_2, one_over_sigma;
    // Calculating the upper incomplete gamma value of (DoF - 1) / 2 with k^2 / 2.
    const double gamma_k;
    // Calculating the lower incomplete gamma value of (DoF - 1) / 2 which will be used for the estimation and,
    // due to being constant, it is better to calculate it a priori.
    double gamma_difference;
    const int points_size, number_of_irwls_iters;
    const double maximum_threshold, max_sigma;

    std::vector<double> residuals, sigma_weights, stored_gamma_values;
    std::vector<int> residuals_idxs;
    // Models fit by weighted least-squares fitting
    std::vector<Mat> sigma_models;
    // Points used in the weighted least-squares fitting
    std::vector<int> sigma_inliers;
    // Weights used in the the weighted least-squares fitting

    double scale_of_stored_gammas;
public:

    SigmaConsensusImpl (const Ptr<Estimator> &estimator_, const Ptr<Error> &error_,
        const Ptr<Quality> &quality_, const Ptr<ModelVerifier> &verifier_,
        int number_of_irwls_iters_, int DoF, double sigma_quantile,
        double upper_incomplete_of_sigma_quantile, double C_, double maximum_thr) :
        estimator (estimator_), quality(quality_),
          error (error_), verifier(verifier_), degrees_of_freedom(DoF), k (sigma_quantile), C(C_),
          sample_size(estimator_->getMinimalSampleSize()), gamma_k (upper_incomplete_of_sigma_quantile),
          points_size (quality_->getPointsSize()), number_of_irwls_iters (number_of_irwls_iters_),
          maximum_threshold(maximum_thr), max_sigma (maximum_thr) {

        dof_minus_one_per_two = (degrees_of_freedom - 1.0) / 2.0;
        two_ad_dof = std::pow(2.0, dof_minus_one_per_two);
        C_times_two_ad_dof = C * two_ad_dof;
        gamma_value = tgamma(dof_minus_one_per_two);
        gamma_difference = gamma_value - gamma_k;
        // Calculate 2 * \sigma_{max}^2 a priori
        squared_sigma_max_2 = max_sigma * max_sigma * 2.0;
        // Divide C * 2^(DoF - 1) by \sigma_{max} a priori
        one_over_sigma = C_times_two_ad_dof / max_sigma;

        residuals = std::vector<double>(points_size);
        residuals_idxs = std::vector<int>(points_size);
        sigma_inliers = std::vector<int>(points_size);
        sigma_weights = std::vector<double>(points_size);
        sigma_models = std::vector<Mat>(estimator->getMaxNumSolutionsNonMinimal());

        if (DoF == 4) {
            scale_of_stored_gammas = scale_of_stored_gammas_n4;
            stored_gamma_values = std::vector<double>(stored_gamma_values_n4,
                    stored_gamma_values_n4+stored_gamma_number+1);
        } else if (DoF == 5) {
            scale_of_stored_gammas = scale_of_stored_gammas_n5;
            stored_gamma_values = std::vector<double>(stored_gamma_values_n5,
                    stored_gamma_values_n5+stored_gamma_number+1);
        } else
            CV_Error(cv::Error::StsNotImplemented, "Sigma values are not generated");
    }

    /*
        this version correspond to https://github.com/danini/magsac with small technical changes.
    */
    bool refineModel (const Mat &in_model, const Score &in_model_score,
                      Mat &new_model, Score &new_model_score) override {
        int residual_cnt = 0;

        // todo: add magsac termination
         if (verifier->isModelGood(in_model)) {
             if (verifier->hasErrors()) {
                 const std::vector<float> &errors = verifier->getErrors();
                 for (int point_idx = 0; point_idx < points_size; ++point_idx) {
                     // Calculate the residual of the current point
                     const auto residual = errors[point_idx];
                     if (max_sigma > residual) {
                         // Store the residual of the current point and its index
                         residuals[residual_cnt] = residual;
                         residuals_idxs[residual_cnt++] = point_idx;
                     }

                     // Interrupt if there is no chance of being better
                     if (residual_cnt + points_size - point_idx < in_model_score.inlier_number)
                         return false;
                 }
             } else {
                error->setModelParameters(in_model);

                for (int point_idx = 0; point_idx < points_size; ++point_idx) {
                    const double residual = error->getError(point_idx);
                    if (max_sigma > residual) {
                        // Store the residual of the current point and its index
                        residuals[residual_cnt] = residual;
                        residuals_idxs[residual_cnt++] = point_idx;
                    }

                    if (residual_cnt + points_size - point_idx < in_model_score.inlier_number)
                        return false;
                }
             }
         } else return false;

        // Initialize the polished model with the initial one
        Mat polished_model;
        in_model.copyTo(polished_model);
        // A flag to determine if the initial model has been updated
        bool updated = false;

        // Do the iteratively re-weighted least squares fitting
        for (int iterations = 0; iterations < number_of_irwls_iters; ++iterations) {
            int sigma_inliers_cnt = 0;
            // If the current iteration is not the first, the set of possibly inliers
            // (i.e., points closer than the maximum threshold) have to be recalculated.
            if (iterations > 0) {
                error->setModelParameters(polished_model);
                // Remove everything from the residual vector
                residual_cnt = 0;

                // Collect the points which are closer than the maximum threshold
                for (int point_idx = 0; point_idx < points_size; ++point_idx) {
                    // Calculate the residual of the current point
                    const double residual = error->getError(point_idx);
                    if (residual < max_sigma) {
                        // Store the residual of the current point and its index
                        residuals[residual_cnt] = residual;
                        residuals_idxs[residual_cnt++] = point_idx;
                    }
                }
                sigma_inliers_cnt = 0;
            }

            // Calculate the weight of each point
            for (int i = 0; i < residual_cnt; i++) {
                const double residual = residuals[i];
                const int idx = residuals_idxs[i];
                // If the residual is ~0, the point fits perfectly and it is handled differently
                if (residual > std::numeric_limits<double>::epsilon()) {
                    // Calculate the squared residual
                    const double squared_residual = residual * residual;
                    // Get the position of the gamma value in the lookup table
                    int x = (int)round(scale_of_stored_gammas * squared_residual
                            / squared_sigma_max_2);

                    // If the sought gamma value is not stored in the lookup, return the closest element
                    if (x >= stored_gamma_number) // actual number of gamma values is 1 more, so >=
                        x  = stored_gamma_number;

                    sigma_inliers[sigma_inliers_cnt] = idx; // store index of point for LSQ
                    sigma_weights[sigma_inliers_cnt++] = one_over_sigma * (stored_gamma_values[x] - gamma_k);
                }
            }

            // If there are fewer than the minimum point close to the model, terminate.
            // Estimate the model parameters using weighted least-squares fitting
            int num_est_models = estimator->estimateModelNonMinimalSample
                    (sigma_inliers, sigma_inliers_cnt, sigma_models, sigma_weights);
            if (num_est_models == 0) {
                // If the estimation failed and the iteration was never successfull,
                // terminate with failure.
                if (iterations == 0)
                    return false;
                // Otherwise, if the iteration was successfull at least one,
                // simply break it.
                break;
            }

            // Update the model parameters
            polished_model = sigma_models[0];
            if (num_est_models > 1) {
                // find best over other models
                Score sigma_best_score = quality->getScore(polished_model);
                for (int m = 1; m < num_est_models; m++) {
                    Score sc = quality->getScore(sigma_models[m]);
                    if (sc.isBetter(sigma_best_score)) {
                        polished_model = sigma_models[m];
                        sigma_best_score = sc;
                    }
                }
            }

            // The model has been updated
            updated = true;
        }

        if (updated) {
            new_model_score = quality->getScore(polished_model);
            new_model = polished_model;
            return true;
        }
        return false;
    }
    Ptr<LocalOptimization> clone(int state) const override {
        return makePtr<SigmaConsensusImpl>(estimator->clone(), error->clone(), quality->clone(),
                verifier->clone(state), number_of_irwls_iters, degrees_of_freedom, k, gamma_k, C,
                maximum_threshold);
    }
};
Ptr<SigmaConsensus>
SigmaConsensus::create(const Ptr<Estimator> &estimator_, const Ptr<Error> &error_,
        const Ptr<Quality> &quality, const Ptr<ModelVerifier> &verifier_,
        int number_of_irwls_iters_, int DoF, double sigma_quantile,
        double upper_incomplete_of_sigma_quantile, double C_, double maximum_thr) {
    return makePtr<SigmaConsensusImpl>(estimator_, error_, quality, verifier_,
            number_of_irwls_iters_, DoF, sigma_quantile, upper_incomplete_of_sigma_quantile,
            C_, maximum_thr);
}

/////////////////////////////////////////// FINAL MODEL POLISHER ////////////////////////
class LeastSquaresPolishingImpl : public LeastSquaresPolishing {
private:
    const Ptr<Estimator> estimator;
    const Ptr<Quality> quality;
    Score score;
    int lsq_iterations;
    std::vector<int> inliers;
    std::vector<Mat> models;
    std::vector<double> weights;
public:

    LeastSquaresPolishingImpl(const Ptr<Estimator> &estimator_, const Ptr<Quality> &quality_,
            int lsq_iterations_) :
            estimator(estimator_), quality(quality_) {
        lsq_iterations = lsq_iterations_;
        // allocate memory for inliers array and models
        inliers = std::vector<int>(quality_->getPointsSize());
        models = std::vector<Mat>(estimator->getMaxNumSolutionsNonMinimal());
    }

    bool polishSoFarTheBestModel(const Mat &model, const Score &best_model_score,
                                 Mat &new_model, Score &out_score) override {
        // get inliers from input model
        int inlier_number = quality->getInliers(model, inliers);
        if (inlier_number < estimator->getMinimalSampleSize())
            return false;

        out_score = Score(); // set the worst case

        // several all-inlier least-squares refines model better than only one but for
        // big amount of points may be too time-consuming.
        for (int lsq_iter = 0; lsq_iter < lsq_iterations; lsq_iter++) {
            bool model_updated = false;

            // estimate non minimal models with all inliers
            const int num_models = estimator->estimateModelNonMinimalSample(inliers,
                                                      inlier_number, models, weights);
            for (int model_idx = 0; model_idx < num_models; model_idx++) {
                score = quality->getScore(models[model_idx]);

                if (best_model_score.isBetter(score))
                    continue;
                if (score.isBetter(out_score)) {
                    models[model_idx].copyTo(new_model);
                    out_score = score;
                    model_updated = true;
                }
            }

            if (!model_updated)
                // if model was not updated at the first iteration then return false
                // otherwise if all-inliers LSQ has not updated model then no sense
                // to do it again -> return true (model was updated before).
                return lsq_iter > 0;

            // if number of inliers doesn't increase more than 5% then break
            if (fabs(static_cast<double>(out_score.inlier_number) - static_cast<double>
                 (best_model_score.inlier_number)) / best_model_score.inlier_number < 0.05)
                return true;

            if (lsq_iter != lsq_iterations - 1)
                // if not the last LSQ normalization then get inliers for next normalization
                inlier_number = quality->getInliers(new_model, inliers);
        }
        return true;
    }
};
Ptr<LeastSquaresPolishing> LeastSquaresPolishing::create (const Ptr<Estimator> &estimator_,
         const Ptr<Quality> &quality_, int lsq_iterations_) {
    return makePtr<LeastSquaresPolishingImpl>(estimator_, quality_, lsq_iterations_);
}
}}
