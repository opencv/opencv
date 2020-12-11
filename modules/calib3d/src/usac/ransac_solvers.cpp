// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"
#include <atomic>

namespace cv { namespace usac {
int mergePoints (InputArray pts1_, InputArray pts2_, Mat &pts, bool ispnp);
void setParameters (int flag, Ptr<Model> &params, EstimationMethod estimator, double thr,
                    int max_iters, double conf, bool mask_needed);

class RansacOutputImpl : public RansacOutput {
private:
    Mat model;
    // vector of number_inliers size
    std::vector<int> inliers;
    // vector of points size, true if inlier, false-outlier
    std::vector<bool> inliers_mask;
    // vector of points size, value of i-th index corresponds to error of i-th point if i is inlier.
    std::vector<double> errors;
    // the best found score of RANSAC
    double score;

    int seconds, milliseconds, microseconds;
    int time_mcs, number_inliers, number_estimated_models, number_good_models;
    int number_iterations; // number of iterations of main RANSAC
public:
    RansacOutputImpl (const Mat &model_, const std::vector<bool> &inliers_mask_,
            int time_mcs_, double score_, int number_inliers_, int number_iterations_,
            int number_estimated_models_, int number_good_models_) {

        model_.copyTo(model);
        inliers_mask = inliers_mask_;
        time_mcs = time_mcs_;
        score = score_;
        number_inliers = number_inliers_;
        number_iterations = number_iterations_;
        number_estimated_models = number_estimated_models_;
        number_good_models = number_good_models_;
        microseconds = time_mcs % 1000;
        milliseconds = ((time_mcs - microseconds)/1000) % 1000;
        seconds = ((time_mcs - 1000*milliseconds - microseconds)/(1000*1000)) % 60;
    }

    /*
     * Return inliers' indices.
     * size of vector = number of inliers
     */
    const std::vector<int> &getInliers() override {
        if (inliers.empty()) {
            inliers.reserve(inliers_mask.size());
            int pt_cnt = 0;
            for (bool is_inlier : inliers_mask) {
                if (is_inlier)
                    inliers.emplace_back(pt_cnt);
                pt_cnt++;
            }
        }
        return inliers;
    }

    // Return inliers mask. Vector of points size. 1-inlier, 0-outlier.
    const std::vector<bool> &getInliersMask() const override { return inliers_mask; }

    int getTimeMicroSeconds() const override {return time_mcs; }
    int getTimeMicroSeconds1() const override {return microseconds; }
    int getTimeMilliSeconds2() const override {return milliseconds; }
    int getTimeSeconds3() const override {return seconds; }
    int getNumberOfInliers() const override { return number_inliers; }
    int getNumberOfMainIterations() const override { return number_iterations; }
    int getNumberOfGoodModels () const override { return number_good_models; }
    int getNumberOfEstimatedModels () const override { return number_estimated_models; }
    const Mat &getModel() const override { return model; }
};

Ptr<RansacOutput> RansacOutput::create(const Mat &model_, const std::vector<bool> &inliers_mask_,
        int time_mcs_, double score_, int number_inliers_, int number_iterations_,
        int number_estimated_models_, int number_good_models_) {
    return makePtr<RansacOutputImpl>(model_, inliers_mask_, time_mcs_, score_, number_inliers_,
            number_iterations_, number_estimated_models_, number_good_models_);
}

class Ransac {
protected:
    const Ptr<const Model> params;
    const Ptr<const Estimator> _estimator;
    const Ptr<Quality> _quality;
    const Ptr<Sampler> _sampler;
    const Ptr<TerminationCriteria> _termination_criteria;
    const Ptr<ModelVerifier> _model_verifier;
    const Ptr<Degeneracy> _degeneracy;
    const Ptr<LocalOptimization> _local_optimization;
    const Ptr<FinalModelPolisher> model_polisher;

    const int points_size, state;
    const bool parallel;
public:

    Ransac (const Ptr<const Model> &params_, int points_size_, const Ptr<const Estimator> &estimator_, const Ptr<Quality> &quality_,
            const Ptr<Sampler> &sampler_, const Ptr<TerminationCriteria> &termination_criteria_,
            const Ptr<ModelVerifier> &model_verifier_, const Ptr<Degeneracy> &degeneracy_,
            const Ptr<LocalOptimization> &local_optimization_, const Ptr<FinalModelPolisher> &model_polisher_,
            bool parallel_=false, int state_ = 0) :

            params (params_), _estimator (estimator_), _quality (quality_), _sampler (sampler_),
            _termination_criteria (termination_criteria_), _model_verifier (model_verifier_),
            _degeneracy (degeneracy_), _local_optimization (local_optimization_),
            model_polisher (model_polisher_), points_size (points_size_), state(state_),
            parallel(parallel_) {}

    bool run(Ptr<RansacOutput> &ransac_output) {
        if (points_size < params->getSampleSize())
            return false;

        const auto begin_time = std::chrono::steady_clock::now();

        // check if LO
        const bool LO = params->getLO() != LocalOptimMethod::LOCAL_OPTIM_NULL;
        const bool is_magsac = params->getLO() == LocalOptimMethod::LOCAL_OPTIM_SIGMA;
        const int max_hyp_test_before_ver = params->getMaxNumHypothesisToTestBeforeRejection();
        const int repeat_magsac = 10, max_iters_before_LO = params->getMaxItersBeforeLO();
        Score best_score;
        Mat best_model;
        int final_iters;

        if (! parallel) {
            auto update_best = [&] (const Mat &new_model, const Score &new_score) {
                best_score = new_score;
                // remember best model
                new_model.copyTo(best_model);
                // update quality and verifier to save evaluation time of a model
                _quality->setBestScore(best_score.score);
                // update verifier
                _model_verifier->update(best_score.inlier_number);
                // update upper bound of iterations
                return _termination_criteria->update(best_model, best_score.inlier_number);
            };
            bool was_LO_run = false;
            Mat non_degenerate_model, lo_model;
            Score current_score, lo_score, non_denegenerate_model_score;

            // reallocate memory for models
            std::vector<Mat> models(_estimator->getMaxNumSolutions());

            // allocate memory for sample
            std::vector<int> sample(_estimator->getMinimalSampleSize());
            int iters = 0, max_iters = params->getMaxIters();
            for (; iters < max_iters; iters++) {
                _sampler->generateSample(sample);
                const int number_of_models = _estimator->estimateModels(sample, models);

                for (int i = 0; i < number_of_models; i++) {
                    if (iters < max_hyp_test_before_ver) {
                        current_score = _quality->getScore(models[i]);
                    } else {
                        if (is_magsac && iters % repeat_magsac == 0) {
                            if (!_local_optimization->refineModel
                                    (models[i], best_score, models[i], current_score))
                                continue;
                        } else if (_model_verifier->isModelGood(models[i])) {
                            if (!_model_verifier->getScore(current_score)) {
                                if (_model_verifier->hasErrors())
                                    current_score = _quality->getScore(_model_verifier->getErrors());
                                else current_score = _quality->getScore(models[i]);
                            }
                        } else continue;
                    }

                    if (current_score.isBetter(best_score)) {
                        if (_degeneracy->recoverIfDegenerate(sample, models[i],
                                non_degenerate_model, non_denegenerate_model_score)) {
                            // check if best non degenerate model is better than so far the best model
                            if (non_denegenerate_model_score.isBetter(best_score))
                                max_iters = update_best(non_degenerate_model, non_denegenerate_model_score);
                            else continue;
                        } else max_iters = update_best(models[i], current_score);

                        if (LO && iters >= max_iters_before_LO) {
                            // do magsac if it wasn't already run
                            if (is_magsac && iters % repeat_magsac == 0 && iters >= max_hyp_test_before_ver) continue; // magsac has already run
                            was_LO_run = true;
                            // update model by Local optimization
                            if (_local_optimization->refineModel
                                    (best_model, best_score, lo_model, lo_score)) {
                                if (lo_score.isBetter(best_score)){
                                    max_iters = update_best(lo_model, lo_score);
                                }
                            }
                        }
                        if (iters > max_iters)
                            break;
                    } // end of if so far the best score
                } // end loop of number of models
                if (LO && !was_LO_run && iters >= max_iters_before_LO) {
                    was_LO_run = true;
                    if (_local_optimization->refineModel(best_model, best_score, lo_model, lo_score))
                        if (lo_score.isBetter(best_score)){
                            max_iters = update_best(lo_model, lo_score);
                        }
                }
            } // end main while loop

            final_iters = iters;
        } else {
            const int MAX_THREADS = getNumThreads();
            const bool is_prosac = params->getSampler() == SamplingMethod::SAMPLING_PROSAC;

            std::atomic_bool success(false);
            std::atomic_int num_hypothesis_tested(0);
            std::atomic_int thread_cnt(0);
            std::vector<Score> best_scores(MAX_THREADS);
            std::vector<Mat> best_models(MAX_THREADS);

            Mutex mutex; // only for prosac

            ///////////////////////////////////////////////////////////////////////////////////////////////////////
            parallel_for_(Range(0, MAX_THREADS), [&](const Range & /*range*/) {
            if (!success) { // cover all if not success to avoid thread creating new variables
                const int thread_rng_id = thread_cnt++;
                int thread_state = state + 10*thread_rng_id;

                Ptr<Estimator> estimator = _estimator->clone();
                Ptr<Degeneracy> degeneracy = _degeneracy->clone(thread_state++);
                Ptr<Quality> quality = _quality->clone();
                Ptr<ModelVerifier> model_verifier = _model_verifier->clone(thread_state++); // update verifier
                Ptr<LocalOptimization> local_optimization;
                if (LO)
                    local_optimization = _local_optimization->clone(thread_state++);
                Ptr<TerminationCriteria> termination_criteria = _termination_criteria->clone();
                Ptr<Sampler> sampler;
                if (!is_prosac)
                   sampler = _sampler->clone(thread_state);

                Mat best_model_thread, non_degenerate_model, lo_model;
                Score best_score_thread, current_score, non_denegenerate_model_score, lo_score,
                      best_score_all_threads;
                std::vector<int> sample(estimator->getMinimalSampleSize());
                std::vector<Mat> models(estimator->getMaxNumSolutions());
                int iters, max_iters = params->getMaxIters();
                auto update_best = [&] (const Score &new_score, const Mat &new_model) {
                    // copy new score to best score
                    best_score_thread = new_score;
                    best_scores[thread_rng_id] = best_score_thread;
                    // remember best model
                    new_model.copyTo(best_model_thread);
                    best_model_thread.copyTo(best_models[thread_rng_id]);
                    best_score_all_threads = best_score_thread;
                    // update upper bound of iterations
                    return termination_criteria->update
                            (best_model_thread, best_score_thread.inlier_number);
                };

                bool was_LO_run = false;
                for (iters = 0; iters < max_iters && !success; iters++) {
                    success = num_hypothesis_tested++ > max_iters;

                    if (iters % 10) {
                        // Synchronize threads. just to speed verification of model.
                        int best_thread_idx = thread_rng_id;
                        bool updated = false;
                        for (int t = 0; t < MAX_THREADS; t++) {
                            if (best_scores[t].isBetter(best_score_all_threads)) {
                                best_score_all_threads = best_scores[t];
                                updated = true;
                                best_thread_idx = t;
                            }
                        }
                        if (updated && best_thread_idx != thread_rng_id) {
                            quality->setBestScore(best_score_all_threads.score);
                            model_verifier->update(best_score_all_threads.inlier_number);
                        }
                    }

                    if (is_prosac) {
                        // use global sampler
                        mutex.lock();
                        _sampler->generateSample(sample);
                        mutex.unlock();
                    } else sampler->generateSample(sample); // use local sampler

                    const int number_of_models = estimator->estimateModels(sample, models);
                    for (int i = 0; i < number_of_models; i++) {
                        if (iters < max_hyp_test_before_ver) {
                            current_score = quality->getScore(models[i]);
                        } else {
                            if (is_magsac && iters % repeat_magsac == 0) {
                                if (local_optimization && !local_optimization->refineModel
                                        (models[i], best_score_thread, models[i], current_score))
                                    continue;
                            } else if (model_verifier->isModelGood(models[i])) {
                                if (!model_verifier->getScore(current_score)) {
                                    if (model_verifier->hasErrors())
                                        current_score = quality->getScore(model_verifier->getErrors());
                                    else current_score = quality->getScore(models[i]);
                                }
                            } else continue;
                        }

                        if (current_score.isBetter(best_score_all_threads)) {
                            if (degeneracy->recoverIfDegenerate(sample, models[i],
                                        non_degenerate_model, non_denegenerate_model_score)) {
                                // check if best non degenerate model is better than so far the best model
                                if (non_denegenerate_model_score.isBetter(best_score_thread))
                                    max_iters = update_best(non_denegenerate_model_score, non_degenerate_model);
                                else continue;
                            } else
                                max_iters = update_best(current_score, models[i]);

                            if (LO && iters >= max_iters_before_LO) {
                                // do magsac if it wasn't already run
                                if (is_magsac && iters % repeat_magsac == 0 && iters >= max_hyp_test_before_ver) continue;
                                was_LO_run = true;
                                // update model by Local optimizaion
                                if (local_optimization->refineModel
                                       (best_model_thread, best_score_thread, lo_model, lo_score))
                                    if (lo_score.isBetter(best_score_thread)) {
                                        max_iters = update_best(lo_score, lo_model);
                                    }
                            }
                            if (num_hypothesis_tested > max_iters) {
                                success = true; break;
                            }
                        } // end of if so far the best score
                    } // end loop of number of models
                    if (LO && !was_LO_run && iters >= max_iters_before_LO) {
                        was_LO_run = true;
                        if (_local_optimization->refineModel(best_model, best_score, lo_model, lo_score))
                            if (lo_score.isBetter(best_score)){
                                max_iters = update_best(lo_score, lo_model);
                            }
                    }
                } // end of loop over iters
            }}); // end parallel
            ///////////////////////////////////////////////////////////////////////////////////////////////////////
            // find best model from all threads' models
            best_score = best_scores[0];
            int best_thread_idx = 0;
            for (int i = 1; i < MAX_THREADS; i++) {
                if (best_scores[i].isBetter(best_score)) {
                    best_score = best_scores[i];
                    best_thread_idx = i;
                }
            }
            best_model = best_models[best_thread_idx];
            final_iters = num_hypothesis_tested;
        }

        if (best_model.empty())
            return false;

        // polish final model
        if (params->getFinalPolisher() != PolishingMethod::NonePolisher) {
            Mat polished_model;
            Score polisher_score;
            if (model_polisher->polishSoFarTheBestModel(best_model, best_score,
                     polished_model, polisher_score))
                if (polisher_score.isBetter(best_score)) {
                    best_score = polisher_score;
                    polished_model.copyTo(best_model);
                }
        }
        // ================= here is ending ransac main implementation ===========================
        std::vector<bool> inliers_mask;
        if (params->isMaskRequired()) {
            inliers_mask = std::vector<bool>(points_size);
            // get final inliers from the best model
            _quality->getInliers(best_model, inliers_mask);
        }
        // Store results
        ransac_output = RansacOutput::create(best_model, inliers_mask,
                static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>
                (std::chrono::steady_clock::now() - begin_time).count()), best_score.score,
                best_score.inlier_number, final_iters, -1, -1);
        return true;
    }
};

/*
 * pts1, pts2 are matrices either N x a, N x b or a x N or b x N, where N > a and N > b
 * pts1 are image points, if pnp pts2 are object points otherwise - image points as well.
 * output is matrix of size N x (a + b)
 * return points_size = N
 */
int mergePoints (InputArray pts1_, InputArray pts2_, Mat &pts, bool ispnp) {
    Mat pts1 = pts1_.getMat(), pts2 = pts2_.getMat();
    auto convertPoints = [] (Mat &points, int pt_dim) {
        points.convertTo(points, CV_32F); // convert points to have float precision
        if (points.channels() > 1)
            points = points.reshape(1, (int)points.total()); // convert point to have 1 channel
        if (points.rows < points.cols)
            transpose(points, points); // transpose so points will be in rows
        CV_CheckGE(points.cols, pt_dim, "Invalid dimension of point");
        if (points.cols != pt_dim) // in case when image points are 3D convert them to 2D
            points = points.colRange(0, pt_dim);
    };

    convertPoints(pts1, 2); // pts1 are always image points
    convertPoints(pts2, ispnp ? 3 : 2); // for PnP points are 3D

    // points are of size [Nx2 Nx2] = Nx4 for H, F, E
    // points are of size [Nx2 Nx3] = Nx5 for PnP
    hconcat(pts1, pts2, pts);
    return pts.rows;
}

void saveMask (OutputArray mask, const std::vector<bool> &inliers_mask) {
    if (mask.needed()) {
        const int points_size = (int) inliers_mask.size();
        mask.create(points_size, 1, CV_8U);
        auto * maskptr = mask.getMat().ptr<uchar>();
        for (int i = 0; i < points_size; i++)
            maskptr[i] = (uchar) inliers_mask[i];
    }
}
void setParameters (Ptr<Model> &params, EstimationMethod estimator, const UsacParams &usac_params,
        bool mask_needed) {
    params = Model::create(usac_params.threshold, estimator, usac_params.sampler,
            usac_params.confidence, usac_params.maxIterations, usac_params.score);
    params->setLocalOptimization(usac_params.loMethod);
    params->setLOSampleSize(usac_params.loSampleSize);
    params->setLOIterations(usac_params.loIterations);
    params->setParallel(usac_params.isParallel);
    params->setNeighborsType(usac_params.neighborsSearch);
    params->setRandomGeneratorState(usac_params.randomGeneratorState);
    params->maskRequired(mask_needed);
}

void setParameters (int flag, Ptr<Model> &params, EstimationMethod estimator, double thr,
        int max_iters, double conf, bool mask_needed) {
    switch (flag) {
        case USAC_DEFAULT:
            params = Model::create(thr, estimator, SamplingMethod::SAMPLING_UNIFORM, conf, max_iters,
                                   ScoreMethod::SCORE_METHOD_MSAC);
            params->setLocalOptimization(LocalOptimMethod ::LOCAL_OPTIM_INNER_AND_ITER_LO);
            break;
        case USAC_MAGSAC:
            params = Model::create(thr, estimator, SamplingMethod::SAMPLING_UNIFORM, conf, max_iters,
                                   ScoreMethod::SCORE_METHOD_MAGSAC);
            params->setLocalOptimization(LocalOptimMethod ::LOCAL_OPTIM_SIGMA);
            params->setLOSampleSize(params->isHomography() ? 75 : 50);
            params->setLOIterations(params->isHomography() ? 15 : 10);
            break;
        case USAC_PARALLEL:
            params = Model::create(thr, estimator, SamplingMethod::SAMPLING_UNIFORM, conf, max_iters,
                                   ScoreMethod::SCORE_METHOD_MSAC);
            params->setParallel(true);
            params->setLocalOptimization(LocalOptimMethod ::LOCAL_OPTIM_INNER_LO);
            break;
        case USAC_ACCURATE:
            params = Model::create(thr, estimator, SamplingMethod::SAMPLING_UNIFORM, conf, max_iters,
                                   ScoreMethod::SCORE_METHOD_MSAC);
            params->setLocalOptimization(LocalOptimMethod ::LOCAL_OPTIM_GC);
            params->setLOSampleSize(20);
            params->setLOIterations(25);
            break;
        case USAC_FAST:
            params = Model::create(thr, estimator, SamplingMethod::SAMPLING_UNIFORM, conf, max_iters,
                                   ScoreMethod::SCORE_METHOD_MSAC);
            params->setLocalOptimization(LocalOptimMethod ::LOCAL_OPTIM_INNER_AND_ITER_LO);
            params->setLOIterations(5);
            params->setLOIterativeIters(3);
            break;
        case USAC_PROSAC:
            params = Model::create(thr, estimator, SamplingMethod::SAMPLING_PROSAC, conf, max_iters,
                                   ScoreMethod::SCORE_METHOD_MSAC);
            params->setLocalOptimization(LocalOptimMethod ::LOCAL_OPTIM_INNER_LO);
            break;
        case USAC_FM_8PTS:
            params = Model::create(thr, EstimationMethod::Fundamental8,SamplingMethod::SAMPLING_UNIFORM,
                    conf, max_iters,ScoreMethod::SCORE_METHOD_MSAC);
            params->setLocalOptimization(LocalOptimMethod ::LOCAL_OPTIM_INNER_LO);
            break;
        default: CV_Error(cv::Error::StsBadFlag, "Incorrect flag for USAC!");
    }
    // do not do too many iterations for PnP
    if (estimator == EstimationMethod::P3P) {
        if (params->getLOInnerMaxIters() > 15)
            params->setLOIterations(15);
        params->setLOIterativeIters(0);
    }

    params->maskRequired(mask_needed);
}

Mat findHomography (InputArray srcPoints, InputArray dstPoints, int method, double thr,
        OutputArray mask, const int max_iters, const double confidence) {
    Ptr<Model> params;
    setParameters(method, params, EstimationMethod::Homography, thr, max_iters, confidence, mask.needed());
    Ptr<RansacOutput> ransac_output;
    if (run(params, srcPoints, dstPoints, params->getRandomGeneratorState(),
            ransac_output, noArray(), noArray(), noArray(), noArray())) {
        saveMask(mask, ransac_output->getInliersMask());
        return ransac_output->getModel() / ransac_output->getModel().at<double>(2,2);
    }
    if (mask.needed()){
        mask.create(std::max(srcPoints.getMat().rows, srcPoints.getMat().cols), 1, CV_8U);
        mask.setTo(Scalar::all(0));
    }
    return Mat();
}

Mat findFundamentalMat( InputArray points1, InputArray points2, int method, double thr,
        double confidence, int max_iters, OutputArray mask ) {
    Ptr<Model> params;
    setParameters(method, params, EstimationMethod::Fundamental, thr, max_iters, confidence, mask.needed());
    Ptr<RansacOutput> ransac_output;
    if (run(params, points1, points2, params->getRandomGeneratorState(),
            ransac_output, noArray(), noArray(), noArray(), noArray())) {
        saveMask(mask, ransac_output->getInliersMask());
        return ransac_output->getModel();
    }
    if (mask.needed()){
        mask.create(std::max(points1.getMat().rows, points1.getMat().cols), 1, CV_8U);
        mask.setTo(Scalar::all(0));
    }
    return Mat();
}

Mat findEssentialMat (InputArray points1, InputArray points2, InputArray cameraMatrix1,
        int method, double prob, double thr, OutputArray mask) {
    Ptr<Model> params;
    setParameters(method, params, EstimationMethod::Essential, thr, 1000, prob, mask.needed());
    Ptr<RansacOutput> ransac_output;
    if (run(params, points1, points2, params->getRandomGeneratorState(),
            ransac_output, cameraMatrix1, cameraMatrix1, noArray(), noArray())) {
        saveMask(mask, ransac_output->getInliersMask());
        return ransac_output->getModel();
    }
    if (mask.needed()){
        mask.create(std::max(points1.getMat().rows, points1.getMat().cols), 1, CV_8U);
        mask.setTo(Scalar::all(0));
    }
    return Mat();
}

bool solvePnPRansac( InputArray objectPoints, InputArray imagePoints,
       InputArray cameraMatrix, InputArray distCoeffs, OutputArray rvec, OutputArray tvec,
       bool /*useExtrinsicGuess*/, int max_iters, float thr, double conf,
       OutputArray mask, int method) {
    Ptr<Model> params;
    setParameters(method, params, cameraMatrix.empty() ? EstimationMethod ::P6P : EstimationMethod ::P3P,
            thr, max_iters, conf, mask.needed());
    Ptr<RansacOutput> ransac_output;
    if (run(params, imagePoints, objectPoints, params->getRandomGeneratorState(),
            ransac_output, cameraMatrix, noArray(), distCoeffs, noArray())) {
        saveMask(mask, ransac_output->getInliersMask());
        const Mat &model = ransac_output->getModel();
        model.col(0).copyTo(rvec);
        model.col(1).copyTo(tvec);
        return true;
    }
    if (mask.needed()){
        mask.create(std::max(objectPoints.getMat().rows, objectPoints.getMat().cols), 1, CV_8U);
        mask.setTo(Scalar::all(0));
    }
    return false;
}

Mat estimateAffine2D(InputArray from, InputArray to, OutputArray mask, int method,
        double thr, int max_iters, double conf, int /*refineIters*/) {
    Ptr<Model> params;
    setParameters(method, params, EstimationMethod ::Affine, thr, max_iters, conf, mask.needed());
    Ptr<RansacOutput> ransac_output;
    if (run(params, from, to, params->getRandomGeneratorState(),
            ransac_output, noArray(), noArray(), noArray(), noArray())) {
        saveMask(mask, ransac_output->getInliersMask());
        return ransac_output->getModel().rowRange(0,2);
    }
    if (mask.needed()){
        mask.create(std::max(from.getMat().rows, from.getMat().cols), 1, CV_8U);
        mask.setTo(Scalar::all(0));
    }
    return Mat();
}

class ModelImpl : public Model {
private:
    // main parameters:
    double threshold, confidence;
    int sample_size, max_iterations;

    EstimationMethod estimator;
    SamplingMethod sampler;
    ScoreMethod score;

    // for neighborhood graph
    int k_nearest_neighbors = 8;//, flann_search_params = 5, num_kd_trees = 1; // for FLANN
    int cell_size = 50; // pixels, for grid neighbors searching
    int radius = 30; // pixels, for radius-search neighborhood graph
    NeighborSearchMethod neighborsType = NeighborSearchMethod::NEIGH_GRID;

    // Local Optimization parameters
    LocalOptimMethod lo = LocalOptimMethod ::LOCAL_OPTIM_INNER_AND_ITER_LO;
    int lo_sample_size=16, lo_inner_iterations=15, lo_iterative_iterations=8,
            lo_thr_multiplier=15, lo_iter_sample_size = 30;

    // Graph cut parameters
    const double spatial_coherence_term = 0.975;

    // apply polisher for final RANSAC model
    PolishingMethod polisher = PolishingMethod ::LSQPolisher;

    // preemptive verification test
    VerificationMethod verifier = VerificationMethod ::SprtVerifier;
    const int max_hypothesis_test_before_verification = 15;

    // sprt parameters
    // lower bound estimate is 1% of inliers
    double sprt_eps = 0.01, sprt_delta = 0.008, avg_num_models, time_for_model_est;

    // estimator error
    ErrorMetric est_error;

    // progressive napsac
    double relax_coef = 0.1;
    // for building neighborhood graphs
    const std::vector<int> grid_cell_number = {16, 8, 4, 2};

    //for final least squares polisher
    int final_lsq_iters = 3;

    bool need_mask = true, is_parallel = false;
    int random_generator_state = 0;
    const int max_iters_before_LO = 100;

    // magsac parameters:
    int DoF = 2;
    double sigma_quantile = 3.04, upper_incomplete_of_sigma_quantile = 0.00419,
        lower_incomplete_of_sigma_quantile = 0.8629, C = 0.5, maximum_thr = 7.5;
public:
    ModelImpl (double threshold_, EstimationMethod estimator_, SamplingMethod sampler_, double confidence_=0.95,
               int max_iterations_=5000, ScoreMethod score_ =ScoreMethod::SCORE_METHOD_MSAC) {
        estimator = estimator_;
        sampler = sampler_;
        confidence = confidence_;
        max_iterations = max_iterations_;
        score = score_;

        switch (estimator_) {
            // time for model estimation is basically a ratio of time need to estimate a model to
            // time needed to verify if a point is consistent with this model
            case (EstimationMethod::Affine):
                avg_num_models = 1; time_for_model_est = 50;
                sample_size = 3; est_error = ErrorMetric ::FORW_REPR_ERR; break;
            case (EstimationMethod::Homography):
                avg_num_models = 1; time_for_model_est = 150;
                sample_size = 4; est_error = ErrorMetric ::FORW_REPR_ERR; break;
            case (EstimationMethod::Fundamental):
                avg_num_models = 2.38; time_for_model_est = 180; maximum_thr = 2.5;
                sample_size = 7; est_error = ErrorMetric ::SAMPSON_ERR; break;
            case (EstimationMethod::Fundamental8):
                avg_num_models = 1; time_for_model_est = 100; maximum_thr = 2.5;
                sample_size = 8; est_error = ErrorMetric ::SAMPSON_ERR; break;
            case (EstimationMethod::Essential):
                avg_num_models = 3.93; time_for_model_est = 1000; maximum_thr = 2.5;
                sample_size = 5; est_error = ErrorMetric ::SGD_ERR; break;
            case (EstimationMethod::P3P):
                avg_num_models = 1.38; time_for_model_est = 800;
                sample_size = 3; est_error = ErrorMetric ::RERPOJ; break;
            case (EstimationMethod::P6P):
                avg_num_models = 1; time_for_model_est = 300;
                sample_size = 6; est_error = ErrorMetric ::RERPOJ; break;
            default: CV_Error(cv::Error::StsNotImplemented, "Estimator has not implemented yet!");
        }

        if (estimator_ == EstimationMethod::P3P || estimator_ == EstimationMethod::P6P) {
            neighborsType = NeighborSearchMethod::NEIGH_FLANN_KNN;
            k_nearest_neighbors = 2;
        }
        if (estimator == EstimationMethod::Fundamental || estimator == EstimationMethod::Essential) {
            lo_sample_size = 21;
            lo_thr_multiplier = 10;
        }
        if (estimator == EstimationMethod::Homography)
            maximum_thr = 8.;
        threshold = threshold_;
    }
    void setVerifier (VerificationMethod verifier_) override { verifier = verifier_; }
    void setPolisher (PolishingMethod polisher_) override { polisher = polisher_; }
    void setParallel (bool is_parallel_) override { is_parallel = is_parallel_; }
    void setError (ErrorMetric error_) override { est_error = error_; }
    void setLocalOptimization (LocalOptimMethod lo_) override { lo = lo_; }
    void setKNearestNeighhbors (int knn_) override { k_nearest_neighbors = knn_; }
    void setNeighborsType (NeighborSearchMethod neighbors) override { neighborsType = neighbors; }
    void setCellSize (int cell_size_) override { cell_size = cell_size_; }
    void setLOIterations (int iters) override { lo_inner_iterations = iters; }
    void setLOIterativeIters (int iters) override {lo_iterative_iterations = iters; }
    void setLOSampleSize (int lo_sample_size_) override { lo_sample_size = lo_sample_size_; }
    void setThresholdMultiplierLO (double thr_mult) override { lo_thr_multiplier = (int) round(thr_mult); }
    void maskRequired (bool need_mask_) override { need_mask = need_mask_; }
    void setRandomGeneratorState (int state) override { random_generator_state = state; }
    bool isMaskRequired () const override { return need_mask; }
    NeighborSearchMethod getNeighborsSearch () const override { return neighborsType; }
    int getKNN () const override { return k_nearest_neighbors; }
    ErrorMetric getError () const override { return est_error; }
    EstimationMethod getEstimator () const override { return estimator; }
    int getSampleSize () const override { return sample_size; }
    int getFinalLSQIterations () const override { return final_lsq_iters; }
    int getDegreesOfFreedom () const override { return DoF; }
    double getSigmaQuantile () const override { return sigma_quantile; }
    double getUpperIncompleteOfSigmaQuantile () const override {
        return upper_incomplete_of_sigma_quantile;
    }
    double getLowerIncompleteOfSigmaQuantile () const override {
        return lower_incomplete_of_sigma_quantile;
    }
    double getC () const override { return C; }
    double getMaximumThreshold () const override { return maximum_thr; }
    double getGraphCutSpatialCoherenceTerm () const override { return spatial_coherence_term; }
    int getLOSampleSize () const override { return lo_sample_size; }
    int getMaxNumHypothesisToTestBeforeRejection() const override {
        return max_hypothesis_test_before_verification;
    }
    PolishingMethod getFinalPolisher () const override { return polisher; }
    int getLOThresholdMultiplier() const override { return lo_thr_multiplier; }
    int getLOIterativeSampleSize() const override { return lo_iter_sample_size; }
    int getLOIterativeMaxIters() const override { return lo_iterative_iterations; }
    int getLOInnerMaxIters() const override { return lo_inner_iterations; }
    LocalOptimMethod getLO () const override { return lo; }
    ScoreMethod getScore () const override { return score; }
    int getMaxIters () const override { return max_iterations; }
    double getConfidence () const override { return confidence; }
    double getThreshold () const override { return threshold; }
    VerificationMethod getVerifier () const override { return verifier; }
    SamplingMethod getSampler () const override { return sampler; }
    int getRandomGeneratorState () const override { return random_generator_state; }
    int getMaxItersBeforeLO () const override { return max_iters_before_LO; }
    double getSPRTdelta () const override { return sprt_delta; }
    double getSPRTepsilon () const override { return sprt_eps; }
    double getSPRTavgNumModels () const override { return avg_num_models; }
    int getCellSize () const override { return cell_size; }
    int getGraphRadius() const override { return radius; }
    double getTimeForModelEstimation () const override { return time_for_model_est; }
    double getRelaxCoef () const override { return relax_coef; }
    const std::vector<int> &getGridCellNumber () const override { return grid_cell_number; }
    bool isParallel () const override { return is_parallel; }
    bool isFundamental () const override {
        return estimator == EstimationMethod ::Fundamental ||
               estimator == EstimationMethod ::Fundamental8;
    }
    bool isHomography () const override { return estimator == EstimationMethod ::Homography; }
    bool isEssential () const override { return estimator == EstimationMethod ::Essential; }
    bool isPnP() const override {
        return estimator == EstimationMethod ::P3P || estimator == EstimationMethod ::P6P;
    }
};

Ptr<Model> Model::create(double threshold_, EstimationMethod estimator_, SamplingMethod sampler_,
                         double confidence_, int max_iterations_, ScoreMethod score_) {
    return makePtr<ModelImpl>(threshold_, estimator_, sampler_, confidence_,
                              max_iterations_, score_);
}

bool run (const Ptr<const Model> &params, InputArray points1, InputArray points2, int state,
       Ptr<RansacOutput> &ransac_output, InputArray K1_, InputArray K2_,
       InputArray dist_coeff1, InputArray dist_coeff2) {
    Ptr<Error> error;
    Ptr<Estimator> estimator;
    Ptr<NeighborhoodGraph> graph;
    Ptr<Degeneracy> degeneracy;
    Ptr<Quality> quality;
    Ptr<ModelVerifier> verifier;
    Ptr<Sampler> sampler;
    Ptr<RandomGenerator> lo_sampler;
    Ptr<TerminationCriteria> termination;
    Ptr<LocalOptimization> lo;
    Ptr<FinalModelPolisher> polisher;
    Ptr<MinimalSolver> min_solver;
    Ptr<NonMinimalSolver> non_min_solver;

    Mat points, K1, K2, calib_points, undist_points1, undist_points2;
    int points_size;
    double threshold = params->getThreshold(), max_thr = params->getMaximumThreshold();
    const int min_sample_size = params->getSampleSize();
    if (params->isPnP()) {
        if (! K1_.empty()) {
            K1 = K1_.getMat(); K1.convertTo(K1, CV_64F);
            if (! dist_coeff1.empty()) {
                // undistortPoints also calibrate points using K
                if (points1.isContinuous())
                     undistortPoints(points1, undist_points1, K1_, dist_coeff1);
                else undistortPoints(points1.getMat().clone(), undist_points1, K1_, dist_coeff1);
                points_size = mergePoints(undist_points1, points2, points, true);
                Utils::normalizeAndDecalibPointsPnP (K1, points, calib_points);
            } else {
                points_size = mergePoints(points1, points2, points, true);
                Utils::calibrateAndNormalizePointsPnP(K1, points, calib_points);
            }
        } else
            points_size = mergePoints(points1, points2, points, true);
    } else {
        if (params->isEssential()) {
            CV_CheckEQ((int)(!K1_.empty() && !K2_.empty()), 1, "Intrinsic matrix must not be empty!");
            K1 = K1_.getMat(); K1.convertTo(K1, CV_64F);
            K2 = K2_.getMat(); K2.convertTo(K2, CV_64F);
            if (! dist_coeff1.empty() || ! dist_coeff2.empty()) {
                // undistortPoints also calibrate points using K
                if (points1.isContinuous())
                     undistortPoints(points1, undist_points1, K1_, dist_coeff1);
                else undistortPoints(points1.getMat().clone(), undist_points1, K1_, dist_coeff1);
                if (points2.isContinuous())
                     undistortPoints(points2, undist_points2, K2_, dist_coeff2);
                else undistortPoints(points2.getMat().clone(), undist_points2, K2_, dist_coeff2);
                points_size = mergePoints(undist_points1, undist_points2, calib_points, false);
            } else {
                points_size = mergePoints(points1, points2, points, false);
                Utils::calibratePoints(K1, K2, points, calib_points);
            }
            threshold = Utils::getCalibratedThreshold(threshold, K1, K2);
            max_thr = Utils::getCalibratedThreshold(max_thr, K1, K2);
        } else
            points_size = mergePoints(points1, points2, points, false);
    }

    // Since error function output squared error distance, so make
    // threshold squared as well
    threshold *= threshold;

    if (params->getSampler() == SamplingMethod::SAMPLING_NAPSAC || params->getLO() == LocalOptimMethod::LOCAL_OPTIM_GC) {
        if (params->getNeighborsSearch() == NeighborSearchMethod::NEIGH_GRID) {
            graph = GridNeighborhoodGraph::create(points, points_size,
                params->getCellSize(), params->getCellSize(),
                params->getCellSize(), params->getCellSize(), 10);
        } else if (params->getNeighborsSearch() == NeighborSearchMethod::NEIGH_FLANN_KNN) {
            graph = FlannNeighborhoodGraph::create(points, points_size,params->getKNN(), false, 5, 1);
        } else if (params->getNeighborsSearch() == NeighborSearchMethod::NEIGH_FLANN_RADIUS) {
            graph = RadiusSearchNeighborhoodGraph::create(points, points_size,
                    params->getGraphRadius(), 5, 1);
        } else CV_Error(cv::Error::StsNotImplemented, "Graph type is not implemented!");
    }

    std::vector<Ptr<NeighborhoodGraph>> layers;
    if (params->getSampler() == SamplingMethod::SAMPLING_PROGRESSIVE_NAPSAC) {
        CV_CheckEQ((int)params->isPnP(), 0, "ProgressiveNAPSAC for PnP is not implemented!");
        const auto &cell_number_per_layer = params->getGridCellNumber();
        layers.reserve(cell_number_per_layer.size());
        const auto * const pts = (float *) points.data;
        float img1_width = 0, img1_height = 0, img2_width = 0, img2_height = 0;
        for (int i = 0; i < 4 * points_size; i += 4) {
            if (pts[i    ] > img1_width ) img1_width  = pts[i    ];
            if (pts[i + 1] > img1_height) img1_height = pts[i + 1];
            if (pts[i + 2] > img2_width ) img2_width  = pts[i + 2];
            if (pts[i + 3] > img2_height) img2_height = pts[i + 3];
        }
        // Create grid graphs (overlapping layes of given cell numbers)
        for (int layer_idx = 0; layer_idx < (int)cell_number_per_layer.size(); layer_idx++) {
            const int cell_number = cell_number_per_layer[layer_idx];
            if (layer_idx > 0)
                if (cell_number_per_layer[layer_idx-1] <= cell_number)
                    CV_Error(cv::Error::StsError, "Progressive NAPSAC sampler: "
                        "Cell number in layers must be in decreasing order!");
            layers.emplace_back(GridNeighborhoodGraph::create(points, points_size,
          (int)(img1_width / (float)cell_number), (int)(img1_height / (float)cell_number),
          (int)(img2_width / (float)cell_number), (int)(img2_height / (float)cell_number), 10));
        }
    }

    // update points by calibrated for Essential matrix after graph is calculated
    if (params->isEssential()) {
        points = calib_points;
        // if maximum calibrated threshold significanlty differs threshold then set upper bound
        if (max_thr > 10*threshold)
            max_thr = sqrt(10*threshold); // max thr will be squared after
    }
    if (max_thr < threshold)
        max_thr = threshold;

    switch (params->getError()) {
        case ErrorMetric::SYMM_REPR_ERR:
            error = ReprojectionErrorSymmetric::create(points); break;
        case ErrorMetric::FORW_REPR_ERR:
            if (params->getEstimator() == EstimationMethod::Affine)
                error = ReprojectionErrorAffine::create(points);
            else error = ReprojectionErrorForward::create(points);
            break;
        case ErrorMetric::SAMPSON_ERR:
            error = SampsonError::create(points); break;
        case ErrorMetric::SGD_ERR:
            error = SymmetricGeometricDistance::create(points); break;
        case ErrorMetric::RERPOJ:
            error = ReprojectionErrorPmatrix::create(points); break;
        default: CV_Error(cv::Error::StsNotImplemented , "Error metric is not implemented!");
    }

    switch (params->getScore()) {
        case ScoreMethod::SCORE_METHOD_RANSAC :
            quality = RansacQuality::create(points_size, threshold, error); break;
        case ScoreMethod::SCORE_METHOD_MSAC :
            quality = MsacQuality::create(points_size, threshold, error); break;
        case ScoreMethod::SCORE_METHOD_MAGSAC :
            quality = MagsacQuality::create(max_thr, points_size, error,
                threshold, params->getDegreesOfFreedom(),  params->getSigmaQuantile(),
                params->getUpperIncompleteOfSigmaQuantile(),
                params->getLowerIncompleteOfSigmaQuantile(), params->getC()); break;
        case ScoreMethod::SCORE_METHOD_LMEDS :
            quality = LMedsQuality::create(points_size, threshold, error); break;
        default: CV_Error(cv::Error::StsNotImplemented, "Score is not imeplemeted!");
    }

    if (params->isHomography()) {
        degeneracy = HomographyDegeneracy::create(points);
        min_solver = HomographyMinimalSolver4ptsGEM::create(points);
        non_min_solver = HomographyNonMinimalSolver::create(points);
        estimator = HomographyEstimator::create(min_solver, non_min_solver, degeneracy);
    } else if (params->isFundamental()) {
        degeneracy = FundamentalDegeneracy::create(state++, quality, points, min_sample_size, 5. /*sqr homogr thr*/);
        if(min_sample_size == 7) min_solver = FundamentalMinimalSolver7pts::create(points);
        else min_solver = FundamentalMinimalSolver8pts::create(points);
        non_min_solver = FundamentalNonMinimalSolver::create(points);
        estimator = FundamentalEstimator::create(min_solver, non_min_solver, degeneracy);
    } else if (params->isEssential()) {
        degeneracy = EssentialDegeneracy::create(points, min_sample_size);
        min_solver = EssentialMinimalSolverStewenius5pts::create(points);
        non_min_solver = EssentialNonMinimalSolver::create(points);
        estimator = EssentialEstimator::create(min_solver, non_min_solver, degeneracy);
    } else if (params->isPnP()) {
        degeneracy = makePtr<Degeneracy>();
        if (min_sample_size == 3) {
            non_min_solver = DLSPnP::create(points, calib_points, K1);
            min_solver = P3PSolver::create(points, calib_points, K1);
        } else {
            min_solver = PnPMinimalSolver6Pts::create(points);
            non_min_solver = PnPNonMinimalSolver::create(points);
        }
        estimator = PnPEstimator::create(min_solver, non_min_solver);
    } else if (params->getEstimator() == EstimationMethod::Affine) {
        degeneracy = makePtr<Degeneracy>();
        min_solver = AffineMinimalSolver::create(points);
        non_min_solver = AffineNonMinimalSolver::create(points);
        estimator = AffineEstimator::create(min_solver, non_min_solver);
    } else CV_Error(cv::Error::StsNotImplemented, "Estimator not implemented!");

    switch (params->getSampler()) {
        case SamplingMethod::SAMPLING_UNIFORM:
            sampler = UniformSampler::create(state++, min_sample_size, points_size); break;
        case SamplingMethod::SAMPLING_PROSAC:
            sampler = ProsacSampler::create(state++, points_size, min_sample_size, 200000); break;
        case SamplingMethod::SAMPLING_PROGRESSIVE_NAPSAC:
            sampler = ProgressiveNapsac::create(state++, points_size, min_sample_size, layers, 20); break;
        case SamplingMethod::SAMPLING_NAPSAC:
            sampler = NapsacSampler::create(state++, points_size, min_sample_size, graph); break;
        default: CV_Error(cv::Error::StsNotImplemented, "Sampler is not implemented!");
    }

    switch (params->getVerifier()) {
        case VerificationMethod::NullVerifier: verifier = ModelVerifier::create(); break;
        case VerificationMethod::SprtVerifier:
            verifier = SPRT::create(state++, error, points_size, params->getScore() == ScoreMethod ::SCORE_METHOD_MAGSAC ? max_thr : threshold,
             params->getSPRTepsilon(), params->getSPRTdelta(), params->getTimeForModelEstimation(),
             params->getSPRTavgNumModels(), params->getScore()); break;
        default: CV_Error(cv::Error::StsNotImplemented, "Verifier is not imeplemented!");
    }

    if (params->getSampler() == SamplingMethod::SAMPLING_PROSAC) {
        termination = ProsacTerminationCriteria::create(sampler.dynamicCast<ProsacSampler>(), error,
                points_size, min_sample_size, params->getConfidence(),
                params->getMaxIters(), 100, 0.05, 0.05, threshold);
    } else if (params->getSampler() == SamplingMethod::SAMPLING_PROGRESSIVE_NAPSAC) {
        if (params->getVerifier() == VerificationMethod::SprtVerifier)
            termination = SPRTPNapsacTermination::create(((SPRT *)verifier.get())->getSPRTvector(),
                    params->getConfidence(), points_size, min_sample_size,
                    params->getMaxIters(), params->getRelaxCoef());
        else
            termination = StandardTerminationCriteria::create (params->getConfidence(),
                    points_size, min_sample_size, params->getMaxIters());
    } else if (params->getVerifier() == VerificationMethod::SprtVerifier) {
        termination = SPRTTermination::create(((SPRT *) verifier.get())->getSPRTvector(),
             params->getConfidence(), points_size, min_sample_size, params->getMaxIters());
    } else
        termination = StandardTerminationCriteria::create
            (params->getConfidence(), points_size, min_sample_size, params->getMaxIters());

    if (params->getLO() != LocalOptimMethod::LOCAL_OPTIM_NULL) {
        lo_sampler = UniformRandomGenerator::create(state++, points_size, params->getLOSampleSize());
        switch (params->getLO()) {
            case LocalOptimMethod::LOCAL_OPTIM_INNER_LO:
                lo = InnerIterativeLocalOptimization::create(estimator, quality, lo_sampler,
                     points_size, threshold, false, params->getLOIterativeSampleSize(),
                     params->getLOInnerMaxIters(), params->getLOIterativeMaxIters(),
                     params->getLOThresholdMultiplier()); break;
            case LocalOptimMethod::LOCAL_OPTIM_INNER_AND_ITER_LO:
                lo = InnerIterativeLocalOptimization::create(estimator, quality, lo_sampler,
                     points_size, threshold, true, params->getLOIterativeSampleSize(),
                     params->getLOInnerMaxIters(), params->getLOIterativeMaxIters(),
                     params->getLOThresholdMultiplier()); break;
            case LocalOptimMethod::LOCAL_OPTIM_GC:
                lo = GraphCut::create(estimator, error, quality, graph, lo_sampler, threshold,
                   params->getGraphCutSpatialCoherenceTerm(), params->getLOInnerMaxIters()); break;
            case LocalOptimMethod::LOCAL_OPTIM_SIGMA:
                lo = SigmaConsensus::create(estimator, error, quality, verifier,
                     params->getLOSampleSize(), params->getLOInnerMaxIters(),
                     params->getDegreesOfFreedom(), params->getSigmaQuantile(),
                     params->getUpperIncompleteOfSigmaQuantile(), params->getC(), max_thr); break;
            default: CV_Error(cv::Error::StsNotImplemented , "Local Optimization is not implemented!");
        }
    }

    if (params->getFinalPolisher() == PolishingMethod::LSQPolisher)
        polisher = LeastSquaresPolishing::create(estimator, quality, params->getFinalLSQIterations());

    Ransac ransac (params, points_size, estimator, quality, sampler,
          termination, verifier, degeneracy, lo, polisher, params->isParallel(), state);
    if (ransac.run(ransac_output)) {
        if (params->isPnP()) {
            // convert R to rodrigues and back and recalculate inliers which due to numerical
            // issues can differ
            Mat out, R, newR, newP, t, rvec;
            if (K1.empty()) {
                usac::Utils::decomposeProjection (ransac_output->getModel(), K1, R, t);
                Rodrigues(R, rvec);
                hconcat(rvec, t, out);
                hconcat(out, K1, out);
            } else {
                const Mat Rt = K1.inv() * ransac_output->getModel();
                t = Rt.col(3);
                Rodrigues(Rt.colRange(0,3), rvec);
                hconcat(rvec, t, out);
            }
            Rodrigues(rvec, newR);
            hconcat(K1 * newR, K1 * t, newP);
            std::vector<bool> inliers_mask(points_size);
            quality->getInliers(newP, inliers_mask);
            ransac_output = RansacOutput::create(out, inliers_mask, 0,0,0,0,0,0);
        }
        return true;
    }
    return false;
}
}}
