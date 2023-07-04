// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"

namespace cv { namespace usac {
int Quality::getInliers(const Ptr<Error> &error, const Mat &model, std::vector<int> &inliers, double threshold) {
    const auto &errors = error->getErrors(model);
    int num_inliers = 0;
    for (int point = 0; point < (int)inliers.size(); point++)
        if (errors[point] < threshold)
            inliers[num_inliers++] = point;
    return num_inliers;
}
int Quality::getInliers(const Ptr<Error> &error, const Mat &model, std::vector<bool> &inliers_mask, double threshold) {
    std::fill(inliers_mask.begin(), inliers_mask.end(), false);
    const auto &errors = error->getErrors(model);
    int num_inliers = 0;
    for (int point = 0; point < (int)inliers_mask.size(); point++)
        if (errors[point] < threshold) {
            inliers_mask[point] = true;
            num_inliers++;
        }
    return num_inliers;
}
int Quality::getInliers (const std::vector<float> &errors, std::vector<bool> &inliers, double threshold) {
    std::fill(inliers.begin(), inliers.end(), false);
    int cnt = 0, inls = 0;
    for (const auto e : errors) {
        if (e < threshold) {
            inliers[cnt] = true;
            inls++;
        }
        cnt++;
    }
    return inls;
}
int Quality::getInliers (const std::vector<float> &errors, std::vector<int> &inliers, double threshold) {
    int cnt = 0, inls = 0;
    for (const auto e : errors) {
        if (e < threshold)
            inliers[inls++] = cnt;
        cnt++;
    }
    return inls;
}

class RansacQualityImpl : public RansacQuality {
private:
    const Ptr<Error> error;
    const int points_size;
    const double threshold;
    double best_score;
public:
    RansacQualityImpl (int points_size_, double threshold_, const Ptr<Error> &error_)
            : error (error_), points_size(points_size_), threshold(threshold_) {
        best_score = std::numeric_limits<double>::max();
    }

    Score getScore (const Mat &model) const override {
        error->setModelParameters(model);
        int inlier_number = 0;
        const auto preemptive_thr = -points_size - best_score;
        for (int point = 0; point < points_size; point++)
            if (error->getError(point) < threshold)
                inlier_number++;
            else if (inlier_number - point < preemptive_thr)
                    break;
        // score is negative inlier number! If less then better
        return {inlier_number, -static_cast<double>(inlier_number)};
    }

    Score getScore (const std::vector<float> &errors) const override {
        int inlier_number = 0;
        for (int point = 0; point < points_size; point++)
            if (errors[point] < threshold)
                inlier_number++;
        // score is negative inlier number! If less then better
        return {inlier_number, -static_cast<double>(inlier_number)};
    }

    void setBestScore(double best_score_) override {
        if (best_score > best_score_) best_score = best_score_;
    }

    int getInliers (const Mat &model, std::vector<int> &inliers) const override
    { return Quality::getInliers(error, model, inliers, threshold); }
    int getInliers (const Mat &model, std::vector<int> &inliers, double thr) const override
    { return Quality::getInliers(error, model, inliers, thr); }
    int getInliers (const Mat &model, std::vector<bool> &inliers_mask) const override
    { return Quality::getInliers(error, model, inliers_mask, threshold); }
    double getThreshold () const override { return threshold; }
    int getPointsSize () const override { return points_size; }
    Ptr<Error> getErrorFnc () const override { return error; }
};

Ptr<RansacQuality> RansacQuality::create(int points_size_, double threshold_,
        const Ptr<Error> &error_) {
    return makePtr<RansacQualityImpl>(points_size_, threshold_, error_);
}

class MsacQualityImpl : public MsacQuality {
protected:
    const Ptr<Error> error;
    const int points_size;
    const double threshold, k_msac;
    double best_score, norm_thr, one_over_thr;
public:
    MsacQualityImpl (int points_size_, double threshold_, const Ptr<Error> &error_, double k_msac_)
            : error (error_), points_size (points_size_), threshold (threshold_), k_msac(k_msac_) {
        best_score = std::numeric_limits<double>::max();
        norm_thr = threshold*k_msac;
        one_over_thr = 1 / norm_thr;
    }

    inline Score getScore (const Mat &model) const override {
        error->setModelParameters(model);
        double err, sum_errors = 0;
        int inlier_number = 0;
        const auto preemptive_thr = points_size + best_score;
        for (int point = 0; point < points_size; point++) {
            err = error->getError(point);
            if (err < norm_thr) {
                sum_errors -= (1 - err * one_over_thr);
                if (err < threshold)
                    inlier_number++;
            } else if (sum_errors + point > preemptive_thr)
                break;
        }
        return {inlier_number, sum_errors};
    }

    Score getScore (const std::vector<float> &errors) const override {
        double sum_errors = 0;
        int inlier_number = 0;
        for (int point = 0; point < points_size; point++) {
            const auto err = errors[point];
            if (err < norm_thr) {
                sum_errors -= (1 - err * one_over_thr);
                if (err < threshold)
                    inlier_number++;
            }
        }
        return {inlier_number, sum_errors};
    }

    void setBestScore(double best_score_) override {
        if (best_score > best_score_) best_score = best_score_;
    }

    int getInliers (const Mat &model, std::vector<int> &inliers) const override
    { return Quality::getInliers(error, model, inliers, threshold); }
    int getInliers (const Mat &model, std::vector<int> &inliers, double thr) const override
    { return Quality::getInliers(error, model, inliers, thr); }
    int getInliers (const Mat &model, std::vector<bool> &inliers_mask) const override
    { return Quality::getInliers(error, model, inliers_mask, threshold); }
    double getThreshold () const override { return threshold; }
    int getPointsSize () const override { return points_size; }
    Ptr<Error> getErrorFnc () const override { return error; }
};
Ptr<MsacQuality> MsacQuality::create(int points_size_, double threshold_,
        const Ptr<Error> &error_, double k_msac) {
    return makePtr<MsacQualityImpl>(points_size_, threshold_, error_, k_msac);
}

class MagsacQualityImpl : public MagsacQuality {
private:
    const Ptr<Error> error;
    const Ptr<GammaValues> gamma_generator;
    const int points_size;
    // for example, maximum standard deviation of noise.
    const double maximum_threshold_sqr, tentative_inlier_threshold;
    // Calculate the gamma value of k
    const double gamma_value_of_k;
    double previous_best_loss;
    float maximum_sigma_2_per_2;
    // Calculating 2^(DoF + 1) / \sigma_{max} which will be used for the estimation and,
    // due to being constant, it is better to calculate it a priori.
    double two_ad_dof_plus_one_per_maximum_sigma, rescale_err, norm_loss;
    const std::vector<double> &stored_complete_gamma_values, &stored_lower_incomplete_gamma_values;
    unsigned int stored_incomplete_gamma_number_min1;
public:

    MagsacQualityImpl (double maximum_thr, int points_size_, const Ptr<Error> &error_,
                       const Ptr<GammaValues> &gamma_generator_,
                       double tentative_inlier_threshold_, int DoF, double sigma_quantile,
                       double upper_incomplete_of_sigma_quantile)
            : error (error_), gamma_generator(gamma_generator_), points_size(points_size_),
            maximum_threshold_sqr(maximum_thr*maximum_thr),
            tentative_inlier_threshold(tentative_inlier_threshold_),
            gamma_value_of_k (upper_incomplete_of_sigma_quantile),
            stored_complete_gamma_values (gamma_generator->getCompleteGammaValues()),
            stored_lower_incomplete_gamma_values (gamma_generator->getIncompleteGammaValues()) {
        previous_best_loss = std::numeric_limits<double>::max();
        const auto maximum_sigma = (float)sqrt(maximum_threshold_sqr) / sigma_quantile;
        const auto maximum_sigma_2 = (float) (maximum_sigma * maximum_sigma);
        maximum_sigma_2_per_2 = maximum_sigma_2 / 2.f;
        const auto maximum_sigma_2_times_2 = maximum_sigma_2 * 2.f;
        two_ad_dof_plus_one_per_maximum_sigma = pow(2.0, (DoF + 1.0)*.5)/maximum_sigma;
        rescale_err = gamma_generator->getScaleOfGammaCompleteValues() / maximum_sigma_2_times_2;
        stored_incomplete_gamma_number_min1 = static_cast<unsigned int>(gamma_generator->getTableSize()-1);

        double max_loss = 1e-10;
        // MAGSAC maximum / minimum loss does not have to be in extremum residuals
        // make 30 iterations to find maximum loss
        const double step = maximum_threshold_sqr / 30;
        double sqr_res = 0;
        while (sqr_res < maximum_threshold_sqr) {
            auto x= static_cast<unsigned int>(rescale_err * sqr_res);
            if (x > stored_incomplete_gamma_number_min1)
                x = stored_incomplete_gamma_number_min1;
            const double loss = two_ad_dof_plus_one_per_maximum_sigma * (maximum_sigma_2_per_2 *
                    stored_lower_incomplete_gamma_values[x] + sqr_res * 0.25 *
                    (stored_complete_gamma_values[x] - gamma_value_of_k));
            if (max_loss < loss)
                max_loss = loss;
            sqr_res += step;
        }
        norm_loss = two_ad_dof_plus_one_per_maximum_sigma / max_loss;
    }

    // https://github.com/danini/magsac
    Score getScore (const Mat &model) const override {
        error->setModelParameters(model);
        double total_loss = 0.0;
        int num_tentative_inliers = 0;
        const auto preemptive_thr = points_size + previous_best_loss;
        for (int point_idx = 0; point_idx < points_size; point_idx++) {
            const float squared_residual = error->getError(point_idx);
            if (squared_residual < tentative_inlier_threshold)
                num_tentative_inliers++;
            if (squared_residual < maximum_threshold_sqr) { // consider point as inlier
                // Get the position of the gamma value in the lookup table
                auto x = static_cast<unsigned int>(rescale_err * squared_residual);
                // If the sought gamma value is not stored in the lookup, return the closest element
                if (x > stored_incomplete_gamma_number_min1)
                    x = stored_incomplete_gamma_number_min1;
                // Calculate the loss implied by the current point
                total_loss -= (1 - (maximum_sigma_2_per_2 *
                    stored_lower_incomplete_gamma_values[x] + squared_residual * 0.25 *
                    (stored_complete_gamma_values[x] - gamma_value_of_k)) * norm_loss);
            } else if (total_loss + point_idx > preemptive_thr)
                break;
        }
        return {num_tentative_inliers, total_loss};
    }

    Score getScore (const std::vector<float> &errors) const override {
        double total_loss = 0.0;
        int num_tentative_inliers = 0;
        for (int point_idx = 0; point_idx < points_size; point_idx++) {
            const float squared_residual = errors[point_idx];
            if (squared_residual < tentative_inlier_threshold)
                num_tentative_inliers++;
            if (squared_residual < maximum_threshold_sqr) {
                auto x = static_cast<unsigned int>(rescale_err * squared_residual);
                if (x > stored_incomplete_gamma_number_min1)
                    x = stored_incomplete_gamma_number_min1;
                total_loss -= (1 - (maximum_sigma_2_per_2 *
                        stored_lower_incomplete_gamma_values[x] + squared_residual * 0.25 *
                        (stored_complete_gamma_values[x] - gamma_value_of_k)) * norm_loss);
            }
        }
        return {num_tentative_inliers, total_loss};
    }

    void setBestScore (double best_loss) override {
        if (previous_best_loss > best_loss) previous_best_loss = best_loss;
    }

    int getInliers (const Mat &model, std::vector<int> &inliers) const override
    { return Quality::getInliers(error, model, inliers, tentative_inlier_threshold); }
    int getInliers (const Mat &model, std::vector<int> &inliers, double thr) const override
    { return Quality::getInliers(error, model, inliers, thr); }
    int getInliers (const Mat &model, std::vector<bool> &inliers_mask) const override
    { return Quality::getInliers(error, model, inliers_mask, tentative_inlier_threshold); }
    double getThreshold () const override { return tentative_inlier_threshold; }
    int getPointsSize () const override { return points_size; }
    Ptr<Error> getErrorFnc () const override { return error; }
};
Ptr<MagsacQuality> MagsacQuality::create(double maximum_thr, int points_size_, const Ptr<Error> &error_,
        const Ptr<GammaValues> &gamma_generator,
        double tentative_inlier_threshold_, int DoF, double sigma_quantile,
        double upper_incomplete_of_sigma_quantile) {
    return makePtr<MagsacQualityImpl>(maximum_thr, points_size_, error_, gamma_generator,
        tentative_inlier_threshold_, DoF, sigma_quantile, upper_incomplete_of_sigma_quantile);
}

class LMedsQualityImpl : public LMedsQuality {
private:
    const Ptr<Error> error;
    const int points_size;
    const double threshold;
public:
    LMedsQualityImpl (int points_size_, double threshold_, const Ptr<Error> &error_) :
        error (error_), points_size (points_size_), threshold (threshold_) {}

    // Finds median of errors.
    Score getScore (const Mat &model) const override {
        std::vector<float> errors = error->getErrors(model);
        int inlier_number = 0;
        for (int point = 0; point < points_size; point++)
            if (errors[point] < threshold)
                inlier_number++;
        // score is median of errors
        return {inlier_number, Utils::findMedian (errors)};
    }
    Score getScore (const std::vector<float> &errors_) const override {
        std::vector<float> errors = errors_;
        int inlier_number = 0;
        for (int point = 0; point < points_size; point++)
            if (errors[point] < threshold)
                inlier_number++;
        // score is median of errors
        return {inlier_number, Utils::findMedian (errors)};
    }

    void setBestScore (double /*best_score*/) override {}

    int getPointsSize () const override { return points_size; }
    int getInliers (const Mat &model, std::vector<int> &inliers) const override
    { return Quality::getInliers(error, model, inliers, threshold); }
    int getInliers (const Mat &model, std::vector<int> &inliers, double thr) const override
    { return Quality::getInliers(error, model, inliers, thr); }
    int getInliers (const Mat &model, std::vector<bool> &inliers_mask) const override
    { return Quality::getInliers(error, model, inliers_mask, threshold); }
    double getThreshold () const override { return threshold; }
    Ptr<Error> getErrorFnc () const override { return error; }
};
Ptr<LMedsQuality> LMedsQuality::create(int points_size_, double threshold_, const Ptr<Error> &error_) {
    return makePtr<LMedsQualityImpl>(points_size_, threshold_, error_);
}

class ModelVerifierImpl : public ModelVerifier {
private:
    Ptr<Quality> quality;
public:
    ModelVerifierImpl (const Ptr<Quality> &q) : quality(q) {}
    inline bool isModelGood(const Mat &model, Score &score) override {
        score = quality->getScore(model);
        return true;
    }
    void update (const Score &/*score*/, int /*iteration*/) override {}
    void reset() override {}
    void updateSPRT (double , double , double , double , double , const Score &) override {}
};
Ptr<ModelVerifier> ModelVerifier::create(const Ptr<Quality> &quality) {
    return makePtr<ModelVerifierImpl>(quality);
}

class AdaptiveSPRTImpl : public AdaptiveSPRT {
private:
    RNG rng;
    const Ptr<Error> err;
    const Ptr<Quality> quality;
    const int points_size;
    int highest_inlier_number, last_iteration;
    // time t_M needed to instantiate a model hypothesis given a sample
    // Let m_S be the number of models that are verified per sample
    const double inlier_threshold, norm_thr, one_over_thr;

    // alpha is false negative rate, alpha = 1 / A
    double t_M, lowest_sum_errors, current_epsilon, current_delta, current_A,
        delta_to_epsilon, complement_delta_to_complement_epsilon,
        time_ver_corr_sprt = 0, time_ver_corr = 0,
        one_over_complement_alpha, avg_num_checked_pts;

    std::vector<SPRT_history> sprt_histories, empty;
    std::vector<int> points_random_pool;
    std::vector<float> errors;

    bool do_sprt, adapt, IS_ADAPTIVE;
    const ScoreMethod score_type;
    double m_S;
public:
    AdaptiveSPRTImpl (int state, const Ptr<Quality> &quality_, int points_size_,
              double inlier_threshold_, double prob_pt_of_good_model, double prob_pt_of_bad_model,
              double time_sample, double avg_num_models, ScoreMethod score_type_,
              double k_mlesac_, bool is_adaptive) : rng(state), err(quality_->getErrorFnc()),
              quality(quality_), points_size(points_size_), inlier_threshold (quality->getThreshold()),
              norm_thr(inlier_threshold_*k_mlesac_), one_over_thr (1/norm_thr), t_M (time_sample),
              score_type (score_type_), m_S (avg_num_models) {

        // Generate array of random points for randomized evaluation
        points_random_pool = std::vector<int> (points_size_);
        // fill values from 0 to points_size-1
        for (int i = 0; i < points_size; i++)
            points_random_pool[i] = i;
        randShuffle(points_random_pool, 1, &rng);

        // reserve (approximately) some space for sprt vector.
        sprt_histories.reserve(20);

        highest_inlier_number = last_iteration = 0;
        lowest_sum_errors = std::numeric_limits<double>::max();
        if (score_type_ != ScoreMethod::SCORE_METHOD_MSAC)
            errors = std::vector<float>(points_size_);
        IS_ADAPTIVE = is_adaptive;
        delta_to_epsilon = one_over_complement_alpha = complement_delta_to_complement_epsilon = current_A = -1;
        avg_num_checked_pts = points_size_;
        adapt = IS_ADAPTIVE;
        do_sprt = !IS_ADAPTIVE;
        if (IS_ADAPTIVE) {
            // all these variables will be initialized later
            current_epsilon = prob_pt_of_good_model;
            current_delta = prob_pt_of_bad_model;
        } else {
            current_epsilon = current_delta = 1e-5;
            createTest(prob_pt_of_good_model, prob_pt_of_bad_model);
        }
    }

    /*
     *                      p(x(r)|Hb)                  p(x(j)|Hb)
     * lambda(j) = Product (----------) = lambda(j-1) * ----------
     *                      p(x(r)|Hg)                  p(x(j)|Hg)
     * Set j = 1
     * 1.  Check whether j-th data point is consistent with the
     * model
     * 2.  Compute the likelihood ratio λj eq. (1)
     * 3.  If λj >  A, decide the model is ’bad’ (model "re-jected"),
     * else increment j or continue testing
     * 4.  If j = N the number of correspondences decide model "accepted"
     *
     * Verifies model and returns model score.

     * Returns true if model is good, false - otherwise.
     * @model: model to verify
     * @current_hypothesis: current RANSAC iteration
     * Return: true if model is good, false - otherwise.
     */
    inline bool isModelGood (const Mat &model, Score &out_score) override {
        // update error object with current model
        bool last_model_is_good = true;
        double sum_errors = 0;
        int tested_inliers = 0;
        if (! do_sprt || adapt) { // if adapt or not sprt then compute model score directly
            out_score = quality->getScore(model);
            tested_inliers = out_score.inlier_number;
            sum_errors = out_score.score;
        } else { // do sprt and not adapt
            err->setModelParameters(model);
            double lambda = 1;
            int random_pool_idx = rng.uniform(0, points_size), tested_point;
            if (score_type == ScoreMethod::SCORE_METHOD_MSAC) {
                const auto preemptive_thr = points_size + lowest_sum_errors;
                for (tested_point = 0; tested_point < points_size; tested_point++) {
                    if (random_pool_idx == points_size)
                        random_pool_idx = 0;
                    const float error = err->getError (points_random_pool[random_pool_idx++]);
                    if (error < inlier_threshold) {
                        tested_inliers++;
                        lambda *= delta_to_epsilon;
                    } else {
                        lambda *= complement_delta_to_complement_epsilon;
                        // since delta is always higher than epsilon, then lambda can increase only
                        // when point is not consistent with model
                        if (lambda > current_A)
                            break;
                    }
                    if (error < norm_thr)
                        sum_errors -= (1 - error * one_over_thr);
                    else if (sum_errors + tested_point > preemptive_thr)
                        break;
                }
            } else { // save errors into array here
                for (tested_point = 0; tested_point < points_size; tested_point++) {
                    if (random_pool_idx == points_size)
                        random_pool_idx = 0;
                    const int pt = points_random_pool[random_pool_idx++];
                    const float error = err->getError (pt);
                    if (error < inlier_threshold) {
                        tested_inliers++;
                        lambda *= delta_to_epsilon;
                    } else {
                        lambda *= complement_delta_to_complement_epsilon;
                        if (lambda > current_A)
                            break;
                    }
                    errors[pt] = error;
                }
            }
            last_model_is_good = tested_point == points_size;
        }
        if (last_model_is_good && do_sprt) {
            out_score.inlier_number = tested_inliers;
            if (score_type == ScoreMethod::SCORE_METHOD_MSAC)
                out_score.score = sum_errors;
            else if (score_type == ScoreMethod::SCORE_METHOD_RANSAC)
                out_score.score = -static_cast<double>(tested_inliers);
            else out_score = quality->getScore(errors);
        }
        return last_model_is_good;
    }

    // update SPRT parameters = called only once inside usac
    void updateSPRT (double time_model_est, double time_corr_ver, double new_avg_models, double new_delta, double new_epsilon, const Score &best_score) override {
        if (adapt) {
            adapt = false;
            m_S = new_avg_models;
            t_M = time_model_est / time_corr_ver;
            time_ver_corr = time_corr_ver;
            time_ver_corr_sprt = time_corr_ver * 1.05;
            createTest(new_epsilon, new_delta);
            highest_inlier_number = best_score.inlier_number;
            lowest_sum_errors = best_score.score;
        }
    }

    const std::vector<SPRT_history> &getSPRTvector () const override { return adapt ? empty : sprt_histories; }
    void update (const Score &score, int iteration) override {
        if (adapt || highest_inlier_number > score.inlier_number)
            return;

        if (sprt_histories.size() == 1 && sprt_histories.back().tested_samples == 0)
            sprt_histories.back().tested_samples = iteration;
        else if (! sprt_histories.empty())
            sprt_histories.back().tested_samples += iteration - last_iteration;

        SPRT_history new_sprt_history;
        new_sprt_history.epsilon = (double)score.inlier_number / points_size;
        highest_inlier_number = score.inlier_number;
        lowest_sum_errors = score.score;
        createTest(static_cast<double>(highest_inlier_number) / points_size, current_delta);
        new_sprt_history.delta = current_delta;
        new_sprt_history.A = current_A;
        sprt_histories.emplace_back(new_sprt_history);
        last_iteration = iteration;
    }
    int avgNumCheckedPts () const override { return do_sprt ? (int)avg_num_checked_pts + 1 : points_size; }
    void reset() override {
        adapt = true;
        do_sprt = false;
        highest_inlier_number = last_iteration = 0;
        lowest_sum_errors = DBL_MAX;
        sprt_histories.clear();
    }
private:
    // Update current epsilon, delta and threshold (A).
    bool createTest (double epsilon, double delta) {
        if (fabs(current_epsilon - epsilon) < FLT_EPSILON && fabs(current_delta - delta) < FLT_EPSILON)
            return false;
        // if epsilon is closed to 1 then set them to 0.99 to avoid numerical problems
        if (epsilon > 0.999999) epsilon = 0.999;
        // delta can't be higher than epsilon, because ratio delta / epsilon will be greater than 1
        if (epsilon < delta) delta = epsilon-0.001;
        // avoid delta going too high as it is very unlikely
        // e.g., 30% of points are consistent with bad model is not very real
        if (delta   > 0.3) delta = 0.3;

        const auto AC = estimateThresholdA (epsilon, delta);
        current_A = AC.first;
        const auto C = AC.second;
        current_delta = delta;
        current_epsilon = epsilon;
        one_over_complement_alpha = 1 / (1 - 1 / current_A);

        delta_to_epsilon = delta / epsilon;
        complement_delta_to_complement_epsilon = (1 - delta) / (1 - epsilon);

        if (IS_ADAPTIVE) {
            avg_num_checked_pts = std::min((log(current_A) / C) * one_over_complement_alpha, (double)points_size);
            do_sprt = time_ver_corr_sprt * avg_num_checked_pts < time_ver_corr * points_size;
        }
        return true;
    }
    std::pair<double,double> estimateThresholdA (double epsilon, double delta) {
        const double C = (1 - delta) * log ((1 - delta) / (1 - epsilon)) + delta * log (delta / epsilon);
        // K = K1/K2 + 1 = (t_M / P_g) / (m_S / (C * P_g)) + 1 = (t_M * C)/m_S + 1
        const double K = t_M * C / m_S + 1;
        double An, An_1 = K;
        // compute A using a recursive relation
        // A* = lim(n->inf)(An), the series typically converges within 4 iterations
        for (int i = 0; i < 10; i++) {
            An = K + log(An_1);
            if (fabs(An - An_1) < FLT_EPSILON)
                break;
            An_1 = An;
        }
        return std::make_pair(An, C);
    }
};
Ptr<AdaptiveSPRT> AdaptiveSPRT::create (int state, const Ptr<Quality> &quality, int points_size_,
            double inlier_threshold_, double prob_pt_of_good_model, double prob_pt_of_bad_model,
            double time_sample, double avg_num_models, ScoreMethod score_type_, double k_mlesac, bool is_adaptive) {
    return makePtr<AdaptiveSPRTImpl>(state, quality, points_size_, inlier_threshold_,
         prob_pt_of_good_model, prob_pt_of_bad_model, time_sample, avg_num_models, score_type_, k_mlesac, is_adaptive);
}
}}
