// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"

namespace cv { namespace usac {
////////////////////////////////// STANDARD TERMINATION ///////////////////////////////////////////
class StandardTerminationCriteriaImpl : public StandardTerminationCriteria {
private:
    const double log_confidence;
    const int points_size, sample_size, MAX_ITERATIONS;
public:
    StandardTerminationCriteriaImpl (double confidence, int points_size_,
                                     int sample_size_, int max_iterations_) :
            log_confidence(log(1 - confidence)), points_size (points_size_),
            sample_size (sample_size_), MAX_ITERATIONS(max_iterations_)  {}

    /*
     * Get upper bound iterations for any sample number
     * n is points size, w is inlier ratio, p is desired probability, k is expceted number of iterations.
     * 1 - p = (1 - w^n)^k,
     * k = log_(1-w^n) (1-p)
     * k = ln (1-p) / ln (1-w^n)
     *
     * w^n is probability that all N points are inliers.
     * (1 - w^n) is probability that at least one point of N is outlier.
     * 1 - p = (1-w^n)^k is probability that in K steps of getting at least one outlier is 1% (5%).
     */
    int update (const Mat &/*model*/, int inlier_number) override {
        const double predicted_iters = log_confidence / log(1 - std::pow
            (static_cast<double>(inlier_number) / points_size, sample_size));

        // if inlier_prob == 1 then log(0) = -inf, predicted_iters == -0
        // if inlier_prob == 0 then log(1) = 0   , predicted_iters == (+-) inf

        if (! std::isinf(predicted_iters) && predicted_iters < MAX_ITERATIONS)
            return static_cast<int>(predicted_iters);
        return MAX_ITERATIONS;
    }

    Ptr<TerminationCriteria> clone () const override {
        return makePtr<StandardTerminationCriteriaImpl>(1-exp(log_confidence), points_size,
                sample_size, MAX_ITERATIONS);
    }
};
Ptr<StandardTerminationCriteria> StandardTerminationCriteria::create(double confidence,
    int points_size_, int sample_size_, int max_iterations_) {
    return makePtr<StandardTerminationCriteriaImpl>(confidence, points_size_,
                        sample_size_, max_iterations_);
}

/////////////////////////////////////// SPRT TERMINATION //////////////////////////////////////////
class SPRTTerminationImpl : public SPRTTermination {
private:
    const std::vector<SPRT_history> &sprt_histories;
    const double log_eta_0;
    const int points_size, sample_size, MAX_ITERATIONS;
public:
    SPRTTerminationImpl (const std::vector<SPRT_history> &sprt_histories_, double confidence,
           int points_size_, int sample_size_, int max_iterations_)
           : sprt_histories (sprt_histories_), log_eta_0(log(1-confidence)),
           points_size (points_size_), sample_size (sample_size_),MAX_ITERATIONS(max_iterations_){}

    /*
     * Termination criterion:
     * l is number of tests
     * n(l) = Product from i = 0 to l ( 1 - P_g (1 - A(i)^(-h(i)))^k(i) )
     * log n(l) = sum from i = 0 to l k(i) * ( 1 - P_g (1 - A(i)^(-h(i))) )
     *
     *        log (n0) - log (n(l-1))
     * k(l) = -----------------------  (9)
     *          log (1 - P_g*A(l)^-1)
     *
     * A is decision threshold
     * P_g is probability of good model.
     * k(i) is number of samples verified by i-th sprt.
     * n0 is typically set to 0.05
     * this equation does not have to be evaluated before nR < n0
     * nR = (1 - P_g)^k
     */
    int update (const Mat &/*model*/, int inlier_size) override {
        if (sprt_histories.empty())
            return std::min(MAX_ITERATIONS, getStandardUpperBound(inlier_size));

        const double epsilon = static_cast<double>(inlier_size) / points_size; // inlier probability
        const double P_g = pow (epsilon, sample_size); // probability of good sample

        double log_eta_lmin1 = 0;

        int total_number_of_tested_samples = 0;
        const int sprts_size_min1 = static_cast<int>(sprt_histories.size())-1;
        if (sprts_size_min1 < 0) return getStandardUpperBound(inlier_size);
        // compute log n(l-1), l is number of tests
        for (int test = 0; test < sprts_size_min1; test++) {
            log_eta_lmin1 += log (1 - P_g * (1 - pow (sprt_histories[test].A,
             -computeExponentH(sprt_histories[test].epsilon, epsilon,sprt_histories[test].delta))))
                         * sprt_histories[test].tested_samples;
            total_number_of_tested_samples += sprt_histories[test].tested_samples;
        }

        // Implementation note: since η > ηR the equation (9) does not have to be evaluated
        // before ηR < η0 is satisfied.
        if (std::pow(1 - P_g, total_number_of_tested_samples) < log_eta_0)
            return std::min(MAX_ITERATIONS, getStandardUpperBound(inlier_size));
        // use decision threshold A for last test (l-th)
        const double predicted_iters_sprt = (log_eta_0 - log_eta_lmin1) /
                log (1 - P_g * (1 - 1 / sprt_histories[sprts_size_min1].A)); // last A
        if (std::isnan(predicted_iters_sprt) || std::isinf(predicted_iters_sprt))
            return getStandardUpperBound(inlier_size);

        if (predicted_iters_sprt < 0) return 0;
        // compare with standard upper bound
        if (predicted_iters_sprt < MAX_ITERATIONS)
            return std::min(static_cast<int>(predicted_iters_sprt),
                    getStandardUpperBound(inlier_size));
        return getStandardUpperBound(inlier_size);
    }

    Ptr<TerminationCriteria> clone () const override {
        return makePtr<SPRTTerminationImpl>(sprt_histories, 1-exp(log_eta_0), points_size,
               sample_size, MAX_ITERATIONS);
    }
private:
    inline int getStandardUpperBound(int inlier_size) const {
        const double predicted_iters = log_eta_0 / log(1 - std::pow
                (static_cast<double>(inlier_size) / points_size, sample_size));
        return (! std::isinf(predicted_iters) && predicted_iters < MAX_ITERATIONS) ?
                static_cast<int>(predicted_iters) : MAX_ITERATIONS;
    }
    /*
     * h(i) must hold
     *
     *     δ(i)                  1 - δ(i)
     * ε (-----)^h(i) + (1 - ε) (--------)^h(i) = 1
     *     ε(i)                  1 - ε(i)
     *
     * ε * a^h + (1 - ε) * b^h = 1
     * Has numerical solution.
     */
    static double computeExponentH (double epsilon, double epsilon_new, double delta) {
        const double a = log (delta / epsilon); // log likelihood ratio
        const double b = log ((1 - delta) / (1 - epsilon));

        const double x0 = log (1 / (1 - epsilon_new)) / b;
        const double v0 = epsilon_new * exp (x0 * a);
        const double x1 = log ((1 - 2*v0) / (1 - epsilon_new)) / b;
        const double v1 = epsilon_new * exp (x1 * a) + (1 - epsilon_new) * exp(x1 * b);
        const double h = x0 - (x0 - x1) / (1 + v0 - v1) * v0;

        if (std::isnan(h))
            // The equation always has solution for h = 0
            // ε * a^0 + (1 - ε) * b^0 = 1
            // ε + 1 - ε = 1 -> 1 = 1
            return 0;
        return h;
    }
};
Ptr<SPRTTermination> SPRTTermination::create(const std::vector<SPRT_history> &sprt_histories_,
    double confidence, int points_size_, int sample_size_, int max_iterations_) {
    return makePtr<SPRTTerminationImpl>(sprt_histories_, confidence, points_size_, sample_size_,
                    max_iterations_);
}

///////////////////////////// PROGRESSIVE-NAPSAC-SPRT TERMINATION /////////////////////////////////
class SPRTPNapsacTerminationImpl : public SPRTPNapsacTermination {
private:
    SPRTTerminationImpl sprt_termination;
    const std::vector<SPRT_history> &sprt_histories;
    const double relax_coef, log_confidence;
    const int points_size, sample_size, MAX_ITERS;
public:

    SPRTPNapsacTerminationImpl (const std::vector<SPRT_history> &sprt_histories_,
            double confidence, int points_size_, int sample_size_,
            int max_iterations_, double relax_coef_)
            : sprt_termination (sprt_histories_, confidence, points_size_, sample_size_,
            max_iterations_), sprt_histories (sprt_histories_),
            relax_coef (relax_coef_), log_confidence(log(1-confidence)),
          points_size (points_size_), sample_size (sample_size_),
          MAX_ITERS (max_iterations_) {}

    int update (const Mat &model, int inlier_number) override {
        int predicted_iterations = sprt_termination.update(model, inlier_number);

        const double inlier_prob = static_cast<double>(inlier_number) / points_size + relax_coef;
        if (inlier_prob >= 1)
            return 0;

        const double predicted_iters = log_confidence / log(1 - std::pow(inlier_prob, sample_size));

        if (! std::isinf(predicted_iters) && predicted_iters < predicted_iterations)
            return static_cast<int>(predicted_iters);
        return predicted_iterations;
    }
    Ptr<TerminationCriteria> clone () const override {
        return makePtr<SPRTPNapsacTerminationImpl>(sprt_histories, 1-exp(log_confidence),
                points_size, sample_size, MAX_ITERS, relax_coef);
    }
};
Ptr<SPRTPNapsacTermination> SPRTPNapsacTermination::create(const std::vector<SPRT_history>&
        sprt_histories_, double confidence, int points_size_, int sample_size_,
        int max_iterations_, double relax_coef_) {
    return makePtr<SPRTPNapsacTerminationImpl>(sprt_histories_, confidence, points_size_,
                   sample_size_, max_iterations_, relax_coef_);
}
////////////////////////////////////// PROSAC TERMINATION /////////////////////////////////////////

class ProsacTerminationCriteriaImpl : public ProsacTerminationCriteria {
private:
    const double log_confidence, beta, non_randomness_phi, inlier_threshold;
    const int MAX_ITERATIONS, points_size, min_termination_length, sample_size;
    const Ptr<ProsacSampler> sampler;

    std::vector<int> non_random_inliers;

    const Ptr<Error> error;
public:
    ProsacTerminationCriteriaImpl (const Ptr<Error> &error_, int points_size_,int sample_size_,
            double confidence, int max_iterations, int min_termination_length_, double beta_,
            double non_randomness_phi_, double inlier_threshold_) : log_confidence
            (log(1-confidence)), beta(beta_), non_randomness_phi(non_randomness_phi_),
            inlier_threshold(inlier_threshold_), MAX_ITERATIONS(max_iterations),
            points_size (points_size_), min_termination_length (min_termination_length_),
            sample_size(sample_size_), error (error_) { init(); }

    ProsacTerminationCriteriaImpl (const Ptr<ProsacSampler> &sampler_,const Ptr<Error> &error_,
            int points_size_, int sample_size_, double confidence, int max_iterations,
            int min_termination_length_, double beta_, double non_randomness_phi_,
            double inlier_threshold_) : log_confidence(log(1-confidence)), beta(beta_),
            non_randomness_phi(non_randomness_phi_), inlier_threshold(inlier_threshold_),
            MAX_ITERATIONS(max_iterations), points_size (points_size_),
            min_termination_length (min_termination_length_), sample_size(sample_size_),
            sampler(sampler_), error (error_) { init(); }

    void init () {
        // m is sample_size
        // N is points_size

        // non-randomness constraint
        // The non-randomness requirement prevents PROSAC
        // from selecting a solution supported by outliers that are
        // by chance consistent with it.  The constraint is typically
        // checked ex-post in standard approaches [1]. The distribution
        // of the cardinalities of sets of random ‘inliers’ is binomial
        // i-th entry - inlier counts for termination up to i-th point (term length = i+1)

        // ------------------------------------------------------------------------
        // initialize the data structures that determine stopping
        // see probabilities description below.

        non_random_inliers = std::vector<int>(points_size, 0);
        std::vector<double> pn_i_arr(points_size);
        const double beta2compl_beta = beta / (1-beta);
        const int step_n = 50, max_n = std::min(points_size, 1200);
        for (int n = sample_size; n <= points_size; n+=step_n) {
            if (n > max_n) {
                // skip expensive calculation
                break;
            }

            // P^R_n(i) = β^(i−m) (1−β)^(n−i+m) (n−m i−m). (7) i = m,...,N
            // initial value for i = m = sample_size
            // P^R_n(i=m)   = β^(0) (1−β)^(n)   (n-m 0) = (1-β)^(n)
            // P^R_n(i=m+1) = β^(1) (1−β)^(n−1) (n−m 1) = P^R_n(i=m) * β   / (1-β)   * (n-m) / 1
            // P^R_n(i=m+2) = β^(2) (1−β)^(n−2) (n−m 2) = P^R_n(i=m) * β^2 / (1-β)^2 * (n-m-1)(n-m) / 2
            // So, for each i=m+1.., P^R_n(i+1) must be calculated as P^R_n(i) * β / (1-β) * (n-i+1) / (i-m)

            pn_i_arr[sample_size-1] = std::pow(1-beta, n);
            double pn_i = pn_i_arr[sample_size-1]; // prob of random inlier set of size i for subset size n
            for (int i = sample_size+1; i <= n; i++) {
                // use recurrent relation to fulfill remaining values
                pn_i *= beta2compl_beta * static_cast<double>(n-i+1) / (i-sample_size);
                // update
                pn_i_arr[i-1] = pn_i;
            }

            // find minimum number of inliers satisfying the non-randomness constraint
            // Imin n = min{j : n∑i=j P^R_n(i) < Ψ }. (8)
            double acc = 0;
            int i_min = sample_size; // there is always sample_size inliers
            for (int i = n; i >= sample_size; i--) {
                acc += pn_i_arr[i-1];
                if (acc < non_randomness_phi) i_min = i;
                else break;
            }
            non_random_inliers[n-1] = i_min;
        }

        // approximate values of binomial distribution
        for (int n = sample_size; n <= points_size; n+=step_n) {
            if (n-1+step_n >= max_n) {
                // copy rest of the values
                std::fill(&non_random_inliers[0]+n-1, &non_random_inliers[0]+points_size, non_random_inliers[n-1]);
                break;
            }
            const int non_rand_n = non_random_inliers[n-1];
            const double step = (double)(non_random_inliers[n-1+step_n] - non_rand_n) / (double)step_n;
            for (int i = 0; i < step_n-1; i++)
                non_random_inliers[n+i] = (int)(non_rand_n + (i+1)*step);
        }
    }
    /*
     * The PROSAC algorithm terminates if the number of inliers I_n*
     * within the set U_n* satisfies the following conditions:
     *
     * • non-randomness – the probability that I_n* out of n* (termination_length)
     * data points are by chance inliers to an arbitrary incorrect model
     * is smaller than Ψ (typically set to 5%)
     *
     * • maximality – the probability that a solution with more than
     * In* inliers in U_n* exists and was not found after k
     * samples is smaller than η0 (typically set to 5%).
     */
    int update (const Mat &model, int inliers_size) override {
        int predicted_iterations = MAX_ITERATIONS;
        /*
         * The termination length n* is chosen to minimize k_n*(η0) subject to I_n* ≥ I_min n*;
         * k_n*(η0) >= log(η0) / log(1 - (I_n* / n*)^m)
         * g(k) <= n, I_n is number of inliers under termination length n.
         */
        const auto &errors = error->getErrors(model);

        // find number of inliers under g(k)
        int num_inliers_under_termination_len = 0;
        for (int pt = 0; pt < min_termination_length; pt++)
            if (errors[pt] < inlier_threshold)
                num_inliers_under_termination_len++;

        for (int termination_len = min_termination_length; termination_len < points_size;termination_len++){
            if (errors[termination_len /* = point*/] < inlier_threshold) {
                num_inliers_under_termination_len++;

                // non-random constraint must satisfy I_n* ≥ I_min n*.
                if (num_inliers_under_termination_len < non_random_inliers[termination_len])
                    continue;

                // add 1 to termination length since num_inliers_under_termination_len is updated
                const double new_max_samples = log_confidence / log(1 -
                        std::pow(static_cast<double>(num_inliers_under_termination_len)
                        / (termination_len+1), sample_size));

                if (! std::isinf(new_max_samples) && predicted_iterations > new_max_samples) {
                    predicted_iterations = static_cast<int>(new_max_samples);
                    if (predicted_iterations == 0) break;
                    if (sampler != nullptr)
                        sampler->setTerminationLength(termination_len);
                }
            }
        }

        // compare also when termination length = points_size,
        // so inliers under termination length is total number of inliers:
        const double predicted_iters = log_confidence / log(1 - std::pow
                (static_cast<double>(inliers_size) / points_size, sample_size));

        if (! std::isinf(predicted_iters) && predicted_iters < predicted_iterations)
            return static_cast<int>(predicted_iters);
        return predicted_iterations;
    }

    Ptr<TerminationCriteria> clone () const override {
        return makePtr<ProsacTerminationCriteriaImpl>(error->clone(),
            points_size, sample_size, 1-exp(log_confidence), MAX_ITERATIONS,
            min_termination_length, beta, non_randomness_phi, inlier_threshold);
    }
};

Ptr<ProsacTerminationCriteria>
ProsacTerminationCriteria::create(const Ptr<ProsacSampler> &sampler, const Ptr<Error> &error,
        int points_size_, int sample_size_, double confidence, int max_iterations,
        int min_termination_length_, double beta, double non_randomness_phi, double inlier_thresh) {
    return makePtr<ProsacTerminationCriteriaImpl> (sampler, error, points_size_, sample_size_,
            confidence, max_iterations, min_termination_length_,
            beta, non_randomness_phi, inlier_thresh);
}
}}
