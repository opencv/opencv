// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"

namespace cv { namespace usac {
class EpipolarGeometryDegeneracyImpl : public EpipolarGeometryDegeneracy {
private:
    Mat points_mat;
    const int min_sample_size;
public:
    explicit EpipolarGeometryDegeneracyImpl (const Mat &points_, int sample_size_) :
        points_mat(points_), min_sample_size (sample_size_)
    {
        CV_DbgAssert(!points_mat.empty() && points_mat.isContinuous());
    }
    /*
     * Do oriented constraint to verify if epipolar geometry is in front or behind the camera.
     * Return: true if all points are in front of the camers w.r.t. tested epipolar geometry - satisfies constraint.
     *         false - otherwise.
     * x'^T F x = 0
     * e' × x' ~+ Fx   <=>  λe' × x' = Fx, λ > 0
     * e  × x ~+ x'^T F
     */
    inline bool isModelValid(const Mat &F_, const std::vector<int> &sample) const override {
        const Vec3d ep = Utils::getRightEpipole(F_);
        const auto * const e = ep.val; // of size 3x1
        const auto * const F = (double *) F_.data;
        const float * points = points_mat.ptr<float>();

        // without loss of generality, let the first point in sample be in front of the camera.
        int pt = 4*sample[0];
        // check only two first elements of vectors (e × x) and (x'^T F)
        // s1 = (x'^T F)[0] = x2 * F11 + y2 * F21 + 1 * F31
        // s2 = (e × x)[0] = e'_2 * 1 - e'_3 * y1
        // sign1 = s1 * s2
        const double sign1 = (F[0]*points[pt+2]+F[3]*points[pt+3]+F[6])*(e[1]-e[2]*points[pt+1]);

        for (int i = 1; i < min_sample_size; i++) {
            pt = 4 * sample[i];
            // if signum of the first point and tested point differs
            // then two points are on different sides of the camera.
            if (sign1*(F[0]*points[pt+2]+F[3]*points[pt+3]+F[6])*(e[1]-e[2]*points[pt+1])<0)
                    return false;
        }
        return true;
    }
};
void EpipolarGeometryDegeneracy::recoverRank (Mat &model, bool is_fundamental_mat) {
    /*
     * Do singular value decomposition.
     * Make last eigen value zero of diagonal matrix of singular values.
     */
    Matx33d U, Vt;
    Vec3d w;
    SVD::compute(model, w, U, Vt, SVD::MODIFY_A);
    if (is_fundamental_mat)
        model = Mat(U * Matx33d(w(0), 0, 0, 0, w(1), 0, 0, 0, 0) * Vt);
    else {
        const double mean_singular_val = (w[0] + w[1]) * 0.5;
        model = Mat(U * Matx33d(mean_singular_val, 0, 0, 0, mean_singular_val, 0, 0, 0, 0) * Vt);
    }
}
Ptr<EpipolarGeometryDegeneracy> EpipolarGeometryDegeneracy::create (const Mat &points_,
        int sample_size_) {
    return makePtr<EpipolarGeometryDegeneracyImpl>(points_, sample_size_);
}

class HomographyDegeneracyImpl : public HomographyDegeneracy {
private:
    Mat points_mat;
    const float TOLERANCE = 2 * FLT_EPSILON; // 2 from area of triangle
public:
    explicit HomographyDegeneracyImpl (const Mat &points_) :
            points_mat(points_)
    {
        CV_DbgAssert(!points_mat.empty() && points_mat.isContinuous());
    }

    inline bool isSampleGood (const std::vector<int> &sample) const override {
        const int smpl1 = 4*sample[0], smpl2 = 4*sample[1], smpl3 = 4*sample[2], smpl4 = 4*sample[3];
        // planar correspondences must lie on the same side of any line from two points in sample
        const float * points = points_mat.ptr<float>();
        const float x1 = points[smpl1], y1 = points[smpl1+1], X1 = points[smpl1+2], Y1 = points[smpl1+3];
        const float x2 = points[smpl2], y2 = points[smpl2+1], X2 = points[smpl2+2], Y2 = points[smpl2+3];
        const float x3 = points[smpl3], y3 = points[smpl3+1], X3 = points[smpl3+2], Y3 = points[smpl3+3];
        const float x4 = points[smpl4], y4 = points[smpl4+1], X4 = points[smpl4+2], Y4 = points[smpl4+3];
        // line from points 1 and 2
        const float ab_cross_x = y1 - y2, ab_cross_y = x2 - x1, ab_cross_z = x1 * y2 - y1 * x2;
        const float AB_cross_x = Y1 - Y2, AB_cross_y = X2 - X1, AB_cross_z = X1 * Y2 - Y1 * X2;

        // check if points 3 and 4 are on the same side of line ab on both images
        if ((ab_cross_x * x3 + ab_cross_y * y3 + ab_cross_z) *
            (AB_cross_x * X3 + AB_cross_y * Y3 + AB_cross_z) < 0)
            return false;
        if ((ab_cross_x * x4 + ab_cross_y * y4 + ab_cross_z) *
            (AB_cross_x * X4 + AB_cross_y * Y4 + AB_cross_z) < 0)
            return false;

        // line from points 3 and 4
        const float cd_cross_x = y3 - y4, cd_cross_y = x4 - x3, cd_cross_z = x3 * y4 - y3 * x4;
        const float CD_cross_x = Y3 - Y4, CD_cross_y = X4 - X3, CD_cross_z = X3 * Y4 - Y3 * X4;

        // check if points 1 and 2 are on the same side of line cd on both images
        if ((cd_cross_x * x1 + cd_cross_y * y1 + cd_cross_z) *
            (CD_cross_x * X1 + CD_cross_y * Y1 + CD_cross_z) < 0)
            return false;
        if ((cd_cross_x * x2 + cd_cross_y * y2 + cd_cross_z) *
            (CD_cross_x * X2 + CD_cross_y * Y2 + CD_cross_z) < 0)
            return false;

        // Checks if points are not collinear
        // If area of triangle constructed with 3 points is less then threshold then points are collinear:
        //           |x1 y1 1|             |x1      y1      1|
        // (1/2) det |x2 y2 1| = (1/2) det |x2-x1   y2-y1   0| = det |x2-x1   y2-y1| < 2 * threshold
        //           |x3 y3 1|             |x3-x1   y3-y1   0|       |x3-x1   y3-y1|
        // for points on the first image
        if (fabsf((x2-x1) * (y3-y1) - (y2-y1) * (x3-x1)) < TOLERANCE) return false; //1,2,3
        if (fabsf((x2-x1) * (y4-y1) - (y2-y1) * (x4-x1)) < TOLERANCE) return false; //1,2,4
        if (fabsf((x3-x1) * (y4-y1) - (y3-y1) * (x4-x1)) < TOLERANCE) return false; //1,3,4
        if (fabsf((x3-x2) * (y4-y2) - (y3-y2) * (x4-x2)) < TOLERANCE) return false; //2,3,4
        // for points on the second image
        if (fabsf((X2-X1) * (Y3-Y1) - (Y2-Y1) * (X3-X1)) < TOLERANCE) return false; //1,2,3
        if (fabsf((X2-X1) * (Y4-Y1) - (Y2-Y1) * (X4-X1)) < TOLERANCE) return false; //1,2,4
        if (fabsf((X3-X1) * (Y4-Y1) - (Y3-Y1) * (X4-X1)) < TOLERANCE) return false; //1,3,4
        if (fabsf((X3-X2) * (Y4-Y2) - (Y3-Y2) * (X4-X2)) < TOLERANCE) return false; //2,3,4

        return true;
    }
};
Ptr<HomographyDegeneracy> HomographyDegeneracy::create (const Mat &points_) {
    return makePtr<HomographyDegeneracyImpl>(points_);
}

class FundamentalDegeneracyViaEImpl : public FundamentalDegeneracyViaE {
private:
    bool is_F_objective;
    std::vector<std::vector<int>> instances = {{0,1,2,3,4}, {2,3,4,5,6}, {0,1,4,5,6}};
    std::vector<int> e_sample;
    const Ptr<Quality> quality;
    Ptr<EpipolarGeometryDegeneracy> e_degen, f_degen;
    Ptr<EssentialMinimalSolver5pts> e_solver;
    std::vector<Mat> e_models;
    const int E_SAMPLE_SIZE = 5;
    Matx33d K2_inv_t, K1_inv;
public:
    FundamentalDegeneracyViaEImpl (const Ptr<Quality> &quality_, const Mat &pts, const Mat &calib_pts, const Matx33d &K1, const Matx33d &K2, bool is_f_objective)
            : quality(quality_) {
        is_F_objective = is_f_objective;
        e_solver = EssentialMinimalSolver5pts::create(calib_pts, false/*svd*/, true/*nister*/);
        f_degen = is_F_objective ? EpipolarGeometryDegeneracy::create(pts, 7) : EpipolarGeometryDegeneracy::create(calib_pts, 7);
        e_degen = EpipolarGeometryDegeneracy::create(calib_pts, E_SAMPLE_SIZE);
        e_sample = std::vector<int>(E_SAMPLE_SIZE);
        if (is_f_objective) {
            K2_inv_t = K2.inv().t();
            K1_inv = K1.inv();
        }
    }
    bool isModelValid (const Mat &F, const std::vector<int> &sample) const override {
        return f_degen->isModelValid(F, sample);
    }
    bool recoverIfDegenerate (const std::vector<int> &sample_7pt, const Mat &/*best*/, const Score &best_score,
            Mat &out_model, Score &out_score) override {
        out_score = Score();
        for (const auto &instance : instances) {
            for (int i = 0; i < E_SAMPLE_SIZE; i++)
                e_sample[i] = sample_7pt[instance[i]];
            const int num_models = e_solver->estimate(e_sample, e_models);
            for (int i = 0; i < num_models; i++) {
                if (e_degen->isModelValid(e_models[i], e_sample)) {
                    const Mat model = is_F_objective ? Mat(K2_inv_t * Matx33d(e_models[i]) * K1_inv) : e_models[i];
                    const auto sc = quality->getScore(model);
                    if (sc.isBetter(out_score)) {
                        out_score = sc;
                        model.copyTo(out_model);
                    }
                }
            }
            if (out_score.isBetter(best_score)) break;
        }
        return true;
    }
};
Ptr<FundamentalDegeneracyViaE> FundamentalDegeneracyViaE::create (const Ptr<Quality> &quality, const Mat &pts,
                                              const Mat &calib_pts, const Matx33d &K1, const Matx33d &K2, bool is_f_objective) {
    return makePtr<FundamentalDegeneracyViaEImpl>(quality, pts, calib_pts, K1, K2, is_f_objective);
}
///////////////////////////////// Fundamental Matrix Degeneracy ///////////////////////////////////
class FundamentalDegeneracyImpl : public FundamentalDegeneracy {
private:
    RNG rng;
    const Ptr<Quality> quality;
    const Ptr<Error> f_error;
    Ptr<Quality> h_repr_quality;
    Mat points_mat;
    const Ptr<ReprojectionErrorForward> h_reproj_error;
    Ptr<EpipolarNonMinimalSolver> f_non_min_solver;
    Ptr<HomographyNonMinimalSolver> h_non_min_solver;
    Ptr<UniformRandomGenerator> random_gen_H;
    const EpipolarGeometryDegeneracyImpl ep_deg;
    // threshold to find inliers for homography model
    const double homography_threshold, log_conf = log(0.05), H_SAMPLE_THR_SQR = 49/*7^2*/, MAX_H_THR = 225/*15^2*/;
    double f_threshold_sqr, best_focal = -1;
    // points (1-7) to verify in sample
    std::vector<std::vector<int>> h_sample {{0,1,2},{3,4,5},{0,1,6},{3,4,6},{2,5,6}};
    std::vector<std::vector<int>> h_sample_ver {{3,4,5,6},{0,1,2,6},{2,3,4,5},{0,1,2,5},{0,1,3,4}};
    std::vector<int> non_planar_supports, h_inliers, h_outliers, h_outliers_eval, f_inliers;
    std::vector<double> weights;
    std::vector<Mat> h_models;
    const int points_size, max_iters_plane_and_parallax, MAX_H_SUBSET = 50, MAX_ITERS_H = 6;
    int num_h_outliers, num_models_used_so_far = 0, estimated_min_non_planar_support,
        num_h_outliers_eval, TENT_MIN_NON_PLANAR_SUPP;
    const int MAX_MODELS_TO_TEST = 21, H_INLS_DEGEN_SAMPLE = 5; // 5 by DEGENSAC, Chum et al.
    Matx33d K, K2, K_inv, K2_inv, K2_inv_t, true_K2_inv, true_K2_inv_t, true_K1_inv, true_K1, true_K2_t;
    Score best_focal_score;
    bool true_K_given, is_principal_pt_set = false;
public:
    FundamentalDegeneracyImpl (int state, const Ptr<Quality> &quality_, const Mat &points_,
                int sample_size_, int plane_and_parallax_iters, double homography_threshold_,
                double f_inlier_thr_sqr, const Mat true_K1_, const Mat true_K2_) :
            rng (state), quality(quality_), f_error(quality_->getErrorFnc()), points_mat(points_),
            h_reproj_error(ReprojectionErrorForward::create(points_)),
            ep_deg (points_, sample_size_), homography_threshold (homography_threshold_),
            points_size (quality_->getPointsSize()),
            max_iters_plane_and_parallax(plane_and_parallax_iters) {
        if (sample_size_ == 8) {
            // add more homography samples to test for 8-points F
            h_sample.emplace_back(std::vector<int>{0, 1, 7}); h_sample_ver.emplace_back(std::vector<int>{2,3,4,5,6});
            h_sample.emplace_back(std::vector<int>{0, 2, 7}); h_sample_ver.emplace_back(std::vector<int>{1,3,4,5,6});
            h_sample.emplace_back(std::vector<int>{3, 5, 7}); h_sample_ver.emplace_back(std::vector<int>{0,1,2,4,6});
            h_sample.emplace_back(std::vector<int>{3, 6, 7}); h_sample_ver.emplace_back(std::vector<int>{0,1,2,4,5});
            h_sample.emplace_back(std::vector<int>{2, 4, 7}); h_sample_ver.emplace_back(std::vector<int>{0,1,3,5,6});
        }
        non_planar_supports = std::vector<int>(MAX_MODELS_TO_TEST);
        h_inliers = std::vector<int>(points_size);
        h_outliers = std::vector<int>(points_size);
        h_outliers_eval = std::vector<int>(points_size);
        f_inliers = std::vector<int>(points_size);
        h_non_min_solver = HomographyNonMinimalSolver::create(points_);
        f_non_min_solver = EpipolarNonMinimalSolver::create(points_, true /*F*/);
        num_h_outliers_eval = num_h_outliers = points_size;
        f_threshold_sqr = f_inlier_thr_sqr;
        h_repr_quality = MsacQuality::create(points_.rows, homography_threshold_, h_reproj_error);
        true_K_given = ! true_K1_.empty() && ! true_K2_.empty();
        if (true_K_given) {
            true_K1 = Matx33d((double *)true_K1_.data);
            true_K2_inv = Matx33d(Mat(true_K2_.inv()));
            true_K2_t = Matx33d(true_K2_).t();
            true_K1_inv = true_K1.inv();
            true_K2_inv_t = true_K2_inv.t();
        }
        random_gen_H = UniformRandomGenerator::create(rng.uniform(0, INT_MAX), points_size, MAX_H_SUBSET);
        estimated_min_non_planar_support = TENT_MIN_NON_PLANAR_SUPP = std::min(5, (int)(0.05*points_size));
    }
    bool estimateHfrom3Points (const Mat &F_best, const std::vector<int> &sample, Mat &H_best) {
        Score H_best_score;
        // find e', null vector of F^T
        const Vec3d e_prime = Utils::getLeftEpipole(F_best);
        const Matx33d A = Math::getSkewSymmetric(e_prime) * Matx33d(F_best);
        bool is_degenerate = false;
        int idx = -1;
        for (const auto &h_i : h_sample) { // only 5 samples
            idx++;
            Matx33d H;
            if (!getH(A, e_prime, 4 * sample[h_i[0]], 4 * sample[h_i[1]], 4 * sample[h_i[2]], H))
                continue;
            h_reproj_error->setModelParameters(Mat(H));
            const auto &ver_pts = h_sample_ver[idx];
            int inliers_in_plane = 3; // 3 points are always inliers
            for (int s : ver_pts)
                if (h_reproj_error->getError(sample[s]) < homography_threshold) {
                    if (++inliers_in_plane >= H_INLS_DEGEN_SAMPLE)
                        break;
                }
            if (inliers_in_plane >= H_INLS_DEGEN_SAMPLE) {
                is_degenerate = true;
                const auto h_score = h_repr_quality->getScore(Mat(H));
                if (h_score.isBetter(H_best_score)) {
                    H_best_score = h_score;
                    H_best = Mat(H);
                }
            }
        }
        if (!is_degenerate)
            return false;
        int h_inls_cnt = optimizeH(H_best, H_best_score);
        for (int iter = 0; iter < 2; iter++) {
            if (h_non_min_solver->estimate(h_inliers, h_inls_cnt, h_models, weights) == 0)
                break;
            const auto h_score = h_repr_quality->getScore(h_models[0]);
            if (h_score.isBetter(H_best_score)) {
                H_best_score = h_score;
                h_models[0].copyTo(H_best);
                h_inls_cnt = h_repr_quality->getInliers(H_best, h_inliers);
            } else break;
        }
        getOutliersH(H_best);
        return true;
    }
    bool optimizeF (const Mat &F, const Score &score, Mat &F_new, Score &new_score) {
        std::vector<Mat> Fs;
        if (f_non_min_solver->estimate(f_inliers, quality->getInliers(F, f_inliers), Fs, weights)) {
            const auto F_polished_score = quality->getScore(f_error->getErrors(Fs[0]));
            if (F_polished_score.isBetter(score)) {
                Fs[0].copyTo(F_new); new_score = F_polished_score;
                return true;
            }
        }
        return false;
    }
    int optimizeH (Mat &H_best, Score &H_best_score) {
        int h_inls_cnt = h_repr_quality->getInliers(H_best, h_inliers);
        random_gen_H->setSubsetSize(h_inls_cnt <= MAX_H_SUBSET ? (int)(0.8*h_inls_cnt) : MAX_H_SUBSET);
        if (random_gen_H->getSubsetSize() >= 4/*min H sample size*/) {
            for (int iter = 0; iter < MAX_ITERS_H; iter++) {
                if (h_non_min_solver->estimate(random_gen_H->generateUniqueRandomSubset(h_inliers, h_inls_cnt), random_gen_H->getSubsetSize(), h_models, weights) == 0)
                    continue;
                const auto h_score = h_repr_quality->getScore(h_models[0]);
                if (h_score.isBetter(H_best_score)) {
                    h_models[0].copyTo(H_best);
                    // if more inliers than previous best
                    if (h_score.inlier_number > H_best_score.inlier_number || h_score.inlier_number >= MAX_H_SUBSET) {
                        h_inls_cnt = h_repr_quality->getInliers(H_best, h_inliers);
                        random_gen_H->setSubsetSize(h_inls_cnt <= MAX_H_SUBSET ? (int)(0.8*h_inls_cnt) : MAX_H_SUBSET);
                    }
                    H_best_score = h_score;
                }
            }
        }
        return h_inls_cnt;
    }

    bool recoverIfDegenerate (const std::vector<int> &sample, const Mat &F_best, const Score &F_best_score,
                              Mat &non_degenerate_model, Score &non_degenerate_model_score) override {
        const auto swapF = [&] (const Mat &_F, const Score &_score) {
            const auto non_min_solver = EpipolarNonMinimalSolver::create(points_mat, true);
            if (! optimizeF(_F, _score, non_degenerate_model, non_degenerate_model_score)) {
                _F.copyTo(non_degenerate_model);
                non_degenerate_model_score = _score;
            }
        };
        Mat F_from_H, F_from_E, H_best; Score F_from_H_score, F_from_E_score;
        if (! estimateHfrom3Points(F_best, sample, H_best)) {
            return false; // non degenerate
        }
        if (true_K_given) {
            if (getFfromTrueK(H_best, F_from_H, F_from_H_score)) {
                if (F_from_H_score.isBetter(F_from_E_score))
                    swapF(F_from_H, F_from_H_score);
                else swapF(F_from_E, F_from_E_score);
                return true;
            } else {
                non_degenerate_model_score = Score();
                return true; // no translation
            }
        }
        const int non_planar_support_degen_F = getNonPlanarSupport(F_best);
        Score F_pl_par_score, F_calib_score; Mat F_pl_par, F_calib;
        if (calibDegensac(H_best, F_calib, F_calib_score, non_planar_support_degen_F, F_best_score)) {
            if (planeAndParallaxRANSAC(H_best, h_outliers, num_h_outliers, max_iters_plane_and_parallax, true,
                    F_best_score, non_planar_support_degen_F, F_pl_par, F_pl_par_score) && F_pl_par_score.isBetter(F_calib_score)
                    && getNonPlanarSupport(F_pl_par) > getNonPlanarSupport(F_calib)) {
                swapF(F_pl_par, F_pl_par_score);
                return true;
            }
            swapF(F_calib, F_calib_score);
            return true;
        } else {
            if (planeAndParallaxRANSAC(H_best, h_outliers, num_h_outliers, max_iters_plane_and_parallax, true,
                    F_best_score, non_planar_support_degen_F, F_pl_par, F_pl_par_score)) {
                swapF(F_pl_par, F_pl_par_score);
                return true;
            }
        }
        if (! isFDegenerate(non_planar_support_degen_F)) {
            return false;
        }
        non_degenerate_model_score = Score();
        return true;
    }

    // RANSAC with plane-and-parallax to find new Fundamental matrix
    bool getFfromTrueK (const Matx33d &H, Mat &F_from_K, Score &F_from_K_score) {
        std::vector<Matx33d> R; std::vector<Vec3d> t;
        if (decomposeHomographyMat(true_K2_inv * H * true_K1, Matx33d::eye(), R, t, noArray()) == 1) {
            // std::cout << "Warning: translation is zero!\n";
            return false; // is degenerate
        }
        // sign of translation does not make difference
        const Mat F1 = Mat(true_K2_inv_t * Math::getSkewSymmetric(t[0]) * R[0] * true_K1_inv);
        const Mat F2 = Mat(true_K2_inv_t * Math::getSkewSymmetric(t[1]) * R[1] * true_K1_inv);
        const auto score_f1 = quality->getScore(f_error->getErrors(F1)), score_f2 = quality->getScore(f_error->getErrors(F2));
        if (score_f1.isBetter(score_f2)) {
            F_from_K = F1; F_from_K_score = score_f1;
        } else {
            F_from_K = F2; F_from_K_score = score_f2;
        }
        return true;
    }
    bool planeAndParallaxRANSAC (const Matx33d &H, std::vector<int> &non_planar_pts, int num_non_planar_pts,
            int max_iters_pl_par, bool use_preemptive, const Score &score_degen_F, int non_planar_support_degen_F,
            Mat &F_new, Score &F_new_score) {
        if (num_non_planar_pts < 2)
            return false;
        num_models_used_so_far = 0; // reset estimation of lambda for plane-and-parallax
        int max_iters = max_iters_pl_par, non_planar_support = 0, iters = 0;
        const float * points = points_mat.ptr<float>();
        std::vector<Matx33d> F_good;
        const double CLOSE_THR = 1.0;
        for (; iters < max_iters; iters++) {
            // draw two random points
            int h_outlier1 = 4 * non_planar_pts[rng.uniform(0, num_non_planar_pts)];
            int h_outlier2 = 4 * non_planar_pts[rng.uniform(0, num_non_planar_pts)];
            while (h_outlier1 == h_outlier2)
                h_outlier2 = 4 * non_planar_pts[rng.uniform(0, num_non_planar_pts)];

            const auto x1 = points[h_outlier1], y1 = points[h_outlier1+1], X1 = points[h_outlier1+2], Y1 = points[h_outlier1+3];
            const auto x2 = points[h_outlier2], y2 = points[h_outlier2+1], X2 = points[h_outlier2+2], Y2 = points[h_outlier2+3];
            if ((fabsf(X1 - X2) < CLOSE_THR && fabsf(Y1 - Y2) < CLOSE_THR) ||
                (fabsf(x1 - x2) < CLOSE_THR && fabsf(y1 - y2) < CLOSE_THR))
                continue;

            // do plane and parallax with outliers of H
            // F = [(p1' x Hp1) x (p2' x Hp2)]_x H = [e']_x H
            Vec3d ep = (Vec3d(X1, Y1, 1).cross(H * Vec3d(x1, y1, 1)))  // l1 = p1' x Hp1
                      .cross((Vec3d(X2, Y2, 1).cross(H * Vec3d(x2, y2, 1))));// l2 = p2' x Hp2
            const Matx33d F = Math::getSkewSymmetric(ep) * H;
            const auto * const f = F.val;
            // check orientation constraint of epipolar lines
            const bool valid = (f[0]*x1+f[1]*y1+f[2])*(ep[1]-ep[2]*Y1) * (f[0]*x2+f[1]*y2+f[2])*(ep[1]-ep[2]*Y2) > 0;
            if (!valid) continue;

            const int num_f_inliers_of_h_outliers = getNonPlanarSupport(Mat(F), num_models_used_so_far >= MAX_MODELS_TO_TEST, non_planar_support);
            if (non_planar_support < num_f_inliers_of_h_outliers) {
                non_planar_support = num_f_inliers_of_h_outliers;
                const double predicted_iters = log_conf / log(1 - pow(static_cast<double>
                    (getNonPlanarSupport(Mat(F), non_planar_pts, num_non_planar_pts)) / num_non_planar_pts, 2));
                if (use_preemptive && ! std::isinf(predicted_iters) && predicted_iters < max_iters)
                    max_iters = static_cast<int>(predicted_iters);
                F_good = { F };
            } else if (non_planar_support == num_f_inliers_of_h_outliers) {
                F_good.emplace_back(F);
            }
        }

        F_new_score = Score();
        for (const auto &F_ : F_good) {
            const auto sc = quality->getScore(f_error->getErrors(Mat(F_)));
            if (sc.isBetter(F_new_score)) {
                F_new_score = sc;
                F_new = Mat(F_);
            }
        }
        if (F_new_score.isBetter(score_degen_F) && non_planar_support > non_planar_support_degen_F)
            return true;
        if (isFDegenerate(non_planar_support))
            return false;
        return true;
    }
    bool calibDegensac (const Matx33d &H, Mat &F_new, Score &F_new_score, int non_planar_support_degen_F, const Score &F_degen_score) {
        const float * points = points_mat.ptr<float>();
        if (! is_principal_pt_set) {
            // estimate principal points from coordinates
            float px1 = 0, py1 = 0, px2 = 0, py2 = 0;
            for (int i = 0; i < points_size; i++) {
                const int idx = 4*i;
                if (px1 < points[idx  ]) px1 = points[idx  ];
                if (py1 < points[idx+1]) py1 = points[idx+1];
                if (px2 < points[idx+2]) px2 = points[idx+2];
                if (py2 < points[idx+3]) py2 = points[idx+3];
            }
            setPrincipalPoint((int)(px1/2)+1, (int)(py1/2)+1, (int)(px2/2)+1, (int)(py2/2)+1);
        }
        std::vector<Mat> R; std::vector<Mat> t; std::vector<Mat> F_good;
        std::vector<double> best_f;
        int non_planar_support_out = 0;
        for (double f = 300; f <= 3000; f += 150.) {
            K(0,0) = K(1,1) = K2(0,0) = K2(1,1) = f;
            const double one_over_f = 1/f;
            K_inv(0,0) = K_inv(1,1) = K2_inv(0,0) = K2_inv(1,1) = K2_inv_t(0,0) = K2_inv_t(1,1) = one_over_f;
            K_inv(0,2) = -K(0,2)*one_over_f; K_inv(1,2) = -K(1,2)*one_over_f;
            K2_inv_t(2,0) = K2_inv(0,2) = -K2(0,2)*one_over_f; K2_inv_t(2,1) = K2_inv(1,2) = -K2(1,2)*one_over_f;
            if (decomposeHomographyMat(K2_inv * H * K, Matx33d::eye(), R, t, noArray()) == 1) continue;
            Mat F1 = Mat(K2_inv_t * Math::getSkewSymmetric(Vec3d(t[0])) * Matx33d(R[0]) * K_inv);
            Mat F2 = Mat(K2_inv_t * Math::getSkewSymmetric(Vec3d(t[2])) * Matx33d(R[2]) * K_inv);
            int non_planar_f1 = getNonPlanarSupport(F1, true, non_planar_support_out),
                non_planar_f2 = getNonPlanarSupport(F2, true, non_planar_support_out);
            if (non_planar_f1 < non_planar_f2) {
                non_planar_f1 = non_planar_f2; F1 = F2;
            }
            if (non_planar_support_out < non_planar_f1) {
                non_planar_support_out = non_planar_f1;
                F_good = {F1};
                best_f = { f };
            } else if (non_planar_support_out == non_planar_f1) {
                F_good.emplace_back(F1);
                best_f.emplace_back(f);
            }
        }
        F_new_score = Score();
        for (int i = 0; i < (int) F_good.size(); i++) {
            const auto sc = quality->getScore(f_error->getErrors(F_good[i]));
            if (sc.isBetter(F_new_score)) {
                F_new_score = sc;
                F_good[i].copyTo(F_new);
                if (sc.isBetter(best_focal_score)) {
                    best_focal = best_f[i]; // save best focal length
                    best_focal_score = sc;
                }
            }
        }
        if (F_degen_score.isBetter(F_new_score) && non_planar_support_out <= non_planar_support_degen_F)
            return false;

        /*
        // logarithmic search -> faster but less accurate
        double f_min = 300, f_max = 3500;
        while (f_max - f_min > 100) {
            const double f_half = (f_max + f_min) * 0.5f, left_half = (f_min + f_half) * 0.5f, right_half = (f_half + f_max) * 0.5f;
            const double inl_in_left = eval_f(left_half), inl_in_right = eval_f(right_half);
            if (inl_in_left > inl_in_right)
                f_max = f_half;
            else f_min = f_half;
        }
        */
        return true;
    }
    void getOutliersH (const Mat &H_best) {
        // get H outliers
        num_h_outliers_eval = num_h_outliers = 0;
        const auto &h_errors = h_reproj_error->getErrors(H_best);
        for (int pt = 0; pt < points_size; pt++)
            if (h_errors[pt] > H_SAMPLE_THR_SQR) {
                h_outliers[num_h_outliers++] = pt;
                if (h_errors[pt] > MAX_H_THR)
                    h_outliers_eval[num_h_outliers_eval++] = pt;
            }
    }

    bool verifyFundamental (const Mat &F_best, const Score &F_score, const std::vector<bool> &inliers_mask, Mat &F_new, Score &new_score) override {
        const int f_sample_size = 3, max_H_iters = 5; // 3.52 = log(0.01) / log(1 - std::pow(0.9, 3));
        int num_f_inliers = 0;
        std::vector<int> inliers(points_size), f_sample(f_sample_size);
        for (int i = 0; i < points_size; i++) if (inliers_mask[i]) inliers[num_f_inliers++] = i;
        const auto sampler = UniformSampler::create(0, f_sample_size, num_f_inliers);
        // find e', null space of F^T
        const Vec3d e_prime = Utils::getLeftEpipole(F_best);
        const Matx33d A = Math::getSkewSymmetric(e_prime) * Matx33d(F_best);
        Score H_best_score; Mat H_best;
        for (int iter = 0; iter < max_H_iters; iter++) {
            sampler->generateSample(f_sample);
            Matx33d H;
            if (!getH(A, e_prime, 4*inliers[f_sample[0]], 4*inliers[f_sample[1]], 4*inliers[f_sample[2]], H))
                continue;
            const auto h_score = h_repr_quality->getScore(Mat(H));
            if (h_score.isBetter(H_best_score)) {
                H_best_score = h_score; H_best = Mat(H);
            }
        }
        if (H_best.empty()) return false; // non-degenerate
        optimizeH(H_best, H_best_score);
        getOutliersH(H_best);
        const int non_planar_support_best_F = getNonPlanarSupport(F_best);
        const bool is_F_degen = isFDegenerate(non_planar_support_best_F);
        Mat F_from_K; Score F_from_K_score;
        bool success = false;
        // generate non-degenerate F even though so-far-the-best one may not be degenerate
        if (true_K_given) {
            // use GT calibration
            if (getFfromTrueK(H_best, F_from_K, F_from_K_score)) {
                new_score = F_from_K_score;
                F_from_K.copyTo(F_new);
                success = true;
            }
        } else {
            // use calibrated DEGENSAC
            if (calibDegensac(H_best, F_from_K, F_from_K_score, non_planar_support_best_F, F_score)) {
                new_score = F_from_K_score;
                F_from_K.copyTo(F_new);
                success = true;
            }
        }
        if (!is_F_degen) {
            return false;
        } else if (success) // F is degenerate
            return true; // but successfully recovered

        // recover degenerate F using plane-and-parallax
        Score plane_parallax_score; Mat F_plane_parallax;
        if (planeAndParallaxRANSAC(H_best, h_outliers, num_h_outliers, 20, true,
                F_score, non_planar_support_best_F, F_plane_parallax, plane_parallax_score)) {
            new_score = plane_parallax_score;
            F_plane_parallax.copyTo(F_new);
            return true;
        }
        // plane-and-parallax failed. A previous non-degenerate so-far-the-best model will be used instead
        new_score = Score();
        return true;
    }
    void setPrincipalPoint (double px_, double py_) override {
        setPrincipalPoint(px_, py_, 0, 0);
    }
    void setPrincipalPoint (double px_, double py_, double px2_, double py2_) override {
        if (px_ > DBL_EPSILON && py_ > DBL_EPSILON) {
            is_principal_pt_set = true;
            K = {1, 0, px_, 0, 1, py_, 0, 0, 1};
            if (px2_ > DBL_EPSILON && py2_ > DBL_EPSILON) K2 = {1, 0, px2_, 0, 1, py2_, 0, 0, 1};
            else K2 = K;
            K_inv = K2_inv = K2_inv_t = Matx33d::eye();
        }
    }
private:
    bool getH (const Matx33d &A, const Vec3d &e_prime, int smpl1, int smpl2, int smpl3, Matx33d &H) {
        const float * points = points_mat.ptr<float>();
        Vec3d p1(points[smpl1  ], points[smpl1+1], 1), p2(points[smpl2  ], points[smpl2+1], 1), p3(points[smpl3  ], points[smpl3+1], 1);
        Vec3d P1(points[smpl1+2], points[smpl1+3], 1), P2(points[smpl2+2], points[smpl2+3], 1), P3(points[smpl3+2], points[smpl3+3], 1);
        const Matx33d M (p1[0], p1[1], 1, p2[0], p2[1], 1, p3[0], p3[1], 1);
        if (p1.cross(p2).dot(p3) * P1.cross(P2).dot(P3) < 0) return false;
        // (x′i × e')
        const Vec3d P1e = P1.cross(e_prime), P2e = P2.cross(e_prime), P3e = P3.cross(e_prime);
        // x′i × (A xi))^T (x′i × e′) / ‖x′i×e′‖^2,
        const Vec3d b (P1.cross(A * p1).dot(P1e) / (P1e[0]*P1e[0]+P1e[1]*P1e[1]+P1e[2]*P1e[2]),
                       P2.cross(A * p2).dot(P2e) / (P2e[0]*P2e[0]+P2e[1]*P2e[1]+P2e[2]*P2e[2]),
                       P3.cross(A * p3).dot(P3e) / (P3e[0]*P3e[0]+P3e[1]*P3e[1]+P3e[2]*P3e[2]));

        H = A - e_prime * (M.inv() * b).t();
        return true;
    }
    int getNonPlanarSupport (const Mat &F, const std::vector<int> &pts, int num_pts) {
        int non_rand_support = 0;
        f_error->setModelParameters(F);
        for (int pt = 0; pt < num_pts; pt++)
            if (f_error->getError(pts[pt]) < f_threshold_sqr)
                non_rand_support++;
        return non_rand_support;
    }

    int getNonPlanarSupport (const Mat &F, bool preemptive=false, int max_so_far=0) {
        int non_rand_support = 0;
        f_error->setModelParameters(F);
        if (preemptive) {
            const auto preemptive_thr = -num_h_outliers_eval + max_so_far;
            for (int pt = 0; pt < num_h_outliers_eval; pt++)
                if (f_error->getError(h_outliers_eval[pt]) < f_threshold_sqr)
                    non_rand_support++;
                else if (non_rand_support - pt < preemptive_thr)
                        break;
        } else {
            for (int pt = 0; pt < num_h_outliers_eval; pt++)
                if (f_error->getError(h_outliers_eval[pt]) < f_threshold_sqr)
                    non_rand_support++;
            if (num_models_used_so_far < MAX_MODELS_TO_TEST && !true_K_given/*for K we know that recovered F cannot be degenerate*/) {
                non_planar_supports[num_models_used_so_far++] = non_rand_support;
                if (num_models_used_so_far == MAX_MODELS_TO_TEST) {
                    getLambda(non_planar_supports, 2.32, num_h_outliers_eval, 0, false, estimated_min_non_planar_support);
                    if (estimated_min_non_planar_support < 3) estimated_min_non_planar_support = 3;
                }
            }
        }
        return non_rand_support;
    }
    inline bool isModelValid(const Mat &F, const std::vector<int> &sample) const override {
        return ep_deg.isModelValid(F, sample);
    }
    bool isFDegenerate (int num_f_inliers_h_outliers) const {
        if (num_models_used_so_far < MAX_MODELS_TO_TEST)
            // the minimum number of non-planar support has not estimated yet -> use tentative
            return num_f_inliers_h_outliers < std::min(TENT_MIN_NON_PLANAR_SUPP, (int)(0.1 * num_h_outliers_eval));
        return num_f_inliers_h_outliers < estimated_min_non_planar_support;
    }
};
Ptr<FundamentalDegeneracy> FundamentalDegeneracy::create (int state, const Ptr<Quality> &quality_,
        const Mat &points_, int sample_size_, int max_iters_plane_and_parallax, double homography_threshold_,
        double f_inlier_thr_sqr, const Mat true_K1, const Mat true_K2) {
    return makePtr<FundamentalDegeneracyImpl>(state, quality_, points_, sample_size_,
              max_iters_plane_and_parallax, homography_threshold_, f_inlier_thr_sqr, true_K1, true_K2);
}


class EssentialDegeneracyImpl : public EssentialDegeneracy {
private:
    const EpipolarGeometryDegeneracyImpl ep_deg;
public:
    explicit EssentialDegeneracyImpl (const Mat &points, int sample_size_) :
        ep_deg (points, sample_size_) {}
    inline bool isModelValid(const Mat &E, const std::vector<int> &sample) const override {
        return ep_deg.isModelValid(E, sample);
    }
};
Ptr<EssentialDegeneracy> EssentialDegeneracy::create (const Mat &points_, int sample_size_) {
    return makePtr<EssentialDegeneracyImpl>(points_, sample_size_);
}
}}
