// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"

namespace cv { namespace usac {
class EpipolarGeometryDegeneracyImpl : public EpipolarGeometryDegeneracy {
private:
    const Mat * points_mat;
    const float * const points; // i-th row xi1 yi1 xi2 yi2
    const int min_sample_size;
public:
    explicit EpipolarGeometryDegeneracyImpl (const Mat &points_, int sample_size_) :
        points_mat(&points_), points ((float*) points_.data), min_sample_size (sample_size_) {}
    /*
     * Do oriented constraint to verify if epipolar geometry is in front or behind the camera.
     * Return: true if all points are in front of the camers w.r.t. tested epipolar geometry - satisfies constraint.
     *         false - otherwise.
     * x'^T F x = 0
     * e' × x' ~+ Fx   <=>  λe' × x' = Fx, λ > 0
     * e  × x ~+ x'^T F
     */
    inline bool isModelValid(const Mat &F_, const std::vector<int> &sample) const override {
        // F is of rank 2, taking cross product of two rows we obtain null vector of F
        Vec3d ec_mat = F_.row(0).cross(F_.row(2));
        auto * ec = ec_mat.val; // of size 3x1

        // e is zero vector, recompute e
        if (ec[0] <= 1.9984e-15 && ec[0] >= -1.9984e-15 &&
            ec[1] <= 1.9984e-15 && ec[1] >= -1.9984e-15 &&
            ec[2] <= 1.9984e-15 && ec[2] >= -1.9984e-15) {
            ec_mat = F_.row(1).cross(F_.row(2));
            ec = ec_mat.val;
        }
        const auto * const F = (double *) F_.data;

        // without loss of generality, let the first point in sample be in front of the camera.
        int pt = 4*sample[0];
        // s1 = F11 * x2 + F21 * y2 + F31 * 1
        // s2 = e'_2 * 1 - e'_3 * y1
        // sign1 = s1 * s2
        const double sign1 = (F[0]*points[pt+2]+F[3]*points[pt+3]+F[6])*(ec[1]-ec[2]*points[pt+1]);

        for (int i = 1; i < min_sample_size; i++) {
            pt = 4 * sample[i];
            // if signum of the first point and tested point differs
            // then two points are on different sides of the camera.
            if (sign1*(F[0]*points[pt+2]+F[3]*points[pt+3]+F[6])*(ec[1]-ec[2]*points[pt+1])<0)
                return false;
        }
        return true;
    }

    Ptr<Degeneracy> clone(int /*state*/) const override {
        return makePtr<EpipolarGeometryDegeneracyImpl>(*points_mat, min_sample_size);
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
    const Mat * points_mat;
    const float * const points;
public:
    explicit HomographyDegeneracyImpl (const Mat &points_) :
            points_mat(&points_), points ((float *)points_.data) {}

    inline bool isSampleGood (const std::vector<int> &sample) const override {
        const int smpl1 = 4*sample[0], smpl2 = 4*sample[1], smpl3 = 4*sample[2], smpl4 = 4*sample[3];
        // planar correspondences must lie on the same side of any line from two points in sample
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
        // (1/2) det |x2 y2 1| = (1/2) det |x2-x1   y2-y1   0| = (1/2) det |x2-x1   y2-y1| < threshold
        //           |x3 y3 1|             |x3-x1   y3-y1   0|             |x3-x1   y3-y1|
        // for points on the first image
        if (fabsf((x2-x1) * (y3-y1) - (y2-y1) * (x3-x1)) * 0.5 < FLT_EPSILON) return false; //1,2,3
        if (fabsf((x2-x1) * (y4-y1) - (y2-y1) * (x4-x1)) * 0.5 < FLT_EPSILON) return false; //1,2,4
        if (fabsf((x3-x1) * (y4-y1) - (y3-y1) * (x4-x1)) * 0.5 < FLT_EPSILON) return false; //1,3,4
        if (fabsf((x3-x2) * (y4-y2) - (y3-y2) * (x4-x2)) * 0.5 < FLT_EPSILON) return false; //2,3,4
        // for points on the second image
        if (fabsf((X2-X1) * (Y3-Y1) - (Y2-Y1) * (X3-X1)) * 0.5 < FLT_EPSILON) return false; //1,2,3
        if (fabsf((X2-X1) * (Y4-Y1) - (Y2-Y1) * (X4-X1)) * 0.5 < FLT_EPSILON) return false; //1,2,4
        if (fabsf((X3-X1) * (Y4-Y1) - (Y3-Y1) * (X4-X1)) * 0.5 < FLT_EPSILON) return false; //1,3,4
        if (fabsf((X3-X2) * (Y4-Y2) - (Y3-Y2) * (X4-X2)) * 0.5 < FLT_EPSILON) return false; //2,3,4

        return true;
    }
    Ptr<Degeneracy> clone(int /*state*/) const override {
        return makePtr<HomographyDegeneracyImpl>(*points_mat);
    }
};
Ptr<HomographyDegeneracy> HomographyDegeneracy::create (const Mat &points_) {
    return makePtr<HomographyDegeneracyImpl>(points_);
}

///////////////////////////////// Fundamental Matrix Degeneracy ///////////////////////////////////
class FundamentalDegeneracyImpl : public FundamentalDegeneracy {
private:
    RNG rng;
    const Ptr<Quality> quality;
    const float * const points;
    const Mat * points_mat;
    const Ptr<ReprojectionErrorForward> h_reproj_error;
    Ptr<HomographyNonMinimalSolver> h_non_min_solver;
    const EpipolarGeometryDegeneracyImpl ep_deg;
    // threshold to find inliers for homography model
    const double homography_threshold, log_conf = log(0.05);
    // points (1-7) to verify in sample
    std::vector<std::vector<int>> h_sample {{0,1,2},{3,4,5},{0,1,6},{3,4,6},{2,5,6}};
    std::vector<int> h_inliers;
    std::vector<double> weights;
    std::vector<Mat> h_models;
    const int points_size, sample_size;
public:

    FundamentalDegeneracyImpl (int state, const Ptr<Quality> &quality_, const Mat &points_,
            int sample_size_, double homography_threshold_) :
            rng (state), quality(quality_), points((float *) points_.data), points_mat(&points_),
            h_reproj_error(ReprojectionErrorForward::create(points_)),
            ep_deg (points_, sample_size_), homography_threshold (homography_threshold_),
            points_size (quality_->getPointsSize()), sample_size (sample_size_) {
        if (sample_size_ == 8) {
            // add more homography samples to test for 8-points F
            h_sample.emplace_back(std::vector<int>{0, 1, 7});
            h_sample.emplace_back(std::vector<int>{0, 2, 7});
            h_sample.emplace_back(std::vector<int>{3, 5, 7});
            h_sample.emplace_back(std::vector<int>{3, 6, 7});
            h_sample.emplace_back(std::vector<int>{2, 4, 7});
        }
        h_inliers = std::vector<int>(points_size);
        h_non_min_solver = HomographyNonMinimalSolver::create(points_);
    }
    inline bool isModelValid(const Mat &F, const std::vector<int> &sample) const override {
        return ep_deg.isModelValid(F, sample);
    }
    bool recoverIfDegenerate (const std::vector<int> &sample, const Mat &F_best,
                 Mat &non_degenerate_model, Score &non_degenerate_model_score) override {
        non_degenerate_model_score = Score(); // set worst case

        // According to Two-view Geometry Estimation Unaffected by a Dominant Plane
        // (http://cmp.felk.cvut.cz/~matas/papers/chum-degen-cvpr05.pdf)
        // only 5 homographies enough to test
        // triplets {1,2,3}, {4,5,6}, {1,2,7}, {4,5,7} and {3,6,7}

        // H = A - e' (M^-1 b)^T
        // A = [e']_x F
        // b_i = (x′i × (A xi))^T (x′i × e′)‖x′i×e′‖^−2,
        // M is a 3×3 matrix with rows x^T_i
        // epipole e' is left nullspace of F s.t. e′^T F=0,

        // find e', null space of F^T
        Vec3d e_prime = F_best.col(0).cross(F_best.col(2));
        if (fabs(e_prime(0)) < 1e-10 && fabs(e_prime(1)) < 1e-10 &&
            fabs(e_prime(2)) < 1e-10) // if e' is zero
            e_prime = F_best.col(1).cross(F_best.col(2));

        const Matx33d A = Math::getSkewSymmetric(e_prime) * Matx33d(F_best);

        Vec3d xi_prime(0,0,1), xi(0,0,1), b;
        Matx33d M(0,0,1,0,0,1,0,0,1); // last column of M is 1

        bool is_model_degenerate = false;
        for (const auto &h_i : h_sample) { // only 5 samples
            for (int pt_i = 0; pt_i < 3; pt_i++) {
                // find b and M
                const int smpl = 4*sample[h_i[pt_i]];
                xi[0] = points[smpl];
                xi[1] = points[smpl+1];
                xi_prime[0] = points[smpl+2];
                xi_prime[1] = points[smpl+3];

                // (x′i × e')
                const Vec3d xprime_X_eprime = xi_prime.cross(e_prime);

                // (x′i × (A xi))
                const Vec3d xprime_X_Ax = xi_prime.cross(A * xi);

                // x′i × (A xi))^T (x′i × e′) / ‖x′i×e′‖^2,
                b[pt_i] = xprime_X_Ax.dot(xprime_X_eprime) /
                           std::pow(norm(xprime_X_eprime), 2);

                // M from x^T
                M(pt_i, 0) = xi[0];
                M(pt_i, 1) = xi[1];
            }

            // compute H
            Matx33d H = A - e_prime * (M.inv() * b).t();

            int inliers_out_plane = 0;
            h_reproj_error->setModelParameters(Mat(H));

            // find inliers from sample, points related to H, x' ~ Hx
            for (int s = 0; s < sample_size; s++)
                if (h_reproj_error->getError(sample[s]) > homography_threshold)
                    if (++inliers_out_plane > 2)
                        break;

            // if there are at least 5 points lying on plane then F is degenerate
            if (inliers_out_plane <= 2) {
                is_model_degenerate = true;

                // update homography by polishing on all inliers
                int h_inls_cnt = 0;
                const auto &h_errors = h_reproj_error->getErrors(Mat(H));
                for (int pt = 0; pt < points_size; pt++)
                    if (h_errors[pt] < homography_threshold)
                        h_inliers[h_inls_cnt++] = pt;
                if (h_non_min_solver->estimate(h_inliers, h_inls_cnt, h_models, weights) != 0)
                    H = Matx33d(h_models[0]);

                Mat newF;
                const Score newF_score = planeAndParallaxRANSAC(H, newF, h_errors);
                if (newF_score.isBetter(non_degenerate_model_score)) {
                    // store non degenerate model
                    non_degenerate_model_score = newF_score;
                    newF.copyTo(non_degenerate_model);
                }
            }
        }
        return is_model_degenerate;
    }
    Ptr<Degeneracy> clone(int state) const override {
        return makePtr<FundamentalDegeneracyImpl>(state, quality->clone(), *points_mat,
            sample_size, homography_threshold);
    }
private:
    // RANSAC with plane-and-parallax to find new Fundamental matrix
    Score planeAndParallaxRANSAC (const Matx33d &H, Mat &best_F, const std::vector<float> &h_errors) {
        int max_iters = 100; // with 95% confidence assume at least 17% of inliers
        Score best_score;
        for (int iters = 0; iters < max_iters; iters++) {
            // draw two random points
            int h_outlier1 = rng.uniform(0, points_size);
            int h_outlier2 = rng.uniform(0, points_size);
            while (h_outlier1 == h_outlier2)
                h_outlier2 = rng.uniform(0, points_size);

            // find outliers of homography H
            if (h_errors[h_outlier1] > homography_threshold &&
                h_errors[h_outlier2] > homography_threshold) {

                // do plane and parallax with outliers of H
                // F = [(p1' x Hp1) x (p2' x Hp2)]_x H
                const Matx33d F = Math::getSkewSymmetric(
                       (Vec3d(points[4*h_outlier1+2], points[4*h_outlier1+3], 1).cross   // p1'
                   (H * Vec3d(points[4*h_outlier1  ], points[4*h_outlier1+1], 1))).cross // Hp1
                       (Vec3d(points[4*h_outlier2+2], points[4*h_outlier2+3], 1).cross   // p2'
                   (H * Vec3d(points[4*h_outlier2  ], points[4*h_outlier2+1], 1)))       // Hp2
                 ) * H;

                const Score score = quality->getScore(Mat(F));
                if (score.isBetter(best_score)) {
                    best_score = score;
                    best_F = Mat(F);
                    const double predicted_iters = log_conf / log(1 - std::pow
                            (static_cast<double>(score.inlier_number) / points_size, 2));

                    if (! std::isinf(predicted_iters) && predicted_iters < max_iters)
                        max_iters = static_cast<int>(predicted_iters);
                }
            }
        }
        return best_score;
    }
};
Ptr<FundamentalDegeneracy> FundamentalDegeneracy::create (int state, const Ptr<Quality> &quality_,
        const Mat &points_, int sample_size_, double homography_threshold_) {
    return makePtr<FundamentalDegeneracyImpl>(state, quality_, points_, sample_size_,
            homography_threshold_);
}

class EssentialDegeneracyImpl : public EssentialDegeneracy {
private:
    const Mat * points_mat;
    const int sample_size;
    const EpipolarGeometryDegeneracyImpl ep_deg;
public:
    explicit EssentialDegeneracyImpl (const Mat &points, int sample_size_) :
            points_mat(&points), sample_size(sample_size_), ep_deg (points, sample_size_) {}
    inline bool isModelValid(const Mat &E, const std::vector<int> &sample) const override {
        return ep_deg.isModelValid(E, sample);
    }
    Ptr<Degeneracy> clone(int /*state*/) const override {
        return makePtr<EssentialDegeneracyImpl>(*points_mat, sample_size);
    }
};
Ptr<EssentialDegeneracy> EssentialDegeneracy::create (const Mat &points_, int sample_size_) {
    return makePtr<EssentialDegeneracyImpl>(points_, sample_size_);
}
}}
