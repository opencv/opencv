// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"
#include "../polynom_solver.h"
#ifdef HAVE_EIGEN
#include <Eigen/Eigen>
#endif

namespace cv { namespace usac {
class FundamentalMinimalSolver7ptsImpl: public FundamentalMinimalSolver7pts {
private:
    Mat points_mat;
    const bool use_ge;
public:
    explicit FundamentalMinimalSolver7ptsImpl (const Mat &points_, bool use_ge_) :
            points_mat (points_), use_ge(use_ge_)
    {
        CV_DbgAssert(!points_mat.empty() && points_mat.isContinuous());
    }

    int estimate (const std::vector<int> &sample, std::vector<Mat> &models) const override {
        const int m = 7, n = 9; // rows, cols
        std::vector<double> a(63); // m*n
        auto * a_ = &a[0];
        const float * points = points_mat.ptr<float>();

        for (int i = 0; i < m; i++ ) {
            const int smpl = 4*sample[i];
            const auto x1 = points[smpl  ], y1 = points[smpl+1],
                       x2 = points[smpl+2], y2 = points[smpl+3];

            (*a_++) = x2*x1;
            (*a_++) = x2*y1;
            (*a_++) = x2;
            (*a_++) = y2*x1;
            (*a_++) = y2*y1;
            (*a_++) = y2;
            (*a_++) = x1;
            (*a_++) = y1;
            (*a_++) = 1;
        }
        double f1[9], f2[9];
        if (use_ge) {
            if (!Math::eliminateUpperTriangular(a, m, n))
                return 0;
            /*
             [a11 a12 a13 a14 a15 a16 a17 a18 a19]
             [  0 a22 a23 a24 a25 a26 a27 a28 a29]
             [  0   0 a33 a34 a35 a36 a37 a38 a39]
             [  0   0   0 a44 a45 a46 a47 a48 a49]
             [  0   0   0   0 a55 a56 a57 a58 a59]
             [  0   0   0   0   0 a66 a67 a68 a69]
             [  0   0   0   0   0   0 a77 a78 a79]
             */

            f1[8] = 1.;
            f1[7] = 0.;
            f1[6] = -a[6*n+8] / a[6*n+6];

            f2[8] = 0.;
            f2[7] = -a[6*n+6] / a[6*n+7];
            f2[6] = 1;

            // start from the last row
            for (int i = m-2; i >= 0; i--) {
                const int row_i = i*n;
                double acc1 = 0, acc2 = 0;
                for (int j = i+1; j < n; j++) {
                    acc1 -= a[row_i + j] * f1[j];
                    acc2 -= a[row_i + j] * f2[j];
                }
                f1[i] = acc1 / a[row_i + i];
                f2[i] = acc2 / a[row_i + i];

                if (std::isnan(f1[i]) || std::isnan(f2[i]))
                    return 0; // due to numerical errors return 0 solutions
            }
        } else {
            Mat U, Vt, D;
            cv::Matx<double, 7, 9> A(&a[0]);
            SVD::compute(A, D, U, Vt, SVD::FULL_UV+SVD::MODIFY_A);
            const auto * const vt = (double *) Vt.data;
            int i1 = 8*9, i2 = 7*9;
            for (int i = 0; i < 9; i++) {
                f1[i] = vt[i1+i];
                f2[i] = vt[i2+i];
            }
        }

        // OpenCV:
        double c[4] = { 0 }, r[3] = { 0 };
        double t0 = 0, t1 = 0, t2 = 0;

        for (int i = 0; i < 9; i++)
            f1[i] -= f2[i];

        t0 = f2[4]*f2[8] - f2[5]*f2[7];
        t1 = f2[3]*f2[8] - f2[5]*f2[6];
        t2 = f2[3]*f2[7] - f2[4]*f2[6];

        c[3] = f2[0]*t0 - f2[1]*t1 + f2[2]*t2;

        c[2] = f1[0]*t0 - f1[1]*t1 + f1[2]*t2 -
               f1[3]*(f2[1]*f2[8] - f2[2]*f2[7]) +
               f1[4]*(f2[0]*f2[8] - f2[2]*f2[6]) -
               f1[5]*(f2[0]*f2[7] - f2[1]*f2[6]) +
               f1[6]*(f2[1]*f2[5] - f2[2]*f2[4]) -
               f1[7]*(f2[0]*f2[5] - f2[2]*f2[3]) +
               f1[8]*(f2[0]*f2[4] - f2[1]*f2[3]);

        t0 = f1[4]*f1[8] - f1[5]*f1[7];
        t1 = f1[3]*f1[8] - f1[5]*f1[6];
        t2 = f1[3]*f1[7] - f1[4]*f1[6];

        c[1] = f2[0]*t0 - f2[1]*t1 + f2[2]*t2 -
               f2[3]*(f1[1]*f1[8] - f1[2]*f1[7]) +
               f2[4]*(f1[0]*f1[8] - f1[2]*f1[6]) -
               f2[5]*(f1[0]*f1[7] - f1[1]*f1[6]) +
               f2[6]*(f1[1]*f1[5] - f1[2]*f1[4]) -
               f2[7]*(f1[0]*f1[5] - f1[2]*f1[3]) +
               f2[8]*(f1[0]*f1[4] - f1[1]*f1[3]);

        c[0] = f1[0]*t0 - f1[1]*t1 + f1[2]*t2;

        // solve the cubic equation; there can be 1 to 3 roots ...
        const int nroots = solve_deg3(c[0], c[1], c[2], c[3], r[0], r[1], r[2]);
        if (nroots < 1) return 0;

        models = std::vector<Mat>(nroots);
        for (int k = 0; k < nroots; k++) {
            models[k] = Mat_<double>(3,3);
            auto * F_ptr = (double *) models[k].data;

            // for each root form the fundamental matrix
            double lambda = r[k], mu = 1;
            double s = f1[8]*lambda + f2[8];

            // normalize each matrix, so that F(3,3) (~F[8]) == 1
            if (fabs(s) > FLT_EPSILON) {
                mu = 1/s;
                lambda *= mu;
                F_ptr[8] = 1;
            } else
                F_ptr[8] = 0;

            for (int i = 0; i < 8; i++)
                F_ptr[i] = f1[i] * lambda + f2[i] * mu;
        }
        return nroots;
    }

    int getMaxNumberOfSolutions () const override { return 3; }
    int getSampleSize() const override { return 7; }
};
Ptr<FundamentalMinimalSolver7pts> FundamentalMinimalSolver7pts::create(const Mat &points, bool use_ge) {
    return makePtr<FundamentalMinimalSolver7ptsImpl>(points, use_ge);
}

class FundamentalMinimalSolver8ptsImpl : public FundamentalMinimalSolver8pts {
private:
    Mat points_mat;
public:
    explicit FundamentalMinimalSolver8ptsImpl (const Mat &points_) :
            points_mat (points_)
    {
        CV_DbgAssert(!points_mat.empty() && points_mat.isContinuous());
    }

    int estimate (const std::vector<int> &sample, std::vector<Mat> &models) const override {
        const int m = 8, n = 9; // rows, cols
        std::vector<double> a(72); // m*n
        auto * a_ = &a[0];
        const float * points = points_mat.ptr<float>();

        for (int i = 0; i < m; i++ ) {
            const int smpl = 4*sample[i];
            const auto x1 = points[smpl  ], y1 = points[smpl+1],
                       x2 = points[smpl+2], y2 = points[smpl+3];

            (*a_++) = x2*x1;
            (*a_++) = x2*y1;
            (*a_++) = x2;
            (*a_++) = y2*x1;
            (*a_++) = y2*y1;
            (*a_++) = y2;
            (*a_++) = x1;
            (*a_++) = y1;
            (*a_++) = 1;
        }

        if (!Math::eliminateUpperTriangular(a, m, n))
            return 0;

        /*
         [a11 a12 a13 a14 a15 a16 a17 a18 a19]
         [  0 a22 a23 a24 a25 a26 a27 a28 a29]
         [  0   0 a33 a34 a35 a36 a37 a38 a39]
         [  0   0   0 a44 a45 a46 a47 a48 a49]
         [  0   0   0   0 a55 a56 a57 a58 a59]
         [  0   0   0   0   0 a66 a67 a68 a69]
         [  0   0   0   0   0   0 a77 a78 a79]
         [  0   0   0   0   0   0   0 a88 a89]

         f9 = 1
         f8 = (-a89*f9) / a88
         f7 = (-a79*f9 - a78*f8) / a77
         f6 = (-a69*f9 - a68*f8 - a69*f9) / a66
         ...
         */

        models = std::vector<Mat>{ Mat_<double>(3,3) };
        auto * f = (double *) models[0].data;
        f[8] = 1.;

        // start from the last row
        for (int i = m-1; i >= 0; i--) {
            double acc = 0;
            for (int j = i+1; j < n; j++)
                acc -= a[i*n+j]*f[j];

            f[i] = acc / a[i*n+i];
            // due to numerical errors return 0 solutions
            if (std::isnan(f[i]))
                return 0;
        }
        return 1;
    }

    int getMaxNumberOfSolutions () const override { return 1; }
    int getSampleSize() const override { return 8; }
};
Ptr<FundamentalMinimalSolver8pts> FundamentalMinimalSolver8pts::create(const Mat &points_) {
    return makePtr<FundamentalMinimalSolver8ptsImpl>(points_);
}

class EpipolarNonMinimalSolverImpl : public EpipolarNonMinimalSolver {
private:
    Mat points_mat;
    const bool do_norm;
    Matx33d _T1, _T2;
    Ptr<NormTransform> normTr = nullptr;
    bool enforce_rank = true, is_fundamental, use_ge;
public:
    explicit EpipolarNonMinimalSolverImpl (const Mat &points_, const Matx33d &T1, const Matx33d &T2, bool use_ge_)
        : points_mat(points_), do_norm(false), _T1(T1), _T2(T2), is_fundamental(true), use_ge(use_ge_) {
        CV_DbgAssert(!points_mat.empty() && points_mat.isContinuous());
    }
    explicit EpipolarNonMinimalSolverImpl (const Mat &points_, bool is_fundamental_) :
        points_mat(points_), do_norm(is_fundamental_), use_ge(false) {
        CV_DbgAssert(!points_mat.empty() && points_mat.isContinuous());
        is_fundamental = is_fundamental_;
        if (is_fundamental)
            normTr = NormTransform::create(points_);
    }
    void enforceRankConstraint (bool enforce) override { enforce_rank = enforce; }
    int estimate (const std::vector<int> &sample, int sample_size, std::vector<Mat>
            &models, const std::vector<double> &weights) const override {
        if (sample_size < getMinimumRequiredSampleSize())
            return 0;

        Matx33d T1, T2;
        Mat norm_points;
        if (do_norm)
            normTr->getNormTransformation(norm_points, sample, sample_size, T1, T2);
        const float * const norm_pts = do_norm ? norm_points.ptr<const float>() : points_mat.ptr<float>();

        if (use_ge) {
            double a[8];
            std::vector<double> AtAb(72, 0); // 8x9
            if (weights.empty()) {
                for (int i = 0; i < sample_size; i++) {
                    const int idx = do_norm ? 4*i : 4*sample[i];
                    const double x1 = norm_pts[idx], y1 = norm_pts[idx+1], x2 = norm_pts[idx+2], y2 = norm_pts[idx+3];
                    a[0] = x2*x1;
                    a[1] = x2*y1;
                    a[2] = x2;
                    a[3] = y2*x1;
                    a[4] = y2*y1;
                    a[5] = y2;
                    a[6] = x1;
                    a[7] = y1;
                    // calculate covariance for eigen
                    for (int row = 0; row < 8; row++) {
                        for (int col = row; col < 8; col++)
                            AtAb[row * 9 + col] += a[row] * a[col];
                        AtAb[row * 9 + 8] += a[row];
                    }
                }
            } else { // use weights
                for (int i = 0; i < sample_size; i++) {
                    const auto weight = weights[i];
                    if (weight < FLT_EPSILON) continue;
                    const int idx = do_norm ? 4*i : 4*sample[i];
                    const double x1 = norm_pts[idx], y1 = norm_pts[idx+1], x2 = norm_pts[idx+2], y2 = norm_pts[idx+3];
                    const double weight_times_x2 = weight * x2, weight_times_y2 = weight * y2;
                    a[0] = weight_times_x2 * x1;
                    a[1] = weight_times_x2 * y1;
                    a[2] = weight_times_x2;
                    a[3] = weight_times_y2 * x1;
                    a[4] = weight_times_y2 * y1;
                    a[5] = weight_times_y2;
                    a[6] = weight * x1;
                    a[7] = weight * y1;
                    // calculate covariance for eigen
                    for (int row = 0; row < 8; row++) {
                        for (int col = row; col < 8; col++)
                            AtAb[row * 9 + col] += a[row] * a[col];
                        AtAb[row * 9 + 8] += a[row];
                    }
                }
            }
            // copy symmetric part of covariance matrix
            for (int j = 1; j < 8; j++)
                for (int z = 0; z < j; z++)
                    AtAb[j*9+z] = AtAb[z*9+j];
            Math::eliminateUpperTriangular(AtAb, 8, 9);
            models = std::vector<Mat>{ Mat_<double>(3,3) };
            auto * f = (double *) models[0].data;
            f[8] = 1.;
            const int m = 8, n = 9;
            // start from the last row
            for (int i = m-1; i >= 0; i--) {
                double acc = 0;
                for (int j = i+1; j < n; j++)
                    acc -= AtAb[i*n+j]*f[j];
                f[i] = acc / AtAb[i*n+i];
                // due to numerical errors return 0 solutions
                if (std::isnan(f[i]))
                    return 0;
            }
        } else {
            // ------- 8 points algorithm with Eigen and covariance matrix --------------
            double a[9] = {0, 0, 0, 0, 0, 0, 0, 0, 1}, AtA[81] = {0}; // 9x9
            if (weights.empty()) {
                for (int i = 0; i < sample_size; i++) {
                    const int idx = do_norm ? 4*i : 4*sample[i];
                    const auto x1 = norm_pts[idx], y1 = norm_pts[idx+1], x2 = norm_pts[idx+2], y2 = norm_pts[idx+3];
                    a[0] = x2*x1;
                    a[1] = x2*y1;
                    a[2] = x2;
                    a[3] = y2*x1;
                    a[4] = y2*y1;
                    a[5] = y2;
                    a[6] = x1;
                    a[7] = y1;
                    // calculate covariance matrix
                    for (int row = 0; row < 9; row++)
                        for (int col = row; col < 9; col++)
                            AtA[row*9+col] += a[row]*a[col];
                }
            } else { // use weights
                for (int i = 0; i < sample_size; i++) {
                    const auto weight = weights[i];
                    if (weight < FLT_EPSILON) continue;
                    const int smpl = do_norm ? 4*i : 4*sample[i];
                    const auto x1 = norm_pts[smpl], y1 = norm_pts[smpl+1], x2 = norm_pts[smpl+2], y2 = norm_pts[smpl+3];
                    const double weight_times_x2 = weight * x2, weight_times_y2 = weight * y2;
                    a[0] = weight_times_x2 * x1;
                    a[1] = weight_times_x2 * y1;
                    a[2] = weight_times_x2;
                    a[3] = weight_times_y2 * x1;
                    a[4] = weight_times_y2 * y1;
                    a[5] = weight_times_y2;
                    a[6] = weight * x1;
                    a[7] = weight * y1;
                    a[8] = weight;
                    for (int row = 0; row < 9; row++)
                        for (int col = row; col < 9; col++)
                            AtA[row*9+col] += a[row]*a[col];
                }
            }
            for (int j = 1; j < 9; j++)
                for (int z = 0; z < j; z++)
                    AtA[j*9+z] = AtA[z*9+j];
#ifdef HAVE_EIGEN
            models = std::vector<Mat>{ Mat_<double>(3,3) };
        // extract the last null-vector
        Eigen::Map<Eigen::Matrix<double, 9, 1>>((double *)models[0].data) = Eigen::JacobiSVD
                <Eigen::Matrix<double, 9, 9>> ((Eigen::Matrix<double, 9, 9>(AtA)),
                        Eigen::ComputeFullV).matrixV().col(8);
#else
            Matx<double, 9, 9> AtA_(AtA), U, Vt;
            Vec<double, 9> W;
            SVD::compute(AtA_, W, U, Vt, SVD::FULL_UV + SVD::MODIFY_A);
            models = std::vector<Mat> { Mat_<double>(3, 3, Vt.val + 72 /*=8*9*/) };
#endif
        }

        if (enforce_rank)
            FundamentalDegeneracy::recoverRank(models[0], is_fundamental);
        if (is_fundamental) {
            const auto * const f = (double *) models[0].data;
            const auto * const t1 = do_norm ? T1.val : _T1.val, * t2 = do_norm ? T2.val : _T2.val;
            // F = T2^T F T1
            models[0] = Mat(Matx33d(t1[0]*t2[0]*f[0],t1[0]*t2[0]*f[1], t2[0]*f[2] + t2[0]*f[0]*t1[2] +
                t2[0]*f[1]*t1[5], t1[0]*t2[0]*f[3],t1[0]*t2[0]*f[4], t2[0]*f[5] + t2[0]*f[3]*t1[2] +
                t2[0]*f[4]*t1[5], t1[0]*(f[6] + f[0]*t2[2] + f[3]*t2[5]), t1[0]*(f[7] + f[1]*t2[2] +
                f[4]*t2[5]), f[8] + t1[2]*(f[6] + f[0]*t2[2] + f[3]*t2[5]) + t1[5]*(f[7] + f[1]*t2[2] +
                f[4]*t2[5]) + f[2]*t2[2] + f[5]*t2[5]));
        }
        return 1;
    }
    int estimate (const std::vector<bool> &/*mask*/, std::vector<Mat> &/*models*/,
            const std::vector<double> &/*weights*/) override {
        return 0;
    }
    int getMinimumRequiredSampleSize() const override { return 8; }
    int getMaxNumberOfSolutions () const override { return 1; }
};
Ptr<EpipolarNonMinimalSolver> EpipolarNonMinimalSolver::create(const Mat &points_, bool is_fundamental) {
    return makePtr<EpipolarNonMinimalSolverImpl>(points_, is_fundamental);
}
Ptr<EpipolarNonMinimalSolver> EpipolarNonMinimalSolver::create(const Mat &points_, const Matx33d &T1, const Matx33d &T2, bool use_ge) {
    return makePtr<EpipolarNonMinimalSolverImpl>(points_, T1, T2, use_ge);
}

class CovarianceEpipolarSolverImpl : public CovarianceEpipolarSolver {
private:
    Mat norm_pts;
    Matx33d T1, T2;
    float * norm_points;
    std::vector<bool> mask;
    int points_size;
    double covariance[81] = {0}, * t1, * t2;
    bool is_fundamental, enforce_rank = true;
public:
    explicit CovarianceEpipolarSolverImpl (const Mat &norm_points_, const Matx33d &T1_, const Matx33d &T2_)
            : norm_pts(norm_points_), T1(T1_), T2(T2_) {
        points_size = norm_points_.rows;
        norm_points = (float *) norm_pts.data;
        t1 = T1.val; t2 = T2.val;
        mask = std::vector<bool>(points_size, false);
        is_fundamental = true;
    }
    explicit CovarianceEpipolarSolverImpl (const Mat &points_, bool is_fundamental_) {
        points_size = points_.rows;
        is_fundamental = is_fundamental_;
        if (is_fundamental) { // normalize image points only for fundamental matrix
            std::vector<int> sample(points_size);
            for (int i = 0; i < points_size; i++) sample[i] = i;
            const Ptr<NormTransform> normTr = NormTransform::create(points_);
            normTr->getNormTransformation(norm_pts, sample, points_size, T1, T2);
            t1 = T1.val; t2 = T2.val;
        } else norm_pts = points_; // otherwise points are normalized by intrinsics
        norm_points = (float *)norm_pts.data;
        mask = std::vector<bool>(points_size, false);
    }
    void enforceRankConstraint (bool enforce_) override { enforce_rank = enforce_; }

    void reset () override {
        std::fill(covariance, covariance+81, 0);
        std::fill(mask.begin(), mask.end(), false);
    }
    /*
     * Find fundamental matrix using 8-point algorithm with covariance matrix and PCA
     */
    int estimate (const std::vector<bool> &new_mask, std::vector<Mat> &models,
                  const std::vector<double> &/*weights*/) override {
        double a[9] = {0, 0, 0, 0, 0, 0, 0, 0, 1};

        for (int i = 0; i < points_size; i++) {
            if (mask[i] != new_mask[i]) {
                const int smpl = 4*i;
                const double x1 = norm_points[smpl  ], y1 = norm_points[smpl+1],
                             x2 = norm_points[smpl+2], y2 = norm_points[smpl+3];

                a[0] = x2*x1;
                a[1] = x2*y1;
                a[2] = x2;
                a[3] = y2*x1;
                a[4] = y2*y1;
                a[5] = y2;
                a[6] = x1;
                a[7] = y1;

                if (mask[i]) // if mask[i] is true then new_mask[i] must be false
                    for (int j = 0; j < 9; j++)
                        for (int z = j; z < 9; z++)
                            covariance[j*9+z] -= a[j]*a[z];
                else
                    for (int j = 0; j < 9; j++)
                        for (int z = j; z < 9; z++)
                            covariance[j*9+z] += a[j]*a[z];
            }
        }
        mask = new_mask;

        // copy symmetric part of covariance matrix
        for (int j = 1; j < 9; j++)
            for (int z = 0; z < j; z++)
                covariance[j*9+z] = covariance[z*9+j];

#ifdef HAVE_EIGEN
        models = std::vector<Mat>{ Mat_<double>(3,3) };
        // extract the last null-vector
        Eigen::Map<Eigen::Matrix<double, 9, 1>>((double *)models[0].data) = Eigen::JacobiSVD
                <Eigen::Matrix<double, 9, 9>> ((Eigen::Matrix<double, 9, 9>(covariance)),
                        Eigen::ComputeFullV).matrixV().col(8);
#else
       Matx<double, 9, 9> AtA_(covariance), U, Vt;
       Vec<double, 9> W;
       SVD::compute(AtA_, W, U, Vt, SVD::FULL_UV + SVD::MODIFY_A);
       models = std::vector<Mat> { Mat_<double>(3, 3, Vt.val + 72 /*=8*9*/) };
#endif
        if (enforce_rank)
            FundamentalDegeneracy::recoverRank(models[0], is_fundamental);
        if (is_fundamental) {
            const auto * const f = (double *) models[0].data;
            // F = T2^T F T1
            models[0] = Mat(Matx33d(t1[0]*t2[0]*f[0],t1[0]*t2[0]*f[1], t2[0]*f[2] + t2[0]*f[0]*t1[2] +
                t2[0]*f[1]*t1[5], t1[0]*t2[0]*f[3],t1[0]*t2[0]*f[4], t2[0]*f[5] + t2[0]*f[3]*t1[2] +
                t2[0]*f[4]*t1[5], t1[0]*(f[6] + f[0]*t2[2] + f[3]*t2[5]), t1[0]*(f[7] + f[1]*t2[2] +
                f[4]*t2[5]), f[8] + t1[2]*(f[6] + f[0]*t2[2] + f[3]*t2[5]) + t1[5]*(f[7] + f[1]*t2[2] +
                f[4]*t2[5]) + f[2]*t2[2] + f[5]*t2[5]));
        }
        return 1;
    }
    int getMinimumRequiredSampleSize() const override { return 8; }
    int getMaxNumberOfSolutions () const override { return 1; }
};
Ptr<CovarianceEpipolarSolver> CovarianceEpipolarSolver::create (const Mat &points, bool is_fundamental) {
    return makePtr<CovarianceEpipolarSolverImpl>(points, is_fundamental);
}
Ptr<CovarianceEpipolarSolver> CovarianceEpipolarSolver::create (const Mat &points, const Matx33d &T1, const Matx33d &T2) {
    return makePtr<CovarianceEpipolarSolverImpl>(points, T1, T2);
}

class LarssonOptimizerImpl : public LarssonOptimizer {
private:
    const Mat &calib_points;
    Matx33d K1, K2, K2_t, K1_inv, K2_inv_t;
    bool is_fundamental;
    BundleOptions opt;
public:
    LarssonOptimizerImpl (const Mat &calib_points_, const Matx33d &K1_, const Matx33d &K2_, int max_iters_, bool is_fundamental_) :
            calib_points(calib_points_), K1(K1_), K2(K2_){
        is_fundamental = is_fundamental_;
        opt.max_iterations = max_iters_;
        opt.loss_scale = Utils::getCalibratedThreshold(std::max(1.5, opt.loss_scale), Mat(K1), Mat(K2));
        if (is_fundamental) {
            K1_inv = K1.inv();
            K2_t = K2.t();
            K2_inv_t = K2_t.inv();
        }
    }

    int estimate (const Mat &model, const std::vector<int> &sample, int sample_size, std::vector<Mat>
            &models, const std::vector<double> &weights) const override {
        if (sample_size < 5) return 0;
        const Matx33d E = is_fundamental ? K2_t * Matx33d(model) * K1 : model;
        RNG rng (sample_size);
        cv::Matx33d R1, R2; cv::Vec3d t;
        cv::decomposeEssentialMat(E, R1, R2, t);
        int positive_depth[4] = {0};
        const auto * const pts_ = (float *) calib_points.data;
        // a few point are enough to test
        // actually due to Sampson error minimization, the input R,t do not really matter
        // for a correct pair there is a slightly faster convergence
        for (int i = 0; i < 3; i++) { // could be 1 point
            const int rand_inl = 4 * sample[rng.uniform(0, sample_size)];
            Vec3d p1 (pts_[rand_inl], pts_[rand_inl+1], 1), p2(pts_[rand_inl+2], pts_[rand_inl+3], 1);
            p1 /= norm(p1); p2 /= norm(p2);
            if (satisfyCheirality(R1,  t, p1, p2)) positive_depth[0]++;
            if (satisfyCheirality(R1, -t, p1, p2)) positive_depth[1]++;
            if (satisfyCheirality(R2,  t, p1, p2)) positive_depth[2]++;
            if (satisfyCheirality(R2, -t, p1, p2)) positive_depth[3]++;
        }
        int corr_idx = 0, max_good_pts = positive_depth[0];
        for (int i = 1; i < 4; i++) {
            if (max_good_pts < positive_depth[i]) {
                max_good_pts = positive_depth[i];
                corr_idx = i;
            }
        }

        CameraPose pose;
        pose.R = corr_idx < 2 ? R1 : R2;
        pose.t = corr_idx % 2 == 1 ? -t : t;
        refine_relpose(calib_points, sample, sample_size, &pose, opt, weights.empty() ? nullptr : &weights[0]);
        Matx33d model_new = Math::getSkewSymmetric(pose.t) * pose.R;
        if (is_fundamental)
            model_new = K2_inv_t * model_new * K1_inv;
        models = std::vector<Mat> { Mat(model_new) };
        return 1;
    }

    int estimate (const std::vector<int>&, int, std::vector<Mat>&, const std::vector<double>&) const override {
        return 0;
    }
    int estimate (const std::vector<bool> &/*mask*/, std::vector<Mat> &/*models*/,
            const std::vector<double> &/*weights*/) override {
        return 0;
    }
    void enforceRankConstraint (bool /*enforce*/) override {}
    int getMinimumRequiredSampleSize() const override { return 5; }
    int getMaxNumberOfSolutions () const override { return 1; }
};
Ptr<LarssonOptimizer> LarssonOptimizer::create(const Mat &calib_points_, const Matx33d &K1, const Matx33d &K2, int max_iters_, bool is_fundamental) {
    return makePtr<LarssonOptimizerImpl>(calib_points_, K1, K2, max_iters_, is_fundamental);
}
}}
