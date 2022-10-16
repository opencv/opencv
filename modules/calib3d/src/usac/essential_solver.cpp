// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"
#if defined(HAVE_EIGEN)
#include <Eigen/Eigen>
#elif defined(HAVE_LAPACK)
#include "opencv_lapack.h"
#endif

namespace cv { namespace usac {
// Essential matrix solver:
/*
* H. Stewenius, C. Engels, and D. Nister. Recent developments on direct relative orientation.
* ISPRS J. of Photogrammetry and Remote Sensing, 60:284,294, 2006
* http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.61.9329&rep=rep1&type=pdf
*/
class EssentialMinimalSolverStewenius5ptsImpl : public EssentialMinimalSolverStewenius5pts {
private:
    // Points must be calibrated K^-1 x
    const Mat * points_mat;
#if defined(HAVE_EIGEN) || defined(HAVE_LAPACK)
    const float * const pts;
#endif
public:
    explicit EssentialMinimalSolverStewenius5ptsImpl (const Mat &points_) :
        points_mat(&points_)
#if defined(HAVE_EIGEN) || defined(HAVE_LAPACK)
        , pts((float*)points_.data)
#endif
        {}

#if defined(HAVE_LAPACK) || defined(HAVE_EIGEN)
    int estimate (const std::vector<int> &sample, std::vector<Mat> &models) const override {
        // (1) Extract 4 null vectors from linear equations of epipolar constraint
        std::vector<double> coefficients(45); // 5 pts=rows, 9 columns
        auto *coefficients_ = &coefficients[0];
        for (int i = 0; i < 5; i++) {
            const int smpl = 4 * sample[i];
            const auto x1 = pts[smpl], y1 = pts[smpl+1], x2 = pts[smpl+2], y2 = pts[smpl+3];
            (*coefficients_++) = x2 * x1;
            (*coefficients_++) = x2 * y1;
            (*coefficients_++) = x2;
            (*coefficients_++) = y2 * x1;
            (*coefficients_++) = y2 * y1;
            (*coefficients_++) = y2;
            (*coefficients_++) = x1;
            (*coefficients_++) = y1;
            (*coefficients_++) = 1;
        }

        const int num_cols = 9, num_e_mat = 4;
        double ee[36]; // 9*4
        // eliminate linear equations
        if (!Math::eliminateUpperTriangular(coefficients, 5, num_cols))
            return 0;
        for (int i = 0; i < num_e_mat; i++)
            for (int j = 5; j < num_cols; j++)
                ee[num_cols * i + j] = (i + 5 == j) ? 1 : 0;
        // use back-substitution
        for (int e = 0; e < num_e_mat; e++) {
            const int curr_e = num_cols * e;
            // start from the last row
            for (int i = 4; i >= 0; i--) {
                const int row_i = i * num_cols;
                double acc = 0;
                for (int j = i + 1; j < num_cols; j++)
                    acc -= coefficients[row_i + j] * ee[curr_e + j];
                ee[curr_e + i] = acc / coefficients[row_i + i];
                // due to numerical errors return 0 solutions
                if (std::isnan(ee[curr_e + i]))
                    return 0;
            }
        }

        const Matx<double, 4, 9> null_space(ee);
        const Matx<double, 4, 1> null_space_mat[3][3] = {
                {null_space.col(0), null_space.col(3), null_space.col(6)},
                {null_space.col(1), null_space.col(4), null_space.col(7)},
                {null_space.col(2), null_space.col(5), null_space.col(8)}};

        // (2) Use the rank constraint and the trace constraint to build ten third-order polynomial
        // equations in the three unknowns. The monomials are ordered in GrLex order and
        // represented in a 10×20 matrix, where each row corresponds to an equation and each column
        // corresponds to a monomial
        Matx<double, 1, 10> eet[3][3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                // compute EE Transpose
                // Shorthand for multiplying the Essential matrix with its transpose.
                eet[i][j] = 2 * (multPolysDegOne(null_space_mat[i][0].val, null_space_mat[j][0].val) +
                                 multPolysDegOne(null_space_mat[i][1].val, null_space_mat[j][1].val) +
                                 multPolysDegOne(null_space_mat[i][2].val, null_space_mat[j][2].val));

        const Matx<double, 1, 10> trace = eet[0][0] + eet[1][1] + eet[2][2];
        Mat_<double> constraint_mat(10, 20);
        // Trace constraint
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Mat(multPolysDegOneAndTwo(eet[i][0].val, null_space_mat[0][j].val) +
                    multPolysDegOneAndTwo(eet[i][1].val, null_space_mat[1][j].val) +
                    multPolysDegOneAndTwo(eet[i][2].val, null_space_mat[2][j].val) -
                    0.5 * multPolysDegOneAndTwo(trace.val, null_space_mat[i][j].val))
                        .copyTo(constraint_mat.row(3 * i + j));

        // Rank = zero determinant constraint
        Mat(multPolysDegOneAndTwo(
                (multPolysDegOne(null_space_mat[0][1].val, null_space_mat[1][2].val) -
                 multPolysDegOne(null_space_mat[0][2].val, null_space_mat[1][1].val)).val,
                 null_space_mat[2][0].val) +
                multPolysDegOneAndTwo(
                    (multPolysDegOne(null_space_mat[0][2].val, null_space_mat[1][0].val) -
                     multPolysDegOne(null_space_mat[0][0].val, null_space_mat[1][2].val)).val,
                     null_space_mat[2][1].val) +
                multPolysDegOneAndTwo(
                    (multPolysDegOne(null_space_mat[0][0].val, null_space_mat[1][1].val) -
                     multPolysDegOne(null_space_mat[0][1].val, null_space_mat[1][0].val)).val,
                     null_space_mat[2][2].val)).copyTo(constraint_mat.row(9));

#ifdef HAVE_EIGEN
        const Eigen::Matrix<double, 10, 20, Eigen::RowMajor> constraint_mat_eig((double *) constraint_mat.data);
        // (3) Compute the Gröbner basis. This turns out to be as simple as performing a
        // Gauss-Jordan elimination on the 10×20 matrix
        const Eigen::Matrix<double, 10, 10> eliminated_mat_eig = constraint_mat_eig.block<10, 10>(0, 0)
                .fullPivLu().solve(constraint_mat_eig.block<10, 10>(0, 10));

        // (4) Compute the 10×10 action matrix for multiplication by one of the un-knowns.
        // This is a simple matter of extracting the correct elements fromthe eliminated
        // 10×20 matrix and organising them to form the action matrix.
        Eigen::Matrix<double, 10, 10> action_mat_eig = Eigen::Matrix<double, 10, 10>::Zero();
        action_mat_eig.block<3, 10>(0, 0) = eliminated_mat_eig.block<3, 10>(0, 0);
        action_mat_eig.block<2, 10>(3, 0) = eliminated_mat_eig.block<2, 10>(4, 0);
        action_mat_eig.row(5) = eliminated_mat_eig.row(7);
        action_mat_eig(6, 0) = -1.0;
        action_mat_eig(7, 1) = -1.0;
        action_mat_eig(8, 3) = -1.0;
        action_mat_eig(9, 6) = -1.0;

        // (5) Compute the left eigenvectors of the action matrix
        Eigen::EigenSolver<Eigen::Matrix<double, 10, 10>> eigensolver(action_mat_eig);
        const Eigen::VectorXcd &eigenvalues = eigensolver.eigenvalues();
        const auto * const eig_vecs_ = (double *) eigensolver.eigenvectors().real().data();
#else
        Matx<double, 10, 10> A = constraint_mat.colRange(0, 10),
                         B = constraint_mat.colRange(10, 20), eliminated_mat;
        if (!solve(A, B, eliminated_mat, DECOMP_LU)) return 0;

        Mat eliminated_mat_dyn = Mat(eliminated_mat);
        Mat action_mat = Mat_<double>::zeros(10, 10);
        eliminated_mat_dyn.rowRange(0,3).copyTo(action_mat.rowRange(0,3));
        eliminated_mat_dyn.rowRange(4,6).copyTo(action_mat.rowRange(3,5));
        eliminated_mat_dyn.row(7).copyTo(action_mat.row(5));
        auto * action_mat_data = (double *) action_mat.data;
        action_mat_data[60] = -1.0; // 6 row, 0 col
        action_mat_data[71] = -1.0; // 7 row, 1 col
        action_mat_data[83] = -1.0; // 8 row, 3 col
        action_mat_data[96] = -1.0; // 9 row, 6 col

        int mat_order = 10, info, lda = 10, ldvl = 10, ldvr = 1, lwork = 100;
        double wr[10], wi[10] = {0}, eig_vecs[100], work[100]; // 10 = mat_order, 100 = lwork
        char jobvl = 'V', jobvr = 'N'; // only left eigen vectors are computed
        OCV_LAPACK_FUNC(dgeev)(&jobvl, &jobvr, &mat_order, action_mat_data, &lda, wr, wi, eig_vecs, &ldvl,
                nullptr, &ldvr, work, &lwork, &info);
        if (info != 0) return 0;
#endif

        models = std::vector<Mat>(); models.reserve(10);

        // Read off the values for the three unknowns at all the solution points and
        // back-substitute to obtain the solutions for the essential matrix.
        for (int i = 0; i < 10; i++)
            // process only real solutions
#ifdef HAVE_EIGEN
            if (eigenvalues(i).imag() == 0) {
                Mat_<double> model(3, 3);
                auto * model_data = (double *) model.data;
                const int eig_i = 20 * i + 12; // eigen stores imaginary values too
                for (int j = 0; j < 9; j++)
                    model_data[j] = ee[j   ] * eig_vecs_[eig_i  ] + ee[j+9 ] * eig_vecs_[eig_i+2] +
                                    ee[j+18] * eig_vecs_[eig_i+4] + ee[j+27] * eig_vecs_[eig_i+6];
#else
            if (wi[i] == 0) {
                Mat_<double> model (3,3);
                auto * model_data = (double *) model.data;
                const int eig_i = 10 * i + 6;
                for (int j = 0; j < 9; j++)
                    model_data[j] = ee[j   ]*eig_vecs[eig_i  ] + ee[j+9 ]*eig_vecs[eig_i+1] +
                                    ee[j+18]*eig_vecs[eig_i+2] + ee[j+27]*eig_vecs[eig_i+3];
#endif
                models.emplace_back(model);
            }
        return static_cast<int>(models.size());
#else
    int estimate (const std::vector<int> &/*sample*/, std::vector<Mat> &/*models*/) const override {
        CV_Error(cv::Error::StsNotImplemented, "To use essential matrix solver LAPACK or Eigen has to be installed!");
#endif
    }

    // number of possible solutions is 0,2,4,6,8,10
    int getMaxNumberOfSolutions () const override { return 10; }
    int getSampleSize() const override { return 5; }
    Ptr<MinimalSolver> clone () const override {
        return makePtr<EssentialMinimalSolverStewenius5ptsImpl>(*points_mat);
    }
private:
    /*
     * Multiply two polynomials of degree one with unknowns x y z
     * @p = (p1 x + p2 y + p3 z + p4) [p1 p2 p3 p4]
     * @q = (q1 x + q2 y + q3 z + q4) [q1 q2 q3 a4]
     * @result is a new polynomial in x^2 xy y^2 xz yz z^2 x y z 1 of size 10
     */
    static inline Matx<double,1,10> multPolysDegOne(const double * const p,
                                                    const double * const q) {
        return
            {p[0]*q[0], p[0]*q[1]+p[1]*q[0], p[1]*q[1], p[0]*q[2]+p[2]*q[0], p[1]*q[2]+p[2]*q[1],
             p[2]*q[2], p[0]*q[3]+p[3]*q[0], p[1]*q[3]+p[3]*q[1], p[2]*q[3]+p[3]*q[2], p[3]*q[3]};
    }

    /*
     * Multiply two polynomials with unknowns x y z
     * @p is of size 10 and @q is of size 4
     * @p = (p1 x^2 + p2 xy + p3 y^2 + p4 xz + p5 yz + p6 z^2 + p7 x + p8 y + p9 z + p10)
     * @q = (q1 x + q2 y + q3 z + a4) [q1 q2 q3 q4]
     * @result is a new polynomial of size 20
     * x^3 x^2y xy^2 y^3 x^2z xyz y^2z xz^2 yz^2 z^3 x^2 xy y^2 xz yz z^2 x y z 1
     */
    static inline Matx<double, 1, 20> multPolysDegOneAndTwo(const double * const p,
                                                            const double * const q) {
        return Matx<double, 1, 20>
           ({p[0]*q[0], p[0]*q[1]+p[1]*q[0], p[1]*q[1]+p[2]*q[0], p[2]*q[1], p[0]*q[2]+p[3]*q[0],
                  p[1]*q[2]+p[3]*q[1]+p[4]*q[0], p[2]*q[2]+p[4]*q[1], p[3]*q[2]+p[5]*q[0],
                  p[4]*q[2]+p[5]*q[1], p[5]*q[2], p[0]*q[3]+p[6]*q[0], p[1]*q[3]+p[6]*q[1]+p[7]*q[0],
                  p[2]*q[3]+p[7]*q[1], p[3]*q[3]+p[6]*q[2]+p[8]*q[0], p[4]*q[3]+p[7]*q[2]+p[8]*q[1],
                  p[5]*q[3]+p[8]*q[2], p[6]*q[3]+p[9]*q[0], p[7]*q[3]+p[9]*q[1], p[8]*q[3]+p[9]*q[2],
                  p[9]*q[3]});
    }
};
Ptr<EssentialMinimalSolverStewenius5pts> EssentialMinimalSolverStewenius5pts::create
        (const Mat &points_) {
    return makePtr<EssentialMinimalSolverStewenius5ptsImpl>(points_);
}

class EssentialNonMinimalSolverImpl : public EssentialNonMinimalSolver {
private:
    const Mat * points_mat;
    const float * const points;
public:
    /*
     * Input calibrated points K^-1 x.
     * Linear 8 points algorithm is used for estimation.
     */
    explicit EssentialNonMinimalSolverImpl (const Mat &points_) :
        points_mat(&points_), points ((float *) points_.data) {}

    int estimate (const std::vector<int> &sample, int sample_size, std::vector<Mat>
            &models, const std::vector<double> &weights) const override {
        if (sample_size < getMinimumRequiredSampleSize())
            return 0;

        // ------- 8 points algorithm with Eigen and covariance matrix --------------
        double a[9] = {0, 0, 0, 0, 0, 0, 0, 0, 1};
        double AtA[81] = {0}; // 9x9

        if (weights.empty()) {
            for (int i = 0; i < sample_size; i++) {
                const int pidx = 4*sample[i];
                const double x1 = points[pidx  ], y1 = points[pidx+1],
                             x2 = points[pidx+2], y2 = points[pidx+3];
                a[0] = x2*x1;
                a[1] = x2*y1;
                a[2] = x2;
                a[3] = y2*x1;
                a[4] = y2*y1;
                a[5] = y2;
                a[6] = x1;
                a[7] = y1;

                // calculate covariance for eigen
                for (int row = 0; row < 9; row++)
                    for (int col = row; col < 9; col++)
                        AtA[row*9+col] += a[row]*a[col];
            }
        } else {
            for (int i = 0; i < sample_size; i++) {
                const int smpl = 4*sample[i];
                const double weight = weights[i];
                const double x1 = points[smpl  ], y1 = points[smpl+1],
                             x2 = points[smpl+2], y2 = points[smpl+3];
                const double weight_times_x2 = weight * x2,
                             weight_times_y2 = weight * y2;

                a[0] = weight_times_x2 * x1;
                a[1] = weight_times_x2 * y1;
                a[2] = weight_times_x2;
                a[3] = weight_times_y2 * x1;
                a[4] = weight_times_y2 * y1;
                a[5] = weight_times_y2;
                a[6] = weight * x1;
                a[7] = weight * y1;
                a[8] = weight;

                // calculate covariance for eigen
                for (int row = 0; row < 9; row++)
                    for (int col = row; col < 9; col++)
                        AtA[row*9+col] += a[row]*a[col];
            }
        }

        // copy symmetric part of covariance matrix
        for (int j = 1; j < 9; j++)
            for (int z = 0; z < j; z++)
                AtA[j*9+z] = AtA[z*9+j];

#ifdef HAVE_EIGEN
        models = std::vector<Mat>{ Mat_<double>(3,3) };
        const Eigen::JacobiSVD<Eigen::Matrix<double, 9, 9>> svd((Eigen::Matrix<double, 9, 9>(AtA)),
                Eigen::ComputeFullV);
        // extract the last nullspace
        Eigen::Map<Eigen::Matrix<double, 9, 1>>((double *)models[0].data) = svd.matrixV().col(8);
#else
        Matx<double, 9, 9> AtA_(AtA), U, Vt;
        Vec<double, 9> W;
        SVD::compute(AtA_, W, U, Vt, SVD::FULL_UV + SVD::MODIFY_A);
        models = std::vector<Mat> { Mat_<double>(3, 3, Vt.val + 72 /*=8*9*/) };
#endif
        FundamentalDegeneracy::recoverRank(models[0], false /*E*/);
        return 1;
    }
    int getMinimumRequiredSampleSize() const override { return 8; }
    int getMaxNumberOfSolutions () const override { return 1; }
    Ptr<NonMinimalSolver> clone () const override {
        return makePtr<EssentialNonMinimalSolverImpl>(*points_mat);
    }
};
Ptr<EssentialNonMinimalSolver> EssentialNonMinimalSolver::create (const Mat &points_) {
    return makePtr<EssentialNonMinimalSolverImpl>(points_);
}
}}