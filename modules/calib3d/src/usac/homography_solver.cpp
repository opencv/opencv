// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"
#ifdef HAVE_EIGEN
#include <Eigen/Eigen>
#endif

namespace cv { namespace usac {
class HomographyMinimalSolver4ptsGEMImpl : public HomographyMinimalSolver4ptsGEM {
private:
    const Mat * points_mat;
    const float * const points;
public:
    explicit HomographyMinimalSolver4ptsGEMImpl (const Mat &points_) :
        points_mat(&points_), points ((float*) points_.data) {}

    int estimate (const std::vector<int>& sample, std::vector<Mat> &models) const override {
        int m = 8, n = 9;
        std::vector<double> A(72, 0);
        int cnt = 0;
        for (int i = 0; i < 4; i++) {
            const int smpl = 4*sample[i];
            const double x1 = points[smpl], y1 = points[smpl+1], x2 = points[smpl+2], y2 = points[smpl+3];

            A[cnt++] = -x1;
            A[cnt++] = -y1;
            A[cnt++] = -1;
            cnt += 3; // skip zeros
            A[cnt++] = x2*x1;
            A[cnt++] = x2*y1;
            A[cnt++] = x2;

            cnt += 3;
            A[cnt++] = -x1;
            A[cnt++] = -y1;
            A[cnt++] = -1;
            A[cnt++] = y2*x1;
            A[cnt++] = y2*y1;
            A[cnt++] = y2;
        }

        if (!Math::eliminateUpperTriangular(A, m, n))
            return 0;

        models = std::vector<Mat>{ Mat_<double>(3,3) };
        auto * h = (double *) models[0].data;
        h[8] = 1.;

        // start from the last row
        for (int i = m-1; i >= 0; i--) {
            double acc = 0;
            for (int j = i+1; j < n; j++)
                acc -= A[i*n+j]*h[j];

            h[i] = acc / A[i*n+i];
            // due to numerical errors return 0 solutions
            if (std::isnan(h[i]))
                return 0;
        }
        return 1;
    }

    int getMaxNumberOfSolutions () const override { return 1; }
    int getSampleSize() const override { return 4; }
    Ptr<MinimalSolver> clone () const override {
        return makePtr<HomographyMinimalSolver4ptsGEMImpl>(*points_mat);
    }
};
Ptr<HomographyMinimalSolver4ptsGEM> HomographyMinimalSolver4ptsGEM::create(const Mat &points_) {
    return makePtr<HomographyMinimalSolver4ptsGEMImpl>(points_);
}

class HomographyNonMinimalSolverImpl : public HomographyNonMinimalSolver {
private:
    const Mat * points_mat;
    const Ptr<NormTransform> normTr;
public:
    explicit HomographyNonMinimalSolverImpl (const Mat &points_) :
        points_mat(&points_), normTr (NormTransform::create(points_)) {}

    /*
     * Find Homography matrix using (weighted) non-minimal estimation.
     * Use Principal Component Analysis. Use normalized points.
     */
    int estimate (const std::vector<int> &sample, int sample_size, std::vector<Mat> &models,
            const std::vector<double> &weights) const override {
        if (sample_size < getMinimumRequiredSampleSize())
            return 0;

        Matx33d T1, T2;
        Mat norm_points_;
        normTr->getNormTransformation(norm_points_, sample, sample_size, T1, T2);

        /*
         * @norm_points is matrix 4 x inlier_size
         * @weights is vector of inliers_size
         * weights[i] is weight of i-th inlier
         */
        const auto * const norm_points = (float *) norm_points_.data;

        double a1[9] = {0, 0, -1, 0, 0, 0, 0, 0, 0},
               a2[9] = {0, 0, 0, 0, 0, -1, 0, 0, 0},
               AtA[81] = {0};

        if (weights.empty()) {
            for (int i = 0; i < sample_size; i++) {
                const int smpl = 4*i;
                const double x1 = norm_points[smpl  ], y1 = norm_points[smpl+1],
                             x2 = norm_points[smpl+2], y2 = norm_points[smpl+3];

                a1[0] = -x1;
                a1[1] = -y1;
                a1[6] = x2*x1;
                a1[7] = x2*y1;
                a1[8] = x2;

                a2[3] = -x1;
                a2[4] = -y1;
                a2[6] = y2*x1;
                a2[7] = y2*y1;
                a2[8] = y2;

                for (int j = 0; j < 9; j++)
                    for (int z = j; z < 9; z++)
                        AtA[j*9+z] += a1[j]*a1[z] + a2[j]*a2[z];
            }
        } else {
            for (int i = 0; i < sample_size; i++) {
                const int smpl = 4*i;
                const double weight = weights[i];
                const double x1 = norm_points[smpl  ], y1 = norm_points[smpl+1],
                             x2 = norm_points[smpl+2], y2 = norm_points[smpl+3];
                const double minus_weight_times_x1 = -weight * x1,
                             minus_weight_times_y1 = -weight * y1,
                                   weight_times_x2 =  weight * x2,
                                   weight_times_y2 =  weight * y2;

                a1[0] = minus_weight_times_x1;
                a1[1] = minus_weight_times_y1;
                a1[2] = -weight;
                a1[6] = weight_times_x2 * x1;
                a1[7] = weight_times_x2 * y1;
                a1[8] = weight_times_x2;

                a2[3] = minus_weight_times_x1;
                a2[4] = minus_weight_times_y1;
                a2[5] = -weight;
                a2[6] = weight_times_y2 * x1;
                a2[7] = weight_times_y2 * y1;
                a2[8] = weight_times_y2;

                for (int j = 0; j < 9; j++)
                    for (int z = j; z < 9; z++)
                        AtA[j*9+z] += a1[j]*a1[z] + a2[j]*a2[z];
            }
        }

        // copy symmetric part of covariance matrix
        for (int j = 1; j < 9; j++)
            for (int z = 0; z < j; z++)
                AtA[j*9+z] = AtA[z*9+j];

#ifdef HAVE_EIGEN
        Mat H = Mat_<double>(3,3);
        Eigen::HouseholderQR<Eigen::Matrix<double, 9, 9>> qr((Eigen::Matrix<double, 9, 9> (AtA)));
        const Eigen::Matrix<double, 9, 9> &Q = qr.householderQ();
        // extract the last nullspace
        Eigen::Map<Eigen::Matrix<double, 9, 1>>((double *)H.data) = Q.col(8);
#else
        Matx<double, 9, 9> Vt;
        Vec<double, 9> D;
        if (! eigen(Matx<double, 9, 9>(AtA), D, Vt)) return 0;
        Mat H = Mat_<double>(3, 3, Vt.val + 72/*=8*9*/);
#endif

        models = std::vector<Mat>{ T2.inv() * H * T1 };
        return 1;
    }

    int getMinimumRequiredSampleSize() const override { return 4; }
    int getMaxNumberOfSolutions () const override { return 1; }
    Ptr<NonMinimalSolver> clone () const override {
        return makePtr<HomographyNonMinimalSolverImpl>(*points_mat);
    }
};
Ptr<HomographyNonMinimalSolver> HomographyNonMinimalSolver::create(const Mat &points_) {
    return makePtr<HomographyNonMinimalSolverImpl>(points_);
}

class AffineMinimalSolverImpl : public AffineMinimalSolver {
private:
    const Mat * points_mat;
    const float * const points;
public:
    explicit AffineMinimalSolverImpl (const Mat &points_) :
            points_mat(&points_), points((float *) points_.data) {}
    /*
        Affine transformation
        x1 y1 1 0  0  0   a   u1
        0  0  0 x1 y1 1   b   v1
        x2 y2 1 0  0  0   c   u2
        0  0  0 x2 y2 1 * d = v2
        x3 y3 1 0  0  0   e   u3
        0  0  0 x3 y3 1   f   v3
    */
    int estimate (const std::vector<int> &sample, std::vector<Mat> &models) const override {
        const int smpl1 = 4*sample[0], smpl2 = 4*sample[1], smpl3 = 4*sample[2];
        const auto
                x1 = points[smpl1], y1 = points[smpl1+1], u1 = points[smpl1+2], v1 = points[smpl1+3],
                x2 = points[smpl2], y2 = points[smpl2+1], u2 = points[smpl2+2], v2 = points[smpl2+3],
                x3 = points[smpl3], y3 = points[smpl3+1], u3 = points[smpl3+2], v3 = points[smpl3+3];

        // covers degeneracy test when all 3 points are collinear.
        // In this case denominator will be 0
        double denominator = x1*y2 - x2*y1 - x1*y3 + x3*y1 + x2*y3 - x3*y2;
        if (fabs(denominator) < FLT_EPSILON) // check if denominator is zero
            return 0;
        denominator = 1. / denominator;

        double a =  (u1*y2 - u2*y1 - u1*y3 + u3*y1 + u2*y3 - u3*y2) * denominator;
        double b = -(u1*x2 - u2*x1 - u1*x3 + u3*x1 + u2*x3 - u3*x2) * denominator;
        double c = u1 - a * x1 - b * y1; // ax1 + by1 + c = u1
        double d =  (v1*y2 - v2*y1 - v1*y3 + v3*y1 + v2*y3 - v3*y2) * denominator;
        double e = -(v1*x2 - v2*x1 - v1*x3 + v3*x1 + v2*x3 - v3*x2) * denominator;
        double f = v1 - d * x1 - e * y1; // dx1 + ey1 + f = v1

        models[0] = Mat(Matx33d(a, b, c, d, e, f, 0, 0, 1));
        return 1;
    }
    int getSampleSize() const override { return 3; }
    int getMaxNumberOfSolutions () const override { return 1; }
    Ptr<MinimalSolver> clone () const override {
        return makePtr<AffineMinimalSolverImpl>(*points_mat);
    }
};
Ptr<AffineMinimalSolver> AffineMinimalSolver::create(const Mat &points_) {
    return makePtr<AffineMinimalSolverImpl>(points_);
}

class AffineNonMinimalSolverImpl : public AffineNonMinimalSolver {
private:
    const Mat * points_mat;
    const float * const points;
    // const NormTransform<double> norm_transform;
public:
    explicit AffineNonMinimalSolverImpl (const Mat &points_) :
            points_mat(&points_), points((float*) points_.data)
    /*, norm_transform(points_)*/ {}

    int estimate (const std::vector<int> &sample, int sample_size, std::vector<Mat> &models,
                  const std::vector<double> &weights) const override {
        // surprisingly normalization of points does not improve the output model
        // Mat norm_points_, T1, T2;
        // norm_transform.getNormTransformation(norm_points_, sample, sample_size, T1, T2);
        // const auto * const n_pts = (double *) norm_points_.data;

        if (sample_size < getMinimumRequiredSampleSize())
            return 0;
        // do Least Squares
        // Ax = b   ->  A^T Ax = A^T b
        // x = (A^T A)^-1 A^T b
        double AtA[36] = {0}, Ab[6] = {0};
        double r1[6] = {0, 0, 1, 0, 0, 0}; // row 1 of A
        double r2[6] = {0, 0, 0, 0, 0, 1}; // row 2 of A

        if (weights.empty())
            for (int p = 0; p < sample_size; p++) {
                // if (weights != nullptr) weight = weights[sample[p]];

                const int smpl = 4*sample[p];
                const double x1=points[smpl], y1=points[smpl+1], x2=points[smpl+2], y2=points[smpl+3];
                // const double x1=n_pts[smpl], y1=n_pts[smpl+1], x2=n_pts[smpl+2], y2=n_pts[smpl+3];

                r1[0] = x1;
                r1[1] = y1;

                r2[3] = x1;
                r2[4] = y1;

                for (int j = 0; j < 6; j++) {
                    for (int z = j; z < 6; z++)
                        AtA[j * 6 + z] += r1[j] * r1[z] + r2[j] * r2[z];
                    Ab[j] += r1[j]*x2 + r2[j]*y2;
                }
            }
        else
            for (int p = 0; p < sample_size; p++) {
                const int smpl = 4*sample[p];
                const double weight = weights[p];
                const double weight_times_x1 = weight * points[smpl  ],
                             weight_times_y1 = weight * points[smpl+1],
                             weight_times_x2 = weight * points[smpl+2],
                             weight_times_y2 = weight * points[smpl+3];

                r1[0] = weight_times_x1;
                r1[1] = weight_times_y1;
                r1[2] = weight;

                r2[3] = weight_times_x1;
                r2[4] = weight_times_y1;
                r2[5] = weight;

                for (int j = 0; j < 6; j++) {
                    for (int z = j; z < 6; z++)
                        AtA[j * 6 + z] += r1[j] * r1[z] + r2[j] * r2[z];
                    Ab[j] += r1[j]*weight_times_x2 + r2[j]*weight_times_y2;
                }
            }

        // copy symmetric part
        for (int j = 1; j < 6; j++)
            for (int z = 0; z < j; z++)
                AtA[j*6+z] = AtA[z*6+j];

        Vec6d aff;
        if (!solve(Matx66d(AtA), Vec6d(Ab), aff))
            return 0;
        models[0] = Mat(Matx33d(aff(0), aff(1), aff(2),
                                aff(3), aff(4), aff(5),
                                0, 0, 1));

        // models[0] = T2.inv() * models[0] * T1;
        return 1;
    }

    int getMinimumRequiredSampleSize() const override { return 3; }
    int getMaxNumberOfSolutions () const override { return 1; }
    Ptr<NonMinimalSolver> clone () const override {
        return makePtr<AffineNonMinimalSolverImpl>(*points_mat);
    }
};
Ptr<AffineNonMinimalSolver> AffineNonMinimalSolver::create(const Mat &points_) {
    return makePtr<AffineNonMinimalSolverImpl>(points_);
}
}}