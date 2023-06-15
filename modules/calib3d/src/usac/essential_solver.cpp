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
/*
* H. Stewenius, C. Engels, and D. Nister. Recent developments on direct relative orientation.
* ISPRS J. of Photogrammetry and Remote Sensing, 60:284,294, 2006
* http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.61.9329&rep=rep1&type=pdf
*
* D. Nister. An efficient solution to the five-point relative pose problem
* IEEE Transactions on Pattern Analysis and Machine Intelligence
* https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.86.8769&rep=rep1&type=pdf
*/
class EssentialMinimalSolver5ptsImpl : public EssentialMinimalSolver5pts {
private:
    // Points must be calibrated K^-1 x
    const Mat points_mat;
    const bool use_svd, is_nister;
public:
    explicit EssentialMinimalSolver5ptsImpl (const Mat &points_, bool use_svd_=false, bool is_nister_=false) :
        points_mat(points_), use_svd(use_svd_), is_nister(is_nister_)
    {
        CV_DbgAssert(!points_mat.empty() && points_mat.isContinuous());
    }

    int estimate (const std::vector<int> &sample, std::vector<Mat> &models) const override {
        const float * pts = points_mat.ptr<float>();
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
        if (use_svd) {
            Matx<double, 5, 9> coeffs (&coefficients[0]);
            Mat D, U, Vt;
            SVDecomp(coeffs, D, U, Vt, SVD::FULL_UV + SVD::MODIFY_A);
            const auto * const vt = (double *) Vt.data;
            for (int i = 0; i < num_e_mat; i++)
                for (int j = 0; j < num_cols; j++)
                    ee[i * num_cols + j] = vt[(8-i)*num_cols+j];
        } else {
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
        }

        const Matx<double, 4, 9> null_space(ee);
        const Matx<double, 4, 1> null_space_mat[3][3] = {
                {null_space.col(0), null_space.col(3), null_space.col(6)},
                {null_space.col(1), null_space.col(4), null_space.col(7)},
                {null_space.col(2), null_space.col(5), null_space.col(8)}};
        Mat_<double> constraint_mat(10, 20);
        Matx<double, 1, 10> eet[3][3];

        // (2) Use the rank constraint and the trace constraint to build ten third-order polynomial
        // equations in the three unknowns. The monomials are ordered in GrLex order and
        // represented in a 10×20 matrix, where each row corresponds to an equation and each column
        // corresponds to a monomial
        if (is_nister) {
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    // compute EE Transpose
                    // Shorthand for multiplying the Essential matrix with its transpose.
                    eet[i][j] = multPolysDegOne(null_space_mat[i][0].val, null_space_mat[j][0].val) +
                                multPolysDegOne(null_space_mat[i][1].val, null_space_mat[j][1].val) +
                                multPolysDegOne(null_space_mat[i][2].val, null_space_mat[j][2].val);

            const Matx<double, 1, 10> trace = 0.5*(eet[0][0] + eet[1][1] + eet[2][2]);
            // Trace constraint
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    Mat(multPolysDegOneAndTwoNister(i == 0 ? (eet[i][0] - trace).val : eet[i][0].val, null_space_mat[0][j].val) +
                        multPolysDegOneAndTwoNister(i == 1 ? (eet[i][1] - trace).val : eet[i][1].val, null_space_mat[1][j].val) +
                        multPolysDegOneAndTwoNister(i == 2 ? (eet[i][2] - trace).val : eet[i][2].val, null_space_mat[2][j].val)).copyTo(constraint_mat.row(1+3 * j + i));

            // Rank = zero determinant constraint
            Mat(multPolysDegOneAndTwoNister(
                    (multPolysDegOne(null_space_mat[0][1].val, null_space_mat[1][2].val) -
                     multPolysDegOne(null_space_mat[0][2].val, null_space_mat[1][1].val)).val,
                        null_space_mat[2][0].val) +
                multPolysDegOneAndTwoNister(
                    (multPolysDegOne(null_space_mat[0][2].val, null_space_mat[1][0].val) -
                     multPolysDegOne(null_space_mat[0][0].val, null_space_mat[1][2].val)).val,
                        null_space_mat[2][1].val) +
                multPolysDegOneAndTwoNister(
                    (multPolysDegOne(null_space_mat[0][0].val, null_space_mat[1][1].val) -
                     multPolysDegOne(null_space_mat[0][1].val, null_space_mat[1][0].val)).val,
                        null_space_mat[2][2].val)).copyTo(constraint_mat.row(0));

            Matx<double, 10, 10> Acoef = constraint_mat.colRange(0, 10),
                                 Bcoef = constraint_mat.colRange(10, 20), A_;

            if (!solve(Acoef, Bcoef, A_, DECOMP_LU)) return 0;

            double b[3 * 13];
            const auto * const a = A_.val;
            for (int i = 0; i < 3; i++) {
                const int r1_idx = i * 2 + 4, r2_idx = i * 2 + 5; // process from 4th row
                for (int j = 0, r1_j = 0, r2_j = 0; j < 13; j++)
                    b[i*13+j] = ((j == 0 || j == 4 || j == 8) ? 0 : a[r1_idx*A_.cols+r1_j++]) - ((j == 3 || j == 7 || j == 12) ? 0 : a[r2_idx*A_.cols+r2_j++]);
            }

            std::vector<double> c(11), rs;
            // filling coefficients of 10-degree polynomial satysfying zero-determinant constraint of essential matrix, ie., det(E) = 0
            // based on "An Efficient Solution to the Five-Point Relative Pose Problem" (David Nister)
            // same as in five-point.cpp
            c[10] = (b[0]*b[17]*b[34]+b[26]*b[4]*b[21]-b[26]*b[17]*b[8]-b[13]*b[4]*b[34]-b[0]*b[21]*b[30]+b[13]*b[30]*b[8]);
            c[9] = (b[26]*b[4]*b[22]+b[14]*b[30]*b[8]+b[13]*b[31]*b[8]+b[1]*b[17]*b[34]-b[13]*b[5]*b[34]+b[26]*b[5]*b[21]-b[0]*b[21]*b[31]-b[26]*b[17]*b[9]-b[1]*b[21]*b[30]+b[27]*b[4]*b[21]+b[0]*b[17]*b[35]-b[0]*b[22]*b[30]+b[13]*b[30]*b[9]+b[0]*b[18]*b[34]-b[27]*b[17]*b[8]-b[14]*b[4]*b[34]-b[13]*b[4]*b[35]-b[26]*b[18]*b[8]);
            c[8] = (b[14]*b[30]*b[9]+b[14]*b[31]*b[8]+b[13]*b[31]*b[9]-b[13]*b[4]*b[36]-b[13]*b[5]*b[35]+b[15]*b[30]*b[8]-b[13]*b[6]*b[34]+b[13]*b[30]*b[10]+b[13]*b[32]*b[8]-b[14]*b[4]*b[35]-b[14]*b[5]*b[34]+b[26]*b[4]*b[23]+b[26]*b[5]*b[22]+b[26]*b[6]*b[21]-b[26]*b[17]*b[10]-b[15]*b[4]*b[34]-b[26]*b[18]*b[9]-b[26]*b[19]*b[8]+b[27]*b[4]*b[22]+b[27]*b[5]*b[21]-b[27]*b[17]*b[9]-b[27]*b[18]*b[8]-b[1]*b[21]*b[31]-b[0]*b[23]*b[30]-b[0]*b[21]*b[32]+b[28]*b[4]*b[21]-b[28]*b[17]*b[8]+b[2]*b[17]*b[34]+b[0]*b[18]*b[35]-b[0]*b[22]*b[31]+b[0]*b[17]*b[36]+b[0]*b[19]*b[34]-b[1]*b[22]*b[30]+b[1]*b[18]*b[34]+b[1]*b[17]*b[35]-b[2]*b[21]*b[30]);
            c[7] = (b[14]*b[30]*b[10]+b[14]*b[32]*b[8]-b[3]*b[21]*b[30]+b[3]*b[17]*b[34]+b[13]*b[32]*b[9]+b[13]*b[33]*b[8]-b[13]*b[4]*b[37]-b[13]*b[5]*b[36]+b[15]*b[30]*b[9]+b[15]*b[31]*b[8]-b[16]*b[4]*b[34]-b[13]*b[6]*b[35]-b[13]*b[7]*b[34]+b[13]*b[30]*b[11]+b[13]*b[31]*b[10]+b[14]*b[31]*b[9]-b[14]*b[4]*b[36]-b[14]*b[5]*b[35]-b[14]*b[6]*b[34]+b[16]*b[30]*b[8]-b[26]*b[20]*b[8]+b[26]*b[4]*b[24]+b[26]*b[5]*b[23]+b[26]*b[6]*b[22]+b[26]*b[7]*b[21]-b[26]*b[17]*b[11]-b[15]*b[4]*b[35]-b[15]*b[5]*b[34]-b[26]*b[18]*b[10]-b[26]*b[19]*b[9]+b[27]*b[4]*b[23]+b[27]*b[5]*b[22]+b[27]*b[6]*b[21]-b[27]*b[17]*b[10]-b[27]*b[18]*b[9]-b[27]*b[19]*b[8]+b[0]*b[17]*b[37]-b[0]*b[23]*b[31]-b[0]*b[24]*b[30]-b[0]*b[21]*b[33]-b[29]*b[17]*b[8]+b[28]*b[4]*b[22]+b[28]*b[5]*b[21]-b[28]*b[17]*b[9]-b[28]*b[18]*b[8]+b[29]*b[4]*b[21]+b[1]*b[19]*b[34]-b[2]*b[21]*b[31]+b[0]*b[20]*b[34]+b[0]*b[19]*b[35]+b[0]*b[18]*b[36]-b[0]*b[22]*b[32]-b[1]*b[23]*b[30]-b[1]*b[21]*b[32]+b[1]*b[18]*b[35]-b[1]*b[22]*b[31]-b[2]*b[22]*b[30]+b[2]*b[17]*b[35]+b[1]*b[17]*b[36]+b[2]*b[18]*b[34]);
            c[6] = (-b[14]*b[6]*b[35]-b[14]*b[7]*b[34]-b[3]*b[22]*b[30]-b[3]*b[21]*b[31]+b[3]*b[17]*b[35]+b[3]*b[18]*b[34]+b[13]*b[32]*b[10]+b[13]*b[33]*b[9]-b[13]*b[4]*b[38]-b[13]*b[5]*b[37]-b[15]*b[6]*b[34]+b[15]*b[30]*b[10]+b[15]*b[32]*b[8]-b[16]*b[4]*b[35]-b[13]*b[6]*b[36]-b[13]*b[7]*b[35]+b[13]*b[31]*b[11]+b[13]*b[30]*b[12]+b[14]*b[32]*b[9]+b[14]*b[33]*b[8]-b[14]*b[4]*b[37]-b[14]*b[5]*b[36]+b[16]*b[30]*b[9]+b[16]*b[31]*b[8]-b[26]*b[20]*b[9]+b[26]*b[4]*b[25]+b[26]*b[5]*b[24]+b[26]*b[6]*b[23]+b[26]*b[7]*b[22]-b[26]*b[17]*b[12]+b[14]*b[30]*b[11]+b[14]*b[31]*b[10]+b[15]*b[31]*b[9]-b[15]*b[4]*b[36]-b[15]*b[5]*b[35]-b[26]*b[18]*b[11]-b[26]*b[19]*b[10]-b[27]*b[20]*b[8]+b[27]*b[4]*b[24]+b[27]*b[5]*b[23]+b[27]*b[6]*b[22]+b[27]*b[7]*b[21]-b[27]*b[17]*b[11]-b[27]*b[18]*b[10]-b[27]*b[19]*b[9]-b[16]*b[5]*b[34]-b[29]*b[17]*b[9]-b[29]*b[18]*b[8]+b[28]*b[4]*b[23]+b[28]*b[5]*b[22]+b[28]*b[6]*b[21]-b[28]*b[17]*b[10]-b[28]*b[18]*b[9]-b[28]*b[19]*b[8]+b[29]*b[4]*b[22]+b[29]*b[5]*b[21]-b[2]*b[23]*b[30]+b[2]*b[18]*b[35]-b[1]*b[22]*b[32]-b[2]*b[21]*b[32]+b[2]*b[19]*b[34]+b[0]*b[19]*b[36]-b[0]*b[22]*b[33]+b[0]*b[20]*b[35]-b[0]*b[23]*b[32]-b[0]*b[25]*b[30]+b[0]*b[17]*b[38]+b[0]*b[18]*b[37]-b[0]*b[24]*b[31]+b[1]*b[17]*b[37]-b[1]*b[23]*b[31]-b[1]*b[24]*b[30]-b[1]*b[21]*b[33]+b[1]*b[20]*b[34]+b[1]*b[19]*b[35]+b[1]*b[18]*b[36]+b[2]*b[17]*b[36]-b[2]*b[22]*b[31]);
            c[5] = (-b[14]*b[6]*b[36]-b[14]*b[7]*b[35]+b[14]*b[31]*b[11]-b[3]*b[23]*b[30]-b[3]*b[21]*b[32]+b[3]*b[18]*b[35]-b[3]*b[22]*b[31]+b[3]*b[17]*b[36]+b[3]*b[19]*b[34]+b[13]*b[32]*b[11]+b[13]*b[33]*b[10]-b[13]*b[5]*b[38]-b[15]*b[6]*b[35]-b[15]*b[7]*b[34]+b[15]*b[30]*b[11]+b[15]*b[31]*b[10]+b[16]*b[31]*b[9]-b[13]*b[6]*b[37]-b[13]*b[7]*b[36]+b[13]*b[31]*b[12]+b[14]*b[32]*b[10]+b[14]*b[33]*b[9]-b[14]*b[4]*b[38]-b[14]*b[5]*b[37]-b[16]*b[6]*b[34]+b[16]*b[30]*b[10]+b[16]*b[32]*b[8]-b[26]*b[20]*b[10]+b[26]*b[5]*b[25]+b[26]*b[6]*b[24]+b[26]*b[7]*b[23]+b[14]*b[30]*b[12]+b[15]*b[32]*b[9]+b[15]*b[33]*b[8]-b[15]*b[4]*b[37]-b[15]*b[5]*b[36]+b[29]*b[5]*b[22]+b[29]*b[6]*b[21]-b[26]*b[18]*b[12]-b[26]*b[19]*b[11]-b[27]*b[20]*b[9]+b[27]*b[4]*b[25]+b[27]*b[5]*b[24]+b[27]*b[6]*b[23]+b[27]*b[7]*b[22]-b[27]*b[17]*b[12]-b[27]*b[18]*b[11]-b[27]*b[19]*b[10]-b[28]*b[20]*b[8]-b[16]*b[4]*b[36]-b[16]*b[5]*b[35]-b[29]*b[17]*b[10]-b[29]*b[18]*b[9]-b[29]*b[19]*b[8]+b[28]*b[4]*b[24]+b[28]*b[5]*b[23]+b[28]*b[6]*b[22]+b[28]*b[7]*b[21]-b[28]*b[17]*b[11]-b[28]*b[18]*b[10]-b[28]*b[19]*b[9]+b[29]*b[4]*b[23]-b[2]*b[22]*b[32]-b[2]*b[21]*b[33]-b[1]*b[24]*b[31]+b[0]*b[18]*b[38]-b[0]*b[24]*b[32]+b[0]*b[19]*b[37]+b[0]*b[20]*b[36]-b[0]*b[25]*b[31]-b[0]*b[23]*b[33]+b[1]*b[19]*b[36]-b[1]*b[22]*b[33]+b[1]*b[20]*b[35]+b[2]*b[19]*b[35]-b[2]*b[24]*b[30]-b[2]*b[23]*b[31]+b[2]*b[20]*b[34]+b[2]*b[17]*b[37]-b[1]*b[25]*b[30]+b[1]*b[18]*b[37]+b[1]*b[17]*b[38]-b[1]*b[23]*b[32]+b[2]*b[18]*b[36]);
            c[4] = (-b[14]*b[6]*b[37]-b[14]*b[7]*b[36]+b[14]*b[31]*b[12]+b[3]*b[17]*b[37]-b[3]*b[23]*b[31]-b[3]*b[24]*b[30]-b[3]*b[21]*b[33]+b[3]*b[20]*b[34]+b[3]*b[19]*b[35]+b[3]*b[18]*b[36]-b[3]*b[22]*b[32]+b[13]*b[32]*b[12]+b[13]*b[33]*b[11]-b[15]*b[6]*b[36]-b[15]*b[7]*b[35]+b[15]*b[31]*b[11]+b[15]*b[30]*b[12]+b[16]*b[32]*b[9]+b[16]*b[33]*b[8]-b[13]*b[6]*b[38]-b[13]*b[7]*b[37]+b[14]*b[32]*b[11]+b[14]*b[33]*b[10]-b[14]*b[5]*b[38]-b[16]*b[6]*b[35]-b[16]*b[7]*b[34]+b[16]*b[30]*b[11]+b[16]*b[31]*b[10]-b[26]*b[19]*b[12]-b[26]*b[20]*b[11]+b[26]*b[6]*b[25]+b[26]*b[7]*b[24]+b[15]*b[32]*b[10]+b[15]*b[33]*b[9]-b[15]*b[4]*b[38]-b[15]*b[5]*b[37]+b[29]*b[5]*b[23]+b[29]*b[6]*b[22]+b[29]*b[7]*b[21]-b[27]*b[20]*b[10]+b[27]*b[5]*b[25]+b[27]*b[6]*b[24]+b[27]*b[7]*b[23]-b[27]*b[18]*b[12]-b[27]*b[19]*b[11]-b[28]*b[20]*b[9]-b[16]*b[4]*b[37]-b[16]*b[5]*b[36]+b[0]*b[19]*b[38]-b[0]*b[24]*b[33]+b[0]*b[20]*b[37]-b[29]*b[17]*b[11]-b[29]*b[18]*b[10]-b[29]*b[19]*b[9]+b[28]*b[4]*b[25]+b[28]*b[5]*b[24]+b[28]*b[6]*b[23]+b[28]*b[7]*b[22]-b[28]*b[17]*b[12]-b[28]*b[18]*b[11]-b[28]*b[19]*b[10]-b[29]*b[20]*b[8]+b[29]*b[4]*b[24]+b[2]*b[18]*b[37]-b[0]*b[25]*b[32]+b[1]*b[18]*b[38]-b[1]*b[24]*b[32]+b[1]*b[19]*b[37]+b[1]*b[20]*b[36]-b[1]*b[25]*b[31]+b[2]*b[17]*b[38]+b[2]*b[19]*b[36]-b[2]*b[24]*b[31]-b[2]*b[22]*b[33]-b[2]*b[23]*b[32]+b[2]*b[20]*b[35]-b[1]*b[23]*b[33]-b[2]*b[25]*b[30]);
            c[3] = (-b[14]*b[6]*b[38]-b[14]*b[7]*b[37]+b[3]*b[19]*b[36]-b[3]*b[22]*b[33]+b[3]*b[20]*b[35]-b[3]*b[23]*b[32]-b[3]*b[25]*b[30]+b[3]*b[17]*b[38]+b[3]*b[18]*b[37]-b[3]*b[24]*b[31]-b[15]*b[6]*b[37]-b[15]*b[7]*b[36]+b[15]*b[31]*b[12]+b[16]*b[32]*b[10]+b[16]*b[33]*b[9]+b[13]*b[33]*b[12]-b[13]*b[7]*b[38]+b[14]*b[32]*b[12]+b[14]*b[33]*b[11]-b[16]*b[6]*b[36]-b[16]*b[7]*b[35]+b[16]*b[31]*b[11]+b[16]*b[30]*b[12]+b[15]*b[32]*b[11]+b[15]*b[33]*b[10]-b[15]*b[5]*b[38]+b[29]*b[5]*b[24]+b[29]*b[6]*b[23]-b[26]*b[20]*b[12]+b[26]*b[7]*b[25]-b[27]*b[19]*b[12]-b[27]*b[20]*b[11]+b[27]*b[6]*b[25]+b[27]*b[7]*b[24]-b[28]*b[20]*b[10]-b[16]*b[4]*b[38]-b[16]*b[5]*b[37]+b[29]*b[7]*b[22]-b[29]*b[17]*b[12]-b[29]*b[18]*b[11]-b[29]*b[19]*b[10]+b[28]*b[5]*b[25]+b[28]*b[6]*b[24]+b[28]*b[7]*b[23]-b[28]*b[18]*b[12]-b[28]*b[19]*b[11]-b[29]*b[20]*b[9]+b[29]*b[4]*b[25]-b[2]*b[24]*b[32]+b[0]*b[20]*b[38]-b[0]*b[25]*b[33]+b[1]*b[19]*b[38]-b[1]*b[24]*b[33]+b[1]*b[20]*b[37]-b[2]*b[25]*b[31]+b[2]*b[20]*b[36]-b[1]*b[25]*b[32]+b[2]*b[19]*b[37]+b[2]*b[18]*b[38]-b[2]*b[23]*b[33]);
            c[2] = (b[3]*b[18]*b[38]-b[3]*b[24]*b[32]+b[3]*b[19]*b[37]+b[3]*b[20]*b[36]-b[3]*b[25]*b[31]-b[3]*b[23]*b[33]-b[15]*b[6]*b[38]-b[15]*b[7]*b[37]+b[16]*b[32]*b[11]+b[16]*b[33]*b[10]-b[16]*b[5]*b[38]-b[16]*b[6]*b[37]-b[16]*b[7]*b[36]+b[16]*b[31]*b[12]+b[14]*b[33]*b[12]-b[14]*b[7]*b[38]+b[15]*b[32]*b[12]+b[15]*b[33]*b[11]+b[29]*b[5]*b[25]+b[29]*b[6]*b[24]-b[27]*b[20]*b[12]+b[27]*b[7]*b[25]-b[28]*b[19]*b[12]-b[28]*b[20]*b[11]+b[29]*b[7]*b[23]-b[29]*b[18]*b[12]-b[29]*b[19]*b[11]+b[28]*b[6]*b[25]+b[28]*b[7]*b[24]-b[29]*b[20]*b[10]+b[2]*b[19]*b[38]-b[1]*b[25]*b[33]+b[2]*b[20]*b[37]-b[2]*b[24]*b[33]-b[2]*b[25]*b[32]+b[1]*b[20]*b[38]);
            c[1] = (b[29]*b[7]*b[24]-b[29]*b[20]*b[11]+b[2]*b[20]*b[38]-b[2]*b[25]*b[33]-b[28]*b[20]*b[12]+b[28]*b[7]*b[25]-b[29]*b[19]*b[12]-b[3]*b[24]*b[33]+b[15]*b[33]*b[12]+b[3]*b[19]*b[38]-b[16]*b[6]*b[38]+b[3]*b[20]*b[37]+b[16]*b[32]*b[12]+b[29]*b[6]*b[25]-b[16]*b[7]*b[37]-b[3]*b[25]*b[32]-b[15]*b[7]*b[38]+b[16]*b[33]*b[11]);
            c[0] = -b[29]*b[20]*b[12]+b[29]*b[7]*b[25]+b[16]*b[33]*b[12]-b[16]*b[7]*b[38]+b[3]*b[20]*b[38]-b[3]*b[25]*b[33];

            const auto poly_solver = SolverPoly::create();
            const int num_roots = poly_solver->getRealRoots(c, rs);

            models = std::vector<Mat>(); models.reserve(num_roots);
            for (int i = 0; i < num_roots; i++) {
                const double z1 = rs[i], z2 = z1*z1, z3 = z2*z1, z4 = z3*z1;
                double bz[9], norm_bz = 0;
                for (int j = 0; j < 3; j++) {
                    double * const br = b + j * 13, * Bz = bz + 3*j;
                    Bz[0] = br[0] * z3 + br[1] * z2 + br[2] * z1 + br[3];
                    Bz[1] = br[4] * z3 + br[5] * z2 + br[6] * z1 + br[7];
                    Bz[2] = br[8] * z4 + br[9] * z3 + br[10] * z2 + br[11] * z1 + br[12];
                    norm_bz += Bz[0]*Bz[0] + Bz[1]*Bz[1] + Bz[2]*Bz[2];
                }

                Matx33d Bz(bz);
                // Bz is rank 2, matrix, so epipole is its null-vector
                Vec3d xy1 = Utils::getRightEpipole(Mat(Bz * (1/sqrt(norm_bz))));

                if (fabs(xy1(2)) < 1e-10) continue;
                Mat_<double> E(3,3);
                double * e_arr = (double *)E.data, x = xy1(0) / xy1(2), y = xy1(1) / xy1(2);
                for (int e_i = 0; e_i < 9; e_i++)
                    e_arr[e_i] = ee[e_i] * x + ee[9+e_i] * y + ee[18+e_i]*z1 + ee[27+e_i];
                models.emplace_back(E);
            }
        } else {
#if defined(HAVE_EIGEN) || defined(HAVE_LAPACK)
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    // compute EE Transpose
                    // Shorthand for multiplying the Essential matrix with its transpose.
                    eet[i][j] = 2 * (multPolysDegOne(null_space_mat[i][0].val, null_space_mat[j][0].val) +
                                     multPolysDegOne(null_space_mat[i][1].val, null_space_mat[j][1].val) +
                                     multPolysDegOne(null_space_mat[i][2].val, null_space_mat[j][2].val));

            const Matx<double, 1, 10> trace = eet[0][0] + eet[1][1] + eet[2][2];
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

            // (4) Compute the 10×10 action matrix for multiplication by one of the unknowns.
            // This is a simple matter of extracting the correct elements from the eliminated
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
#else
            CV_Error(cv::Error::StsNotImplemented, "To run essential matrix estimation of Stewenius method you need to have either Eigen or LAPACK installed! Or switch to Nister algorithm");
            return 0;
#endif
        }
        return static_cast<int>(models.size());
    }

    // number of possible solutions is 0,2,4,6,8,10
    int getMaxNumberOfSolutions () const override { return 10; }
    int getSampleSize() const override { return 5; }
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
    static inline Matx<double, 1, 20> multPolysDegOneAndTwoNister(const double * const p,
                                                            const double * const q) {
        // permutation {0, 3, 1, 2, 4, 10, 6, 12, 5, 11, 7, 13, 16, 8, 14, 17, 9, 15, 18, 19};
        return Matx<double, 1, 20>
           ({p[0]*q[0], p[2]*q[1], p[0]*q[1]+p[1]*q[0], p[1]*q[1]+p[2]*q[0], p[0]*q[2]+p[3]*q[0],
            p[0]*q[3]+p[6]*q[0], p[2]*q[2]+p[4]*q[1], p[2]*q[3]+p[7]*q[1], p[1]*q[2]+p[3]*q[1]+p[4]*q[0],
            p[1]*q[3]+p[6]*q[1]+p[7]*q[0], p[3]*q[2]+p[5]*q[0], p[3]*q[3]+p[6]*q[2]+p[8]*q[0],
            p[6]*q[3]+p[9]*q[0], p[4]*q[2]+p[5]*q[1], p[4]*q[3]+p[7]*q[2]+p[8]*q[1], p[7]*q[3]+p[9]*q[1],
            p[5]*q[2], p[5]*q[3]+p[8]*q[2], p[8]*q[3]+p[9]*q[2], p[9]*q[3]});
    }
};
Ptr<EssentialMinimalSolver5pts> EssentialMinimalSolver5pts::create
        (const Mat &points_, bool use_svd, bool is_nister) {
    return makePtr<EssentialMinimalSolver5ptsImpl>(points_, use_svd, is_nister);
}
}}
