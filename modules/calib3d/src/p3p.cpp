#include <cstring>
#include <cmath>
#include <iostream>

#include "polynom_solver.h"
#include "p3p.h"


using namespace cv;

void p3p::calibrateAndNormalizePointsPnP(const Mat &opoints_, const Mat &ipoints_) {
    Mat ipoints = ipoints_.clone(), opoints = opoints_.clone();

    auto convertPoints = [] (Mat &points, int pt_dim) {
        points.convertTo(points, CV_64F); // convert points to have float precision
        if (points.channels() > 1)
            points = points.reshape(1, (int)points.total()); // convert point to have 1 channel
        if (points.rows < points.cols)
            transpose(points, points); // transpose so points will be in rows
        CV_CheckGE(points.cols, pt_dim, "Invalid dimension of point");
        if (points.cols != pt_dim) // in case when image points are 3D convert them to 2D
            points = points.colRange(0, pt_dim);
    };

    convertPoints(ipoints, 2);
    convertPoints(opoints, 3);

    points_mat(0, 0) = opoints.at<double>(0, 0);
    points_mat(0, 1) = opoints.at<double>(0, 1);
    points_mat(0, 2) = opoints.at<double>(0, 2);

    points_mat(1, 0) = opoints.at<double>(1, 0);
    points_mat(1, 1) = opoints.at<double>(1, 1);
    points_mat(1, 2) = opoints.at<double>(1, 2);

    points_mat(2, 0) = opoints.at<double>(2, 0);
    points_mat(2, 1) = opoints.at<double>(2, 1);
    points_mat(2, 2) = opoints.at<double>(2, 2);

    for (int i = 0; i < ipoints.rows; i++) {
        const double k_inv_u = ipoints.at<double>(i, 0);
        const double k_inv_v = ipoints.at<double>(i, 1);
        const double norm = 1.f / sqrtf(k_inv_u*k_inv_u + k_inv_v*k_inv_v + 1);
        calib_norm_points_mat(i, 0) = k_inv_u * norm;
        calib_norm_points_mat(i, 1) = k_inv_v * norm;
        calib_norm_points_mat(i, 2) =           norm;
    }
}

p3p::p3p() :
    points_mat(), calib_norm_points_mat()
{
}

int p3p::estimate(std::vector<Mat>& Rs, std::vector<Mat>& ts, const cv::Mat &opoints, const cv::Mat &ipoints) {
   /*
    * The description of this solution can be found here:
    * http://cmp.felk.cvut.cz/~pajdla/gvg/GVG-2016-Lecture.pdf
    * pages: 51-59
    */
    CV_INSTRUMENT_REGION();
    calibrateAndNormalizePointsPnP(opoints, ipoints);

    const Vec3d X1 (points_mat(0, 0), points_mat(0, 1), points_mat(0, 2));
    const Vec3d X2 (points_mat(1, 0), points_mat(1, 1), points_mat(1, 2));
    const Vec3d X3 (points_mat(2, 0), points_mat(2, 1), points_mat(2, 2));

    // find distance between world points d_ij = ||Xi - Xj||
    const double d12 = norm(X1 - X2);
    const double d23 = norm(X2 - X3);
    const double d31 = norm(X3 - X1);

    const double VAL_THR = 1e-4;
    if (d12 < VAL_THR || d23 < VAL_THR || d31 < VAL_THR)
        return 0;

    const Vec3d cx1 (calib_norm_points_mat(0, 0), calib_norm_points_mat(0, 1), calib_norm_points_mat(0, 2));
    const Vec3d cx2 (calib_norm_points_mat(1, 0), calib_norm_points_mat(1, 1), calib_norm_points_mat(1, 2));
    const Vec3d cx3 (calib_norm_points_mat(2, 0), calib_norm_points_mat(2, 1), calib_norm_points_mat(2, 2));

    // find cosine angles, cos(x1,x2) = K^-1 x1.dot(K^-1 x2) / (||K^-1 x1|| * ||K^-1 x2||)
    // calib_norm_points are already K^-1 x / ||K^-1 x||, so we perform only dot product
    const double c12 = cx1(0)*cx2(0) + cx1(1)*cx2(1) + cx1(2)*cx2(2);
    const double c23 = cx2(0)*cx3(0) + cx2(1)*cx3(1) + cx2(2)*cx3(2);
    const double c31 = cx3(0)*cx1(0) + cx3(1)*cx1(1) + cx3(2)*cx1(2);

    Matx33d Z, Zw;
    auto * z = Z.val, * zw = Zw.val;

    // find coefficients of polynomial a4 x^4 + ... + a0 = 0
    const double c12_p2 = c12*c12, c23_p2 = c23*c23, c31_p2 = c31*c31;
    const double d12_p2 = d12*d12, d12_p4 = d12_p2*d12_p2;
    const double d23_p2 = d23*d23, d23_p4 = d23_p2*d23_p2, d23_p6 = d23_p2*d23_p4, d23_p8 = d23_p4*d23_p4;
    const double d31_p2 = d31*d31, d31_p4 = d31_p2*d31_p2;
    const double a4 = -4*d23_p4*d12_p2*d31_p2*c23_p2+d23_p8-2*d23_p6*d12_p2-2*d23_p6*d31_p2+d23_p4*d12_p4+2*d23_p4*d12_p2*d31_p2+d23_p4*d31_p4;
    const double a3 = 8*d23_p4*d12_p2*d31_p2*c12*c23_p2+4*d23_p6*d12_p2*c31*c23-4*d23_p4*d12_p4*c31*c23+4*d23_p4*d12_p2*d31_p2*c31*c23-4*d23_p8*c12+4*d23_p6*d12_p2*c12+8*d23_p6*d31_p2*c12-4*d23_p4*d12_p2*d31_p2*c12-4*d23_p4*d31_p4*c12;
    const double a2 = -8*d23_p6*d12_p2*c31*c12*c23-8*d23_p4*d12_p2*d31_p2*c31*c12*c23+4*d23_p8*c12_p2-4*d23_p6*d12_p2*c31_p2-8*d23_p6*d31_p2*c12_p2+4*d23_p4*d12_p4*c31_p2+4*d23_p4*d12_p4*c23_p2-4*d23_p4*d12_p2*d31_p2*c23_p2+4*d23_p4*d31_p4*c12_p2+2*d23_p8-4*d23_p6*d31_p2-2*d23_p4*d12_p4+2*d23_p4*d31_p4;
    const double a1 = 8*d23_p6*d12_p2*c31_p2*c12+4*d23_p6*d12_p2*c31*c23-4*d23_p4*d12_p4*c31*c23+4*d23_p4*d12_p2*d31_p2*c31*c23-4*d23_p8*c12-4*d23_p6*d12_p2*c12+8*d23_p6*d31_p2*c12+4*d23_p4*d12_p2*d31_p2*c12-4*d23_p4*d31_p4*c12;
    const double a0 = -4*d23_p6*d12_p2*c31_p2+d23_p8-2*d23_p4*d12_p2*d31_p2+2*d23_p6*d12_p2+d23_p4*d31_p4+d23_p4*d12_p4-2*d23_p6*d31_p2;

    double roots[4] = {0};
    int num_roots = solve_deg4(a4, a3, a2, a1, a0, roots[0], roots[1], roots[2], roots[3]);

    Rs.reserve(num_roots);
    ts.reserve(num_roots);
    int nb_solutions = 0;
    for (double root : roots) {
        if (root <= 0) continue;

        const double n12 = root, n12_p2 = n12 * n12;
        const double n13 = (d12_p2*(d23_p2-d31_p2*n12_p2)+(d23_p2-d31_p2)*(d23_p2*(1+n12_p2-2*n12*c12)-d12_p2*n12_p2))
                            / (2*d12_p2*(d23_p2*c31 - d31_p2*c23*n12) + 2*(d31_p2-d23_p2)*d12_p2*c23*n12);
        const double n1 = d12 / sqrt(1 + n12_p2 - 2*n12*c12); // 1+n12^2-2n12c12 is always > 0
        const double n2 = n1 * n12;
        const double n3 = n1 * n13;

        if (n1 <= 0 || n2 <= 0 || n3 <= 0)
            continue;
        // compute and check errors
        if (fabs((sqrt(n1*n1 + n2*n2 - 2*n1*n2*c12) - d12) / d12) > VAL_THR ||
            fabs((sqrt(n2*n2 + n3*n3 - 2*n2*n3*c23) - d23) / d23) > VAL_THR ||
            fabs((sqrt(n3*n3 + n1*n1 - 2*n3*n1*c31) - d31) / d31) > VAL_THR)
            continue;

        const Vec3d nX1 = n1 * cx1;
        Vec3d Z2 = n2 * cx2 - nX1; Z2 /= norm(Z2);
        Vec3d Z3 = n3 * cx3 - nX1; Z3 /= norm(Z3);
        Vec3d Z1 = Z2.cross(Z3);   Z1 /= norm(Z1);
        const Vec3d Z3crZ1 = Z3.cross(Z1);

        z[0] = Z1(0);     z[3] = Z1(1);     z[6] = Z1(2);
        z[1] = Z2(0);     z[4] = Z2(1);     z[7] = Z2(2);
        z[2] = Z3crZ1(0); z[5] = Z3crZ1(1); z[8] = Z3crZ1(2);

        Vec3d Zw2 = (X2 - X1) / d12;
        Vec3d Zw3 = (X3 - X1) / d31;
        Vec3d Zw1 = Zw2.cross(Zw3); Zw1 /= norm(Zw1);
        const Vec3d Z3crZ1w = Zw3.cross(Zw1);

        zw[0] = Zw1(0);     zw[3] = Zw1(1);     zw[6] = Zw1(2);
        zw[1] = Zw2(0);     zw[4] = Zw2(1);     zw[7] = Zw2(2);
        zw[2] = Z3crZ1w(0); zw[5] = Z3crZ1w(1); zw[8] = Z3crZ1w(2);

        const Matx33d R = Z * Zw.inv();
        Rs.push_back(cv::Mat(R));
        ts.push_back(cv::Mat(-R * (X1 - R.t() * nX1)));
        nb_solutions++;
    }

    return nb_solutions;
}
