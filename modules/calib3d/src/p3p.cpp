#include <cstring>
#include <cmath>
#include <iostream>

#include "p3p.h"


using namespace cv;

// Copyright (c) 2020, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Author: Yaqing Ding
//         Mark Shachkov

// https://github.com/PoseLib/PoseLib/blob/79fe59ada3122c50383ac06e043a5e04072c6711/PoseLib/solvers/p3p.cc
namespace yaqding
{

static bool solve_cubic_single_real(double c2, double c1, double c0, double &root) {
    double a = c1 - c2 * c2 / 3.0;
    double b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
    double c = b * b / 4.0 + a * a * a / 27.0;
    if (c != 0) {
        if (c > 0) {
            c = std::sqrt(c);
            b *= -0.5;
            root = std::cbrt(b + c) + std::cbrt(b - c) - c2 / 3.0;
            return true;
        } else {
            c = 3.0 * b / (2.0 * a) * std::sqrt(-3.0 / a);
            root = 2.0 * std::sqrt(-a / 3.0) * std::cos(std::acos(c) / 3.0) - c2 / 3.0;
        }
    } else {
        root = -c2 / 3.0 + (a != 0 ? (3.0 * b / a) : 0);
    }
    return false;
}

static bool root2real(double b, double c, double &r1, double &r2) {
    const double THRESHOLD = -1.0e-12;
    double v = b * b - 4.0 * c;
    if (v < THRESHOLD) {
        r1 = r2 = -0.5 * b;
        return v >= 0;
    }
    if (v > THRESHOLD && v < 0.0) {
        r1 = -0.5 * b;
        r2 = -2;
        return true;
    }

    double y = std::sqrt(v);
    if (b < 0) {
        r1 = 0.5 * (-b + y);
        r2 = 0.5 * (-b - y);
    } else {
        r1 = 2.0 * c / (-b + y);
        r2 = 2.0 * c / (-b - y);
    }
    return true;
}

static std::array<Vec3d, 2> compute_pq(Matx33d C) {
    std::array<Vec3d, 2> pq;
    Matx33d C_adj;

    C_adj(0, 0) = C(1, 2) * C(2, 1) - C(1, 1) * C(2, 2);
    C_adj(1, 1) = C(0, 2) * C(2, 0) - C(0, 0) * C(2, 2);
    C_adj(2, 2) = C(0, 1) * C(1, 0) - C(0, 0) * C(1, 1);
    C_adj(0, 1) = C(0, 1) * C(2, 2) - C(0, 2) * C(2, 1);
    C_adj(0, 2) = C(0, 2) * C(1, 1) - C(0, 1) * C(1, 2);
    C_adj(1, 0) = C_adj(0, 1);
    C_adj(1, 2) = C(0, 0) * C(1, 2) - C(0, 2) * C(1, 0);
    C_adj(2, 0) = C_adj(0, 2);
    C_adj(2, 1) = C_adj(1, 2);

    Matx31d v;
    if (C_adj(0, 0) > C_adj(1, 1)) {
        if (C_adj(0, 0) > C_adj(2, 2)) {
            v = C_adj.col(0) / std::sqrt(C_adj(0, 0));
        } else {
            v = C_adj.col(2) / std::sqrt(C_adj(2, 2));
        }
    } else if (C_adj(1, 1) > C_adj(2, 2)) {
        v = C_adj.col(1) / std::sqrt(C_adj(1, 1));
    } else {
        v = C_adj.col(2) / std::sqrt(C_adj(2, 2));
    }

    C(0, 1) -= v(2);
    C(0, 2) += v(1);
    C(1, 2) -= v(0);
    C(1, 0) += v(2);
    C(2, 0) -= v(1);
    C(2, 1) += v(0);

    pq[0](0) = C.col(0)(0);
    pq[0](1) = C.col(0)(1);
    pq[0](2) = C.col(0)(2);
    pq[1](0) = C.row(0)(0);
    pq[1](1) = C.row(0)(1);
    pq[1](2) = C.row(0)(2);

    return pq;
}

// Performs a few Newton steps on the equations
static void refine_lambda(double &lambda1, double &lambda2, double &lambda3, const double a12, const double a13,
                   const double a23, const double b12, const double b13, const double b23) {
    for (int iter = 0; iter < 5; ++iter) {
        double r1 = (lambda1 * lambda1 - 2.0 * lambda1 * lambda2 * b12 + lambda2 * lambda2 - a12);
        double r2 = (lambda1 * lambda1 - 2.0 * lambda1 * lambda3 * b13 + lambda3 * lambda3 - a13);
        double r3 = (lambda2 * lambda2 - 2.0 * lambda2 * lambda3 * b23 + lambda3 * lambda3 - a23);

        if (std::abs(r1) + std::abs(r2) + std::abs(r3) < 1e-10)
            return;

        double x11 = lambda1 - lambda2 * b12;
        double x12 = lambda2 - lambda1 * b12;
        double x21 = lambda1 - lambda3 * b13;
        double x23 = lambda3 - lambda1 * b13;
        double x32 = lambda2 - lambda3 * b23;
        double x33 = lambda3 - lambda2 * b23;
        double detJ = 0.5 / (x11 * x23 * x32 + x12 * x21 * x33); // half minus inverse determinant

        // This uses the closed form of the inverse for the jacobian.
        // Due to the zero elements this actually becomes quite nice.
        lambda1 += (-x23 * x32 * r1 - x12 * x33 * r2 + x12 * x23 * r3) * detJ;
        lambda2 += (-x21 * x33 * r1 + x11 * x33 * r2 - x11 * x23 * r3) * detJ;
        lambda3 += (x21 * x32 * r1 - x11 * x32 * r2 - x12 * x21 * r3) * detJ;
    }
}

};

void p3p::calibrateAndNormalizePointsPnP(const Mat &opoints_, const Mat &ipoints_) {
    auto convertPoints = [] (const Mat &points_input, Mat &points, int pt_dim) {
        points_input.convertTo(points, CV_64F); // convert points to have float precision
        if (points.channels() > 1)
            points = points.reshape(1, (int)points.total()); // convert point to have 1 channel
        if (points.rows < points.cols)
            transpose(points, points); // transpose so points will be in rows
        CV_CheckGE(points.cols, pt_dim, "Invalid dimension of point");
        if (points.cols != pt_dim) // in case when image points are 3D convert them to 2D
            points = points.colRange(0, pt_dim);
    };

    Mat ipoints;
    convertPoints(ipoints_, ipoints, 2);
    for (int i = 0; i < ipoints.rows; i++) {
        const double k_inv_u = ipoints.at<double>(i, 0);
        const double k_inv_v = ipoints.at<double>(i, 1);
        double x_norm = 1.0 / sqrt(k_inv_u*k_inv_u + k_inv_v*k_inv_v + 1);
        x_copy[i](0) = k_inv_u * x_norm;
        x_copy[i](1) = k_inv_v * x_norm;
        x_copy[i](2) =           x_norm;
    }

    Mat opoints;
    convertPoints(opoints_, opoints, 3);
    X_copy[0](0) = opoints.at<double>(0, 0);
    X_copy[0](1) = opoints.at<double>(0, 1);
    X_copy[0](2) = opoints.at<double>(0, 2);

    X_copy[1](0) = opoints.at<double>(1, 0);
    X_copy[1](1) = opoints.at<double>(1, 1);
    X_copy[1](2) = opoints.at<double>(1, 2);

    X_copy[2](0) = opoints.at<double>(2, 0);
    X_copy[2](1) = opoints.at<double>(2, 1);
    X_copy[2](2) = opoints.at<double>(2, 2);
}

p3p::p3p() :
    x_copy(), X_copy()
{
}

int p3p::estimate(std::vector<Mat>& Rs, std::vector<Mat>& ts, const cv::Mat& opoints, const cv::Mat& ipoints) {
    CV_INSTRUMENT_REGION();
    calibrateAndNormalizePointsPnP(opoints, ipoints);

    Rs.reserve(4);
    ts.reserve(4);

    Vec3d X01 = X_copy[0] - X_copy[1];
    Vec3d X02 = X_copy[0] - X_copy[2];
    Vec3d X12 = X_copy[1] - X_copy[2];

    double a01 = norm(X01, NORM_L2SQR);
    double a02 = norm(X02, NORM_L2SQR);
    double a12 = norm(X12, NORM_L2SQR);

    std::array<Vec3d, 3> X = {X_copy[0], X_copy[1], X_copy[2]};
    std::array<Vec3d, 3> x = {x_copy[0], x_copy[1], x_copy[2]};

    // Switch X,x so that BC is the largest distance among {X01, X02, X12}
    if (a01 > a02) {
        if (a01 > a12) {
            std::swap(x[0], x[2]);
            std::swap(X[0], X[2]);
            std::swap(a01, a12);
            X01 = -X12;
            X02 = -X02;
        }
    } else if (a02 > a12) {
        std::swap(x[0], x[1]);
        std::swap(X[0], X[1]);
        std::swap(a02, a12);
        X01 = -X01;
        X02 = X12;
    }

    const double a12d = 1.0 / a12;
    const double a = a01 * a12d;
    const double b = a02 * a12d;

    const double m01 = x[0].dot(x[1]);
    const double m02 = x[0].dot(x[2]);
    const double m12 = x[1].dot(x[2]);

    // Ugly parameters to simplify the calculation
    const double m12sq = -m12 * m12 + 1.0;
    const double m02sq = -1.0 + m02 * m02;
    const double m01sq = -1.0 + m01 * m01;
    const double ab = a * b;
    const double bsq = b * b;
    const double asq = a * a;
    const double m013 = -2.0 + 2.0 * m01 * m02 * m12;
    const double bsqm12sq = bsq * m12sq;
    const double asqm12sq = asq * m12sq;
    const double abm12sq = 2.0 * ab * m12sq;

    const double k3_inv = 1.0 / (bsqm12sq + b * m02sq);
    const double k2 = k3_inv * ((-1.0 + a) * m02sq + abm12sq + bsqm12sq + b * m013);
    const double k1 = k3_inv * (asqm12sq + abm12sq + a * m013 + (-1.0 + b) * m01sq);
    const double k0 = k3_inv * (asqm12sq + a * m01sq);

    double s;
    bool G = yaqding::solve_cubic_single_real(k2, k1, k0, s);

    Matx33d C;
    C(0, 0) = -a + s * (1 - b);
    C(0, 1) = -m02 * s;
    C(0, 2) = a * m12 + b * m12 * s;
    C(1, 0) = C(0, 1);
    C(1, 1) = s + 1;
    C(1, 2) = -m01;
    C(2, 0) = C(0, 2);
    C(2, 1) = C(1, 2);
    C(2, 2) = -a - b * s + 1;

    std::array<Vec3d, 2> pq = yaqding::compute_pq(C);

    // XX << X01, X02, X01.cross(X02);
    // XX = XX.inverse().eval();
    Matx33d XX;
    XX(0,0) = X01(0);   XX(1,0) = X01(1);   XX(2,0) = X01(2);
    XX(0,1) = X02(0);   XX(1,1) = X02(1);   XX(2,1) = X02(2);
    Vec3d X01_X02 = X01.cross(X02);
    XX(0,2) = X01_X02(0);   XX(1,2) = X01_X02(1);   XX(2,2) = X01_X02(2);
    XX = XX.inv();

    int n_sols = 0;
    for (int i = 0; i < 2; ++i) {
        // [p0 p1 p2] * [1; x; y] = 0, or [p0 p1 p2] * [d2; d0; d1] = 0
        double p0 = pq[i](0);
        double p1 = pq[i](1);
        double p2 = pq[i](2);
        // here we run into trouble if p0 is zero,
        // so depending on which is larger, we solve for either d0 or d1
        // The case p0 = p1 = 0 is degenerate and can be ignored
        bool switch_12 = std::abs(p0) <= std::abs(p1);

        if (switch_12) {
            // eliminate d0
            double w0 = -p0 / p1;
            double w1 = -p2 / p1;
            double ca = 1.0 / (w1 * w1 - b);
            double cb = 2.0 * (b * m12 - m02 * w1 + w0 * w1) * ca;
            double cc = (w0 * w0 - 2 * m02 * w0 - b + 1.0) * ca;
            double taus[2];

            if (!yaqding::root2real(cb, cc, taus[0], taus[1]))
                continue;

            for (double tau : taus) {
                if (tau <= 0)
                    continue;

                // positive only
                double d2 = std::sqrt(a12 / (tau * (tau - 2.0 * m12) + 1.0));
                double d1 = tau * d2;
                double d0 = (w0 * d2 + w1 * d1);
                if (d0 < 0)
                    continue;

                yaqding::refine_lambda(d0, d1, d2, a01, a02, a12, m01, m02, m12);
                Vec3d v1 = d0 * x[0] - d1 * x[1];
                Vec3d v2 = d0 * x[0] - d2 * x[2];
                // YY << v1, v2, v1.cross(v2);
                Matx33d YY;
                YY(0,0) = v1(0);   YY(1,0) = v1(1);   YY(2,0) = v1(2);
                YY(0,1) = v2(0);   YY(1,1) = v2(1);   YY(2,1) = v2(2);
                Vec3d v1_v2 = v1.cross(v2);
                YY(0,2) = v1_v2(0);   YY(1,2) = v1_v2(1);   YY(2,2) = v1_v2(2);

                // output->emplace_back(R, d0 * x[0] - R * X[0]);
                Matx33d R = (YY * XX);
                Rs.push_back(Mat(R));
                Vec3d trans = (d0 * x[0] - R * X[0]);
                ts.push_back(Mat(trans));
                ++n_sols;
            }
        } else {
            double w0 = -p1 / p0;
            double w1 = -p2 / p0;
            double ca = 1.0 / (-a * w1 * w1 + 2 * a * m12 * w1 - a + 1);
            double cb = 2 * (a * m12 * w0 - m01 - a * w0 * w1) * ca;
            double cc = (1 - a * w0 * w0) * ca;

            double taus[2];
            if (!yaqding::root2real(cb, cc, taus[0], taus[1]))
                continue;

            for (double tau : taus) {
                if (tau <= 0)
                    continue;

                double d0 = std::sqrt(a01 / (tau * (tau - 2.0 * m01) + 1.0));
                double d1 = tau * d0;
                double d2 = w0 * d0 + w1 * d1;

                if (d2 < 0)
                    continue;

                yaqding::refine_lambda(d0, d1, d2, a01, a02, a12, m01, m02, m12);
                Vec3d v1 = d0 * x[0] - d1 * x[1];
                Vec3d v2 = d0 * x[0] - d2 * x[2];
                // YY << v1, v2, v1.cross(v2);
                Matx33d YY;
                YY(0,0) = v1(0);   YY(1,0) = v1(1);   YY(2,0) = v1(2);
                YY(0,1) = v2(0);   YY(1,1) = v2(1);   YY(2,1) = v2(2);
                Vec3d v1_v2 = v1.cross(v2);
                YY(0,2) = v1_v2(0);   YY(1,2) = v1_v2(1);   YY(2,2) = v1_v2(2);

                // output->emplace_back(R, d0 * x[0] - R * X[0]);
                Matx33d R = (YY * XX);
                Rs.push_back(Mat(R));
                Vec3d trans = (d0 * x[0] - R * X[0]);
                ts.push_back(Mat(trans));
                ++n_sols;
            }
        }

        if (n_sols > 0 && G)
            break;
    }

    return n_sols;
}
