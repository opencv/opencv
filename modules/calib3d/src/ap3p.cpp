#include "precomp.hpp"
#include "ap3p.h"

#include <cmath>
#include <complex>
#if defined (_MSC_VER) && (_MSC_VER <= 1700)
static inline double cbrt(double x) { return (double)cv::cubeRoot((float)x); };
#endif

namespace {
void solveQuartic(const double *factors, double *realRoots) {
    const double &a4 = factors[0];
    const double &a3 = factors[1];
    const double &a2 = factors[2];
    const double &a1 = factors[3];
    const double &a0 = factors[4];

    double a4_2 = a4 * a4;
    double a3_2 = a3 * a3;
    double a4_3 = a4_2 * a4;
    double a2a4 = a2 * a4;

    double p4 = (8 * a2a4 - 3 * a3_2) / (8 * a4_2);
    double q4 = (a3_2 * a3 - 4 * a2a4 * a3 + 8 * a1 * a4_2) / (8 * a4_3);
    double r4 = (256 * a0 * a4_3 - 3 * (a3_2 * a3_2) - 64 * a1 * a3 * a4_2 + 16 * a2a4 * a3_2) / (256 * (a4_3 * a4));

    double p3 = ((p4 * p4) / 12 + r4) / 3; // /=-3
    double q3 = (72 * r4 * p4 - 2 * p4 * p4 * p4 - 27 * q4 * q4) / 432; // /=2

    double t; // *=2
    std::complex<double> w;
    if (q3 >= 0)
        w = -std::sqrt(static_cast<std::complex<double> >(q3 * q3 - p3 * p3 * p3)) - q3;
    else
        w = std::sqrt(static_cast<std::complex<double> >(q3 * q3 - p3 * p3 * p3)) - q3;
    if (w.imag() == 0.0) {
        w.real(std::cbrt(w.real()));
        t = 2.0 * (w.real() + p3 / w.real());
    } else {
        w = pow(w, 1.0 / 3);
        t = 4.0 * w.real();
    }

    std::complex<double> sqrt_2m = sqrt(static_cast<std::complex<double> >(-2 * p4 / 3 + t));
    double B_4A = -a3 / (4 * a4);
    double complex1 = 4 * p4 / 3 + t;
#if defined(__clang__) && defined(__arm__) && (__clang_major__ == 3 || __clang_major__ == 4) && !defined(__ANDROID__)
    // details: https://github.com/opencv/opencv/issues/11135
    // details: https://github.com/opencv/opencv/issues/11056
    std::complex<double> complex2 = 2 * q4;
    complex2 = std::complex<double>(complex2.real() / sqrt_2m.real(), 0);
#else
    std::complex<double> complex2 = 2 * q4 / sqrt_2m;
#endif
    double sqrt_2m_rh = sqrt_2m.real() / 2;
    double sqrt1 = sqrt(-(complex1 + complex2)).real() / 2;
    realRoots[0] = B_4A + sqrt_2m_rh + sqrt1;
    realRoots[1] = B_4A + sqrt_2m_rh - sqrt1;
    double sqrt2 = sqrt(-(complex1 - complex2)).real() / 2;
    realRoots[2] = B_4A - sqrt_2m_rh + sqrt2;
    realRoots[3] = B_4A - sqrt_2m_rh - sqrt2;
}

void polishQuarticRoots(const double *coeffs, double *roots) {
    const int iterations = 2;
    for (int i = 0; i < iterations; ++i) {
        for (int j = 0; j < 4; ++j) {
            double error =
                    (((coeffs[0] * roots[j] + coeffs[1]) * roots[j] + coeffs[2]) * roots[j] + coeffs[3]) * roots[j] +
                    coeffs[4];
            double
                    derivative =
                    ((4 * coeffs[0] * roots[j] + 3 * coeffs[1]) * roots[j] + 2 * coeffs[2]) * roots[j] + coeffs[3];
            roots[j] -= error / derivative;
        }
    }
}

inline void vect_cross(const double *a, const double *b, double *result) {
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = -(a[0] * b[2] - a[2] * b[0]);
    result[2] = a[0] * b[1] - a[1] * b[0];
}

inline double vect_dot(const double *a, const double *b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline double vect_norm(const double *a) {
    return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

inline void vect_scale(const double s, const double *a, double *result) {
    result[0] = a[0] * s;
    result[1] = a[1] * s;
    result[2] = a[2] * s;
}

inline void vect_sub(const double *a, const double *b, double *result) {
    result[0] = a[0] - b[0];
    result[1] = a[1] - b[1];
    result[2] = a[2] - b[2];
}

inline void vect_divide(const double *a, const double d, double *result) {
    result[0] = a[0] / d;
    result[1] = a[1] / d;
    result[2] = a[2] / d;
}

inline void mat_mult(const double a[3][3], const double b[3][3], double result[3][3]) {
    result[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0];
    result[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1];
    result[0][2] = a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2];

    result[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0];
    result[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1];
    result[1][2] = a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2];

    result[2][0] = a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0];
    result[2][1] = a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1];
    result[2][2] = a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2];
}
}

namespace cv {
void ap3p::init_inverse_parameters() {
    inv_fx = 1. / fx;
    inv_fy = 1. / fy;
    cx_fx = cx / fx;
    cy_fy = cy / fy;
}

ap3p::ap3p(cv::Mat cameraMatrix) {
    if (cameraMatrix.depth() == CV_32F)
        init_camera_parameters<float>(cameraMatrix);
    else
        init_camera_parameters<double>(cameraMatrix);
    init_inverse_parameters();
}

ap3p::ap3p(double _fx, double _fy, double _cx, double _cy) {
    fx = _fx;
    fy = _fy;
    cx = _cx;
    cy = _cy;
    init_inverse_parameters();
}

// This algorithm is from "Tong Ke, Stergios Roumeliotis, An Efficient Algebraic Solution to the Perspective-Three-Point Problem" (Accepted by CVPR 2017)
// See https://arxiv.org/pdf/1701.08237.pdf
// featureVectors: The 3 bearing measurements (normalized) stored as column vectors
// worldPoints: The positions of the 3 feature points stored as column vectors
// solutionsR: 4 possible solutions of rotation matrix of the world w.r.t the camera frame
// solutionsT: 4 possible solutions of translation of the world origin w.r.t the camera frame
int ap3p::computePoses(const double featureVectors[3][4],
                       const double worldPoints[3][4],
                       double solutionsR[4][3][3],
                       double solutionsT[4][3],
                       bool p4p) {

    //world point vectors
    double w1[3] = {worldPoints[0][0], worldPoints[1][0], worldPoints[2][0]};
    double w2[3] = {worldPoints[0][1], worldPoints[1][1], worldPoints[2][1]};
    double w3[3] = {worldPoints[0][2], worldPoints[1][2], worldPoints[2][2]};
    // k1
    double u0[3];
    vect_sub(w1, w2, u0);

    double nu0 = vect_norm(u0);
    double k1[3];
    vect_divide(u0, nu0, k1);
    // bi
    double b1[3] = {featureVectors[0][0], featureVectors[1][0], featureVectors[2][0]};
    double b2[3] = {featureVectors[0][1], featureVectors[1][1], featureVectors[2][1]};
    double b3[3] = {featureVectors[0][2], featureVectors[1][2], featureVectors[2][2]};
    // k3,tz
    double k3[3];
    vect_cross(b1, b2, k3);
    double nk3 = vect_norm(k3);
    vect_divide(k3, nk3, k3);

    double tz[3];
    vect_cross(b1, k3, tz);
    // ui,vi
    double v1[3];
    vect_cross(b1, b3, v1);
    double v2[3];
    vect_cross(b2, b3, v2);

    double u1[3];
    vect_sub(w1, w3, u1);
    // coefficients related terms
    double u1k1 = vect_dot(u1, k1);
    double k3b3 = vect_dot(k3, b3);
    // f1i
    double f11 = k3b3;
    double f13 = vect_dot(k3, v1);
    double f15 = -u1k1 * f11;
    //delta
    double nl[3];
    vect_cross(u1, k1, nl);
    double delta = vect_norm(nl);
    vect_divide(nl, delta, nl);
    f11 *= delta;
    f13 *= delta;
    // f2i
    double u2k1 = u1k1 - nu0;
    double f21 = vect_dot(tz, v2);
    double f22 = nk3 * k3b3;
    double f23 = vect_dot(k3, v2);
    double f24 = u2k1 * f22;
    double f25 = -u2k1 * f21;
    f21 *= delta;
    f22 *= delta;
    f23 *= delta;
    double g1 = f13 * f22;
    double g2 = f13 * f25 - f15 * f23;
    double g3 = f11 * f23 - f13 * f21;
    double g4 = -f13 * f24;
    double g5 = f11 * f22;
    double g6 = f11 * f25 - f15 * f21;
    double g7 = -f15 * f24;
    double coeffs[5] = {g5 * g5 + g1 * g1 + g3 * g3,
                        2 * (g5 * g6 + g1 * g2 + g3 * g4),
                        g6 * g6 + 2 * g5 * g7 + g2 * g2 + g4 * g4 - g1 * g1 - g3 * g3,
                        2 * (g6 * g7 - g1 * g2 - g3 * g4),
                        g7 * g7 - g2 * g2 - g4 * g4};
    double s[4];
    solveQuartic(coeffs, s);
    polishQuarticRoots(coeffs, s);

    double temp[3];
    vect_cross(k1, nl, temp);

    double Ck1nl[3][3] =
            {{k1[0], nl[0], temp[0]},
             {k1[1], nl[1], temp[1]},
             {k1[2], nl[2], temp[2]}};

    double Cb1k3tzT[3][3] =
            {{b1[0], b1[1], b1[2]},
             {k3[0], k3[1], k3[2]},
             {tz[0], tz[1], tz[2]}};

    double b3p[3];
    vect_scale((delta / k3b3), b3, b3p);

    double X3 = worldPoints[0][3];
    double Y3 = worldPoints[1][3];
    double Z3 = worldPoints[2][3];
    double mu3 = featureVectors[0][3];
    double mv3 = featureVectors[1][3];
    double reproj_errors[4];

    int nb_solutions = 0;
    for (int i = 0; i < 4; ++i) {
        double ctheta1p = s[i];
        if (abs(ctheta1p) > 1)
            continue;
        double stheta1p = sqrt(1 - ctheta1p * ctheta1p);
        stheta1p = (k3b3 > 0) ? stheta1p : -stheta1p;
        double ctheta3 = g1 * ctheta1p + g2;
        double stheta3 = g3 * ctheta1p + g4;
        double ntheta3 = stheta1p / ((g5 * ctheta1p + g6) * ctheta1p + g7);
        ctheta3 *= ntheta3;
        stheta3 *= ntheta3;

        double C13[3][3] =
                {{ctheta3,            0,         -stheta3},
                 {stheta1p * stheta3, ctheta1p,  stheta1p * ctheta3},
                 {ctheta1p * stheta3, -stheta1p, ctheta1p * ctheta3}};

        double temp_matrix[3][3];
        double R[3][3];
        mat_mult(Ck1nl, C13, temp_matrix);
        mat_mult(temp_matrix, Cb1k3tzT, R);

        // R' * p3
        double rp3[3] =
                {w3[0] * R[0][0] + w3[1] * R[1][0] + w3[2] * R[2][0],
                 w3[0] * R[0][1] + w3[1] * R[1][1] + w3[2] * R[2][1],
                 w3[0] * R[0][2] + w3[1] * R[1][2] + w3[2] * R[2][2]};

        double pxstheta1p[3];
        vect_scale(stheta1p, b3p, pxstheta1p);

        vect_sub(pxstheta1p, rp3, solutionsT[nb_solutions]);

        solutionsR[nb_solutions][0][0] = R[0][0];
        solutionsR[nb_solutions][1][0] = R[0][1];
        solutionsR[nb_solutions][2][0] = R[0][2];
        solutionsR[nb_solutions][0][1] = R[1][0];
        solutionsR[nb_solutions][1][1] = R[1][1];
        solutionsR[nb_solutions][2][1] = R[1][2];
        solutionsR[nb_solutions][0][2] = R[2][0];
        solutionsR[nb_solutions][1][2] = R[2][1];
        solutionsR[nb_solutions][2][2] = R[2][2];

        if (p4p) {
            double X3p = solutionsR[nb_solutions][0][0] * X3 + solutionsR[nb_solutions][0][1] * Y3 + solutionsR[nb_solutions][0][2] * Z3 + solutionsT[nb_solutions][0];
            double Y3p = solutionsR[nb_solutions][1][0] * X3 + solutionsR[nb_solutions][1][1] * Y3 + solutionsR[nb_solutions][1][2] * Z3 + solutionsT[nb_solutions][1];
            double Z3p = solutionsR[nb_solutions][2][0] * X3 + solutionsR[nb_solutions][2][1] * Y3 + solutionsR[nb_solutions][2][2] * Z3 + solutionsT[nb_solutions][2];
            double mu3p = X3p / Z3p;
            double mv3p = Y3p / Z3p;
            reproj_errors[nb_solutions] = (mu3p - mu3) * (mu3p - mu3) + (mv3p - mv3) * (mv3p - mv3);
        }

        nb_solutions++;
    }

    //sort the solutions
    if (p4p) {
        for (int i = 1; i < nb_solutions; i++) {
            for (int j = i; j > 0 && reproj_errors[j-1] > reproj_errors[j]; j--) {
                std::swap(reproj_errors[j], reproj_errors[j-1]);
                std::swap(solutionsR[j], solutionsR[j-1]);
                std::swap(solutionsT[j], solutionsT[j-1]);
            }
        }
    }

    return nb_solutions;
}

bool ap3p::solve(cv::Mat &R, cv::Mat &tvec, const cv::Mat &opoints, const cv::Mat &ipoints) {
    CV_INSTRUMENT_REGION();

    double rotation_matrix[3][3] = {}, translation[3] = {};
    std::vector<double> points;
    if (opoints.depth() == ipoints.depth()) {
        if (opoints.depth() == CV_32F)
            extract_points<cv::Point3f, cv::Point2f>(opoints, ipoints, points);
        else
            extract_points<cv::Point3d, cv::Point2d>(opoints, ipoints, points);
    } else if (opoints.depth() == CV_32F)
        extract_points<cv::Point3f, cv::Point2d>(opoints, ipoints, points);
    else
        extract_points<cv::Point3d, cv::Point2f>(opoints, ipoints, points);

    bool result = solve(rotation_matrix, translation,
                        points[0], points[1], points[2], points[3], points[4],
                        points[5], points[6], points[7], points[8], points[9],
                        points[10], points[11], points[12], points[13],points[14],
                        points[15], points[16], points[17], points[18], points[19]);
    cv::Mat(3, 1, CV_64F, translation).copyTo(tvec);
    cv::Mat(3, 3, CV_64F, rotation_matrix).copyTo(R);
    return result;
}

int ap3p::solve(std::vector<cv::Mat> &Rs, std::vector<cv::Mat> &tvecs, const cv::Mat &opoints, const cv::Mat &ipoints) {
    CV_INSTRUMENT_REGION();

    double rotation_matrix[4][3][3] = {}, translation[4][3] = {};
    std::vector<double> points;
    if (opoints.depth() == ipoints.depth()) {
        if (opoints.depth() == CV_32F)
            extract_points<cv::Point3f, cv::Point2f>(opoints, ipoints, points);
        else
            extract_points<cv::Point3d, cv::Point2d>(opoints, ipoints, points);
    } else if (opoints.depth() == CV_32F)
        extract_points<cv::Point3f, cv::Point2d>(opoints, ipoints, points);
    else
        extract_points<cv::Point3d, cv::Point2f>(opoints, ipoints, points);

    const bool p4p = std::max(opoints.checkVector(3, CV_32F), opoints.checkVector(3, CV_64F)) == 4;
    int solutions = solve(rotation_matrix, translation,
                          points[0], points[1], points[2], points[3], points[4],
                          points[5], points[6], points[7], points[8], points[9],
                          points[10], points[11], points[12], points[13], points[14],
                          points[15], points[16], points[17], points[18], points[19],
                          p4p);

    for (int i = 0; i < solutions; i++) {
        cv::Mat R, tvec;
        cv::Mat(3, 1, CV_64F, translation[i]).copyTo(tvec);
        cv::Mat(3, 3, CV_64F, rotation_matrix[i]).copyTo(R);

        Rs.push_back(R);
        tvecs.push_back(tvec);
    }

    return solutions;
}

bool
ap3p::solve(double R[3][3], double t[3],
            double mu0, double mv0, double X0, double Y0, double Z0,
            double mu1, double mv1, double X1, double Y1, double Z1,
            double mu2, double mv2, double X2, double Y2, double Z2,
            double mu3, double mv3, double X3, double Y3, double Z3) {
    double Rs[4][3][3] = {}, ts[4][3] = {};

    const bool p4p = true;
    int n = solve(Rs, ts, mu0, mv0, X0, Y0, Z0, mu1, mv1, X1, Y1, Z1, mu2, mv2, X2, Y2, Z2, mu3, mv3, X3, Y3, Z3, p4p);
    if (n == 0)
        return false;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
            R[i][j] = Rs[0][i][j];
        t[i] = ts[0][i];
    }

    return true;
}

int ap3p::solve(double R[4][3][3], double t[4][3],
                double mu0, double mv0, double X0, double Y0, double Z0,
                double mu1, double mv1, double X1, double Y1, double Z1,
                double mu2, double mv2, double X2, double Y2, double Z2,
                double mu3, double mv3, double X3, double Y3, double Z3,
                bool p4p) {
    double mk0, mk1, mk2;
    double norm;

    mu0 = inv_fx * mu0 - cx_fx;
    mv0 = inv_fy * mv0 - cy_fy;
    norm = sqrt(mu0 * mu0 + mv0 * mv0 + 1);
    mk0 = 1. / norm;
    mu0 *= mk0;
    mv0 *= mk0;

    mu1 = inv_fx * mu1 - cx_fx;
    mv1 = inv_fy * mv1 - cy_fy;
    norm = sqrt(mu1 * mu1 + mv1 * mv1 + 1);
    mk1 = 1. / norm;
    mu1 *= mk1;
    mv1 *= mk1;

    mu2 = inv_fx * mu2 - cx_fx;
    mv2 = inv_fy * mv2 - cy_fy;
    norm = sqrt(mu2 * mu2 + mv2 * mv2 + 1);
    mk2 = 1. / norm;
    mu2 *= mk2;
    mv2 *= mk2;

    mu3 = inv_fx * mu3 - cx_fx;
    mv3 = inv_fy * mv3 - cy_fy;
    double mk3 = 1; //not used

    double featureVectors[3][4] = {{mu0, mu1, mu2, mu3},
                                   {mv0, mv1, mv2, mv3},
                                   {mk0, mk1, mk2, mk3}};
    double worldPoints[3][4] = {{X0, X1, X2, X3},
                                {Y0, Y1, Y2, Y3},
                                {Z0, Z1, Z2, Z3}};

    return computePoses(featureVectors, worldPoints, R, t, p4p);
}
}
