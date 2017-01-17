#include "P3p.h"

#include <cmath>
#include <complex>

using namespace std;

// featureVectors: The 3 bearing measurements (normalized) stored as column vectors
// worldPoints: The positions of the 3 feature points stored as column vectors
// solutions: Output of this function. Column i (i=0,4,8,12) is the solution for the camera position, and column i+1 to
//            i+3 h
int P3p::computePoses(const double featureVectors[3][3],
                      const double worldPoints[3][3],
                      double solutions[3][16]) {

    //world point vectors
    double w1[3] = {worldPoints[0][0], worldPoints[1][0], worldPoints[2][0]};
    double w2[3] = {worldPoints[0][1], worldPoints[1][1], worldPoints[2][1]};
    double w3[3] = {worldPoints[0][2], worldPoints[1][2], worldPoints[2][2]};
    // k1
    double u0[3];
    this->vect_sub(w1, w2, u0);

    double nu0 = this->vect_norm(u0);
    double k1[3];
    this->vect_divide(u0, nu0, k1);
    // bi
    double b1[3] = {featureVectors[0][0], featureVectors[1][0], featureVectors[2][0]};
    double b2[3] = {featureVectors[0][1], featureVectors[1][1], featureVectors[2][1]};
    double b3[3] = {featureVectors[0][2], featureVectors[1][2], featureVectors[2][2]};
    // k3,tz
    double k3[3];
    this->vect_cross(b1, b2, k3);
    double nk3 = this->vect_norm(k3);
    this->vect_divide(k3, nk3, k3);


    double tz[3];
    this->vect_cross(b1, k3, tz);
    // ui,vi
    double v1[3];
    this->vect_cross(b1, b3, v1);
    double v2[3];
    this->vect_cross(b2, b3, v2);

    double u1[3];
    this->vect_sub(w1, w3, u1);
    // coefficients related terms
    double u1k1 = this->vect_dot(u1, k1);
    double k3b3 = this->vect_dot(k3, b3);
    // f1i
    double f11 = k3b3;
    double f13 = this->vect_dot(k3, v1);
    double f15 = -u1k1 * f11;
    //delta
    double nl[3];
    this->vect_cross(u1, k1, nl);
    double delta = this->vect_norm(nl);
    this->vect_divide(nl, delta, nl);
    f11 *= delta;
    f13 *= delta;
    // f2i
    double u2k1 = u1k1 - nu0;
    double f21 = this->vect_dot(tz, v2);
    double f22 = nk3 * k3b3;
    double f23 = this->vect_dot(k3, v2);
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
    this->solveQuartic(coeffs, s);
    this->polishQuarticRoots(coeffs, s);


    double temp[3];
    this->vect_cross(k1, nl, temp);

    double Ck1nl[3][3] =
            {{k1[0], nl[0], temp[0]},
             {k1[1], nl[1], temp[1]},
             {k1[2], nl[2], temp[2]}};

    double Cb1k3tzT[3][3] =
            {{b1[0], b1[1], b1[2]},
             {k3[0], k3[1], k3[2]},
             {tz[0], tz[1], tz[2]}};

    double b3p[3];
    this->vect_scale((delta / k3b3), b3, b3p);

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
        this->mat_mult(Ck1nl, C13, temp_matrix);
        this->mat_mult(temp_matrix, Cb1k3tzT, R);

        solutions[0][i * 4 + 1] = R[0][0];
        solutions[1][i * 4 + 1] = R[1][0];
        solutions[2][i * 4 + 1] = R[2][0];
        solutions[0][i * 4 + 2] = R[0][1];
        solutions[1][i * 4 + 2] = R[1][1];
        solutions[2][i * 4 + 2] = R[2][1];
        solutions[0][i * 4 + 3] = R[0][2];
        solutions[1][i * 4 + 3] = R[1][2];
        solutions[2][i * 4 + 3] = R[2][2];

        // R * b3p
        double p[3] =
                {b3p[0] * R[0][0] + b3p[1] * R[0][1] + b3p[2] * R[0][2],
                 b3p[0] * R[1][0] + b3p[1] * R[1][1] + b3p[2] * R[1][2],
                 b3p[0] * R[2][0] + b3p[1] * R[2][1] + b3p[2] * R[2][2]};

        double pxstheta1p[3];
        this->vect_scale(stheta1p, p, pxstheta1p);

        this->vect_sub(w3, pxstheta1p, temp);

        solutions[0][i * 4] = temp[0];
        solutions[1][i * 4] = temp[1];
        solutions[2][i * 4] = temp[2];
    }

    return 0;
}

int P3p::solveQuartic(const double *factors, double *realRoots) {
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
    complex<double> w;
    if (q3 >= 0)
        w = -sqrt(static_cast<complex<double>>(q3 * q3 - p3 * p3 * p3)) - q3;
    else
        w = sqrt(static_cast<complex<double>>(q3 * q3 - p3 * p3 * p3)) - q3;
    if (w.imag() == 0.0) {
        w.real(cbrt(w.real()));
        t = 2.0 * (w.real() + p3 / w.real());
    } else {
        w = pow(w, 1.0 / 3);
        t = 4.0 * w.real();
    }

    complex<double> sqrt_2m = sqrt(static_cast<complex<double>>(-2 * p4 / 3 + t));
    double B_4A = -a3 / (4 * a4);
    double complex1 = 4 * p4 / 3 + t;
    complex<double> complex2 = 2 * q4 / sqrt_2m;
    double sqrt_2m_rh = sqrt_2m.real() / 2;
    double sqrt1 = sqrt(-(complex1 + complex2)).real() / 2;
    realRoots[0] = B_4A + sqrt_2m_rh + sqrt1;
    realRoots[1] = B_4A + sqrt_2m_rh - sqrt1;
    double sqrt2 = sqrt(-(complex1 - complex2)).real() / 2;
    realRoots[2] = B_4A - sqrt_2m_rh + sqrt2;
    realRoots[3] = B_4A - sqrt_2m_rh - sqrt2;

    return 0;
}

void P3p::polishQuarticRoots(const double *coeffs, double *roots) {
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

inline void P3p::vect_cross(const double *a, const double *b, double *result) {
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = -(a[0] * b[2] - a[2] * b[0]);
    result[2] = a[0] * b[1] - a[1] * b[0];
}

inline double P3p::vect_dot(const double *a, const double *b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline double P3p::vect_norm(const double *a) {
    return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

inline void P3p::vect_scale(const double s, const double *a, double *result) {
    result[0] = a[0] * s;
    result[1] = a[1] * s;
    result[2] = a[2] * s;
}

inline void P3p::vect_sub(const double *a, const double *b, double *result) {
    result[0] = a[0] - b[0];
    result[1] = a[1] - b[1];
    result[2] = a[2] - b[2];
}

inline void P3p::vect_divide(const double *a, const double d, double *result) {
    result[0] = a[0] / d;
    result[1] = a[1] / d;
    result[2] = a[2] / d;
}

inline void P3p::mat_mult(const double a[3][3], const double b[3][3], double result[3][3]) {
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

