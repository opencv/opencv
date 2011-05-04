#include "focal_estimators.hpp"
#include "util.hpp"

using namespace std;
using namespace cv;

void focalsFromHomography(const Mat& H, double &f0, double &f1, bool &f0_ok, bool &f1_ok)
{
    CV_Assert(H.type() == CV_64F && H.size() == Size(3, 3));

    const double h[9] =
    {
        H.at<double>(0, 0), H.at<double>(0, 1), H.at<double>(0, 2),
        H.at<double>(1, 0), H.at<double>(1, 1), H.at<double>(1, 2),
        H.at<double>(2, 0), H.at<double>(2, 1), H.at<double>(2, 2)
    };

    f1_ok = true;
    double denom1 = h[6] * h[7];
    double denom2 = (h[7] - h[6]) * (h[7] + h[6]);
    if (max(abs(denom1), abs(denom2)) < 1e-5)
        f1_ok = false;
    else
    {
        double val1 = -(h[0] * h[1] + h[3] * h[4]) / denom1;
        double val2 = (h[0] * h[0] + h[3] * h[3] - h[1] * h[1] - h[4] * h[4]) / denom2;
        if (val1 < val2)
            swap(val1, val2);
        if (val1 > 0 && val2 > 0)
            f1 = sqrt(abs(denom1) > abs(denom2) ? val1 : val2);
        else if (val1 > 0)
            f1 = sqrt(val1);
        else
            f1_ok = false;
    }

    f0_ok = true;
    denom1 = h[0] * h[3] + h[1] * h[4];
    denom2 = h[0] * h[0] + h[1] * h[1] - h[3] * h[3] - h[4] * h[4];
    if (max(abs(denom1), abs(denom2)) < 1e-5)
        f0_ok = false;
    else
    {
        double val1 = -h[2] * h[5] / denom1;
        double val2 = (h[5] * h[5] - h[2] * h[2]) / denom2;
        if (val1 < val2)
            swap(val1, val2);
        if (val1 > 0 && val2 > 0)
            f0 = sqrt(abs(denom1) > abs(denom2) ? val1 : val2);
        else if (val1 > 0)
            f0 = sqrt(val1);
        else
            f0_ok = false;
    }
}


bool focalsFromFundamental(const Mat &F, double &f0, double &f1)
{
    CV_Assert(F.type() == CV_64F);
    CV_Assert(F.size() == Size(3, 3));

    Mat Ft = F.t();
    Mat k = Mat::zeros(3, 1, CV_64F);
    k.at<double>(2, 0) = 1.f;

    // 1. Compute quantities
    double a = normL2sq(F*Ft*k) / normL2sq(Ft*k);
    double b = normL2sq(Ft*F*k) / normL2sq(F*k);
    double c = sqr(k.dot(F*k)) / (normL2sq(Ft*k) * normL2sq(F*k));
    double d = k.dot(F*Ft*F*k) / k.dot(F*k);
    double A = 1/c + a - 2*d;
    double B = 1/c + b - 2*d;
    double P = 2*(1/c - 2*d + 0.5*normL2sq(F));
    double Q = -(A + B)/c + 0.5*(normL2sq(F*Ft) - 0.5*sqr(normL2sq(F)));

    // 2. Solve quadratic equation Z*Z*a_ + Z*b_ + c_ = 0
    double a_ = 1 + c*P;
    double b_ = -(c*P*P + 2*P + 4*c*Q);
    double c_ = P*P + 4*c*P*Q + 12*A*B;
    double D = b_*b_ - 4*a_*c_;
    if (abs(D) < 1e-5)
        D = 0;
    else if (D < 0)
        return false;
    double D_sqrt = sqrt(D);
    double Z0 = (-b_ - D_sqrt) / (2*a_);
    double Z1 = (-b_ + D_sqrt) / (2*a_);

    // 3. Choose solution
    double w0 = abs(Z0*Z0*Z0 - 3*P*Z0*Z0 + 2*(P*P + 2*Q)*Z0 - 4*(P*Q + 4*A*B/c));
    double w1 = abs(Z1*Z1*Z1 - 3*P*Z1*Z1 + 2*(P*P + 2*Q)*Z1 - 4*(P*Q + 4*A*B/c));
    double Z = Z0;
    if (w1 < w0)
        Z = Z1;

    // 4.
    double X = -1/c*(1 + 2*B/(Z - P));
    double Y = -1/c*(1 + 2*A/(Z - P));

    // 5. Compute focal lengths
    f0 = 1/sqrt(1 + X/normL2sq(Ft*k));
    f1 = 1/sqrt(1 + Y/normL2sq(F*k));

    return true;
}
