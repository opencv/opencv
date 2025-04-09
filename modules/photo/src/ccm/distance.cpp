// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Author: Longbu Wang <wanglongbu@huawei.com.com>
//         Jinheng Zhang <zhangjinheng1@huawei.com>
//         Chenqi Shan <shanchenqi@huawei.com>

#include "distance.hpp"

namespace cv {
namespace ccm {

double deltaCIE76(const Vec3d& lab1, const Vec3d& lab2) { return norm(lab1 - lab2); };

double deltaCIE94(const Vec3d& lab1, const Vec3d& lab2, const double& kH,
        const double& kC, const double& kL, const double& k1, const double& k2)
{
    double dl = lab1[0] - lab2[0];
    double c1 = sqrt(pow(lab1[1], 2) + pow(lab1[2], 2));
    double c2 = sqrt(pow(lab2[1], 2) + pow(lab2[2], 2));
    double dc = c1 - c2;
    double da = lab1[1] - lab2[1];
    double db = lab1[2] - lab2[2];
    double dh = pow(da, 2) + pow(db, 2) - pow(dc, 2);
    double sc = 1.0 + k1 * c1;
    double sh = 1.0 + k2 * c1;
    double sl = 1.0;
    double res = pow(dl / (kL * sl), 2) + pow(dc / (kC * sc), 2) + dh / pow(kH * sh, 2);

    return res > 0 ? sqrt(res) : 0;
}

double deltaCIE94GraphicArts(const Vec3d& lab1, const Vec3d& lab2)
{
    return deltaCIE94(lab1, lab2);
}

double toRad(const double& degree) { return degree / 180 * CV_PI; };

double deltaCIE94Textiles(const Vec3d& lab1, const Vec3d& lab2)
{
    return deltaCIE94(lab1, lab2, 1.0, 1.0, 2.0, 0.048, 0.014);
}

double deltaCIEDE2000_(const Vec3d& lab1, const Vec3d& lab2, const double& kL,
        const double& kC, const double& kH)
{
    double deltaLApo = lab2[0] - lab1[0];
    double lBarApo = (lab1[0] + lab2[0]) / 2.0;
    double C1 = sqrt(pow(lab1[1], 2) + pow(lab1[2], 2));
    double C2 = sqrt(pow(lab2[1], 2) + pow(lab2[2], 2));
    double cBar = (C1 + C2) / 2.0;
    double G = sqrt(pow(cBar, 7) / (pow(cBar, 7) + pow(25, 7)));
    double a1Apo = lab1[1] + lab1[1] / 2.0 * (1.0 - G);
    double a2Apo = lab2[1] + lab2[1] / 2.0 * (1.0 - G);
    double c1Apo = sqrt(pow(a1Apo, 2) + pow(lab1[2], 2));
    double c2Apo = sqrt(pow(a2Apo, 2) + pow(lab2[2], 2));
    double cBarApo = (c1Apo + c2Apo) / 2.0;
    double deltaCApo = c2Apo - c1Apo;

    double h1Apo;
    if (c1Apo == 0)
    {
        h1Apo = 0.0;
    }
    else
    {
        h1Apo = atan2(lab1[2], a1Apo);
        if (h1Apo < 0.0)
            h1Apo += 2. * CV_PI;
    }

    double h2Apo;
    if (c2Apo == 0)
    {
        h2Apo = 0.0;
    }
    else
    {
        h2Apo = atan2(lab2[2], a2Apo);
        if (h2Apo < 0.0)
            h2Apo += 2. * CV_PI;
    }

    double deltaHApo;
    if (abs(h2Apo - h1Apo) <= CV_PI)
    {
        deltaHApo = h2Apo - h1Apo;
    }
    else if (h2Apo <= h1Apo)
    {
        deltaHApo = h2Apo - h1Apo + 2. * CV_PI;
    }
    else
    {
        deltaHApo = h2Apo - h1Apo - 2. * CV_PI;
    }

    double hBarApo;
    if (c1Apo == 0 || c2Apo == 0)
    {
        hBarApo = h1Apo + h2Apo;
    }
    else if (abs(h1Apo - h2Apo) <= CV_PI)
    {
        hBarApo = (h1Apo + h2Apo) / 2.0;
    }
    else if (h1Apo + h2Apo < 2. * CV_PI)
    {
        hBarApo = (h1Apo + h2Apo + 2. * CV_PI) / 2.0;
    }
    else
    {
        hBarApo = (h1Apo + h2Apo - 2. * CV_PI) / 2.0;
    }

    double deltaH_Apo = 2.0 * sqrt(c1Apo * c2Apo) * sin(deltaHApo / 2.0);
    double T = 1.0 - 0.17 * cos(hBarApo - toRad(30.)) + 0.24 * cos(2.0 * hBarApo) + 0.32 * cos(3.0 * hBarApo + toRad(6.0)) - 0.2 * cos(4.0 * hBarApo - toRad(63.0));
    double sC = 1.0 + 0.045 * cBarApo;
    double sH = 1.0 + 0.015 * cBarApo * T;
    double sL = 1.0 + ((0.015 * pow(lBarApo - 50.0, 2.0)) / sqrt(20.0 + pow(lBarApo - 50.0, 2.0)));
    double rC = 2.0 * sqrt(pow(cBarApo, 7.0) / (pow(cBarApo, 7.0) + pow(25, 7)));
    double rT = -sin(toRad(60.0) * exp(-pow((hBarApo - toRad(275.0)) / toRad(25.0), 2.0))) * rC;
    double res = (pow(deltaLApo / (kL * sL), 2.0) + pow(deltaCApo / (kC * sC), 2.0) + pow(deltaH_Apo / (kH * sH), 2.0) + rT * (deltaCApo / (kC * sC)) * (deltaH_Apo / (kH * sH)));
    return res > 0 ? sqrt(res) : 0;
}

double deltaCIEDE2000(const Vec3d& lab1, const Vec3d& lab2)
{
    return deltaCIEDE2000_(lab1, lab2);
}

double deltaCMC(const Vec3d& lab1, const Vec3d& lab2, const double& kL, const double& kC)
{
    double dL = lab2[0] - lab1[0];
    double da = lab2[1] - lab1[1];
    double db = lab2[2] - lab1[2];
    double C1 = sqrt(pow(lab1[1], 2.0) + pow(lab1[2], 2.0));
    double C2 = sqrt(pow(lab2[1], 2.0) + pow(lab2[2], 2.0));
    double dC = C2 - C1;
    double dH = sqrt(pow(da, 2) + pow(db, 2) - pow(dC, 2));

    double H1;
    if (C1 == 0.)
    {
        H1 = 0.0;
    }
    else
    {
        H1 = atan2(lab1[2], lab1[1]);
        if (H1 < 0.0)
            H1 += 2. * CV_PI;
    }

    double F = pow(C1, 2) / sqrt(pow(C1, 4) + 1900);
    double T = (H1 > toRad(164) && H1 <= toRad(345))
            ? 0.56 + abs(0.2 * cos(H1 + toRad(168)))
            : 0.36 + abs(0.4 * cos(H1 + toRad(35)));
    double sL = lab1[0] < 16. ? 0.511 : (0.040975 * lab1[0]) / (1.0 + 0.01765 * lab1[0]);
    double sC = (0.0638 * C1) / (1.0 + 0.0131 * C1) + 0.638;
    double sH = sC * (F * T + 1.0 - F);

    return sqrt(pow(dL / (kL * sL), 2.0) + pow(dC / (kC * sC), 2.0) + pow(dH / sH, 2.0));
}

double deltaCMC1To1(const Vec3d& lab1, const Vec3d& lab2)
{
    return deltaCMC(lab1, lab2);
}

double deltaCMC2To1(const Vec3d& lab1, const Vec3d& lab2)
{
    return deltaCMC(lab1, lab2, 2, 1);
}

Mat distance(Mat src, Mat ref, DistanceType distanceType)
{
    switch (distanceType)
    {
    case cv::ccm::DISTANCE_CIE76:
        return distanceWise(src, ref, deltaCIE76);
    case cv::ccm::DISTANCE_CIE94_GRAPHIC_ARTS:
        return distanceWise(src, ref, deltaCIE94GraphicArts);
    case cv::ccm::DISTANCE_CIE94_TEXTILES:
        return distanceWise(src, ref, deltaCIE94Textiles);
    case cv::ccm::DISTANCE_CIE2000:
        return distanceWise(src, ref, deltaCIEDE2000);
    case cv::ccm::DISTANCE_CMC_1TO1:
        return distanceWise(src, ref, deltaCMC1To1);
    case cv::ccm::DISTANCE_CMC_2TO1:
        return distanceWise(src, ref, deltaCMC2To1);
    case cv::ccm::DISTANCE_RGB:
        return distanceWise(src, ref, deltaCIE76);
    case cv::ccm::DISTANCE_RGBL:
        return distanceWise(src, ref, deltaCIE76);
    default:
        CV_Error(Error::StsBadArg, "Wrong distanceType!" );
        break;
    }
};

}
}  // namespace ccm