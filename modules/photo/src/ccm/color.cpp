// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Author: Longbu Wang <wanglongbu@huawei.com.com>
//         Jinheng Zhang <zhangjinheng1@huawei.com>
//         Chenqi Shan <shanchenqi@huawei.com>

#include "color.hpp"

namespace cv {
namespace ccm {
Color::Color()
    : colors(Mat())
    , cs(std::make_shared<ColorSpaceBase>())
{}
Color::Color(Mat colors_, enum ColorSpace cs_)
    : colors(colors_)
    , cs(GetCS::getInstance().getCS(cs_))
{}

Color::Color(Mat colors_, enum ColorSpace cs_, Mat colored_)
    : colors(colors_)
    , cs(GetCS::getInstance().getCS(cs_))
    , colored(colored_)
{
    grays = ~colored;
}
Color::Color(Mat colors_, const ColorSpaceBase& cs_, Mat colored_)
    : colors(colors_)
    , cs(std::make_shared<ColorSpaceBase>(cs_))
    , colored(colored_)
{
    grays = ~colored;
}

Color::Color(Mat colors_, const ColorSpaceBase& cs_)
    : colors(colors_)
    , cs(std::make_shared<ColorSpaceBase>(cs_))
{}

Color::Color(Mat colors_, std::shared_ptr<ColorSpaceBase> cs_)
    : colors(colors_)
    , cs(cs_)
{}

Color Color::to(const ColorSpaceBase& other, CAM method, bool save)
{
    if (history.count(other) == 1)
    {
        return *history[other];
    }
    if (cs->relate(other))
    {
        return Color(cs->relation(other).run(colors), other);
    }
    Operations ops;
    ops.add(cs->to).add(XYZ(cs->io).cam(other.io, method)).add(other.from);
    std::shared_ptr<Color> color(new Color(ops.run(colors), other));
    if (save)
    {
        history[other] = color;
    }
    return *color;
}

Color Color::to(ColorSpace other, CAM method, bool save)
{
    return to(*GetCS::getInstance().getCS(other), method, save);
}

Mat Color::channel(Mat m, int i)
{
    Mat dchannels[3];
    split(m, dchannels);
    return dchannels[i];
}

Mat Color::toGray(IO io, CAM method, bool save)
{
    XYZ xyz = *XYZ::get(io);
    return channel(this->to(xyz, method, save).colors, 1);
}

Mat Color::toLuminant(IO io, CAM method, bool save)
{
    Lab lab = *Lab::get(io);
    return channel(this->to(lab, method, save).colors, 0);
}

Mat Color::diff(Color& other, DistanceType method)
{
    return diff(other, cs->io, method);
}

Mat Color::diff(Color& other, IO io, DistanceType method)
{
    Lab lab = *Lab::get(io);
    switch (method)
    {
    case cv::ccm::DISTANCE_CIE76:
    case cv::ccm::DISTANCE_CIE94_GRAPHIC_ARTS:
    case cv::ccm::DISTANCE_CIE94_TEXTILES:
    case cv::ccm::DISTANCE_CIE2000:
    case cv::ccm::DISTANCE_CMC_1TO1:
    case cv::ccm::DISTANCE_CMC_2TO1:
        return distance(to(lab).colors, other.to(lab).colors, method);
    case cv::ccm::DISTANCE_RGB:
        return distance(to(*cs->nl).colors, other.to(*cs->nl).colors, method);
    case cv::ccm::DISTANCE_RGBL:
        return distance(to(*cs->l).colors, other.to(*cs->l).colors, method);
    default:
        CV_Error(Error::StsBadArg, "Wrong method!" );
        break;
    }
}

void Color::getGray(double JDN)
{
    if (!grays.empty())
    {
        return;
    }
    Mat lab = to(COLOR_SPACE_LAB_D65_2).colors;
    Mat gray(colors.size(), colors.type());
    int fromto[] = { 0, 0, -1, 1, -1, 2 };
    mixChannels(&lab, 1, &gray, 1, fromto, 3);
    Mat d = distance(lab, gray, DISTANCE_CIE2000);
    this->grays = d < JDN;
    this->colored = ~grays;
}

Color Color::operator[](Mat mask)
{
    return Color(maskCopyTo(colors, mask), cs);
}

Mat GetColor::getColorChecker(const double* checker, int row)
{
    Mat res(row, 1, CV_64FC3);
    for (int i = 0; i < row; ++i)
    {
        res.at<Vec3d>(i, 0) = Vec3d(checker[3 * i], checker[3 * i + 1], checker[3 * i + 2]);
    }
    return res;
}

Mat GetColor::getColorCheckerMask(const uchar* checker, int row)
{
    Mat res(row, 1, CV_8U);
    for (int i = 0; i < row; ++i)
    {
        res.at<uchar>(i, 0) = checker[i];
    }
    return res;
}

Color GetColor::getColor(ColorCheckerType const_color)
{

    /** @brief Data is from https://www.imatest.com/wp-content/uploads/2011/11/Lab-data-Iluminate-D65-D50-spectro.xls
           see Miscellaneous.md for details.
*/
    static const double ColorChecker2005_LAB_D50_2[24][3] = { { 37.986, 13.555, 14.059 },
        { 65.711, 18.13, 17.81 },
        { 49.927, -4.88, -21.925 },
        { 43.139, -13.095, 21.905 },
        { 55.112, 8.844, -25.399 },
        { 70.719, -33.397, -0.199 },
        { 62.661, 36.067, 57.096 },
        { 40.02, 10.41, -45.964 },
        { 51.124, 48.239, 16.248 },
        { 30.325, 22.976, -21.587 },
        { 72.532, -23.709, 57.255 },
        { 71.941, 19.363, 67.857 },
        { 28.778, 14.179, -50.297 },
        { 55.261, -38.342, 31.37 },
        { 42.101, 53.378, 28.19 },
        { 81.733, 4.039, 79.819 },
        { 51.935, 49.986, -14.574 },
        { 51.038, -28.631, -28.638 },
        { 96.539, -0.425, 1.186 },
        { 81.257, -0.638, -0.335 },
        { 66.766, -0.734, -0.504 },
        { 50.867, -0.153, -0.27 },
        { 35.656, -0.421, -1.231 },
        { 20.461, -0.079, -0.973 } };

    static const uchar ColorChecker2005_COLORED_MASK[24] = { 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0 };
    static const double Vinyl_LAB_D50_2[18][3] = { { 100, 0.00520000001, -0.0104 },
        { 73.0833969, -0.819999993, -2.02099991 },
        { 62.493, 0.425999999, -2.23099995 },
        { 50.4640007, 0.446999997, -2.32399988 },
        { 37.7970009, 0.0359999985, -1.29700005 },
        { 0, 0, 0 },
        { 51.5880013, 73.5179977, 51.5690002 },
        { 93.6989975, -15.7340002, 91.9420013 },
        { 69.4079971, -46.5940018, 50.4869995 },
        { 66.61000060000001, -13.6789999, -43.1720009 },
        { 11.7110004, 16.9799995, -37.1759987 },
        { 51.973999, 81.9440002, -8.40699959 },
        { 40.5489998, 50.4399986, 24.8490009 },
        { 60.8160019, 26.0690002, 49.4420013 },
        { 52.2529984, -19.9500008, -23.9960003 },
        { 51.2859993, 48.4700012, -15.0579996 },
        { 68.70700069999999, 12.2959995, 16.2129993 },
        { 63.6839981, 10.2930002, 16.7639999 } };
    static const uchar Vinyl_COLORED_MASK[18] = { 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1 };
    static const double DigitalSG_LAB_D50_2[140][3] = { { 96.55, -0.91, 0.57 },
        { 6.43, -0.06, -0.41 },
        { 49.7, -0.18, 0.03 },
        { 96.5, -0.89, 0.59 },
        { 6.5, -0.06, -0.44 },
        { 49.66, -0.2, 0.01 },
        { 96.52, -0.91, 0.58 },
        { 6.49, -0.02, -0.28 },
        { 49.72, -0.2, 0.04 },
        { 96.43, -0.91, 0.67 },
        { 49.72, -0.19, 0 },
        { 32.6, 51.58, -10.85 },
        { 60.75, 26.22, -18.6 },
        { 28.69, 48.28, -39 },
        { 49.38, -15.43, -48.48 },
        { 60.63, -30.77, -26.23 },
        { 19.29, -26.37, -6.15 },
        { 60.15, -41.77, -12.6 },
        { 21.42, 1.67, 8.79 },
        { 49.69, -0.2, 0.01 },
        { 6.5, -0.03, -0.67 },
        { 21.82, 17.33, -18.35 },
        { 41.53, 18.48, -37.26 },
        { 19.99, -0.16, -36.29 },
        { 60.16, -18.45, -31.42 },
        { 19.94, -17.92, -20.96 },
        { 60.68, -6.05, -32.81 },
        { 50.81, -49.8, -9.63 },
        { 60.65, -39.77, 20.76 },
        { 6.53, -0.03, -0.43 },
        { 96.56, -0.91, 0.59 },
        { 84.19, -1.95, -8.23 },
        { 84.75, 14.55, 0.23 },
        { 84.87, -19.07, -0.82 },
        { 85.15, 13.48, 6.82 },
        { 84.17, -10.45, 26.78 },
        { 61.74, 31.06, 36.42 },
        { 64.37, 20.82, 18.92 },
        { 50.4, -53.22, 14.62 },
        { 96.51, -0.89, 0.65 },
        { 49.74, -0.19, 0.03 },
        { 31.91, 18.62, 21.99 },
        { 60.74, 38.66, 70.97 },
        { 19.35, 22.23, -58.86 },
        { 96.52, -0.91, 0.62 },
        { 6.66, 0, -0.3 },
        { 76.51, 20.81, 22.72 },
        { 72.79, 29.15, 24.18 },
        { 22.33, -20.7, 5.75 },
        { 49.7, -0.19, 0.01 },
        { 6.53, -0.05, -0.61 },
        { 63.42, 20.19, 19.22 },
        { 34.94, 11.64, -50.7 },
        { 52.03, -44.15, 39.04 },
        { 79.43, 0.29, -0.17 },
        { 30.67, -0.14, -0.53 },
        { 63.6, 14.44, 26.07 },
        { 64.37, 14.5, 17.05 },
        { 60.01, -44.33, 8.49 },
        { 6.63, -0.01, -0.47 },
        { 96.56, -0.93, 0.59 },
        { 46.37, -5.09, -24.46 },
        { 47.08, 52.97, 20.49 },
        { 36.04, 64.92, 38.51 },
        { 65.05, 0, -0.32 },
        { 40.14, -0.19, -0.38 },
        { 43.77, 16.46, 27.12 },
        { 64.39, 17, 16.59 },
        { 60.79, -29.74, 41.5 },
        { 96.48, -0.89, 0.64 },
        { 49.75, -0.21, 0.01 },
        { 38.18, -16.99, 30.87 },
        { 21.31, 29.14, -27.51 },
        { 80.57, 3.85, 89.61 },
        { 49.71, -0.2, 0.03 },
        { 60.27, 0.08, -0.41 },
        { 67.34, 14.45, 16.9 },
        { 64.69, 16.95, 18.57 },
        { 51.12, -49.31, 44.41 },
        { 49.7, -0.2, 0.02 },
        { 6.67, -0.05, -0.64 },
        { 51.56, 9.16, -26.88 },
        { 70.83, -24.26, 64.77 },
        { 48.06, 55.33, -15.61 },
        { 35.26, -0.09, -0.24 },
        { 75.16, 0.25, -0.2 },
        { 44.54, 26.27, 38.93 },
        { 35.91, 16.59, 26.46 },
        { 61.49, -52.73, 47.3 },
        { 6.59, -0.05, -0.5 },
        { 96.58, -0.9, 0.61 },
        { 68.93, -34.58, -0.34 },
        { 69.65, 20.09, 78.57 },
        { 47.79, -33.18, -30.21 },
        { 15.94, -0.42, -1.2 },
        { 89.02, -0.36, -0.48 },
        { 63.43, 25.44, 26.25 },
        { 65.75, 22.06, 27.82 },
        { 61.47, 17.1, 50.72 },
        { 96.53, -0.89, 0.66 },
        { 49.79, -0.2, 0.03 },
        { 85.17, 10.89, 17.26 },
        { 89.74, -16.52, 6.19 },
        { 84.55, 5.07, -6.12 },
        { 84.02, -13.87, -8.72 },
        { 70.76, 0.07, -0.35 },
        { 45.59, -0.05, 0.23 },
        { 20.3, 0.07, -0.32 },
        { 61.79, -13.41, 55.42 },
        { 49.72, -0.19, 0.02 },
        { 6.77, -0.05, -0.44 },
        { 21.85, 34.37, 7.83 },
        { 42.66, 67.43, 48.42 },
        { 60.33, 36.56, 3.56 },
        { 61.22, 36.61, 17.32 },
        { 62.07, 52.8, 77.14 },
        { 72.42, -9.82, 89.66 },
        { 62.03, 3.53, 57.01 },
        { 71.95, -27.34, 73.69 },
        { 6.59, -0.04, -0.45 },
        { 49.77, -0.19, 0.04 },
        { 41.84, 62.05, 10.01 },
        { 19.78, 29.16, -7.85 },
        { 39.56, 65.98, 33.71 },
        { 52.39, 68.33, 47.84 },
        { 81.23, 24.12, 87.51 },
        { 81.8, 6.78, 95.75 },
        { 71.72, -16.23, 76.28 },
        { 20.31, 14.45, 16.74 },
        { 49.68, -0.19, 0.05 },
        { 96.48, -0.88, 0.68 },
        { 49.69, -0.18, 0.03 },
        { 6.39, -0.04, -0.33 },
        { 96.54, -0.9, 0.67 },
        { 49.72, -0.18, 0.05 },
        { 6.49, -0.03, -0.41 },
        { 96.51, -0.9, 0.69 },
        { 49.7, -0.19, 0.07 },
        { 6.47, 0, -0.38 },
        { 96.46, -0.89, 0.7 } };

    switch (const_color)
    {

    case cv::ccm::COLORCHECKER_MACBETH:
    {
        Mat ColorChecker2005_LAB_D50_2_ = GetColor::getColorChecker(*ColorChecker2005_LAB_D50_2, 24);
        Mat ColorChecker2005_COLORED_MASK_ = GetColor::getColorCheckerMask(ColorChecker2005_COLORED_MASK, 24);
        Color Macbeth_D50_2 = Color(ColorChecker2005_LAB_D50_2_, COLOR_SPACE_LAB_D50_2, ColorChecker2005_COLORED_MASK_);
        return Macbeth_D50_2;
    }

    case cv::ccm::COLORCHECKER_VINYL:
    {
        Mat Vinyl_LAB_D50_2__ = GetColor::getColorChecker(*Vinyl_LAB_D50_2, 18);
        Mat Vinyl_COLORED_MASK__ = GetColor::getColorCheckerMask(Vinyl_COLORED_MASK, 18);
        Color Vinyl_D50_2 = Color(Vinyl_LAB_D50_2__, COLOR_SPACE_LAB_D50_2, Vinyl_COLORED_MASK__);
        return Vinyl_D50_2;
    }

    case cv::ccm::COLORCHECKER_DIGITAL_SG:
    {
        Mat DigitalSG_LAB_D50_2__ = GetColor::getColorChecker(*DigitalSG_LAB_D50_2, 140);
        Color DigitalSG_D50_2 = Color(DigitalSG_LAB_D50_2__, COLOR_SPACE_LAB_D50_2);
        return DigitalSG_D50_2;
    }
    }
    CV_Error(Error::StsNotImplemented, "");
}

}
}  // namespace cv::ccm
