// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Author: Longbu Wang <wanglongbu@huawei.com.com>
//         Jinheng Zhang <zhangjinheng1@huawei.com>
//         Chenqi Shan <shanchenqi@huawei.com>

#include "colorspace.hpp"
#include "operations.hpp"
#include "illumobserver.hpp"

namespace cv {
namespace ccm {
static const std::vector<double>& getIlluminants(const IllumObserver& illumobserver)
{
    static const std::map<IllumObserver, std::vector<double>> illuminants = {
        { IllumObserver::getIllumObservers(A_2), { 1.098466069456375, 1, 0.3558228003436005 } },
        { IllumObserver::getIllumObservers(A_10), { 1.111420406956693, 1, 0.3519978321919493 } },
        { IllumObserver::getIllumObservers(D50_2), { 0.9642119944211994, 1, 0.8251882845188288 } },
        { IllumObserver::getIllumObservers(D50_10), { 0.9672062750333777, 1, 0.8142801513128616 } },
        { IllumObserver::getIllumObservers(D55_2), { 0.956797052643698, 1, 0.9214805860173273 } },
        { IllumObserver::getIllumObservers(D55_10), { 0.9579665682254781, 1, 0.9092525159847462 } },
        { IllumObserver::getIllumObservers(D65_2), { 0.95047, 1., 1.08883 } },
        { IllumObserver::getIllumObservers(D65_10), { 0.94811, 1., 1.07304 } },
        { IllumObserver::getIllumObservers(D75_2), { 0.9497220898840717, 1, 1.226393520724154 } },
        { IllumObserver::getIllumObservers(D75_10), { 0.9441713925645873, 1, 1.2064272211720228 } },
        { IllumObserver::getIllumObservers(E_2), { 1., 1., 1. } },
        { IllumObserver::getIllumObservers(E_10), { 1., 1., 1. } },
    };
    auto it = illuminants.find(illumobserver);
    CV_Assert(it != illuminants.end());
    return it->second;
};

/* @brief Basic class for ColorSpaceBase.
 */
bool ColorSpaceBase::relate(const ColorSpaceBase& other) const
{
    return (type == other.type) && (illumobserver == other.illumobserver);
};

Operations ColorSpaceBase::relation(const ColorSpaceBase& /*other*/) const
{
    return Operations::getIdentityOps();
}

bool ColorSpaceBase::operator<(const ColorSpaceBase& other) const
{
    return (illumobserver < other.illumobserver || (illumobserver == other.illumobserver && type < other.type) || (illumobserver == other.illumobserver && type == other.type && linear < other.linear));
}

/* @brief Base of RGB color space;
 *        the argument values are from AdobeRGB;
 *        Data from https://en.wikipedia.org/wiki/Adobe_RGB_color_space
 */
Operations RGBBase_::relation(const ColorSpaceBase& other) const
{
    if (linear == other.linear)
    {
        return Operations::getIdentityOps();
    }
    if (linear)
    {
        return Operations({ Operation([this](Mat rgbl) -> Mat { return fromLFunc(rgbl); }) });
    }
    return Operations({ Operation([this](Mat rgb) -> Mat { return toLFunc(rgb); })});
}

/* @brief Initial operations.
 */
void RGBBase_::init()
{
    setParameter();
    calLinear();
    calM();
    calOperations();
}

/* @brief Produce color space instance with linear and non-linear versions.
 * @param rgbl type of RGBBase_.
 */
void RGBBase_::bind(RGBBase_& rgbl)
{
    init();
    rgbl.init();
    l = &rgbl;
    rgbl.l = &rgbl;
    nl = this;
    rgbl.nl = this;
}

/* @brief Calculation of M_RGBL2XYZ_base.
 */
void RGBBase_::calM()
{
    Mat XYZr, XYZg, XYZb, XYZ_rgbl, Srgb;
    XYZr = Mat(xyY2XYZ({ xr, yr }), true);
    XYZg = Mat(xyY2XYZ({ xg, yg }), true);
    XYZb = Mat(xyY2XYZ({ xb, yb }), true);
    merge(std::vector<Mat> { XYZr, XYZg, XYZb }, XYZ_rgbl);
    XYZ_rgbl = XYZ_rgbl.reshape(1, (int)XYZ_rgbl.total());
    Mat XYZw = Mat(getIlluminants(illumobserver), true);
    XYZw = XYZw.reshape(1, (int)XYZw.total());
    solve(XYZ_rgbl, XYZw, Srgb);
    merge(std::vector<Mat> { Srgb.at<double>(0) * XYZr, Srgb.at<double>(1) * XYZg,
                  Srgb.at<double>(2) * XYZb },
            M_to);
    M_to = M_to.reshape(1, (int)M_to.total());
    M_from = M_to.inv();
};

/* @brief operations to or from XYZ.
 */
void RGBBase_::calOperations()
{
    if (linear)
    {
        to = Operations({ Operation(M_to.t()) });
        from = Operations({ Operation(M_from.t()) });
    }
    else
    {
        // rgb -> rgbl
        to = Operations({ Operation([this](Mat rgb) -> Mat { return toLFunc(rgb); }), Operation(M_to.t()) });
        // rgbl -> rgb
        from = Operations({ Operation(M_from.t()), Operation([this](Mat rgbl) -> Mat { return fromLFunc(rgbl); }) });
    }
}

Mat RGBBase_::toLFunc(Mat& /*rgb*/) const { return Mat(); }

Mat RGBBase_::fromLFunc(Mat& /*rgbl*/, Mat dst) const { return dst; }

/* @brief Base of Adobe RGB color space;
 */

Mat AdobeRGBBase_::toLFunc(Mat& rgb) const
{
    Mat out;
    gammaCorrection(rgb, out, gamma);
    return out;
}

Mat AdobeRGBBase_::fromLFunc(Mat& rgbl, Mat dst) const
{
    gammaCorrection(rgbl, dst, 1. / gamma);
    return dst;
}

/* @brief Base of sRGB color space;
 */

void sRGBBase_::calLinear()
{
    alpha = a + 1;
    K0 = a / (gamma - 1);
    phi = (pow(alpha, gamma) * pow(gamma - 1, gamma - 1)) / (pow(a, gamma - 1) * pow(gamma, gamma));
    beta = K0 / phi;
}

/* @brief Used by toLFunc.
 */
double sRGBBase_::toLFuncEW(double x) const
{
    if (x > K0)
    {
        return pow(((x + alpha - 1) / alpha), gamma);
    }
    else if (x >= -K0)
    {
        return x / phi;
    }
    else
    {
        return -(pow(((-x + alpha - 1) / alpha), gamma));
    }
}

/* @brief Linearization.
 * @param rgb the input array, type of cv::Mat.
 * @return the output array, type of cv::Mat.
 */
Mat sRGBBase_::toLFunc(Mat& rgb) const
{
    return elementWise(rgb,
            [this](double a_) -> double { return toLFuncEW(a_); });
}

/* @brief Used by fromLFunc.
 */
double sRGBBase_::fromLFuncEW(double x) const
{
    if (x > beta)
    {
        return alpha * pow(x, 1 / gamma) - (alpha - 1);
    }
    else if (x >= -beta)
    {
        return x * phi;
    }
    else
    {
        return -(alpha * pow(-x, 1 / gamma) - (alpha - 1));
    }
}

/* @brief Delinearization.
 * @param rgbl the input array, type of cv::Mat.
 * @return the output array, type of cv::Mat.
 */
Mat sRGBBase_::fromLFunc(Mat& rgbl, Mat dst) const
{
    return elementWise(rgbl, [this](double a_) -> double { return fromLFuncEW(a_); }, dst);
}

/* @brief sRGB color space.
 *        data from https://en.wikipedia.org/wiki/SRGB.
 */
void sRGB_::setParameter()
{
    xr = 0.64;
    yr = 0.33;
    xg = 0.3;
    yg = 0.6;
    xb = 0.15;
    yb = 0.06;
    a = 0.055;
    gamma = 2.4;
}

/* @brief Adobe RGB color space.
 */
void AdobeRGB_::setParameter()
{
    xr = 0.64;
    yr = 0.33;
    xg = 0.21;
    yg = 0.71;
    xb = 0.15;
    yb = 0.06;
    gamma = 2.2;
}

/* @brief Wide-gamut RGB color space.
 *        data from https://en.wikipedia.org/wiki/Wide-gamut_RGB_color_space.
 */
void WideGamutRGB_::setParameter()
{
    xr = 0.7347;
    yr = 0.2653;
    xg = 0.1152;
    yg = 0.8264;
    xb = 0.1566;
    yb = 0.0177;
    gamma = 2.2;
}

/* @brief ProPhoto RGB color space.
 *        data from https://en.wikipedia.org/wiki/ProPhoto_RGB_color_space.
 */
void ProPhotoRGB_::setParameter()
{
    xr = 0.734699;
    yr = 0.265301;
    xg = 0.159597;
    yg = 0.840403;
    xb = 0.036598;
    yb = 0.000105;
    gamma = 1.8;
}

/* @brief DCI-P3 RGB color space.
 *        data from https://en.wikipedia.org/wiki/DCI-P3.
 */

void DCI_P3_RGB_::setParameter()
{
    xr = 0.68;
    yr = 0.32;
    xg = 0.265;
    yg = 0.69;
    xb = 0.15;
    yb = 0.06;
    gamma = 2.2;
}

/* @brief Apple RGB color space.
 *        data from
 * http://www.brucelindbloom.com/index.html?WorkingSpaceInfo.html.
 */
void AppleRGB_::setParameter()
{
    xr = 0.625;
    yr = 0.34;
    xg = 0.28;
    yg = 0.595;
    xb = 0.155;
    yb = 0.07;
    gamma = 1.8;
}

/* @brief REC_709 RGB color space.
 *        data from https://en.wikipedia.org/wiki/Rec._709.
 */
void REC_709_RGB_::setParameter()
{
    xr = 0.64;
    yr = 0.33;
    xg = 0.3;
    yg = 0.6;
    xb = 0.15;
    yb = 0.06;
    a = 0.099;
    gamma = 1 / 0.45;
}

/* @brief REC_2020 RGB color space.
 *        data from https://en.wikipedia.org/wiki/Rec._2020.
 */

void REC_2020_RGB_::setParameter()
{
    xr = 0.708;
    yr = 0.292;
    xg = 0.17;
    yg = 0.797;
    xb = 0.131;
    yb = 0.046;
    a = 0.09929682680944;
    gamma = 1 / 0.45;
}

Operations XYZ::cam(IllumObserver dio, ChromaticAdaptationType method)
{
    return (illumobserver == dio) ? Operations()
                       : Operations({ Operation(cam_(illumobserver, dio, method).t()) });
}
Mat XYZ::cam_(IllumObserver sio, IllumObserver dio, ChromaticAdaptationType method) const
{
    static std::map<std::tuple<IllumObserver, IllumObserver, ChromaticAdaptationType>, Mat> cams;

    if (sio == dio)
    {
        return Mat::eye(cv::Size(3, 3), CV_64FC1);
    }
    if (cams.count(std::make_tuple(dio, sio, method)) == 1)
    {
        return cams[std::make_tuple(dio, sio, method)];
    }
    /* @brief XYZ color space.
    *        Chromatic adaption matrices.
    */

    static const Mat Von_Kries = (Mat_<double>(3, 3) << 0.40024, 0.7076, -0.08081, -0.2263, 1.16532, 0.0457, 0., 0., 0.91822);
    static const Mat Bradford = (Mat_<double>(3, 3) << 0.8951, 0.2664, -0.1614, -0.7502, 1.7135, 0.0367, 0.0389, -0.0685, 1.0296);
    static const std::map<ChromaticAdaptationType, std::vector<Mat>> MAs = {
        { IDENTITY, { Mat::eye(Size(3, 3), CV_64FC1), Mat::eye(Size(3, 3), CV_64FC1) } },
        { VON_KRIES, { Von_Kries, Von_Kries.inv() } },
        { BRADFORD, { Bradford, Bradford.inv() } }
    };

    // Function from http://www.brucelindbloom.com/index.html?ColorCheckerRGB.html.
    Mat XYZws = Mat(getIlluminants(dio));
    Mat XYZWd = Mat(getIlluminants(sio));
    XYZws = XYZws.reshape(1, (int)XYZws.total());
    XYZWd = XYZWd.reshape(1, (int)XYZWd.total());
    Mat MA = MAs.at(method)[0];
    Mat MA_inv = MAs.at(method)[1];
    Mat M = MA_inv * Mat::diag((MA * XYZws) / (MA * XYZWd)) * MA;
    cams[std::make_tuple(dio, sio, method)] = M;
    cams[std::make_tuple(sio, dio, method)] = M.inv();
    return M;
}

std::shared_ptr<XYZ> XYZ::get(IllumObserver illumobserver)
{
    static std::map<IllumObserver, std::shared_ptr<XYZ>> xyz_cs;

    if (xyz_cs.count(illumobserver) == 1)
    {
        return xyz_cs[illumobserver];
    }
    std::shared_ptr<XYZ> XYZ_CS = std::make_shared<XYZ>(illumobserver);
    xyz_cs[illumobserver] = XYZ_CS;
    return xyz_cs[illumobserver];
}

/* @brief Lab color space.
 */
Lab::Lab(IllumObserver illumobserver_)
    : ColorSpaceBase(illumobserver_, "Lab", true)
{
    to = { Operation([this](Mat src) -> Mat { return tosrc(src); }) };
    from = { Operation([this](Mat src) -> Mat { return fromsrc(src); }) };
}

Vec3d Lab::fromxyz(const Vec3d& xyz)
{
    auto& il = getIlluminants(illumobserver);
    double x = xyz[0] / il[0],
           y = xyz[1] / il[1],
           z = xyz[2] / il[2];
    auto f = [](double t) -> double {
        return t > T0 ? std::cbrt(t) : (M * t + C);
    };
    double fx = f(x), fy = f(y), fz = f(z);
    return { 116. * fy - 16., 500 * (fx - fy), 200 * (fy - fz) };
}

/* @brief Calculate From.
 * @param src the input array, type of cv::Mat.
 * @return the output array, type of cv::Mat
 */
Mat Lab::fromsrc(Mat& src)
{
    return channelWise(src,
            [this](cv::Vec3d a) -> cv::Vec3d { return fromxyz(a); });
}

Vec3d Lab::tolab(const Vec3d& lab)
{
    auto f_inv = [](double t) -> double {
        return t > DELTA ? pow(t, 3.0) : (t - C) / M;
    };
    double L = (lab[0] + 16.) / 116., a = lab[1] / 500., b = lab[2] / 200.;
    auto& il = getIlluminants(illumobserver);
    return { il[0] * f_inv(L + a),
        il[1] * f_inv(L),
        il[2] * f_inv(L - b) };
}

/* @brief Calculate To.
 * @param src the input array, type of cv::Mat.
 * @return the output array, type of cv::Mat
 */
Mat Lab::tosrc(Mat& src)
{
    return channelWise(src,
            [this](cv::Vec3d a) -> cv::Vec3d { return tolab(a); });
}

std::shared_ptr<Lab> Lab::get(IllumObserver illumobserver)
{
    static std::map<IllumObserver, std::shared_ptr<Lab>> 	lab_cs;

    if (lab_cs.count(illumobserver) == 1)
    {
        return lab_cs[illumobserver];
    }
    std::shared_ptr<Lab> Lab_CS(new Lab(illumobserver));
    lab_cs[illumobserver] = Lab_CS;
    return lab_cs[illumobserver];
}

GetCS::GetCS()
{
    // nothing
}

GetCS& GetCS::getInstance()
{
    static GetCS instance;
    return instance;
}

std::shared_ptr<RGBBase_> GetCS::getRgb(enum ColorSpace cs_name)
{
    switch (cs_name)
    {
    case cv::ccm::COLOR_SPACE_SRGB:
        if (map_cs.find(COLOR_SPACE_SRGB) == map_cs.end())
        {
            std::shared_ptr<sRGB_> sRGB_CS(new sRGB_(false));
            std::shared_ptr<sRGB_> sRGBL_CS(new sRGB_(true));
            (*sRGB_CS).bind(*sRGBL_CS);
            map_cs[COLOR_SPACE_SRGB] = sRGB_CS;
            map_cs[COLOR_SPACE_SRGBL] = sRGBL_CS;
        }
        return std::dynamic_pointer_cast<RGBBase_>(map_cs[COLOR_SPACE_SRGB]);

    case cv::ccm::COLOR_SPACE_ADOBE_RGB:
        if (map_cs.find(COLOR_SPACE_ADOBE_RGB) == map_cs.end())
        {
            std::shared_ptr<AdobeRGB_> AdobeRGB_CS(new AdobeRGB_(false));
            std::shared_ptr<AdobeRGB_> AdobeRGBL_CS(new AdobeRGB_(true));
            (*AdobeRGB_CS).bind(*AdobeRGBL_CS);
            map_cs[COLOR_SPACE_ADOBE_RGB] = AdobeRGB_CS;
            map_cs[COLOR_SPACE_ADOBE_RGBL] = AdobeRGBL_CS;
        }
        return std::dynamic_pointer_cast<RGBBase_>(map_cs[COLOR_SPACE_ADOBE_RGB]);

    case cv::ccm::COLOR_SPACE_WIDE_GAMUT_RGB:
    {
        if (map_cs.count(cs_name) < 1)
        {
            std::shared_ptr<WideGamutRGB_> WideGamutRGB_CS(new WideGamutRGB_(false));
            std::shared_ptr<WideGamutRGB_> WideGamutRGBL_CS(new WideGamutRGB_(true));
            (*WideGamutRGB_CS).bind(*WideGamutRGBL_CS);
            map_cs[COLOR_SPACE_WIDE_GAMUT_RGB] = WideGamutRGB_CS;
            map_cs[COLOR_SPACE_WIDE_GAMUT_RGBL] = WideGamutRGBL_CS;
        }
        break;
    }
    case cv::ccm::COLOR_SPACE_PRO_PHOTO_RGB:
    {
        if (map_cs.count(cs_name) < 1)
        {
            std::shared_ptr<ProPhotoRGB_> ProPhotoRGB_CS(new ProPhotoRGB_(false));
            std::shared_ptr<ProPhotoRGB_> ProPhotoRGBL_CS(new ProPhotoRGB_(true));
            (*ProPhotoRGB_CS).bind(*ProPhotoRGBL_CS);
            map_cs[COLOR_SPACE_PRO_PHOTO_RGB] = ProPhotoRGB_CS;
            map_cs[COLOR_SPACE_PRO_PHOTO_RGBL] = ProPhotoRGBL_CS;
        }
        break;
    }
    case cv::ccm::COLOR_SPACE_DCI_P3_RGB:
    {
        if (map_cs.count(cs_name) < 1)
        {
            std::shared_ptr<DCI_P3_RGB_> DCI_P3_RGB_CS(new DCI_P3_RGB_(false));
            std::shared_ptr<DCI_P3_RGB_> DCI_P3_RGBL_CS(new DCI_P3_RGB_(true));
            (*DCI_P3_RGB_CS).bind(*DCI_P3_RGBL_CS);
            map_cs[COLOR_SPACE_DCI_P3_RGB] = DCI_P3_RGB_CS;
            map_cs[COLOR_SPACE_DCI_P3_RGBL] = DCI_P3_RGBL_CS;
        }
        break;
    }
    case cv::ccm::COLOR_SPACE_APPLE_RGB:
    {
        if (map_cs.count(cs_name) < 1)
        {
            std::shared_ptr<AppleRGB_> AppleRGB_CS(new AppleRGB_(false));
            std::shared_ptr<AppleRGB_> AppleRGBL_CS(new AppleRGB_(true));
            (*AppleRGB_CS).bind(*AppleRGBL_CS);
            map_cs[COLOR_SPACE_APPLE_RGB] = AppleRGB_CS;
            map_cs[COLOR_SPACE_APPLE_RGBL] = AppleRGBL_CS;
        }
        break;
    }
    case cv::ccm::COLOR_SPACE_REC_709_RGB:
    {
        if (map_cs.count(cs_name) < 1)
        {
            std::shared_ptr<REC_709_RGB_> REC_709_RGB_CS(new REC_709_RGB_(false));
            std::shared_ptr<REC_709_RGB_> REC_709_RGBL_CS(new REC_709_RGB_(true));
            (*REC_709_RGB_CS).bind(*REC_709_RGBL_CS);
            map_cs[COLOR_SPACE_REC_709_RGB] = REC_709_RGB_CS;
            map_cs[COLOR_SPACE_REC_709_RGBL] = REC_709_RGBL_CS;
        }
        break;
    }
    case cv::ccm::COLOR_SPACE_REC_2020_RGB:
    {
        if (map_cs.count(cs_name) < 1)
        {
            std::shared_ptr<REC_2020_RGB_> REC_2020_RGB_CS(new REC_2020_RGB_(false));
            std::shared_ptr<REC_2020_RGB_> REC_2020_RGBL_CS(new REC_2020_RGB_(true));
            (*REC_2020_RGB_CS).bind(*REC_2020_RGBL_CS);
            map_cs[COLOR_SPACE_REC_2020_RGB] = REC_2020_RGB_CS;
            map_cs[COLOR_SPACE_REC_2020_RGBL] = REC_2020_RGBL_CS;
        }
        break;
    }
    case cv::ccm::COLOR_SPACE_SRGBL:
    case cv::ccm::COLOR_SPACE_ADOBE_RGBL:
    case cv::ccm::COLOR_SPACE_WIDE_GAMUT_RGBL:
    case cv::ccm::COLOR_SPACE_PRO_PHOTO_RGBL:
    case cv::ccm::COLOR_SPACE_DCI_P3_RGBL:
    case cv::ccm::COLOR_SPACE_APPLE_RGBL:
    case cv::ccm::COLOR_SPACE_REC_709_RGBL:
    case cv::ccm::COLOR_SPACE_REC_2020_RGBL:
        CV_Error(Error::StsBadArg, "linear RGB colorspaces are not supported, you should assigned as normal RGB color space");
        break;

    default:
        CV_Error(Error::StsBadArg, "Only RGB color spaces are supported");
    }
    return (std::dynamic_pointer_cast<RGBBase_>)(map_cs[cs_name]);
}

std::shared_ptr<ColorSpaceBase> GetCS::getCS(enum ColorSpace cs_name)
{
    switch (cs_name)
    {
    case cv::ccm::COLOR_SPACE_SRGB:
    case cv::ccm::COLOR_SPACE_SRGBL:
        if (map_cs.find(COLOR_SPACE_SRGB) == map_cs.end())
        {
            std::shared_ptr<sRGB_> sRGB_CS(new sRGB_(false));
            std::shared_ptr<sRGB_> sRGBL_CS(new sRGB_(true));
            (*sRGB_CS).bind(*sRGBL_CS);
            map_cs[COLOR_SPACE_SRGB] = sRGB_CS;
            map_cs[COLOR_SPACE_SRGBL] = sRGBL_CS;
        }
        return map_cs[cs_name];

    case cv::ccm::COLOR_SPACE_ADOBE_RGB:
    case cv::ccm::COLOR_SPACE_ADOBE_RGBL:
    {
        if (map_cs.count(cs_name) < 1)
        {
            std::shared_ptr<AdobeRGB_> AdobeRGB_CS(new AdobeRGB_(false));
            std::shared_ptr<AdobeRGB_> AdobeRGBL_CS(new AdobeRGB_(true));
            (*AdobeRGB_CS).bind(*AdobeRGBL_CS);
            map_cs[COLOR_SPACE_ADOBE_RGB] = AdobeRGB_CS;
            map_cs[COLOR_SPACE_ADOBE_RGBL] = AdobeRGBL_CS;
        }
        break;
    }
    case cv::ccm::COLOR_SPACE_WIDE_GAMUT_RGB:
    case cv::ccm::COLOR_SPACE_WIDE_GAMUT_RGBL:
    {
        if (map_cs.count(cs_name) < 1)
        {
            std::shared_ptr<WideGamutRGB_> WideGamutRGB_CS(new WideGamutRGB_(false));
            std::shared_ptr<WideGamutRGB_> WideGamutRGBL_CS(new WideGamutRGB_(true));
            (*WideGamutRGB_CS).bind(*WideGamutRGBL_CS);
            map_cs[COLOR_SPACE_WIDE_GAMUT_RGB] = WideGamutRGB_CS;
            map_cs[COLOR_SPACE_WIDE_GAMUT_RGBL] = WideGamutRGBL_CS;
        }
        break;
    }
    case cv::ccm::COLOR_SPACE_PRO_PHOTO_RGB:
    case cv::ccm::COLOR_SPACE_PRO_PHOTO_RGBL:
    {
        if (map_cs.count(cs_name) < 1)
        {
            std::shared_ptr<ProPhotoRGB_> ProPhotoRGB_CS(new ProPhotoRGB_(false));
            std::shared_ptr<ProPhotoRGB_> ProPhotoRGBL_CS(new ProPhotoRGB_(true));
            (*ProPhotoRGB_CS).bind(*ProPhotoRGBL_CS);
            map_cs[COLOR_SPACE_PRO_PHOTO_RGB] = ProPhotoRGB_CS;
            map_cs[COLOR_SPACE_PRO_PHOTO_RGBL] = ProPhotoRGBL_CS;
        }
        break;
    }
    case cv::ccm::COLOR_SPACE_DCI_P3_RGB:
    case cv::ccm::COLOR_SPACE_DCI_P3_RGBL:
    {
        if (map_cs.count(cs_name) < 1)
        {
            std::shared_ptr<DCI_P3_RGB_> DCI_P3_RGB_CS(new DCI_P3_RGB_(false));
            std::shared_ptr<DCI_P3_RGB_> DCI_P3_RGBL_CS(new DCI_P3_RGB_(true));
            (*DCI_P3_RGB_CS).bind(*DCI_P3_RGBL_CS);
            map_cs[COLOR_SPACE_DCI_P3_RGB] = DCI_P3_RGB_CS;
            map_cs[COLOR_SPACE_DCI_P3_RGBL] = DCI_P3_RGBL_CS;
        }
        break;
    }
    case cv::ccm::COLOR_SPACE_APPLE_RGB:
    case cv::ccm::COLOR_SPACE_APPLE_RGBL:
    {
        if (map_cs.count(cs_name) < 1)
        {
            std::shared_ptr<AppleRGB_> AppleRGB_CS(new AppleRGB_(false));
            std::shared_ptr<AppleRGB_> AppleRGBL_CS(new AppleRGB_(true));
            (*AppleRGB_CS).bind(*AppleRGBL_CS);
            map_cs[COLOR_SPACE_APPLE_RGB] = AppleRGB_CS;
            map_cs[COLOR_SPACE_APPLE_RGBL] = AppleRGBL_CS;
        }
        break;
    }
    case cv::ccm::COLOR_SPACE_REC_709_RGB:
    case cv::ccm::COLOR_SPACE_REC_709_RGBL:
    {
        if (map_cs.count(cs_name) < 1)
        {
            std::shared_ptr<REC_709_RGB_> REC_709_RGB_CS(new REC_709_RGB_(false));
            std::shared_ptr<REC_709_RGB_> REC_709_RGBL_CS(new REC_709_RGB_(true));
            (*REC_709_RGB_CS).bind(*REC_709_RGBL_CS);
            map_cs[COLOR_SPACE_REC_709_RGB] = REC_709_RGB_CS;
            map_cs[COLOR_SPACE_REC_709_RGBL] = REC_709_RGBL_CS;
        }
        break;
    }
    case cv::ccm::COLOR_SPACE_REC_2020_RGB:
    case cv::ccm::COLOR_SPACE_REC_2020_RGBL:
    {
        if (map_cs.count(cs_name) < 1)
        {
            std::shared_ptr<REC_2020_RGB_> REC_2020_RGB_CS(new REC_2020_RGB_(false));
            std::shared_ptr<REC_2020_RGB_> REC_2020_RGBL_CS(new REC_2020_RGB_(true));
            (*REC_2020_RGB_CS).bind(*REC_2020_RGBL_CS);
            map_cs[COLOR_SPACE_REC_2020_RGB] = REC_2020_RGB_CS;
            map_cs[COLOR_SPACE_REC_2020_RGBL] = REC_2020_RGBL_CS;
        }
        break;
    }
    case cv::ccm::COLOR_SPACE_XYZ_D65_2:
        return XYZ::get(IllumObserver::getIllumObservers(D65_2));
        break;
    case cv::ccm::COLOR_SPACE_XYZ_D50_2:
        return XYZ::get(IllumObserver::getIllumObservers(D50_2));
        break;
    case cv::ccm::COLOR_SPACE_XYZ_D65_10:
        return XYZ::get(IllumObserver::getIllumObservers(D65_10));
        break;
    case cv::ccm::COLOR_SPACE_XYZ_D50_10:
        return XYZ::get(IllumObserver::getIllumObservers(D50_10));
        break;
    case cv::ccm::COLOR_SPACE_XYZ_A_2:
        return XYZ::get(IllumObserver::getIllumObservers(A_2));
        break;
    case cv::ccm::COLOR_SPACE_XYZ_A_10:
        return XYZ::get(IllumObserver::getIllumObservers(A_10));
        break;
    case cv::ccm::COLOR_SPACE_XYZ_D55_2:
        return XYZ::get(IllumObserver::getIllumObservers(D55_2));
        break;
    case cv::ccm::COLOR_SPACE_XYZ_D55_10:
        return XYZ::get(IllumObserver::getIllumObservers(D55_10));
        break;
    case cv::ccm::COLOR_SPACE_XYZ_D75_2:
        return XYZ::get(IllumObserver::getIllumObservers(D75_2));
        break;
    case cv::ccm::COLOR_SPACE_XYZ_D75_10:
        return XYZ::get(IllumObserver::getIllumObservers(D75_10));
        break;
    case cv::ccm::COLOR_SPACE_XYZ_E_2:
        return XYZ::get(IllumObserver::getIllumObservers(E_2));
        break;
    case cv::ccm::COLOR_SPACE_XYZ_E_10:
        return XYZ::get(IllumObserver::getIllumObservers(E_10));
        break;
    case cv::ccm::COLOR_SPACE_LAB_D65_2:
        return Lab::get(IllumObserver::getIllumObservers(D65_2));
        break;
    case cv::ccm::COLOR_SPACE_LAB_D50_2:
        return Lab::get(IllumObserver::getIllumObservers(D50_2));
        break;
    case cv::ccm::COLOR_SPACE_LAB_D65_10:
        return Lab::get(IllumObserver::getIllumObservers(D65_10));
        break;
    case cv::ccm::COLOR_SPACE_LAB_D50_10:
        return Lab::get(IllumObserver::getIllumObservers(D50_10));
        break;
    case cv::ccm::COLOR_SPACE_LAB_A_2:
        return Lab::get(IllumObserver::getIllumObservers(A_2));
        break;
    case cv::ccm::COLOR_SPACE_LAB_A_10:
        return Lab::get(IllumObserver::getIllumObservers(A_10));
        break;
    case cv::ccm::COLOR_SPACE_LAB_D55_2:
        return Lab::get(IllumObserver::getIllumObservers(D55_2));
        break;
    case cv::ccm::COLOR_SPACE_LAB_D55_10:
        return Lab::get(IllumObserver::getIllumObservers(D55_10));
        break;
    case cv::ccm::COLOR_SPACE_LAB_D75_2:
        return Lab::get(IllumObserver::getIllumObservers(D75_2));
        break;
    case cv::ccm::COLOR_SPACE_LAB_D75_10:
        return Lab::get(IllumObserver::getIllumObservers(D75_10));
        break;
    case cv::ccm::COLOR_SPACE_LAB_E_2:
        return Lab::get(IllumObserver::getIllumObservers(E_2));
        break;
    case cv::ccm::COLOR_SPACE_LAB_E_10:
        return Lab::get(IllumObserver::getIllumObservers(E_10));
        break;
    default:
        break;
    }

    return map_cs[cs_name];
}

}
}  // namespace cv::ccm
