// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Author: Longbu Wang <wanglongbu@huawei.com.com>
//         Jinheng Zhang <zhangjinheng1@huawei.com>
//         Chenqi Shan <shanchenqi@huawei.com>

#ifndef __OPENCV_CCM_COLORSPACE_HPP__
#define __OPENCV_CCM_COLORSPACE_HPP__

#include "operations.hpp"
#include "io.hpp"
#include "opencv2/photo.hpp"

namespace cv {
namespace ccm {

/** @brief Basic class for ColorSpace.
*/
class ColorSpaceBase
{
public:
    typedef std::function<Mat(Mat)> MatFunc;
    IO io;
    std::string type;
    bool linear;
    Operations to;
    Operations from;
    ColorSpaceBase* l;
    ColorSpaceBase* nl;

    ColorSpaceBase() {};

    ColorSpaceBase(IO io_, std::string type_, bool linear_)
        : io(io_)
        , type(type_)
        , linear(linear_) {};

    virtual ~ColorSpaceBase()
    {
        l = 0;
        nl = 0;
    };
    virtual bool relate(const ColorSpaceBase& other) const;

    virtual Operations relation(const ColorSpaceBase& /*other*/) const;

    bool operator<(const ColorSpaceBase& other) const;
};

/** @brief Base of RGB color space;
           the argument values are from AdobeRGB;
           Data from https://en.wikipedia.org/wiki/Adobe_RGB_color_space
*/

class RGBBase_ : public ColorSpaceBase
{
public:
    // primaries
    double xr;
    double yr;
    double xg;
    double yg;
    double xb;
    double yb;
    Mat M_to;
    Mat M_from;

    using ColorSpaceBase::ColorSpaceBase;

    /** @brief There are 3 kinds of relationships for RGB:
               1. Different types;    - no operation
               1. Same type, same linear; - copy
               2. Same type, different linear, self is nonlinear; - 2 toL
               3. Same type, different linear, self is linear - 3 fromL
        @param other type of ColorSpaceBase.
        @return Operations.
    */
    Operations relation(const ColorSpaceBase& other) const CV_OVERRIDE;

    /** @brief Initial operations.
    */
    void init();
    /** @brief Produce color space instance with linear and non-linear versions.
        @param rgbl type of RGBBase_.
    */
    void bind(RGBBase_& rgbl);

    virtual Mat toLFunc(Mat& /*rgb*/) const;

    virtual Mat fromLFunc(Mat& /*rgbl*/, Mat dst=Mat()) const;
private:
    virtual void setParameter() {};

    /** @brief Calculation of M_RGBL2XYZ_base.
    */
    virtual void calM();

    /** @brief operations to or from XYZ.
    */
    virtual void calOperations();

    virtual void calLinear() {};
};

/** @brief Base of Adobe RGB color space;
*/
class AdobeRGBBase_ : public RGBBase_

{
public:
    using RGBBase_::RGBBase_;
    double gamma;

private:
    Mat toLFunc(Mat& rgb) const CV_OVERRIDE;
    Mat fromLFunc(Mat& rgbl, Mat dst=Mat()) const CV_OVERRIDE;
};

/** @brief Base of sRGB color space;
*/
class sRGBBase_ : public RGBBase_

{
public:
    using RGBBase_::RGBBase_;
    double a;
    double gamma;
    double alpha;
    double beta;
    double phi;
    double K0;

private:
    /** @brief linearization parameters
    */
    virtual void calLinear() CV_OVERRIDE;
    /** @brief Used by toLFunc.
    */
    double toLFuncEW(double& x) const;

    /** @brief Linearization.
        @param rgb the input array, type of cv::Mat.
        @return the output array, type of cv::Mat.
    */
    Mat toLFunc(Mat& rgb) const CV_OVERRIDE;

    /** @brief Used by fromLFunc.
    */
    double fromLFuncEW(const double& x) const;

    /** @brief Delinearization.
        @param rgbl the input array, type of cv::Mat.
        @return the output array, type of cv::Mat.
    */
    Mat fromLFunc(Mat& rgbl, Mat dst=Mat()) const CV_OVERRIDE;
};

/** @brief sRGB color space.
           data from https://en.wikipedia.org/wiki/SRGB.
*/
class sRGB_ : public sRGBBase_

{
public:
    sRGB_(bool linear_)
        : sRGBBase_(IO::getIOs(D65_2), "sRGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE;
};

/** @brief Adobe RGB color space.
*/
class AdobeRGB_ : public AdobeRGBBase_
{
public:
    AdobeRGB_(bool linear_ = false)
        : AdobeRGBBase_(IO::getIOs(D65_2), "AdobeRGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE;
};

/** @brief Wide-gamut RGB color space.
           data from https://en.wikipedia.org/wiki/Wide-gamut_RGB_color_space.
*/
class WideGamutRGB_ : public AdobeRGBBase_
{
public:
    WideGamutRGB_(bool linear_ = false)
        : AdobeRGBBase_(IO::getIOs(D50_2), "WideGamutRGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE;
};

/** @brief ProPhoto RGB color space.
           data from https://en.wikipedia.org/wiki/ProPhoto_RGB_color_space.
*/

class ProPhotoRGB_ : public AdobeRGBBase_
{
public:
    ProPhotoRGB_(bool linear_ = false)
        : AdobeRGBBase_(IO::getIOs(D50_2), "ProPhotoRGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE;
};

/** @brief DCI-P3 RGB color space.
           data from https://en.wikipedia.org/wiki/DCI-P3.
*/
class DCI_P3_RGB_ : public AdobeRGBBase_
{
public:
    DCI_P3_RGB_(bool linear_ = false)
        : AdobeRGBBase_(IO::getIOs(D65_2), "DCI_P3_RGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE;
};

/** @brief Apple RGB color space.
           data from http://www.brucelindbloom.com/index.html?WorkingSpaceInfo.html.
*/
class AppleRGB_ : public AdobeRGBBase_
{
public:
    AppleRGB_(bool linear_ = false)
        : AdobeRGBBase_(IO::getIOs(D65_2), "AppleRGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE;
};

/** @brief REC_709 RGB color space.
           data from https://en.wikipedia.org/wiki/Rec._709.
*/
class REC_709_RGB_ : public sRGBBase_
{
public:
    REC_709_RGB_(bool linear_)
        : sRGBBase_(IO::getIOs(D65_2), "REC_709_RGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE;
};

/** @brief REC_2020 RGB color space.
           data from https://en.wikipedia.org/wiki/Rec._2020.
*/
class REC_2020_RGB_ : public sRGBBase_
{
public:
    REC_2020_RGB_(bool linear_)
        : sRGBBase_(IO::getIOs(D65_2), "REC_2020_RGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE;
};

/** @brief Enum of the possible types of CAMs.
*/
enum CAM
{
    IDENTITY,
    VON_KRIES,
    BRADFORD
};


/** @brief XYZ color space.
           Chromatic adaption matrices.
*/
class XYZ : public ColorSpaceBase
{
public:
    XYZ(IO io_)
        : ColorSpaceBase(io_, "XYZ", true) {};
    Operations cam(IO dio, CAM method = BRADFORD);
    static std::shared_ptr<XYZ> get(IO io);

private:
    /** @brief Get cam.
        @param sio the input IO of src.
        @param dio the input IO of dst.
        @param method type of CAM.
        @return the output array, type of cv::Mat.
    */
    Mat cam_(IO sio, IO dio, CAM method = BRADFORD) const;
};

/** @brief Lab color space.
*/
class Lab : public ColorSpaceBase
{
public:
    Lab(IO io_);
    static std::shared_ptr<Lab> get(IO io);

private:
    static constexpr double delta = (6. / 29.);
    static constexpr double m = 1. / (3. * delta * delta);
    static constexpr double t0 = delta * delta * delta;
    static constexpr double c = 4. / 29.;

    Vec3d fromxyz(Vec3d& xyz);

    /** @brief Calculate From.
        @param src the input array, type of cv::Mat.
        @return the output array, type of cv::Mat
    */
    Mat fromsrc(Mat& src);

    Vec3d tolab(Vec3d& lab);

    /** @brief Calculate To.
        @param src the input array, type of cv::Mat.
        @return the output array, type of cv::Mat
    */
    Mat tosrc(Mat& src);
};

class GetCS
{
protected:
    std::map<enum COLOR_SPACE, std::shared_ptr<ColorSpaceBase>> map_cs;

    GetCS();  // singleton, use getInstance()
public:
    static GetCS& getInstance();

    std::shared_ptr<RGBBase_> getRgb(enum COLOR_SPACE cs_name);
    std::shared_ptr<ColorSpaceBase> getCS(enum COLOR_SPACE cs_name);
};

}
}  // namespace cv::ccm

#endif