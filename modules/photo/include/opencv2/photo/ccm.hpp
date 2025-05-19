// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Author: Longbu Wang <wanglongbu@huawei.com.com>
//         Jinheng Zhang <zhangjinheng1@huawei.com>
//         Chenqi Shan <shanchenqi@huawei.com>

#ifndef OPENCV_PHOTO_CCM_HPP
#define OPENCV_PHOTO_CCM_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace cv
{
namespace ccm
{

/** @defgroup ccm Color Correction module
@{
*/

/** @brief Enum of the possible types of ccm.
*/
enum CcmType
{
    CCM_LINEAR,   ///< Uses a \f$3\times3\f$ matrix to linearly transform RGB values without offsets.
    CCM_AFFINE,   ///< Uses a \f$4\times3\f$ matrix to affine transform RGB values with both scaling and offset terms.
};

/** @brief Enum of the possible types of initial method.
*/
enum InitialMethodType
{
    INITIAL_METHOD_WHITE_BALANCE,      ///< The white balance method. The initial value is:\n
                                       /// \f$
                                       /// M_{CCM}=
                                       /// \begin{bmatrix}
                                       /// k_R & 0 & 0\\
                                       /// 0 & k_G & 0\\
                                       /// 0 & 0 & k_B\\
                                       /// \end{bmatrix}
                                       /// \f$\n
                                       /// where\n
                                       /// \f$
                                       /// k_R=mean(R_{li}')/mean(R_{li})\\
                                       /// k_G=mean(G_{li}')/mean(G_{li})\\
                                       /// k_B=mean(B_{li}')/mean(B_{li})
                                       /// \f$
    INITIAL_METHOD_LEAST_SQUARE,       ///< The least square method is an optimal solution under the linear RGB distance function
};
/** @brief  Macbeth and Vinyl ColorChecker with 2deg D50
*/
enum ColorCheckerType {
    COLORCHECKER_MACBETH,                ///< Macbeth ColorChecker
    COLORCHECKER_VINYL,                  ///< DKK ColorChecker
    COLORCHECKER_DIGITAL_SG,              ///< DigitalSG ColorChecker with 140 squares
};
enum ColorSpace {
    COLOR_SPACE_SRGB,                       ///< https://en.wikipedia.org/wiki/SRGB , RGB color space
    COLOR_SPACE_SRGBL,                      ///< https://en.wikipedia.org/wiki/SRGB , linear RGB color space
    COLOR_SPACE_ADOBE_RGB,                  ///< https://en.wikipedia.org/wiki/Adobe_RGB_color_space , RGB color space
    COLOR_SPACE_ADOBE_RGBL,                 ///< https://en.wikipedia.org/wiki/Adobe_RGB_color_space , linear RGB color space
    COLOR_SPACE_WIDE_GAMUT_RGB,             ///< https://en.wikipedia.org/wiki/Wide-gamut_RGB_color_space , RGB color space
    COLOR_SPACE_WIDE_GAMUT_RGBL,            ///< https://en.wikipedia.org/wiki/Wide-gamut_RGB_color_space , linear RGB color space
    COLOR_SPACE_PRO_PHOTO_RGB,              ///< https://en.wikipedia.org/wiki/ProPhoto_RGB_color_space , RGB color space
    COLOR_SPACE_PRO_PHOTO_RGBL,             ///< https://en.wikipedia.org/wiki/ProPhoto_RGB_color_space , linear RGB color space
    COLOR_SPACE_DCI_P3_RGB,                 ///< https://en.wikipedia.org/wiki/DCI-P3 , RGB color space
    COLOR_SPACE_DCI_P3_RGBL,                ///< https://en.wikipedia.org/wiki/DCI-P3 , linear RGB color space
    COLOR_SPACE_APPLE_RGB,                  ///< http://www.brucelindbloom.com/index.html?WorkingSpaceInfo.html , RGB color space
    COLOR_SPACE_APPLE_RGBL,                 ///< http://www.brucelindbloom.com/index.html?WorkingSpaceInfo.html , linear RGB color space
    COLOR_SPACE_REC_709_RGB,                ///< https://en.wikipedia.org/wiki/Rec._709 , RGB color space
    COLOR_SPACE_REC_709_RGBL,               ///< https://en.wikipedia.org/wiki/Rec._709 , linear RGB color space
    COLOR_SPACE_REC_2020_RGB,               ///< https://en.wikipedia.org/wiki/Rec._2020 , RGB color space
    COLOR_SPACE_REC_2020_RGBL,              ///< https://en.wikipedia.org/wiki/Rec._2020 , linear RGB color space
    COLOR_SPACE_XYZ_D65_2,                  ///< https://en.wikipedia.org/wiki/CIE_1931_color_space , XYZ color space, D65 illuminant, 2 degree
    COLOR_SPACE_XYZ_D50_2,                  ///< https://en.wikipedia.org/wiki/CIE_1931_color_space , XYZ color space, D50 illuminant, 2 degree
    COLOR_SPACE_XYZ_D65_10,                 ///< https://en.wikipedia.org/wiki/CIE_1931_color_space , XYZ color space, D65 illuminant, 10 degree
    COLOR_SPACE_XYZ_D50_10,                 ///< https://en.wikipedia.org/wiki/CIE_1931_color_space , XYZ color space, D50 illuminant, 10 degree
    COLOR_SPACE_XYZ_A_2,                    ///< https://en.wikipedia.org/wiki/CIE_1931_color_space , XYZ color space, A illuminant, 2 degree
    COLOR_SPACE_XYZ_A_10,                   ///< https://en.wikipedia.org/wiki/CIE_1931_color_space , XYZ color space, A illuminant, 10 degree
    COLOR_SPACE_XYZ_D55_2,                  ///< https://en.wikipedia.org/wiki/CIE_1931_color_space , XYZ color space, D55 illuminant, 2 degree
    COLOR_SPACE_XYZ_D55_10,                 ///< https://en.wikipedia.org/wiki/CIE_1931_color_space , XYZ color space, D55 illuminant, 10 degree
    COLOR_SPACE_XYZ_D75_2,                  ///< https://en.wikipedia.org/wiki/CIE_1931_color_space , XYZ color space, D75 illuminant, 2 degree
    COLOR_SPACE_XYZ_D75_10,                 ///< https://en.wikipedia.org/wiki/CIE_1931_color_space , XYZ color space, D75 illuminant, 10 degree
    COLOR_SPACE_XYZ_E_2,                    ///< https://en.wikipedia.org/wiki/CIE_1931_color_space , XYZ color space, E illuminant, 2 degree
    COLOR_SPACE_XYZ_E_10,                   ///< https://en.wikipedia.org/wiki/CIE_1931_color_space , XYZ color space, E illuminant, 10 degree
    COLOR_SPACE_LAB_D65_2,                  ///< https://en.wikipedia.org/wiki/CIELAB_color_space , Lab color space, D65 illuminant, 2 degree
    COLOR_SPACE_LAB_D50_2,                  ///< https://en.wikipedia.org/wiki/CIELAB_color_space , Lab color space, D50 illuminant, 2 degree
    COLOR_SPACE_LAB_D65_10,                 ///< https://en.wikipedia.org/wiki/CIELAB_color_space , Lab color space, D65 illuminant, 10 degree
    COLOR_SPACE_LAB_D50_10,                 ///< https://en.wikipedia.org/wiki/CIELAB_color_space , Lab color space, D50 illuminant, 10 degree
    COLOR_SPACE_LAB_A_2,                    ///< https://en.wikipedia.org/wiki/CIELAB_color_space , Lab color space, A illuminant, 2 degree
    COLOR_SPACE_LAB_A_10,                   ///< https://en.wikipedia.org/wiki/CIELAB_color_space , Lab color space, A illuminant, 10 degree
    COLOR_SPACE_LAB_D55_2,                  ///< https://en.wikipedia.org/wiki/CIELAB_color_space , Lab color space, D55 illuminant, 2 degree
    COLOR_SPACE_LAB_D55_10,                 ///< https://en.wikipedia.org/wiki/CIELAB_color_space , Lab color space, D55 illuminant, 10 degree
    COLOR_SPACE_LAB_D75_2,                  ///< https://en.wikipedia.org/wiki/CIELAB_color_space , Lab color space, D75 illuminant, 2 degree
    COLOR_SPACE_LAB_D75_10,                 ///< https://en.wikipedia.org/wiki/CIELAB_color_space , Lab color space, D75 illuminant, 10 degree
    COLOR_SPACE_LAB_E_2,                    ///< https://en.wikipedia.org/wiki/CIELAB_color_space , Lab color space, E illuminant, 2 degree
    COLOR_SPACE_LAB_E_10                    ///< https://en.wikipedia.org/wiki/CIELAB_color_space , Lab color space, E illuminant, 10 degree
};

/** @brief Linearization transformation type
*/
enum LinearizationType
{

    LINEARIZATION_IDENTITY,                  ///<no change is made
    LINEARIZATION_GAMMA,                      ///<gamma correction; Need assign a value to gamma simultaneously
    LINEARIZATION_COLORPOLYFIT,               ///<polynomial fitting channels respectively; Need assign a value to deg simultaneously
    LINEARIZATION_COLORLOGPOLYFIT,            ///<logarithmic polynomial fitting channels respectively; Need assign a value to deg simultaneously
    LINEARIZATION_GRAYPOLYFIT,                ///<grayscale polynomial fitting; Need assign a value to deg and dst_whites simultaneously
    LINEARIZATION_GRAYLOGPOLYFIT              ///<grayscale Logarithmic polynomial fitting;  Need assign a value to deg and dst_whites simultaneously
};

/** @brief Enum of possible functions to calculate the distance between colors.

See https://en.wikipedia.org/wiki/Color_difference for details
*/
enum DistanceType
{
    DISTANCE_CIE76,                      ///<The 1976 formula is the first formula that related a measured color difference to a known set of CIELAB coordinates.
    DISTANCE_CIE94_GRAPHIC_ARTS,         ///<The 1976 definition was extended to address perceptual non-uniformities.
    DISTANCE_CIE94_TEXTILES,
    DISTANCE_CIE2000,
    DISTANCE_CMC_1TO1,                   ///<In 1984, the Colour Measurement Committee of the Society of Dyers and Colourists defined a difference measure, also based on the L*C*h color model.
    DISTANCE_CMC_2TO1,
    DISTANCE_RGB,                        ///<Euclidean distance of rgb color space
    DISTANCE_RGBL                        ///<Euclidean distance of rgbl color space
};

/**
 * @brief Applies gamma correction to the input image.
 * @param src Input image.
 * @param dst Output image.
 * @param gamma Gamma correction greater than zero.
 */
CV_EXPORTS_W void gammaCorrection(InputArray src, OutputArray dst, double gamma);

/** @brief Core class of ccm model

Produce a ColorCorrectionModel instance for inference
*/
class CV_EXPORTS_W ColorCorrectionModel
{
public:
    CV_WRAP ColorCorrectionModel();

    /** @brief Color Correction Model

        Supported list of color cards:
        - @ref COLORCHECKER_MACBETH, the Macbeth ColorChecker
        - @ref COLORCHECKER_VINYL, the DKK ColorChecker
        - @ref COLORCHECKER_DIGITAL_SG, the DigitalSG ColorChecker with 140 squares

        @param src detected colors of ColorChecker patches;
                    the color type is RGB not BGR, and the color values are in [0, 1];
        @param constColor the Built-in color card
    */
    CV_WRAP ColorCorrectionModel(InputArray src, int constColor);

    /** @brief Color Correction Model
        @param src detected colors of ColorChecker patches;
                the color type is RGB not BGR, and the color values are in [0, 1];
        @param colors the reference color values, the color values are in [0, 1].
        @param refColorSpace the corresponding color space
                If the color type is some RGB, the format is RGB not BGR;
    */
    CV_WRAP ColorCorrectionModel(InputArray src, InputArray colors, ColorSpace refColorSpace);

    /** @brief Color Correction Model
        @param src detected colors of ColorChecker patches;
                    the color type is RGB not BGR, and the color values are in [0, 1];
        @param colors the reference color values, the color values are in [0, 1].
        @param refColorSpace the corresponding color space
                    If the color type is some RGB, the format is RGB not BGR;
        @param coloredPatchesMask binary mask indicating which patches are colored (non-gray) patches
    */
    CV_WRAP ColorCorrectionModel(InputArray src, InputArray colors, ColorSpace refColorSpace, InputArray coloredPatchesMask);

    /** @brief set ColorSpace
        @note It should be some RGB color space;
        Supported list of color cards:
        - @ref COLOR_SPACE_SRGB
        - @ref COLOR_SPACE_ADOBE_RGB
        - @ref COLOR_SPACE_WIDE_GAMUT_RGB
        - @ref COLOR_SPACE_PRO_PHOTO_RGB
        - @ref COLOR_SPACE_DCI_P3_RGB
        - @ref COLOR_SPACE_APPLE_RGB
        - @ref COLOR_SPACE_REC_709_RGB
        - @ref COLOR_SPACE_REC_2020_RGB
        @param cs the absolute color space that detected colors convert to;
              default: @ref COLOR_SPACE_SRGB
    */
    CV_WRAP void setColorSpace(ColorSpace cs);

    /** @brief set ccmType
    @param ccmType the shape of color correction matrix(CCM);
                    default: @ref CCM_LINEAR
    */
    CV_WRAP void setCcmType(CcmType ccmType);

    /** @brief set Distance
    @param distance the type of color distance;
                    default: @ref DISTANCE_CIE2000
    */
    CV_WRAP void setDistance(DistanceType distance);

    /** @brief set Linear
    @param linearizationType the method of linearization;
                       default: @ref LINEARIZATION_GAMMA
    */
    CV_WRAP void setLinearization(LinearizationType linearizationType);

    /** @brief set Gamma

    @note only valid when linear is set to "gamma";

    @param gamma the gamma value of gamma correction;
                 default: 2.2;
    */
    CV_WRAP void setLinearizationGamma(double gamma);

    /** @brief set degree
        @note only valid when linear is set to
        - @ref LINEARIZATION_COLORPOLYFIT
        - @ref LINEARIZATION_GRAYPOLYFIT
        - @ref LINEARIZATION_COLORLOGPOLYFIT
        - @ref LINEARIZATION_GRAYLOGPOLYFIT

        @param deg the degree of linearization polynomial
            default: 3

    */
    CV_WRAP void setLinearizationDegree(int deg);

    /** @brief set SaturatedThreshold.
                The colors in the closed interval [lower, upper] are reserved to participate
                in the calculation of the loss function and initialization parameters
        @param lower the lower threshold to determine saturation;
                default: 0;
        @param upper the upper threshold to determine saturation;
                default: 0
    */
    CV_WRAP void setSaturatedThreshold(double lower, double upper);

    /** @brief set WeightsList
    @param weightsList the list of weight of each color;
                        default: empty array
    */
    CV_WRAP void setWeightsList(const Mat& weightsList);

    /** @brief set WeightCoeff
    @param weightsCoeff the exponent number of L* component of the reference color in CIE Lab color space;
                         default: 0
    */
    CV_WRAP void setWeightCoeff(double weightsCoeff);

    /** @brief set InitialMethod
    @param initialMethodType the method of calculating CCM initial value;
            default: INITIAL_METHOD_LEAST_SQUARE
    */
    CV_WRAP void setInitialMethod(InitialMethodType initialMethodType);

    /** @brief set MaxCount
    @param maxCount used in MinProblemSolver-DownhillSolver;
        Terminal criteria to the algorithm;
                     default: 5000;
    */
    CV_WRAP void setMaxCount(int maxCount);

    /** @brief set Epsilon
    @param epsilon used in MinProblemSolver-DownhillSolver;
        Terminal criteria to the algorithm;
                   default: 1e-4;
    */
    CV_WRAP void setEpsilon(double epsilon);

    /** @brief Set whether the input image is in RGB color space
    @param rgb If true, the model expects input images in RGB format.
                 If false, input is assumed to be in BGR (default).
    */
    CV_WRAP void setRGB(bool rgb);

    /** @brief make color correction */
    CV_WRAP Mat compute();

    CV_WRAP Mat getColorCorrectionMatrix() const;
    CV_WRAP double getLoss() const;
    CV_WRAP Mat getSrcLinearRGB() const;
    CV_WRAP Mat getRefLinearRGB() const;
    CV_WRAP Mat getMask() const;
    CV_WRAP Mat getWeights() const;

    /** @brief Applies color correction to the input image using a fitted color correction matrix.
     *
     * The conventional ranges for R, G, and B channel values are:
     -   0 to 255 for CV_8U images
     -   0 to 65535 for CV_16U images
     -   0 to 1 for CV_32F images
        @param src Input 8-bit, 16-bit unsigned or 32-bit float 3-channel image..
        @param dst Output image of the same size and datatype as src.
        @param islinear default false.
    */
    CV_WRAP void correctImage(InputArray src, OutputArray dst, bool islinear = false);

    CV_WRAP void write(cv::FileStorage& fs) const;
    CV_WRAP void read(const cv::FileNode& node);

    class Impl;
private:
    std::shared_ptr<Impl> p;
};

CV_EXPORTS void write(cv::FileStorage& fs, const std::string&, const ColorCorrectionModel& ccm);
CV_EXPORTS void read(const cv::FileNode& node, ColorCorrectionModel& ccm, const ColorCorrectionModel& defaultValue = ColorCorrectionModel());

//! @} ccm
} // namespace ccm
} // namespace cv

#endif