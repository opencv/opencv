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
enum CCM_TYPE
{
    CCM_3x3,   ///< The CCM with the shape \f$3\times3\f$ performs linear transformation on color values.
    CCM_4x3,   ///< The CCM with the shape \f$4\times3\f$ performs affine transformation.
};

/** @brief Enum of the possible types of initial method.
*/
enum INITIAL_METHOD_TYPE
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
                        /// k_R=mean(G_{li}')/mean(G_{li})\\
                        /// k_R=mean(B_{li}')/mean(B_{li})
                        /// \f$
    INITIAL_METHOD_LEAST_SQUARE,       ///<the least square method is an optimal solution under the linear RGB distance function
};
/** @brief  Macbeth and Vinyl ColorChecker with 2deg D50
*/
enum CONST_COLOR {
    COLORCHECKER_Macbeth,                ///< Macbeth ColorChecker
    COLORCHECKER_Vinyl,                  ///< DKK ColorChecker
    COLORCHECKER_DigitalSG,              ///< DigitalSG ColorChecker with 140 squares
};
enum COLOR_SPACE {
    COLOR_SPACE_sRGB,                       ///< https://en.wikipedia.org/wiki/SRGB , RGB color space
    COLOR_SPACE_sRGBL,                      ///< https://en.wikipedia.org/wiki/SRGB , linear RGB color space
    COLOR_SPACE_AdobeRGB,                   ///< https://en.wikipedia.org/wiki/Adobe_RGB_color_space , RGB color space
    COLOR_SPACE_AdobeRGBL,                  ///< https://en.wikipedia.org/wiki/Adobe_RGB_color_space , linear RGB color space
    COLOR_SPACE_WideGamutRGB,               ///< https://en.wikipedia.org/wiki/Wide-gamut_RGB_color_space , RGB color space
    COLOR_SPACE_WideGamutRGBL,              ///< https://en.wikipedia.org/wiki/Wide-gamut_RGB_color_space , linear RGB color space
    COLOR_SPACE_ProPhotoRGB,                ///< https://en.wikipedia.org/wiki/ProPhoto_RGB_color_space , RGB color space
    COLOR_SPACE_ProPhotoRGBL,               ///< https://en.wikipedia.org/wiki/ProPhoto_RGB_color_space , linear RGB color space
    COLOR_SPACE_DCI_P3_RGB,                 ///< https://en.wikipedia.org/wiki/DCI-P3 , RGB color space
    COLOR_SPACE_DCI_P3_RGBL,                ///< https://en.wikipedia.org/wiki/DCI-P3 , linear RGB color space
    COLOR_SPACE_AppleRGB,                   ///< https://en.wikipedia.org/wiki/RGB_color_space , RGB color space
    COLOR_SPACE_AppleRGBL,                  ///< https://en.wikipedia.org/wiki/RGB_color_space , linear RGB color space
    COLOR_SPACE_REC_709_RGB,                ///< https://en.wikipedia.org/wiki/Rec._709 , RGB color space
    COLOR_SPACE_REC_709_RGBL,               ///< https://en.wikipedia.org/wiki/Rec._709 , linear RGB color space
    COLOR_SPACE_REC_2020_RGB,               ///< https://en.wikipedia.org/wiki/Rec._2020 , RGB color space
    COLOR_SPACE_REC_2020_RGBL,              ///< https://en.wikipedia.org/wiki/Rec._2020 , linear RGB color space
    COLOR_SPACE_XYZ_D65_2,                  ///< https://en.wikipedia.org/wiki/CIE_1931_color_space , non-RGB color space
    COLOR_SPACE_XYZ_D65_10,                 ///< non-RGB color space
    COLOR_SPACE_XYZ_D50_2,                  ///< non-RGB color space
    COLOR_SPACE_XYZ_D50_10,                 ///< non-RGB color space
    COLOR_SPACE_XYZ_A_2,                    ///< non-RGB color space
    COLOR_SPACE_XYZ_A_10,                   ///< non-RGB color space
    COLOR_SPACE_XYZ_D55_2,                  ///< non-RGB color space
    COLOR_SPACE_XYZ_D55_10,                 ///< non-RGB color space
    COLOR_SPACE_XYZ_D75_2,                  ///< non-RGB color space
    COLOR_SPACE_XYZ_D75_10,                 ///< non-RGB color space
    COLOR_SPACE_XYZ_E_2,                    ///< non-RGB color space
    COLOR_SPACE_XYZ_E_10,                   ///< non-RGB color space
    COLOR_SPACE_Lab_D65_2,                  ///< https://en.wikipedia.org/wiki/CIELAB_color_space , non-RGB color space
    COLOR_SPACE_Lab_D65_10,                 ///< non-RGB color space
    COLOR_SPACE_Lab_D50_2,                  ///< non-RGB color space
    COLOR_SPACE_Lab_D50_10,                 ///< non-RGB color space
    COLOR_SPACE_Lab_A_2,                    ///< non-RGB color space
    COLOR_SPACE_Lab_A_10,                   ///< non-RGB color space
    COLOR_SPACE_Lab_D55_2,                  ///< non-RGB color space
    COLOR_SPACE_Lab_D55_10,                 ///< non-RGB color space
    COLOR_SPACE_Lab_D75_2,                  ///< non-RGB color space
    COLOR_SPACE_Lab_D75_10,                 ///< non-RGB color space
    COLOR_SPACE_Lab_E_2,                    ///< non-RGB color space
    COLOR_SPACE_Lab_E_10,                   ///< non-RGB color space
};

/** @brief Linearization transformation type
*/
enum LINEAR_TYPE
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
enum DISTANCE_TYPE
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

/** @brief Core class of ccm model

Produce a ColorCorrectionModel instance for inference
*/
class CV_EXPORTS_W ColorCorrectionModel
{
public:
    /** @brief Color Correction Model

        Supported list of color cards:
        - @ref COLORCHECKER_Macbeth, the Macbeth ColorChecker
        - @ref COLORCHECKER_Vinyl, the DKK ColorChecker
        - @ref COLORCHECKER_DigitalSG, the DigitalSG ColorChecker with 140 squares

        @param src detected colors of ColorChecker patches;\n
                    the color type is RGB not BGR, and the color values are in [0, 1];
        @param constcolor the Built-in color card
    */
    CV_WRAP ColorCorrectionModel(const Mat& src, CONST_COLOR constcolor);

    /** @brief Color Correction Model
        @param src detected colors of ColorChecker patches;\n
                the color type is RGB not BGR, and the color values are in [0, 1];
        @param colors the reference color values, the color values are in [0, 1].\n
        @param ref_cs the corresponding color space
                If the color type is some RGB, the format is RGB not BGR;\n
    */
    CV_WRAP ColorCorrectionModel(const Mat& src, Mat colors, COLOR_SPACE ref_cs);

    /** @brief Color Correction Model
        @param src detected colors of ColorChecker patches;\n
                    the color type is RGB not BGR, and the color values are in [0, 1];
        @param colors the reference color values, the color values are in [0, 1].
        @param ref_cs the corresponding color space
                    If the color type is some RGB, the format is RGB not BGR;
        @param colored mask of colored color
    */
    CV_WRAP ColorCorrectionModel(const Mat& src, Mat colors, COLOR_SPACE ref_cs, Mat colored);

    /** @brief set ColorSpace
        @note It should be some RGB color space;
        Supported list of color cards:
        - @ref COLOR_SPACE_sRGB
        - @ref COLOR_SPACE_AdobeRGB
        - @ref COLOR_SPACE_WideGamutRGB
        - @ref COLOR_SPACE_ProPhotoRGB
        - @ref COLOR_SPACE_DCI_P3_RGB
        - @ref COLOR_SPACE_AppleRGB
        - @ref COLOR_SPACE_REC_709_RGB
        - @ref COLOR_SPACE_REC_2020_RGB
        @param cs the absolute color space that detected colors convert to;\n
              default: @ref COLOR_SPACE_sRGB
    */
    CV_WRAP void setColorSpace(COLOR_SPACE cs);

    /** @brief set ccm_type
    @param ccm_type the shape of color correction matrix(CCM);\n
                    default: @ref CCM_3x3
    */
    CV_WRAP void setCCM_TYPE(CCM_TYPE ccm_type);

    /** @brief set Distance
    @param distance the type of color distance;\n
                    default: @ref DISTANCE_CIE2000
    */
    CV_WRAP void setDistance(DISTANCE_TYPE distance);

    /** @brief set Linear
    @param linear_type the method of linearization;\n
                       default: @ref LINEARIZATION_GAMMA
    */
    CV_WRAP void setLinear(LINEAR_TYPE linear_type);

    /** @brief set Gamma

    @note only valid when linear is set to "gamma";\n

    @param gamma the gamma value of gamma correction;\n
                 default: 2.2;
    */
    CV_WRAP void setLinearGamma(const double& gamma);

    /** @brief set degree
        @note only valid when linear is set to
        - @ref LINEARIZATION_COLORPOLYFIT
        - @ref LINEARIZATION_GRAYPOLYFIT
        - @ref LINEARIZATION_COLORLOGPOLYFIT
        - @ref LINEARIZATION_GRAYLOGPOLYFIT

        @param deg the degree of linearization polynomial;\n
            default: 3

    */
    CV_WRAP void setLinearDegree(const int& deg);

    /** @brief set SaturatedThreshold.
                The colors in the closed interval [lower, upper] are reserved to participate
                in the calculation of the loss function and initialization parameters
        @param lower the lower threshold to determine saturation;\n
                default: 0;
        @param upper the upper threshold to determine saturation;\n
                default: 0
    */
    CV_WRAP void setSaturatedThreshold(const double& lower, const double& upper);

    /** @brief set WeightsList
    @param weights_list the list of weight of each color;\n
                        default: empty array
    */
    CV_WRAP void setWeightsList(const Mat& weights_list);

    /** @brief set WeightCoeff
    @param weights_coeff the exponent number of L* component of the reference color in CIE Lab color space;\n
                         default: 0
    */
    CV_WRAP void setWeightCoeff(const double& weights_coeff);

    /** @brief set InitialMethod
    @param initial_method_type the method of calculating CCM initial value;\n
            default: INITIAL_METHOD_LEAST_SQUARE
    */
    CV_WRAP void setInitialMethod(INITIAL_METHOD_TYPE initial_method_type);

    /** @brief set MaxCount
    @param max_count used in MinProblemSolver-DownhillSolver;\n
        Terminal criteria to the algorithm;\n
                     default: 5000;
    */
    CV_WRAP void setMaxCount(const int& max_count);

    /** @brief set Epsilon
    @param epsilon used in MinProblemSolver-DownhillSolver;\n
        Terminal criteria to the algorithm;\n
                   default: 1e-4;
    */
    CV_WRAP void setEpsilon(const double& epsilon);

    /** @brief make color correction */
    CV_WRAP void run();

    CV_WRAP Mat getCCM() const;
    CV_WRAP double getLoss() const;
    CV_WRAP Mat get_src_rgbl() const;
    CV_WRAP Mat get_dst_rgbl() const;
    CV_WRAP Mat getMask() const;
    CV_WRAP Mat getWeights() const;

    /** @brief Infer using fitting ccm.
        @param img the input image.
        @param islinear default false.
        @return the output array.
    */
    CV_WRAP Mat infer(const Mat& img, bool islinear = false);

    class Impl;
private:
    std::shared_ptr<Impl> p;
};

//! @} ccm
} // namespace ccm
} // namespace cv

#endif