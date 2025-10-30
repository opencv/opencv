/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2025, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_IMGPROC_ICC_HPP
#define OPENCV_IMGPROC_ICC_HPP

#include "opencv2/core.hpp"

/**
@defgroup imgproc_color_icc ICC Color Management

This module provides advanced color management capabilities using ICC profiles,
including support for ICC v2/v4/v5 and iccMAX specifications.

@note ICC v5/iccMAX support enables:
- Multi-channel color spaces (beyond RGB)
- High-precision floating-point processing
- Color appearance models (CAM02/CAM16)
- HDR and wide-gamut workflows
- Spectral and device-independent color representation

@{
*/

namespace cv {

/** @brief ICC Profile version types
*/
enum IccProfileVersion {
    ICC_PROFILE_V2    = 0x02000000,  //!< ICC Profile version 2
    ICC_PROFILE_V4    = 0x04000000,  //!< ICC Profile version 4
    ICC_PROFILE_V5    = 0x05000000,  //!< ICC Profile version 5
    ICC_PROFILE_MAX   = 0x05100000   //!< iccMAX (ICC v5.1+)
};

/** @brief Color appearance model types for perceptual transformations
*/
enum ColorAppearanceModel {
    CAM_NONE          = 0,   //!< No color appearance model
    CAM02             = 1,   //!< CIECAM02 color appearance model
    CAM16             = 2    //!< CAM16 color appearance model (CIECAM02 successor)
};

/** @brief ICC rendering intents
*/
enum IccRenderingIntent {
    ICC_PERCEPTUAL            = 0,   //!< Perceptual rendering intent
    ICC_RELATIVE_COLORIMETRIC = 1,   //!< Relative colorimetric rendering intent
    ICC_SATURATION            = 2,   //!< Saturation rendering intent
    ICC_ABSOLUTE_COLORIMETRIC = 3    //!< Absolute colorimetric rendering intent
};

/** @brief ICC Profile class representing an ICC color profile
 *
 * This class encapsulates ICC v2/v4/v5 and iccMAX profile data and provides
 * methods for profile parsing, validation, and color transformation setup.
 *
 * Supports:
 * - Multi-channel color spaces
 * - High-precision floating-point processing
 * - Color appearance models (CAM02/CAM16)
 * - Perceptual adaptation and viewing conditions
 *
 * @note Python bindings are temporarily disabled for this class.
 */
class CV_EXPORTS IccProfile {
public:
    /** @brief Default constructor
     */
    IccProfile();

    /** @brief Constructor from file path
     * @param filename Path to ICC profile file
     */
    explicit IccProfile(const String& filename);

    /** @brief Constructor from profile data
     * @param data ICC profile data buffer
     */
    explicit IccProfile(const std::vector<uchar>& data);

    /** @brief Constructor from InputArray
     * @param data ICC profile data as InputArray
     */
    explicit IccProfile(InputArray data);

    /** @brief Destructor
     */
    ~IccProfile();

    /** @brief Load ICC profile from file
     * @param filename Path to ICC profile file
     * @return true if profile loaded successfully
     */
    bool load(const String& filename);

    /** @brief Load ICC profile from data buffer
     * @param data ICC profile data buffer
     * @return true if profile loaded successfully
     */
    bool load(const std::vector<uchar>& data);

    /** @brief Load ICC profile from InputArray
     * @param data ICC profile data as InputArray
     * @return true if profile loaded successfully
     */
    bool load(InputArray data);

    /** @brief Check if profile is valid and loaded
     * @return true if profile is valid
     */
    bool isValid() const;

    /** @brief Get ICC profile version
     * @return ICC profile version
     */
    IccProfileVersion getVersion() const;

    /** @brief Get profile color space signature
     * @return Color space signature as string
     */
    String getColorSpace() const;

    /** @brief Get profile connection space signature
     * @return PCS signature as string
     */
    String getPCS() const;

    /** @brief Get number of input channels
     * @return Number of input channels
     */
    int getInputChannels() const;

    /** @brief Get number of output channels
     * @return Number of output channels
     */
    int getOutputChannels() const;

    /** @brief Check if profile supports floating-point processing
     * @return true if floating-point processing is supported
     */
    bool supportsFloat() const;

    /** @brief Check if profile supports HDR processing
     * @return true if HDR processing is supported
     */
    bool supportsHDR() const;

    /** @brief Get profile description
     * @return Profile description string
     */
    String getDescription() const;

private:
    class Impl;
    Ptr<Impl> p;
};

/** @brief Viewing conditions for color appearance models
 *
 * @note Python bindings are temporarily disabled for this struct.
 */
struct CV_EXPORTS ViewingConditions {
    /** @brief White point chromaticity (xy)
     */
    Point2f whitePoint;

    /** @brief Adapting luminance in cd/m²
     */
    float adaptingLuminance;

    /** @brief Background luminance ratio (0-1)
     */
    float backgroundLuminance;

    /** @brief Surround condition (0=dark, 1=dim, 2=average)
     */
    int surround;

    /** @brief Discounting illuminant (true/false)
     */
    bool discountIlluminant;

    /** @brief Default constructor with standard viewing conditions
     */
    ViewingConditions();
};

/** @brief Convert colors using ICC profile transformation
 *
 * This function performs color space conversion using ICC profiles with support
 * for multi-channel data, floating-point precision, and color appearance models.
 *
 * @param src Source image
 * @param dst Destination image
 * @param srcProfile Source ICC profile
 * @param dstProfile Destination ICC profile
 * @param intent Rendering intent (default: perceptual)
 * @param cam Color appearance model (default: none)
 * @param viewingConditions Viewing conditions for appearance model
 *
 * @note For optimal results with HDR/wide-gamut content:
 * - Use floating-point input/output (CV_32F or CV_64F)
 * - Apply appropriate color appearance model
 * - Specify viewing conditions when using CAM02/CAM16
 */
CV_EXPORTS void colorProfileTransform(InputArray src, OutputArray dst,
                                    const IccProfile& srcProfile,
                                    const IccProfile& dstProfile,
                                    IccRenderingIntent intent = ICC_PERCEPTUAL,
                                    ColorAppearanceModel cam = CAM_NONE,
                                    const ViewingConditions& viewingConditions = ViewingConditions());

/** @brief Convert single color using ICC profile transformation
 *
 * @param src Source color values
 * @param dst Destination color values
 * @param srcProfile Source ICC profile
 * @param dstProfile Destination ICC profile
 * @param intent Rendering intent
 * @param cam Color appearance model
 * @param viewingConditions Viewing conditions
 */
CV_EXPORTS void colorProfileTransformSingle(InputArray src, OutputArray dst,
                                           const IccProfile& srcProfile,
                                           const IccProfile& dstProfile,
                                           IccRenderingIntent intent = ICC_PERCEPTUAL,
                                           ColorAppearanceModel cam = CAM_NONE,
                                           const ViewingConditions& viewingConditions = ViewingConditions());

/** @brief Create standard ICC profiles for common color spaces
 *
 * @param colorSpace Color space name ("sRGB", "Adobe RGB", "ProPhoto RGB", "Rec2020", etc.)
 * @return ICC profile for the specified color space
 */
CV_EXPORTS IccProfile createStandardProfile(const String& colorSpace);

/** @brief Get default viewing conditions for different environments
 *
 * @param environment Environment type ("office", "print", "cinema", "outdoors")
 * @return Viewing conditions appropriate for the environment
 */
CV_EXPORTS ViewingConditions getStandardViewingConditions(const String& environment);

//! @} imgproc_color_icc

} // namespace cv

#endif // OPENCV_IMGPROC_ICC_HPP
