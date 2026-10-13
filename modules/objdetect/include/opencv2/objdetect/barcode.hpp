// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (c) 2020-2021 darkliang wangberlinT Certseeds

#ifndef OPENCV_OBJDETECT_BARCODE_HPP
#define OPENCV_OBJDETECT_BARCODE_HPP

#include <opencv2/core.hpp>
#include <opencv2/objdetect/graphical_code_detector.hpp>

namespace cv {
namespace barcode {

//! @addtogroup objdetect_barcode
//! @{

class CV_EXPORTS_W_SIMPLE BarcodeDetector : public cv::GraphicalCodeDetector
{
public:
    /** @brief Initialize the BarcodeDetector. Super resolution is disabled.
    */
    CV_WRAP BarcodeDetector();

    /** @brief Initialize the BarcodeDetector with a Super Resolution model.
     *
     * Loads a Super Resolution DNN model in ONNX format, used to upscale small/low-quality
     * barcode crops before decoding for better quality.
     *
     * @note Caffe models (`sr.prototxt` / `sr.caffemodel`) are no longer supported; convert
     * the model to ONNX (a converted `sr.onnx` is available from
     * https://github.com/WeChatCV/opencv_3rdparty/tree/wechat_qrcode).
     *
     * @param super_resolution_model_path path to a single-file ONNX Super Resolution model.
     */
    CV_WRAP BarcodeDetector(CV_WRAP_FILE_PATH const std::string &super_resolution_model_path);
    ~BarcodeDetector();

    /** @brief Decodes barcode in image once it's found by the detect() method.
     *
     * @param img grayscale or color (BGR) image containing bar code.
     * @param points vector of rotated rectangle vertices found by detect() method (or some other algorithm).
     * For N detected barcodes, the dimensions of this array should be [N][4].
     * Order of four points in vector<Point2f> is bottomLeft, topLeft, topRight, bottomRight.
     * @param decoded_info UTF8-encoded output vector of string or empty vector of string if the codes cannot be decoded.
     * @param decoded_type vector strings, specifies the type of these barcodes
     * @return true if at least one valid barcode have been found
     */
    CV_WRAP bool decodeWithType(InputArray img,
                             InputArray points,
                             CV_OUT std::vector<std::string> &decoded_info,
                             CV_OUT std::vector<std::string> &decoded_type) const;

    /** @brief Both detects and decodes barcode

     * @param img grayscale or color (BGR) image containing barcode.
     * @param decoded_info UTF8-encoded output vector of string(s) or empty vector of string if the codes cannot be decoded.
     * @param decoded_type vector of strings, specifies the type of these barcodes
     * @param points optional output vector of vertices of the found  barcode rectangle. Will be empty if not found.
     * @return true if at least one valid barcode have been found
     */
    CV_WRAP bool detectAndDecodeWithType(InputArray img,
                                      CV_OUT std::vector<std::string> &decoded_info,
                                      CV_OUT std::vector<std::string> &decoded_type,
                                      OutputArray points = noArray()) const;

    /** @brief Get detector downsampling threshold.
     *
     * @return detector downsampling threshold
     */
    CV_WRAP double getDownsamplingThreshold() const;

    /** @brief Set detector downsampling threshold.
     *
     * By default, the detect method resizes the input image to this limit if the smallest image size is is greater than the threshold.
     * Increasing this value can improve detection accuracy and the number of results at the expense of performance.
     * Correlates with detector scales. Setting this to a large value will disable downsampling.
     * @param thresh downsampling limit to apply (default 512)
     * @see setDetectorScales
     */
    CV_WRAP BarcodeDetector& setDownsamplingThreshold(double thresh);

    /** @brief Returns detector box filter sizes.
     *
     * @param sizes output parameter for returning the sizes.
     */
    CV_WRAP void getDetectorScales(CV_OUT std::vector<float>& sizes) const;

    /** @brief Set detector box filter sizes.
     *
     * Adjusts the value and the number of box filters used in the detect step.
     * The filter sizes directly correlate with the expected line widths for a barcode. Corresponds to expected barcode distance.
     * If the downsampling limit is increased, filter sizes need to be adjusted in an inversely proportional way.
     * @param sizes box filter sizes, relative to minimum dimension of the image (default [0.01, 0.03, 0.06, 0.08])
     */
    CV_WRAP BarcodeDetector& setDetectorScales(const std::vector<float>& sizes);

    /** @brief Get detector gradient magnitude threshold.
     *
     * @return detector gradient magnitude threshold.
     */
    CV_WRAP double getGradientThreshold() const;

    /** @brief Set detector gradient magnitude threshold.
     *
     * Sets the coherence threshold for detected bounding boxes.
     * Increasing this value will generate a closer fitted bounding box width and can reduce false-positives.
     * Values between 16 and 1024 generally work, while too high of a value will remove valid detections.
     * @param thresh gradient magnitude threshold (default 64).
     */
    CV_WRAP BarcodeDetector& setGradientThreshold(double thresh);
};

/** @brief Linear 1D barcode encoder for GS1 symbologies (EAN-13, EAN-8, UPC-A).
 *
 * This class generates standard binary raster images (CV_8UC1) representing linear barcodes conforming to
 * GS1 General Specifications.
 *
 * ### Input String and Checksum Policy Contracts
 * The input string must consist solely of decimal numeric ASCII characters ('0' - '9'). Leading zeros are
 * significant and preserved.
 *
 * - **APPEND_CHECKSUM**: The input string must strictly contain only the payload digits without a check digit:
 *   - EAN_13: exactly 12 digits. The 13th digit (GS1 modulo-10 check digit) is computed and appended.
 *   - EAN_8: exactly 7 digits. The 8th digit (GS1 modulo-10 check digit) is computed and appended.
 *   - UPC_A: exactly 11 digits. The 12th digit (GS1 modulo-10 check digit) is computed and appended.
 *   @note APPEND_CHECKSUM does not accept complete codes that already include a check digit; providing full-length
 *   inputs under APPEND_CHECKSUM throws cv::Error::StsBadArg.
 *
 * - **VALIDATE_CHECKSUM**: The input string must be the complete barcode string including the check digit:
 *   - EAN_13: exactly 13 digits.
 *   - EAN_8: exactly 8 digits.
 *   - UPC_A: exactly 12 digits.
 *   The check digit is validated against the GS1 modulo-10 calculation. If the check digit is incorrect,
 *   cv::Error::StsBadArg is thrown.
 *
 * ### Output Raster and Geometry Contract
 * - **Matrix Type**: Single-channel 8-bit unsigned char (CV_8UC1).
 * - **Pixel Values**: 0 for black bars (binary 1 modules) and 255 for white spaces / quiet zones (binary 0 modules).
 * - **Height**: Exactly `params.bar_height` rows (must be >= 1).
 * - **Width**: Exactly `total_modules * params.module_width` columns (must be >= 1 and not overflow signed int).
 * - **Quiet Zones and Total Module Count**:
 *   - **EAN_13**: 11 left quiet modules + 95 symbol modules (3 start guard + 42 left digits [6 x 7] + 5 center guard + 42 right digits [6 x 7] + 3 stop guard) + 7 right quiet modules = 113 total modules. Total width = `113 * module_width` pixels.
 *   - **EAN_8**: 7 left quiet modules + 67 symbol modules (3 start guard + 28 left digits [4 x 7] + 5 center guard + 28 right digits [4 x 7] + 3 stop guard) + 7 right quiet modules = 81 total modules. Total width = `81 * module_width` pixels.
 *   - **UPC_A**: 9 left quiet modules + 95 symbol modules (3 start guard + 42 left digits [6 x 7] + 5 center guard + 42 right digits [6 x 7] + 3 stop guard) + 9 right quiet modules = 113 total modules. Total width = `113 * module_width` pixels.
 */
class CV_EXPORTS_W BarcodeEncoder
{
protected:
    BarcodeEncoder(); // use ::create()
public:
    virtual ~BarcodeEncoder();

    /** @brief Barcode symbology type. */
    enum Type {
        NONE = 0,   //!< Unspecified symbology (invalid for encoding).
        EAN_8 = 1,  //!< European Article Number 8 (7-digit payload / 8-digit full code).
        EAN_13 = 2, //!< European Article Number 13 (12-digit payload / 13-digit full code).
        UPC_A = 3   //!< Universal Product Code A (11-digit payload / 12-digit full code).
    };

    /** @brief Checksum policy for barcode generation. */
    enum ChecksumPolicy {
        /** @brief Calculate and append the GS1 modulo-10 check digit.
         *
         * Requires payload-only input string (12 digits for EAN-13, 7 for EAN-8, 11 for UPC-A).
         * Complete codes with existing check digits are rejected with cv::Error::StsBadArg.
         */
        APPEND_CHECKSUM = 0,

        /** @brief Validate the supplied GS1 modulo-10 check digit on a complete barcode string.
         *
         * Requires full-length input string including check digit (13 digits for EAN-13, 8 for EAN-8, 12 for UPC-A).
         * Check digit mismatch causes cv::Error::StsBadArg to be thrown.
         */
        VALIDATE_CHECKSUM = 1
    };

    /** @brief Barcode encoder parameters. */
    struct CV_EXPORTS_W_SIMPLE Params
    {
        CV_WRAP Params();

        //! Barcode type (symbology). Default is EAN_13.
        CV_PROP_RW BarcodeEncoder::Type type;

        //! Checksum handling policy. Default is APPEND_CHECKSUM.
        CV_PROP_RW BarcodeEncoder::ChecksumPolicy checksum_policy;

        //! Module width in pixels. Must be >= 1 and not exceed INT_MAX / total_modules. Default is 2.
        CV_PROP_RW int module_width;

        //! Bar height in pixels. Must be >= 1. Default is 50.
        CV_PROP_RW int bar_height;
    };

    /** @brief Creates a BarcodeEncoder instance with specified parameters.
     * @param parameters Barcode encoder parameters BarcodeEncoder::Params
     * @return Smart pointer to the created BarcodeEncoder instance.
     */
    static CV_WRAP
    Ptr<BarcodeEncoder> create(const BarcodeEncoder::Params& parameters = BarcodeEncoder::Params());

    /** @brief Generates a binary raster barcode image from an input string.
     *
     * The input string must contain only ASCII decimal digits ('0' - '9').
     * Accepted string lengths and check digit expectations depend strictly on the configured
     * @ref Type and @ref ChecksumPolicy:
     * - EAN-13: 12 digits for APPEND_CHECKSUM, 13 digits for VALIDATE_CHECKSUM.
     * - EAN-8:  7 digits for APPEND_CHECKSUM, 8 digits for VALIDATE_CHECKSUM.
     * - UPC-A:  11 digits for APPEND_CHECKSUM, 12 digits for VALIDATE_CHECKSUM.
     *
     * The output matrix is allocated as CV_8UC1 with dimensions:
     * - rows: `bar_height`
     * - cols: `total_modules * module_width` (113 * module_width for EAN-13 and UPC-A; 81 * module_width for EAN-8).
     * Pixel value 0 represents black bars; pixel value 255 represents white spaces and quiet zones.
     *
     * @param encoded_info Input string of digits to encode.
     * @param output Output Array allocated as CV_8UC1 binary raster.
     * @throws cv::Exception with Error::StsBadArg if encoded_info is empty, contains non-digits,
     *         has an incorrect length for the symbology/checksum policy, has an invalid check digit
     *         under VALIDATE_CHECKSUM, or if module_width/bar_height are invalid or cause integer overflow.
     * @throws cv::Exception with Error::StsNotImplemented if type is NONE or unsupported.
     */
    CV_WRAP virtual void encode(const String& encoded_info, OutputArray output) = 0;
};

//! @}

}} // cv::barcode::

#endif // OPENCV_OBJDETECT_BARCODE_HPP
