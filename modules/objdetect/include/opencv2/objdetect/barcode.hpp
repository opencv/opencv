// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (c) 2020-2021 darkliang wangberlinT Certseeds

#ifndef OPENCV_BARCODE_HPP
#define OPENCV_BARCODE_HPP

#include <opencv2/core.hpp>
#include <opencv2/objdetect/graphical_code_detector.hpp>

namespace cv {
namespace barcode {

//! @addtogroup objdetect_barcode
//! @{

enum BarcodeType
{
    Barcode_NONE,
    Barcode_EAN_8,
    Barcode_EAN_13,
    Barcode_UPC_A,
    Barcode_UPC_E,
    Barcode_UPC_EAN_EXTENSION
};

template <typename S>
static inline S &operator<<(S &out, const BarcodeType &barcode_type)
{
    switch (barcode_type)
    {
        case BarcodeType::Barcode_EAN_8:
            out << "EAN_8";
            break;
        case BarcodeType::Barcode_EAN_13:
            out << "EAN_13";
            break;
        case BarcodeType::Barcode_UPC_E:
            out << "UPC_E";
            break;
        case BarcodeType::Barcode_UPC_A:
            out << "UPC_A";
            break;
        case BarcodeType::Barcode_UPC_EAN_EXTENSION:
            out << "UPC_EAN_EXTENSION";
            break;
        default:
            out << "NONE";
    }
    return out;
}

class CV_EXPORTS_W_SIMPLE BarcodeDetector : public cv::GraphicalCodeDetector
{
public:
    /**
     * @brief Initialize the BarcodeDetector.
     * Parameters allow to load _optional_ Super Resolution DNN model for better quality.
     * @param prototxt_path prototxt file path for the super resolution model
     * @param model_path model file path for the super resolution model
     */
    CV_WRAP BarcodeDetector(const std::string &prototxt_path = "", const std::string &model_path = "");
    ~BarcodeDetector();

    /** @brief Decodes barcode in image once it's found by the detect() method.
     *
     * @param img grayscale or color (BGR) image containing bar code.
     * @param points vector of rotated rectangle vertices found by detect() method (or some other algorithm).
     * For N detected barcodes, the dimensions of this array should be [N][4].
     * Order of four points in vector<Point2f> is bottomLeft, topLeft, topRight, bottomRight.
     * @param decoded_info UTF8-encoded output vector of string or empty vector of string if the codes cannot be decoded.
     * @param decoded_type vector of BarcodeType, specifies the type of these barcodes
     */
    CV_WRAP bool decodeExtra(InputArray img,
                             InputArray points,
                             CV_OUT std::vector<std::string> &decoded_info,
                             CV_OUT std::vector<BarcodeType> &decoded_type) const;

    /** @brief Both detects and decodes barcode

     * @param img grayscale or color (BGR) image containing barcode.
     * @param decoded_info UTF8-encoded output vector of string(s) or empty vector of string if the codes cannot be decoded.
     * @param decoded_type vector of BarcodeType, specifies the type of these barcodes
     * @param points optional output vector of vertices of the found  barcode rectangle. Will be empty if not found.
     */
    CV_WRAP bool detectAndDecodeExtra(InputArray img,
                                      CV_OUT std::vector<std::string> &decoded_info,
                                      CV_OUT std::vector<BarcodeType> &decoded_type,
                                      OutputArray points = noArray()) const;
};
//! @}

}} // cv::barcode::

#endif // OPENCV_BARCODE_HPP
