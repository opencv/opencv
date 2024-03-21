/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#ifndef OPENCV_OBJDETECT_HPP
#define OPENCV_OBJDETECT_HPP

#include "opencv2/core.hpp"
#include "opencv2/objdetect/aruco_detector.hpp"
#include "opencv2/objdetect/graphical_code_detector.hpp"

/**
@defgroup objdetect Object Detection

@{
    @defgroup objdetect_barcode Barcode detection and decoding
    @defgroup objdetect_qrcode QRCode detection and encoding
    @defgroup objdetect_dnn_face DNN-based face detection and recognition

    Check @ref tutorial_dnn_face "the corresponding tutorial" for more details.

    @defgroup objdetect_common Common functions and classes
    @defgroup objdetect_aruco ArUco markers and boards detection for robust camera pose estimation
    @{
        ArUco Marker Detection
        Square fiducial markers (also known as Augmented Reality Markers) are useful for easy,
        fast and robust camera pose estimation.

        The main functionality of ArucoDetector class is detection of markers in an image. If the markers are grouped
        as a board, then you can try to recover the missing markers with ArucoDetector::refineDetectedMarkers().
        ArUco markers can also be used for advanced chessboard corner finding. To do this, group the markers in the
        CharucoBoard and find the corners of the chessboard with the CharucoDetector::detectBoard().

        The implementation is based on the ArUco Library by R. Muñoz-Salinas and S. Garrido-Jurado @cite Aruco2014.

        Markers can also be detected based on the AprilTag 2 @cite wang2016iros fiducial detection method.

        @sa @cite Aruco2014
        This code has been originally developed by Sergio Garrido-Jurado as a project
        for Google Summer of Code 2015 (GSoC 15).
    @}

@}
 */

namespace cv
{
//! @addtogroup objdetect_qrcode
//! @{

class CV_EXPORTS_W QRCodeEncoder {
protected:
    QRCodeEncoder();  // use ::create()
public:
    virtual ~QRCodeEncoder();

    enum EncodeMode {
        MODE_AUTO              = -1,
        MODE_NUMERIC           = 1, // 0b0001
        MODE_ALPHANUMERIC      = 2, // 0b0010
        MODE_BYTE              = 4, // 0b0100
        MODE_ECI               = 7, // 0b0111
        MODE_KANJI             = 8, // 0b1000
        MODE_STRUCTURED_APPEND = 3  // 0b0011
    };

    enum CorrectionLevel {
        CORRECT_LEVEL_L = 0,
        CORRECT_LEVEL_M = 1,
        CORRECT_LEVEL_Q = 2,
        CORRECT_LEVEL_H = 3
    };

    enum ECIEncodings {
        ECI_UTF8 = 26
    };

    /** @brief QR code encoder parameters. */
    struct CV_EXPORTS_W_SIMPLE Params
    {
        CV_WRAP Params();

        //! The optional version of QR code (by default - maximum possible depending on the length of the string).
        CV_PROP_RW int version;

        //! The optional level of error correction (by default - the lowest).
        CV_PROP_RW CorrectionLevel correction_level;

        //! The optional encoding mode - Numeric, Alphanumeric, Byte, Kanji, ECI or Structured Append.
        CV_PROP_RW EncodeMode mode;

        //! The optional number of QR codes to generate in Structured Append mode.
        CV_PROP_RW int structure_number;
    };

    /** @brief Constructor
    @param parameters QR code encoder parameters QRCodeEncoder::Params
    */
    static CV_WRAP
    Ptr<QRCodeEncoder> create(const QRCodeEncoder::Params& parameters = QRCodeEncoder::Params());

    /** @brief Generates QR code from input string.
     @param encoded_info Input string to encode.
     @param qrcode Generated QR code.
    */
    CV_WRAP virtual void encode(const String& encoded_info, OutputArray qrcode) = 0;

    /** @brief Generates QR code from input string in Structured Append mode. The encoded message is splitting over a number of QR codes.
     @param encoded_info Input string to encode.
     @param qrcodes Vector of generated QR codes.
    */
    CV_WRAP virtual void encodeStructuredAppend(const String& encoded_info, OutputArrayOfArrays qrcodes) = 0;

};
class CV_EXPORTS_W_SIMPLE QRCodeDetector : public GraphicalCodeDetector
{
public:
    CV_WRAP QRCodeDetector();

    /** @brief sets the epsilon used during the horizontal scan of QR code stop marker detection.
     @param epsX Epsilon neighborhood, which allows you to determine the horizontal pattern
     of the scheme 1:1:3:1:1 according to QR code standard.
    */
    CV_WRAP QRCodeDetector& setEpsX(double epsX);
    /** @brief sets the epsilon used during the vertical scan of QR code stop marker detection.
     @param epsY Epsilon neighborhood, which allows you to determine the vertical pattern
     of the scheme 1:1:3:1:1 according to QR code standard.
     */
    CV_WRAP QRCodeDetector& setEpsY(double epsY);

    /** @brief use markers to improve the position of the corners of the QR code
     *
     * alignmentMarkers using by default
     */
    CV_WRAP QRCodeDetector& setUseAlignmentMarkers(bool useAlignmentMarkers);

    /** @brief Decodes QR code on a curved surface in image once it's found by the detect() method.

     Returns UTF8-encoded output string or empty string if the code cannot be decoded.
     @param img grayscale or color (BGR) image containing QR code.
     @param points Quadrangle vertices found by detect() method (or some other algorithm).
     @param straight_qrcode The optional output image containing rectified and binarized QR code
     */
    CV_WRAP cv::String decodeCurved(InputArray img, InputArray points, OutputArray straight_qrcode = noArray());

    /** @brief Both detects and decodes QR code on a curved surface

     @param img grayscale or color (BGR) image containing QR code.
     @param points optional output array of vertices of the found QR code quadrangle. Will be empty if not found.
     @param straight_qrcode The optional output image containing rectified and binarized QR code
     */
    CV_WRAP std::string detectAndDecodeCurved(InputArray img, OutputArray points=noArray(),
                                              OutputArray straight_qrcode = noArray());
};

class CV_EXPORTS_W_SIMPLE QRCodeDetectorAruco : public GraphicalCodeDetector {
public:
    CV_WRAP QRCodeDetectorAruco();

    struct CV_EXPORTS_W_SIMPLE Params {
        CV_WRAP Params();

        /** @brief The minimum allowed pixel size of a QR module in the smallest image in the image pyramid, default 4.f */
        CV_PROP_RW float minModuleSizeInPyramid;

        /** @brief The maximum allowed relative rotation for finder patterns in the same QR code, default pi/12 */
        CV_PROP_RW float maxRotation;

        /** @brief The maximum allowed relative mismatch in module sizes for finder patterns in the same QR code, default 1.75f */
        CV_PROP_RW float maxModuleSizeMismatch;

        /** @brief The maximum allowed module relative mismatch for timing pattern module, default 2.f
         *
         * If relative mismatch of timing pattern module more this value, penalty points will be added.
         * If a lot of penalty points are added, QR code will be rejected. */
        CV_PROP_RW float maxTimingPatternMismatch;

        /** @brief The maximum allowed percentage of penalty points out of total pins in timing pattern, default 0.4f */
        CV_PROP_RW float maxPenalties;

        /** @brief The maximum allowed relative color mismatch in the timing pattern, default 0.2f*/
        CV_PROP_RW float maxColorsMismatch;

        /** @brief The algorithm find QR codes with almost minimum timing pattern score and minimum size, default 0.9f
         *
         * The QR code with the minimum "timing pattern score" and minimum "size" is selected as the best QR code.
         * If for the current QR code "timing pattern score" * scaleTimingPatternScore < "previous timing pattern score" and "size" < "previous size", then
         * current QR code set as the best QR code. */
        CV_PROP_RW float scaleTimingPatternScore;
    };

    /** @brief QR code detector constructor for Aruco-based algorithm. See cv::QRCodeDetectorAruco::Params */
    CV_WRAP explicit QRCodeDetectorAruco(const QRCodeDetectorAruco::Params& params);

    /** @brief Detector parameters getter. See cv::QRCodeDetectorAruco::Params */
    CV_WRAP const QRCodeDetectorAruco::Params& getDetectorParameters() const;

    /** @brief Detector parameters setter. See cv::QRCodeDetectorAruco::Params */
    CV_WRAP QRCodeDetectorAruco& setDetectorParameters(const QRCodeDetectorAruco::Params& params);

    /** @brief Aruco detector parameters are used to search for the finder patterns. */
    CV_WRAP const aruco::DetectorParameters& getArucoParameters() const;

    /** @brief Aruco detector parameters are used to search for the finder patterns. */
    CV_WRAP void setArucoParameters(const aruco::DetectorParameters& params);
};

//! @}
}

#include "opencv2/objdetect/face.hpp"
#include "opencv2/objdetect/charuco_detector.hpp"
#include "opencv2/objdetect/barcode.hpp"

#endif
