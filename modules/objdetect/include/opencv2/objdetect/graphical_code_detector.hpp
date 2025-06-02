// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#ifndef OPENCV_OBJDETECT_GRAPHICAL_CODE_DETECTOR_HPP
#define OPENCV_OBJDETECT_GRAPHICAL_CODE_DETECTOR_HPP

#include <opencv2/core.hpp>

namespace cv {

//! @addtogroup objdetect_common
//! @{

class CV_EXPORTS_W_SIMPLE GraphicalCodeDetector {
public:
    CV_DEPRECATED_EXTERNAL  // avoid using in C++ code, will be moved to "protected" (need to fix bindings first)
    GraphicalCodeDetector();

    GraphicalCodeDetector(const GraphicalCodeDetector&) = default;
    GraphicalCodeDetector(GraphicalCodeDetector&&) = default;
    GraphicalCodeDetector& operator=(const GraphicalCodeDetector&) = default;
    GraphicalCodeDetector& operator=(GraphicalCodeDetector&&) = default;

    /** @brief Detects graphical code in image and returns the quadrangle containing the code.
     @param img grayscale or color (BGR) image containing (or not) graphical code.
     @param points Output vector of vertices of the minimum-area quadrangle containing the code.
     */
    CV_WRAP bool detect(InputArray img, OutputArray points) const;

    /** @brief Decodes graphical code in image once it's found by the detect() method.

     Returns UTF8-encoded output string or empty string if the code cannot be decoded.
     @param img grayscale or color (BGR) image containing graphical code.
     @param points Quadrangle vertices found by detect() method (or some other algorithm).
     @param straight_code The optional output image containing binarized code, will be empty if not found.
     */
    CV_WRAP std::string decode(InputArray img, InputArray points, OutputArray straight_code = noArray()) const;

    /** @brief Both detects and decodes graphical code

     @param img grayscale or color (BGR) image containing graphical code.
     @param points optional output array of vertices of the found graphical code quadrangle, will be empty if not found.
     @param straight_code The optional output image containing binarized code
     */
    CV_WRAP std::string detectAndDecode(InputArray img, OutputArray points = noArray(),
                                        OutputArray straight_code = noArray()) const;


    /** @brief Detects graphical codes in image and returns the vector of the quadrangles containing the codes.
     @param img grayscale or color (BGR) image containing (or not) graphical codes.
     @param points Output vector of vector of vertices of the minimum-area quadrangle containing the codes.
     */
    CV_WRAP bool detectMulti(InputArray img, OutputArray points) const;

    /** @brief Decodes graphical codes in image once it's found by the detect() method.
     @param img grayscale or color (BGR) image containing graphical codes.
     @param decoded_info UTF8-encoded output vector of string or empty vector of string if the codes cannot be decoded.
     @param points vector of Quadrangle vertices found by detect() method (or some other algorithm).
     @param straight_code The optional output vector of images containing binarized codes
     */
    CV_WRAP bool decodeMulti(InputArray img, InputArray points, CV_OUT std::vector<std::string>& decoded_info,
                             OutputArrayOfArrays straight_code = noArray()) const;

    /** @brief Both detects and decodes graphical codes
    @param img grayscale or color (BGR) image containing graphical codes.
    @param decoded_info UTF8-encoded output vector of string or empty vector of string if the codes cannot be decoded.
    @param points optional output vector of vertices of the found graphical code quadrangles. Will be empty if not found.
    @param straight_code The optional vector of images containing binarized codes

    - If there are QR codes encoded with a Structured Append mode on the image and all of them detected and decoded correctly,
    method writes a full message to position corresponds to 0-th code in a sequence. The rest of QR codes from the same sequence
    have empty string.
    */
    CV_WRAP bool detectAndDecodeMulti(InputArray img, CV_OUT std::vector<std::string>& decoded_info, OutputArray points = noArray(),
                                      OutputArrayOfArrays straight_code = noArray()) const;

#ifdef OPENCV_BINDINGS_PARSER
    CV_WRAP_AS(detectAndDecodeBytes) NativeByteArray detectAndDecode(InputArray img, OutputArray points = noArray(),
                                                                     OutputArray straight_code = noArray()) const;
    CV_WRAP_AS(decodeBytes) NativeByteArray decode(InputArray img, InputArray points, OutputArray straight_code = noArray()) const;
    // CV_WRAP_AS(decodeBytesMulti) bool decodeMulti(InputArray img, InputArray points, CV_OUT std::vector<NativeByteArray>& decoded_info,
    //                                               OutputArrayOfArrays straight_code = noArray()) const;
    // CV_WRAP_AS(detectAndDecodeBytesMulti) bool detectAndDecodeMulti(InputArray img, CV_OUT std::vector<NativeByteArray>& decoded_info, OutputArray points = noArray(),
    //                                                                 OutputArrayOfArrays straight_code = noArray()) const;
#endif

    struct Impl;
protected:
    Ptr<Impl> p;
};

//! @}

}

#endif
