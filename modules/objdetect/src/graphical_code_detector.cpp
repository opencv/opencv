// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "opencv2/objdetect/graphical_code_detector.hpp"
#include "graphical_code_detector_impl.hpp"

namespace cv {

GraphicalCodeDetector::GraphicalCodeDetector() {}

bool GraphicalCodeDetector::detect(InputArray img, OutputArray points) const {
    CV_Assert(p);
    return p->detect(img, points);
}

std::string GraphicalCodeDetector::decode(InputArray img, InputArray points, OutputArray straight_code) const {
    CV_Assert(p);
    return p->decode(img, points, straight_code);
}

std::string GraphicalCodeDetector::detectAndDecode(InputArray img, OutputArray points, OutputArray straight_code) const {
    CV_Assert(p);
    return p->detectAndDecode(img, points, straight_code);
}

bool GraphicalCodeDetector::detectMulti(InputArray img, OutputArray points) const {
    CV_Assert(p);
    return p->detectMulti(img, points);
}

bool GraphicalCodeDetector::decodeMulti(InputArray img, InputArray points, std::vector<std::string>& decoded_info,
                                       OutputArrayOfArrays straight_code) const {
    CV_Assert(p);
    return p->decodeMulti(img, points, decoded_info, straight_code);
}

bool GraphicalCodeDetector::detectAndDecodeMulti(InputArray img, std::vector<std::string>& decoded_info, OutputArray points,
                                                OutputArrayOfArrays straight_code) const {
    CV_Assert(p);
    return p->detectAndDecodeMulti(img, decoded_info, points, straight_code);
}

}
