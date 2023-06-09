// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#ifndef OPENCV_OBJDETECT_GRAPHICAL_CODE_DETECTOR_IMPL_HPP
#define OPENCV_OBJDETECT_GRAPHICAL_CODE_DETECTOR_IMPL_HPP

#include <opencv2/core.hpp>

namespace cv {

struct GraphicalCodeDetector::Impl {
    virtual ~Impl() {}
    virtual bool detect(InputArray img, OutputArray points) const = 0;
    virtual std::string decode(InputArray img, InputArray points, OutputArray straight_code) const = 0;
    virtual std::string detectAndDecode(InputArray img, OutputArray points, OutputArray straight_code) const = 0;
    virtual bool detectMulti(InputArray img, OutputArray points) const = 0;
    virtual bool decodeMulti(InputArray img, InputArray points, std::vector<std::string>& decoded_info,
                             OutputArrayOfArrays straight_code) const = 0;
    virtual bool detectAndDecodeMulti(InputArray img, std::vector<std::string>& decoded_info,
                                      OutputArray points, OutputArrayOfArrays straight_code) const = 0;
};

}

#endif