// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation


#ifndef OPENCV_GAPI_GMETA_CHECK_HPP
#define OPENCV_GAPI_GMETA_CHECK_HPP

#include <ostream>

#include <opencv2/gapi/gmetaarg.hpp>

namespace cv
{
bool validate_input_meta_arg(const GMetaArg& meta, std::ostream* tracer = nullptr);
bool validate_input_meta(const GMatDesc& meta, std::ostream* tracer = nullptr);
}

#endif //OPENCV_GAPI_GMETA_CHECK_HPP
