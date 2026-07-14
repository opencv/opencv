// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#ifdef HAVE_PLAIDML

#ifndef OPENCV_GAPI_PLAIDML_UTIL_HPP
#define OPENCV_GAPI_PLAIDML_UTIL_HPP

#include <plaidml2/core/ffi.h>

namespace cv
{
namespace util
{
namespace plaidml
{

inline plaidml_datatype depth_from_ocv(int depth)
{
    switch(depth)
    {
        case CV_8U  : return PLAIDML_DATA_UINT8;
        case CV_8S  : return PLAIDML_DATA_INT8;
        case CV_16U : return PLAIDML_DATA_UINT16;
        case CV_16S : return PLAIDML_DATA_INT16;
        case CV_32S : return PLAIDML_DATA_INT32;
        case CV_32F : return PLAIDML_DATA_FLOAT32;
        case CV_64F : return PLAIDML_DATA_FLOAT64;
        default: util::throw_error("Unrecognized OpenCV depth");
    }
};

}
}
}
#endif // OPENCV_GAPI_PLAIDML_UTIL_HPP

#endif // HAVE_PLAIDML
