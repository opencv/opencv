// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_PREPROC_UTILS_HPP
#define GAPI_STREAMING_ONEVPL_PREPROC_UTILS_HPP

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/onevpl_export.hpp"

#include <opencv2/gapi/gframe.hpp>

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
namespace utils {

cv::MediaFormat fourcc_to_MediaFormat(int value);
int MediaFormat_to_fourcc(cv::MediaFormat value);
int MediaFormat_to_chroma(cv::MediaFormat value);

mfxFrameInfo to_mfxFrameInfo(const cv::GFrameDesc& frame_info);
} // namespace utils
} // namespace cv
} // namespace gapi
} // namespace wip
} // namespace onevpl
#endif // #ifdef HAVE_ONEVPL
#endif // GAPI_STREAMING_ONEVPL_PREPROC_UTILS_HPP
