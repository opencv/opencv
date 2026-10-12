// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef _CODERS_SPLAT_H_
#define _CODERS_SPLAT_H_

#include <opencv2/core.hpp>
#include <string>

namespace cv {

// Headerless, no point cloud attributes, so it stays outside BasePointCloudDecoder.
bool readSplatFile(const std::string& filename, Mat& splats);

} /* namespace cv */

#endif
