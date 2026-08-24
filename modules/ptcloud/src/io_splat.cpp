// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"

#include "io_splat.hpp"
#include "splat.hpp"
#include <opencv2/core/utils/logger.hpp>

#include <fstream>
#include <vector>

namespace cv {

bool readSplatFile(const std::string& filename, Mat& splats)
{
    splats.release();

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file)
    {
        CV_LOG_ERROR(NULL, "Failed to open '" << filename << "'");
        return false;
    }

    const std::streamoff size = file.tellg();
    if (size <= 0 || size % (int)splat::PACKED_STRIDE != 0)
    {
        CV_LOG_ERROR(NULL, "'" << filename << "' is not a .splat file: size " << size
                     << " is not a positive multiple of " << (int)splat::PACKED_STRIDE);
        return false;
    }

    file.seekg(0);
    std::vector<uchar> buf((size_t)size);
    if (!file.read((char*)buf.data(), size))
    {
        CV_LOG_ERROR(NULL, "Failed to read '" << filename << "'");
        return false;
    }

    decodeGaussianSplatsPacked(buf, splats);
    return true;
}

} /* namespace cv */
