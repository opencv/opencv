// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _CODERS_SPLAT_H_
#define _CODERS_SPLAT_H_

#include <opencv2/core.hpp>
#include <string>

namespace cv {

// Headerless, no point cloud attributes, so it stays outside BasePointCloudDecoder.
class SplatDecoder
{
public:
    void setSource(const std::string& filename) noexcept { m_filename = filename; }

    bool readSplats(Mat& splats);

protected:
    std::string m_filename;
};

} /* namespace cv */

#endif
