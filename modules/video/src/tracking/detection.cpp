// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"

namespace cv {


Detection::Detection()
{
    // nothing
}


Detection::Detection(cv::Rect2f tlwh, int classId, float score)
{
    rect = tlwh;
    classLabel = classId;
    classScore = score;
}



Detection::~Detection()
{
    // nothing
}

}  // namespace cv