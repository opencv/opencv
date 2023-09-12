// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#ifndef __OPENCV_OBJDETECT_ARUCO_UTILS_HPP__
#define __OPENCV_OBJDETECT_ARUCO_UTILS_HPP__

#include <opencv2/core.hpp>
#include <vector>

namespace cv {
namespace aruco {

/**
 * @brief Copy the contents of a corners vector to an OutputArray, settings its size.
 */
void _copyVector2Output(std::vector<std::vector<Point2f> > &vec, OutputArrayOfArrays out, const float scale = 1.f);

/**
  * @brief Convert input image to gray if it is a 3-channels image
  */
void _convertToGrey(InputArray _in, OutputArray _out);

template<typename T>
inline bool readParameter(const std::string& name, T& parameter, const FileNode& node)
{
    if (!node.empty() && !node[name].empty()) {
        node[name] >> parameter;
        return true;
    }
    return false;
}

template<typename T>
inline bool readWriteParameter(const std::string& name, T& parameter, const FileNode* readNode, FileStorage* writeStorage)
{
    if (readNode)
        return readParameter(name, parameter, *readNode);
    CV_Assert(writeStorage);
    *writeStorage << name << parameter;
    return true;
}

}
}
#endif
