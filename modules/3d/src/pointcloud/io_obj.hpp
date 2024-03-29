// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _CODERS_OBJ_H_
#define _CODERS_OBJ_H_

#include "io_base.hpp"
#include <unordered_set>

namespace cv {

class ObjDecoder CV_FINAL : public BasePointCloudDecoder
{
public:
    void readData(std::vector<Point3f>& points, std::vector<Point3f>& normals, std::vector<Point3f>& rgb) CV_OVERRIDE;
    void readData(std::vector<Point3f>& points, std::vector<Point3f>& normals, std::vector<Point3f>& rgb,
                  std::vector<Point3f>& texCoords, int& nTexCoords,
                  std::vector<std::vector<int32_t>>& indices, int flags) CV_OVERRIDE;

protected:
    static std::unordered_set<std::string> m_unsupportedKeys;
};

class ObjEncoder CV_FINAL : public BasePointCloudEncoder
{
public:
    void writeData(const std::vector<Point3f>& points, const std::vector<Point3f>& normals, const std::vector<Point3f>& rgb,
                   const std::vector<Point3f>& texCoords, int nTexCoords,
                   const std::vector<std::vector<int32_t>>& indices) CV_OVERRIDE;

};

} /* namespace cv */

#endif
