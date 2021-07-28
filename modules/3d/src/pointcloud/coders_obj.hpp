// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _CODERS_OBJ_H_
#define _CODERS_OBJ_H_

#include "coders_base.hpp"

namespace cv
{
namespace pc
{

class ObjDecoder CV_FINAL : public BasePointCloudDecoder
{
public:
    void readData(std::vector<Point3f> &points, std::vector<Point3f> &normals, std::vector<std::vector<int32_t>> &indices) CV_OVERRIDE;

};

class ObjEncoder CV_FINAL : public BasePointCloudEncoder
{
public:
    void writeData(std::vector<Point3f> &points, std::vector<Point3f> &normals, std::vector<std::vector<int32_t>> &indices) CV_OVERRIDE;

};

}

}

#endif
