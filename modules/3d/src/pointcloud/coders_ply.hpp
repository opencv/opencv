// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _CODERS_PLY_H_
#define _CODERS_PLY_H_

#include "coders_base.hpp"
#include <istream>
#include <vector>
#include <string>

namespace cv
{
namespace pc
{

enum class DataFormat
{
    ASCII,
    BinaryLittleEndian,
    BinaryBigEndian
};

class PlyDecoder CV_FINAL : public BasePointCloudDecoder
{
public:
    void readData(std::vector<Point3f> &points, std::vector<Point3f> &normals, std::vector<std::vector<int32_t>> &indices) CV_OVERRIDE;

protected:
    void parseHeader(std::ifstream &file);
    void parseBody(std::ifstream &file, std::vector<Point3f> &points, std::vector<Point3f> &normals);

    DataFormat m_inputDataFormat;
    size_t m_vertexCount{0};
    bool m_hasColour{false};
    bool m_hasNormal{false};
};

class PlyEncoder CV_FINAL : public BasePointCloudEncoder
{
public:
    void writeData(std::vector<Point3f> &points, std::vector<Point3f> &normals, std::vector<std::vector<int32_t>> &indices) CV_OVERRIDE;

};

}

}

#endif
