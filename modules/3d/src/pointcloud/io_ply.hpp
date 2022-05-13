// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _CODERS_PLY_H_
#define _CODERS_PLY_H_

#include "io_base.hpp"
#include <istream>
#include <vector>

namespace cv {

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
    bool parseHeader(std::ifstream &file);
    void parseBody(std::ifstream &file, std::vector<Point3f> &points, std::vector<Point3f> &normals);

    DataFormat m_inputDataFormat;
    size_t m_vertexCount{0};
    bool m_hasColour{false};
    bool m_hasNormal{false};
};

class PlyEncoder CV_FINAL : public BasePointCloudEncoder
{
public:
    void writeData(const std::vector<Point3f> &points, const std::vector<Point3f> &normals, const std::vector<std::vector<int32_t>> &indices) CV_OVERRIDE;

};

} /* namespace cv */

#endif
