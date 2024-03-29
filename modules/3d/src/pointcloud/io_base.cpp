// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "io_base.hpp"

namespace cv {

void BasePointCloudDecoder::setSource(const std::string& filename) noexcept
{
    m_filename = filename;
}

void BasePointCloudDecoder::readData(std::vector<Point3f> &points, std::vector<Point3f> &normals, std::vector<Point3f> &rgb)
{
    std::vector<std::vector<int32_t>> indices;
    std::vector<Point3f> texCoords;
    int nTexCoords;
    readData(points, normals, rgb, texCoords, nTexCoords, indices, READ_AS_IS_FLAG);
}

void BasePointCloudEncoder::setDestination(const std::string& filename) noexcept
{
    m_filename = filename;
}

void BasePointCloudEncoder::writeData(const std::vector<Point3f> &points, const std::vector<Point3f> &normals, const std::vector<Point3f> &rgb)
{
    std::vector<std::vector<int32_t>> indices;
    std::vector<Point3f> texCoords;
    writeData(points, normals, rgb, texCoords, 0, indices);
}

} /* namespace cv */
