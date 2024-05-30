// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _CODERS_BASE_H_
#define _CODERS_BASE_H_

#include <vector>
#include <memory>
#include <cstdint>

#include <opencv2/core.hpp>

namespace cv {

class BasePointCloudDecoder;
class BasePointCloudEncoder;
using PointCloudDecoder = std::unique_ptr<BasePointCloudDecoder>;
using PointCloudEncoder = std::unique_ptr<BasePointCloudEncoder>;

// for OBJ files: to read vertices, normals and texture coords as they are given in the file
// or duplicate them according to faces
const int READ_AS_IS_FLAG = 1;

///////////////////////////////// base class for decoders ////////////////////////
class BasePointCloudDecoder
{
public:
    virtual ~BasePointCloudDecoder() = default;

    virtual void setSource(const std::string& filename) noexcept;
    virtual void readData(std::vector<Point3f>& points, std::vector<Point3f>& normals, std::vector<Point3f>& rgb);
    virtual void readData(std::vector<Point3f>& points, std::vector<Point3f>& normals, std::vector<Point3f>& rgb,
                          std::vector<Point3f>& texCoords, int& nTexCoords,
                          std::vector<std::vector<int32_t>>& indices, int flags) = 0;

protected:
    std::string m_filename;
};

///////////////////////////////// base class for encoders ////////////////////////
class BasePointCloudEncoder
{
public:
    virtual ~BasePointCloudEncoder() = default;

    virtual void setDestination(const std::string& filename) noexcept;
    virtual void writeData(const std::vector<Point3f>& points, const std::vector<Point3f>& normals, const std::vector<Point3f>& rgb);
    virtual void writeData(const std::vector<Point3f>& points, const std::vector<Point3f>& normals, const std::vector<Point3f>& rgb,
                           const std::vector<Point3f>& texCoords, int nTexCoords,
                           const std::vector<std::vector<int32_t>>& indices) = 0;

protected:
    std::string m_filename;
};

} /* namespace cv */

#endif
