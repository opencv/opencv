// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Author: Yuhang Wang <yuhangwang0012@gmail.com>
//         Chengwei Ye <broweigg@gmail.com>
//         Zhangjie Cheng <zhangjiec01@gmail.com>

#ifndef OPENCV_PCC_H
#define OPENCV_PCC_H

#include <vector>
#include "opencv2/core.hpp"
#include "octree.hpp"
#include "opencv2/3d.hpp"

namespace cv {


/** @brief class to serialize or deserialize pointcloud data
 *
 * A class for "encoding" pointcloud data to a
 * meaningful, unevenly distributed char vector,
 * so that it can be further compressed by Entropy coding.
 * And the opposite "decoding" way as well.
 *
 * The current implementation is to represent pointcloud as Octree,
 * then traverse OctreeNodes to get vector of "occupancy code".
*/
class OctreeSerializeCoder {
private:
    Octree *octree;
    float resolution;
public:

    OctreeSerializeCoder();

    OctreeSerializeCoder(float resolution);

    //! encode Pointcloud data to serialized char vector
    void encode(const std::vector<Point3f> &pointCloud,
                        std::vector<unsigned char> &serializedVector);

    //! decode Pointcloud data from serialized char vector
    void decode(const std::vector<unsigned char> &serializedVector,
                        std::vector<Point3f> &pointCloud);
};

/** @brief Class to reduce vectorized data size by EntropyCoding
 *
 * The algorithm used here is Range Coding Algorithm.
 *
*/
class EntropyCoder {
public:
    //! encode char vector to bit stream
    void encodeCharVectorToStream(const std::vector<unsigned char> &inputCharVector,
                                  std::ostream &outputStream);

    //! decode char vector from bit stream
    void decodeStreamToCharVector(const std::istream &inputStream,
                                  std::vector<unsigned char> &outputCharVector);
};

/** @brief pointcloud compression class
 *
 * This class enables user to do compression and decompression to pointcloud,
 * currently based on Octree,
 * may support other method (like kd-tree, etc.) in future if necessary.
 *
*/
class PointCloudCompression{
private:
    OctreeSerializeCoder _coder;
    EntropyCoder _entropyCoder;
public:
    /** @brief User compress the pointcloud to stream
     *
     * @param pointCloud the pointcloud to compress
     * @param outputStream the output compressed bit stream destination
    */
    void compress(const std::vector<Point3f> &pointCloud, std::ostream &outputStream);

    /** @brief User decompress(recover) pointcloud from stream
     *
     * @param inputStream the input compressed bit stream source
     * @param pointCloud the output pointcloud
    */
    void decompress(std::istream &inputStream, const std::vector<Point3f> &pointCloud);
};

}

#endif //OPENCV_PCC_H
