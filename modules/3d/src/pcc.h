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
 * A class for "encoding" pointCloud data to a
 * meaningful, unevenly distributed char vector,
 * so that it can be further compressed by Entropy coding.
 * And the opposite "decoding" way as well.
 *
 * The current implementation is to represent pointCloud as Octree,
 * then traverse OctreeNodes to get vector of "occupancy code".
*/
class OctreeSerializeCoder {
private:
    Octree *octree;
public:
    //! Default constructor.
    OctreeSerializeCoder(){
        this->octree=new Octree();
    };

    /** @brief encode pointCloud data to serialized char vector
    *
    * Based on the specified resolution, an octree is constructed from the point cloud data.
    * The octree is then serialized into meaningful, unevenly distributed char vector, which
    * is used for entropy coding to further compress the data. At the same time, the octree
    * information is written to the outputStream header.
    * @param pointCloud The point cloud data.
    * @param serializedVector Used for storing the serialized char vector.
    * @param resolution The size of the Octree leaf node.
    * @param outputStream The output stream, will be written to point cloud compressed file.
    */
    void encode(const std::vector<Point3f> &pointCloud, const std::vector<Point3f> &colorAttribute,
                std::vector<unsigned char> &serializedVector, double resolution, double qStep,
                std::ostream &outputStream);

    /** @brief decode pointCloud data from serialized char vector
    *
    * Based on the specified resolution and origin, an octree is restored from the serialized char vector.
    * Then, the point cloud data is extracted from the octree.
    * @param pointCloud Used for storing the point cloud data.
    * @param serializedVector The serialized char vector.
    * @param resolution The size of the Octree leaf node.
    * @param origin The vertex of the cube represented by the octree root node.
    */
    void decode(std::vector<Point3f> &pointCloud, const std::vector<unsigned char> &serializedVector,
                double resolution, Point3f &origin);
};

/** @brief Class to reduce vectorized data size by EntropyCoding
 *
 * The algorithm used here is Range Coding Algorithm.
 *
*/
class EntropyCoder {
public:
    /** @brief encode char vector to bit stream.
    *
    * @param inputCharVector Char vector for entropy encoding.
    * @param outputStream The output stream, will be written to point cloud compressed file.
    */
    static void encodeCharVectorToStream(std::vector<unsigned char> &inputCharVector,
                                  std::ostream &outputStream);

    /** @brief decode char vector from bit stream
    *
    * @param inputStream The point cloud compressed file.
    * @param outputCharVector The output Char vector, used for storing the char vector.
    */
    static void decodeStreamToCharVector(std::istream &inputStream,
                                  std::vector<unsigned char> &outputCharVector);
};

/** @brief pointCloud compression class
 *
 * This class enables user to do compression and decompression to pointCloud,
 * currently based on Octree,
 * may support other method (like kd-tree, etc.) in future if necessary.
 *
*/
class PointCloudCompression{
private:
    OctreeSerializeCoder _coder=OctreeSerializeCoder();
    EntropyCoder _entropyCoder=EntropyCoder();
public:
    /** @brief User compress the pointcloud to stream.
     *
     * @param pointCloud the pointcloud to compress.
     * @param resolution the size of the leaf node.
     * @param outputStream the output compressed bit stream destination.
    */
    void compress(const std::vector<Point3f> &pointCloud, double resolution, std::ostream &outputStream,
                  const std::vector<Point3f> &colorAttribute = std::vector<Point3f>(), double qStep = -1.0);

    /** @brief User decompress(recover) pointcloud from stream.
     *
     * @param inputStream the input compressed bit stream source.
     * @param pointCloud the output pointcloud.
    */
    void decompress(std::istream &inputStream, std::vector<Point3f> &pointCloud);
};

}

#endif //OPENCV_PCC_H
