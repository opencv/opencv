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
    * @param colorAttribute The color attribute data.
    * @param serializedVector Used for storing the serialized char vector.
    * @param resolution The size of the Octree leaf node.
    * @param outputStream The output stream, will be written to point cloud compressed file.
    */
    void encode(const std::vector<Point3f> &pointCloud, const std::vector<Point3f> &colorAttribute,
                std::vector<unsigned char> &serializedVector, double resolution, std::ostream &outputStream);

    /** @brief decode pointCloud data from serialized char vector
    *
    * Based on the specified resolution and origin, an octree is restored from the serialized char vector.
    * Then, the point cloud data is extracted from the octree.
    * @param serializedVector The serialized char vector.
    * @param resolution The size of the Octree leaf node.
    * @param origin The vertex of the cube represented by the octree root node.
    * @param maxDepth The depth of the Octree leaf node.
    */
    void decode(const std::vector<unsigned char> &serializedVector, std::vector<Point3f> &pointCloud, double resolution, Point3f &origin, size_t maxDepth);

    /** @brief encode color data to serialized char vector
    *
    * Based on the specified quantization step, an octree's color attribute is encoded to a serialized char vector.
    * @param qStep Parameter for quantization.
    * @param colorCode The serialized char vector for color attribute.
    */
    void encodeColor(float qStep, std::vector<unsigned char> &colorCode);

    /** @brief decode color data from serialized char vector
    *
    * Based on the specified quantization step, an octree's color attribute is restored from the serialized char vector.
    * @param qStep Parameter for quantization.
    * @param colorCode The input serialized char vector for color attribute.
    */
    void decodeColor(float qStep, const std::vector<unsigned char> &colorCode, std::vector<Point3f> &colorAttribute);

    /** @brief get the octree instance **/
    Octree *getOctree() const { return octree; };
};

/** @brief to select EntropyCoder's coding Algorithm
*
*/
enum class EntropyCodingMethod {
    RANGE_CODING_METHOD,
    ZLIB_METHOD
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
private:
    //! Select EntropyCoding Algorithm.
    static EntropyCodingMethod codingMethod;
};


/** @brief Traversing octree by BFS.
 *
 * By traversing the octree, the octree is represented as vector for further compression.
 * Each node is represented as a unsigned char.
 * @param root the root node of octree.
 * @param serializedVectorOut The vector obtained after traversing the octree.
*/
void traverse(OctreeNode &root, std::vector<unsigned char> &serializedVectorOut);

/** @brief Restore octree from vector.
 *
 * @param root the root node of octree.
 * @param serializedVectorOut The vector obtained by traversing the octree.
*/
void restore(OctreeNode &root, const std::vector<unsigned char> &serializedVectorIn);

/** @brief 3D Haar Transform
 *
 * @param node the current octree node.
 * @param haarCoefficients The vector storing Haar coefficients.
 * @param cubes The vector of octree nodes to avoid wasting memory.
 * @param N The index.
*/
void Haar3DRecursive(OctreeNode *node, std::vector<Point3f> &haarCoefficients, std::vector<OctreeNode *> &cubes,
                         size_t &N);
/** @brief Inverse 3D Haar Transform
 *
 * @param node the current octree node.
 * @param haarCoefficients The vector storing Haar coefficients.
 * @param cubes The vector of octree nodes to avoid wasting memory.
 * @param N The index.
*/
void invHaar3DRecursive(OctreeNode *node, std::vector<Point3f> &haarCoefficients, std::vector<OctreeNode *> &cubes,
                            size_t &N, std::vector<Point3f> &colorAttribute);

}

#endif //OPENCV_PCC_H
