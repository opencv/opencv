// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021, Huawei Technologies Co., Ltd. All rights reserved.
// Third party copyrights are property of their respective owners.
//
// Author: Zihao Mu <zihaomu6@gmail.com>
//         Liangqian Kong <chargerKong@126.com>
//         Longbu Wang <riskiest@gmail.com>

#ifndef OPENCV_3D_SRC_OCTREE_HPP
#define OPENCV_3D_SRC_OCTREE_HPP

#include <vector>
#include <array>
#include "opencv2/core.hpp"

namespace cv
{
// Forward declaration
class OctreeKey;

/** @brief OctreeNode for Octree.

The class OctreeNode represents the node of the octree. Each node contains 8 children, which are used to divide the
space cube into eight parts. Each octree node represents a cube.
And these eight children will have a fixed order, the order is described as follows:

For illustration, assume,
    rootNode: origin == (0, 0, 0), size == 2
 Then,
    children[0]: origin == (0, 0, 0), size == 1
    children[1]: origin == (1, 0, 0), size == 1, along X-axis next to child 0
    children[2]: origin == (0, 1, 0), size == 1, along Y-axis next to child 0
    children[3]: origin == (1, 1, 0), size == 1, in X-Y plane
    children[4]: origin == (0, 0, 1), size == 1, along Z-axis next to child 0
    children[5]: origin == (1, 0, 1), size == 1, in X-Z plane
    children[6]: origin == (0, 1, 1), size == 1, in Y-Z plane
    children[7]: origin == (1, 1, 1), size == 1, furthest from child 0

There are two kinds of nodes in an octree, intermediate nodes and leaf nodes, which are distinguished by isLeaf.
Intermediate nodes are used to contain leaf nodes, and leaf nodes will contain pointers to all pointcloud data
within the node, which will be used for octree indexing and mapping from point clouds to octree. Note that,
in an octree, each leaf node contains at least one point cloud data. Similarly, every intermediate OctreeNode
contains at least one non-empty child pointer, except for the root node.
*/
class OctreeNode
{
public:

    /**
    * There are multiple constructors to create OctreeNode.
    * */
    OctreeNode();

    /** @overload
    *
    * @param _depth The depth of the current node. The depth of the root node is 0, and the leaf node is equal
    * to the depth of Octree.
    * @param _size The length of the OctreeNode. In space, every OctreeNode represents a cube.
    * @param _origin The absolute coordinates of the center of the cube.
    * @param _parentIndex The serial number of the child of the current node in the parent node,
    * the range is (-1~7). Among them, only the root node's _parentIndex is -1.
    */
    OctreeNode(int _depth, double _size, const Point3f& _origin, int _parentIndex);

    //! returns true if the rootNode is NULL.
    bool empty() const;

    bool isPointInBound(const Point3f& _point) const;

    bool overlap(const Point3f& query, float squareRadius) const;

    void KNNSearchRecurse(const Point3f& query, const int K, float& smallestDist, std::vector<std::tuple<float, Point3f, Point3f>>& candidatePoint) const;

    //! Contains 8 pointers to its 8 children.
    std::array<Ptr<OctreeNode>, 8> children;

    //! Point to the parent node of the current node. The root node has no parent node and the value is NULL.
    OctreeNode* parent;

    //! The depth of the current node. The depth of the root node is 0, and the leaf node is equal to the depth of Octree.
    int depth;

    //! The length of the OctreeNode. In space, every OctreeNode represents a cube.
    double size;

    //! Absolute coordinates of the smallest point of the cube.
    //! And the center of cube is `center = origin + Point3f(size/2, size/2, size/2)`.
    Point3f origin;

    //! RAHTCoefficient of octree node, used for color attribute compression.
    Point3f RAHTCoefficient = { };

    /**  The list of 6 adjacent neighbor node.
        *    index mapping:
        *     +z                        [101]
        *      |                          |    [110]
        *      |                          |  /
        *      O-------- +x    [001]----{000} ----[011]
        *     /                       /   |
        *    /                   [010]    |
        *  +y                           [100]
        *  index 000, 111 are reserved
        */
    std::array<Ptr<OctreeNode>, 8> neigh;

    /**  The serial number of the child of the current node in the parent node,
    * the range is (-1~7). Among them, only the root node's _parentIndex is -1.
    */
    int parentIndex;

    //! If the OctreeNode is LeafNode.
    bool isLeaf = false;

    //! Contains pointers to all point cloud data in this node.
    std::vector<Point3f> pointList;

    //! color attribute of octree node.
    std::vector<Point3f> colorList;
};

/** @brief Key for pointCloud, used to compute the child node index through bit operations.

When building the octree, the point cloud data is firstly voxelized/discretized: by inserting
all the points into a voxel coordinate system. For example, when resolution is set to 0.01, a point
with coordinate Point3f(0.251,0.502,0.753) would be transformed to:(0.251/0.01,0.502/0.01,0.753/0.01)
=(25,50,75). And the OctreeKey will be (x_key:1_1001,y_key:11_0010,z_key:100_1011). Assume the Octree->depth
is 100_0000, It can quickly calculate the index of the child nodes at each layer.
layer    Depth Mask   x&Depth Mask    y&Depth Mask    z&Depth Mask    Child Index(0-7)
1        100_0000     0               0               1               4
2        10_0000      0               1               0               2
3        1_0000       1               1               0               3
4        1000         1               0               1               5
5        100          0               0               0               0
6        10           0               1               1               6
7        1            1               0               1               5
*/

class OctreeKey
{
public:
    size_t x_key;
    size_t y_key;
    size_t z_key;

public:
    OctreeKey() : x_key(0), y_key(0), z_key(0) { }
    OctreeKey(size_t x, size_t y, size_t z) : x_key(x), y_key(y), z_key(z) { }

    /** @brief compute the child node index through bit operations.
    *
    * @param mask The mask of specify layer.
    * @return the index of child(0-7)
    */
    inline unsigned char findChildIdxByMask(size_t mask) const
    {
        return static_cast<unsigned char>((!!(z_key & mask))<<2) | ((!!(y_key & mask))<<1) | (!!(x_key & mask));
    }

    /** @brief get occupancy code from node.
    *
    * The occupancy code type is unsigned char that represents whether the eight child nodes of the octree node exist
    * If a octree node has 3 child which indexes are 0,1,7, then the occupancy code of this node is 1000_0011
    * @param node The octree node.
    * @return the occupancy code(0000_0000-1111_1111)
    */
    static inline unsigned char getBitPattern(OctreeNode &node)
    {
        unsigned char res = 0;
        for (unsigned char i = 0; i < node.children.size(); i++)
        {
            res |= static_cast<unsigned char>((!node.children[i].empty()) << i);
        }
        return res;
    }
};

}
#endif //OPENCV_3D_SRC_OCTREE_HPP