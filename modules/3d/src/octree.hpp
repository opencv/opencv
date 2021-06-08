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
#include "opencv2/core.hpp"

namespace cv {

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
class OctreeNode{
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

    bool isPointInBound(const Point3f& _point, const Point3f& _origin, double _size) const;

    bool isPointInBound(const Point3f& _point) const;

    //! Contains 8 pointers to its 8 children.
    std::vector< Ptr<OctreeNode> > children;

    //! Point to the parent node of the current node. The root node has no parent node and the value is NULL.
    Ptr<OctreeNode> parent = nullptr;

    //! The depth of the current node. The depth of the root node is 0, and the leaf node is equal to the depth of Octree.
    int depth;

    //! The length of the OctreeNode. In space, every OctreeNode represents a cube.
    double size;

    //! Absolute coordinates of the smallest point of the cube.
    //! And the center of cube is `center = origin + Point3f(size/2, size/2, size/2)`.
    Point3f origin;

    /**  The serial number of the child of the current node in the parent node,
    * the range is (-1~7). Among them, only the root node's _parentIndex is -1.
    */
    int parentIndex;

    //! If the OctreeNode is LeafNode.
    bool isLeaf = false;

    //! Contains pointers to all point cloud data in this node.
    std::vector<Point3f> pointList;
};

}
#endif //OPENCV_3D_SRC_OCTREE_HPP