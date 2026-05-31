// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_3D_MST_HPP
#define OPENCV_3D_MST_HPP

#include <vector>

namespace cv
{

/**
 * @brief Represents an edge in a graph for Minimum Spanning Tree (MST) computation.
 *
 * Each edge connects two nodes (source and target) and has an associated weight.
 */
struct CV_EXPORTS_W_SIMPLE MSTEdge
{
    CV_PROP_RW int source, target;
    CV_PROP_RW double weight;
};

/**
 * @brief Represents the algorithms available for building a Minimum Spanning Tree (MST).
 *
 * More algorithms may be added in the future.
 */
enum MSTAlgorithm
{
    MST_PRIM = 0,
    MST_KRUSKAL = 1
};

/**
 * @brief Builds a Minimum Spanning Tree (MST) using the specified algorithm (see @ref MSTAlgorithm).
 *
 * Supports graphs with negative edge weights. Self-loop edges (edges where source and target are the
 * same) are ignored. If multiple edges exist between the same pair of nodes, only the one with the
 * lowest weight is considered. If the graph is disconnected or input is invalid, the function
 * returns false.
 *
 * @note The @p root parameter is ignored for algorithms that do not require a starting node.
 * @note Additional MST algorithms may be supported in the future via the @p algorithm parameter
 * (see @ref MSTAlgorithm).
 *
 * @param numNodes Number of nodes in the graph (must be greater than 0).
 * @param inputEdges Input vector of edges representing the graph.
 * @param[out] resultingEdges Output vector to store the edges of the resulting MST.
 * @param algorithm Specifies which algorithm to use to compute the MST (see @ref MSTAlgorithm).
 * @param root Starting node for the MST algorithm (only used for certain algorithms).
 * @return true if a valid MST was successfully built; false otherwise.
 * @throws cv::Error (StsBadArg) if an invalid algorithm is specified.
 */
CV_EXPORTS_W bool buildMST(
    int numNodes,
    const std::vector<MSTEdge>& inputEdges,
    CV_OUT std::vector<MSTEdge>& resultingEdges,
    MSTAlgorithm algorithm,
    int root = 0
);

} // namespace cv

#endif // include guard
