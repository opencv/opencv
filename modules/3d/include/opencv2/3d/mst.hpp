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
 * Currently supports Prim's and Kruskal's algorithms.
 * More algorithms may be added in the future.
 */
enum MSTAlgorithm
{
    MST_PRIM = 0,
    MST_KRUSKAL = 1
};

/**
 * @brief Builds a Minimum Spanning Tree (MST) using the selected algorithm (Prim or Kruskal).
 *
 * Additional algorithms may be supported in the future via the @p algorithm parameter.
 *
 * @param numNodes Number of nodes in the graph.
 * @param inputEdges Input vector of edges representing the graph.
 * @param[out] resultingEdges Output vector to store the edges of the resulting MST.
 * @param algorithm Specifies which algorithm to use (e.g., MST_PRIM, MST_KRUSKAL).
 * @param root Root node for Prim's algorithm (ignored if Kruskal or other algorithms are selected).
 * @return true if the MST was successfully built, false otherwise.
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
