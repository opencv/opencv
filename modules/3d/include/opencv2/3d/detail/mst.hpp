// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_3D_DETAIL_MST_HPP
#define OPENCV_3D_DETAIL_MST_HPP

#include <vector>

namespace cv
{
namespace detail
{

// represents an edge in a graph for MST computation
struct CV_EXPORTS MSTEdge
{
    size_t source, target;
    double weight;
};

// builds a MST using Prim's algorithm
CV_EXPORTS std::vector<MSTEdge> buildMSTPrim(
    const size_t numNodes,
    const std::vector<MSTEdge>& edges,
    size_t root = 0
);

// builds a MST using Kruskal's algorithm
CV_EXPORTS std::vector<MSTEdge> buildMSTKruskal(
    const size_t numNodes,
    const std::vector<MSTEdge>& edges
);

} // namespace detail
} // namespace cv

#endif // include guard
