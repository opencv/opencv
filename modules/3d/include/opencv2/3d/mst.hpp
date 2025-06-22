// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_3D_DETAIL_MST_HPP
#define OPENCV_3D_DETAIL_MST_HPP

#include <vector>

namespace cv
{

// represents an edge in a graph for MST computation
struct CV_EXPORTS_W_SIMPLE MSTEdge
{
    int source, target;
    double weight;
};

enum MSTAlgorithm
{
    MST_PRIM = 0,
    MST_KRUSKAL = 1
};

// builds a MST using the selected algorithm (Prim or Kruskal).
CV_EXPORTS std::vector<MSTEdge> buildMST(
    int numNodes,
    const std::vector<MSTEdge>& edges,
    MSTAlgorithm algorithm,
    int root = 0 // ignored if Kruskal is selected
);

} // namespace cv

#endif // include guard
