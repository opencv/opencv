#ifndef __OPENCV_STITCHING_UTIL_INL_HPP__
#define __OPENCV_STITCHING_UTIL_INL_HPP__

#include <queue>
#include "util.hpp" // Make your IDE see declarations

template <typename B>
B Graph::forEach(B body) const
{
    for (int i = 0; i < numVertices(); ++i)
    {
        std::list<GraphEdge>::const_iterator edge = edges_[i].begin();
        for (; edge != edges_[i].end(); ++edge)
            body(*edge);
    }
    return body;
}


template <typename B>
B Graph::walkBreadthFirst(int from, B body) const
{
    std::vector<bool> was(numVertices(), false);
    std::queue<int> vertices;

    was[from] = true;
    vertices.push(from);

    while (!vertices.empty())
    {
        int vertex = vertices.front();
        vertices.pop();

        std::list<GraphEdge>::const_iterator edge = edges_[vertex].begin();
        for (; edge != edges_[vertex].end(); ++edge)
        {
            if (!was[edge->to])
            {
                body(*edge);
                was[edge->to] = true;
                vertices.push(edge->to);
            }
        }
    }

    return body;
}


//////////////////////////////////////////////////////////////////////////////
// Some auxiliary math functions

static inline
float normL2(const cv::Point3f& a, const cv::Point3f& b)
{
    return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z);
}


static inline
double normL2sq(const cv::Mat &r)
{
    return r.dot(r);
}


template <typename T>
static inline
T sqr(T x)
{
    return x * x;
}

#endif // __OPENCV_STITCHING_UTIL_INL_HPP__
