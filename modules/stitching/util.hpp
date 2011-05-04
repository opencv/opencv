#ifndef _OPENCV_STITCHING_UTIL_HPP_
#define _OPENCV_STITCHING_UTIL_HPP_

#include <vector>
#include <list>
#include <opencv2/core/core.hpp>

#define ENABLE_LOG 1

#if ENABLE_LOG
  #include <iostream>
  #define LOG(msg) std::cout << msg;
#else
  #define LOG(msg)
#endif

#define LOGLN(msg) LOG(msg << std::endl)


class DjSets
{
public:
    DjSets(int n = 0) { create(n); }

    void create(int n);
    int find(int elem);
    int merge(int set1, int set2);

    std::vector<int> parent;
    std::vector<int> size;

private:
    std::vector<int> rank_;
};


struct GraphEdge
{
    GraphEdge(int from, int to, float weight) 
        : from(from), to(to), weight(weight) {}
    bool operator <(const GraphEdge& other) const { return weight < other.weight; }
    bool operator >(const GraphEdge& other) const { return weight > other.weight; }

    int from, to;
    float weight;
};


class Graph
{
public:
    Graph(int num_vertices = 0) { create(num_vertices); }

    void create(int num_vertices) { edges_.assign(num_vertices, std::list<GraphEdge>()); }

    int numVertices() const { return static_cast<int>(edges_.size()); }

    void addEdge(int from, int to, float weight);

    template <typename B>
    B forEach(B body) const;

    template <typename B> 
    B walkBreadthFirst(int from, B body) const;
    
private:
    std::vector< std::list<GraphEdge> > edges_;
};

#include "util_inl.hpp"

#endif // _OPENCV_STITCHING_UTIL_HPP_
