// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include <opencv2/3d/mst.hpp>
#include <queue>
#include <tuple>

namespace
{

struct DSU
{
    std::vector<int> parent, rank;
    DSU(int n) : parent(n), rank(n, 0)
    {
        for (int i = 0; i < n; ++i)
            parent[i] = i;
    }

    int find(int x)
    {
        if (parent[x] != x)
            parent[x] = find(parent[x]);
        return parent[x];
    }

    void unite(int x, int y)
    {
        int rootX = find(x), rootY = find(y);
        if (rootX != rootY)
        {
            if (rank[rootX] < rank[rootY])
            {
                parent[rootX] = rootY;
            } else if (rank[rootX] > rank[rootY])
            {
                parent[rootY] = rootX;
            } else
            {
                parent[rootY] = rootX;
                ++rank[rootX];
            }
        }
    }
};

bool weightComparator(const cv::MSTEdge& a, const cv::MSTEdge& b)
{
    if (a.weight != b.weight)
        return a.weight < b.weight;
    if (a.source != b.source)
        return a.source < b.source;
    return a.target < b.target;
}

bool buildMSTKruskal(int numNodes,
                    const std::vector<cv::MSTEdge>& edges,
                    std::vector<cv::MSTEdge>& resultingEdges)
{
    std::vector<cv::MSTEdge> sortedEdges = edges;
    std::sort(sortedEdges.begin(), sortedEdges.end(), weightComparator);
    DSU dsu(numNodes);

    for (const auto &e : sortedEdges)
    {
        int u = e.source, v = e.target;
        if (u >= numNodes || v >= numNodes)
            return false;
        if (dsu.find(u) != dsu.find(v))
        {
            resultingEdges.push_back(e);
            dsu.unite(u, v);
        }
    }

    return true;
}

bool buildMSTPrim(int numNodes,
                  const std::vector<cv::MSTEdge>& edges,
                  std::vector<cv::MSTEdge>& resultingEdges,
                  int root)
{
    std::vector<bool> inMST(numNodes, false);
    std::vector<std::vector<cv::MSTEdge>> adj(numNodes);
    for (const auto& e : edges)
    {
        int u = e.source, v = e.target;
        if (u >= numNodes || v >= numNodes)
            return false;
        adj[u].push_back({u, v, e.weight});
        adj[v].push_back({v, u, e.weight});
    }

    using HeapElem = std::tuple<double, int, int>; // (weight, from, to)
    std::priority_queue<HeapElem, std::vector<HeapElem>, std::greater<HeapElem>> pq;

    inMST[root] = true;
    for (const auto& e : adj[root])
    {
        pq.emplace(e.weight, root, e.target);
    }

    while (!pq.empty())
    {
        auto [w, u, v] = pq.top();
        pq.pop();

        if (inMST[v])
            continue;

        inMST[v] = true;
        resultingEdges.push_back({u, v, w});
        for (const auto& e : adj[v])
        {
            if (!inMST[e.target])
                pq.emplace(e.weight, v, e.target);
        }
    }

    return true;
}

} // unamed namespace

namespace cv
{

bool buildMST(int numNodes,
              const std::vector<cv::MSTEdge>& inputEdges,
              std::vector<cv::MSTEdge>& resultingEdges,
              MSTAlgorithm algorithm,
              int root)
{
    CV_TRACE_FUNCTION();

    resultingEdges.clear();
    if (numNodes <= 0 || inputEdges.empty() || root >= numNodes)
        return false;

    bool result = false;
    switch (algorithm)
    {
        case MST_PRIM:
            result = buildMSTPrim(numNodes, inputEdges, resultingEdges, root);
            break;
        case MST_KRUSKAL:
            result = buildMSTKruskal(numNodes, inputEdges, resultingEdges);
            break;
        default:
            CV_LOG_INFO(NULL, "Invalid MST algorithm specified");
            result = false;
    }

    return (result && resultingEdges.size() >= static_cast<size_t>(numNodes - 1));
}

} // namespace cv
