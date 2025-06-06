// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "opencv2/3d/detail/mst.hpp"
#include <queue>
#include <tuple>

namespace
{
    struct DSU {
        std::vector<size_t> parent, rank;
        DSU(size_t n) : parent(n), rank(n, 0) {
            for (size_t i = 0; i < n; ++i)
                parent[i] = i;
        }

        size_t find(size_t x) {
            if (parent[x] != x)
                parent[x] = find(parent[x]);
            return parent[x];
        }

        void unite(size_t x, size_t y) {
            size_t rootX = find(x);
            size_t rootY = find(y);
            if (rootX != rootY) {
                if (rank[rootX] < rank[rootY]) {
                    parent[rootX] = rootY;
                } else if (rank[rootX] > rank[rootY]) {
                    parent[rootY] = rootX;
                } else {
                    parent[rootY] = rootX;
                    ++rank[rootX];
                }
            }
        }
    };

    bool weightComparator(const cv::detail::MSTEdge& a, const cv::detail::MSTEdge& b) {
        if (a.weight != b.weight)
            return a.weight < b.weight;
        if (a.source != b.source)
            return a.source < b.source;
        if (a.target != b.target)
            return a.target < b.target;
        return false;
    }
} // unamed namespace

namespace cv
{
namespace detail
{
std::vector<cv::detail::MSTEdge> buildMSTKruskal(size_t numNodes,
                                                 const std::vector<cv::detail::MSTEdge>& edges)
{
    std::vector<cv::detail::MSTEdge> mst;
    if (numNodes == 0 || edges.empty())
        return mst;

    std::vector<cv::detail::MSTEdge> sortedEdges = edges;
    std::sort(sortedEdges.begin(), sortedEdges.end(), weightComparator);
    DSU dsu(numNodes);

    for (auto &e : sortedEdges) {
        size_t u = e.source, v = e.target;
        CV_Assert(u < numNodes && v < numNodes);
        if (dsu.find(u) != dsu.find(v)) {
            mst.push_back(e);
            dsu.unite(u, v);
        }
    }

    return mst;
}

std::vector<cv::detail::MSTEdge> buildMSTPrim(size_t numNodes,
                                              const std::vector<cv::detail::MSTEdge>& edges,
                                              size_t root)
{
    std::vector<cv::detail::MSTEdge> mst;
    if (numNodes == 0 || edges.empty() || root >= numNodes) return mst;

    std::vector<bool> inMST(numNodes, false);
    std::vector<std::vector<cv::detail::MSTEdge>> adj(numNodes);
    for (const auto& e : edges) {
        size_t u = e.source, v = e.target;
        CV_Assert(u < numNodes && v < numNodes);
        adj[u].push_back({u, v, e.weight});
        adj[v].push_back({v, u, e.weight});
    }

    // Min-heap: (weight, from, to)
    using HeapElem = std::tuple<double, size_t, size_t>;
    std::priority_queue<HeapElem, std::vector<HeapElem>, std::greater<HeapElem>> pq;

    inMST[root] = true;
    for (const auto& e : adj[root]) {
        pq.emplace(e.weight, root, e.target);
    }

    while (!pq.empty()) {
        auto [w, u, v] = pq.top();
        pq.pop();

        if (inMST[v])
            continue;

        inMST[v] = true;
        mst.push_back({u, v, w});
        for (const auto& e : adj[v]) {
            if (!inMST[e.target]) {
                pq.emplace(e.weight, v, e.target);
            }
        }
    }

    return mst;
}

} // namespace detail
} // namespace cv
