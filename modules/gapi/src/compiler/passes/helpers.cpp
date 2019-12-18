// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include <algorithm>     // copy
#include <unordered_map>
#include <unordered_set>

#include <ade/util/filter_range.hpp>

#include <opencv2/gapi/own/assert.hpp> // GAPI_Assert
#include "compiler/passes/helpers.hpp"

namespace {
namespace Cycles
{
    // FIXME: This code is taken directly from ADE.
    // export a bool(ade::Graph) function with pass instead
    enum class TraverseState
    {
        visiting,
        visited,
    };
    using state_t = std::unordered_map<ade::Node*, TraverseState>;

    bool inline checkCycle(state_t& state, const ade::NodeHandle& node)
    {
        GAPI_Assert(nullptr != node);
        state[node.get()] = TraverseState::visiting;
        for (auto adj: node->outNodes())
        {
            auto it = state.find(adj.get());
            if (state.end() == it) // not visited
            {
                // FIXME: use std::stack instead on-stack recursion
                if (checkCycle(state, adj))
                {
                    return true; // detected! (deeper frame)
                }
            }
            else if (TraverseState::visiting == it->second)
            {
                return true; // detected! (this frame)
            }
        }
        state[node.get()] = TraverseState::visited;
        return false; // not detected
    }

    bool inline hasCycles(const ade::Graph &graph)
    {
        state_t state;
        bool detected = false;
        for (auto node: graph.nodes())
        {
            if (state.end() == state.find(node.get()))
            {
                // not yet visited during recursion
                detected |= checkCycle(state, node);
                if (detected) break;
            }
        }
        return detected;
    }
} // namespace Cycles

namespace TopoSort
{
    using sorted_t = std::vector<ade::NodeHandle>;
    using visited_t = std::unordered_set<ade::Node*>;

    struct NonEmpty final
    {
        bool operator()(const ade::NodeHandle& node) const
        {
            return nullptr != node;
        }
    };

    void inline visit(sorted_t& sorted, visited_t& visited, const ade::NodeHandle& node)
    {
        if (visited.end() == visited.find(node.get()))
        {
            for (auto adj: node->inNodes())
            {
                visit(sorted, visited, adj);
            }
            sorted.push_back(node);
            visited.insert(node.get());
        }
    }

    sorted_t inline topoSort(const ade::Graph &g)
    {
        sorted_t sorted;
        visited_t visited;
        for (auto node: g.nodes())
        {
            visit(sorted, visited, node);
        }

        auto r = ade::util::filter<NonEmpty>(ade::util::toRange(sorted));
        return sorted_t(r.begin(), r.end());
    }
} // namespace TopoSort

} // anonymous namespace

bool cv::gimpl::pass_helpers::hasCycles(const ade::Graph &g)
{
    return Cycles::hasCycles(g);
}

std::vector<ade::NodeHandle> cv::gimpl::pass_helpers::topoSort(const ade::Graph &g)
{
    return TopoSort::topoSort(g);
}
