// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"
#include <opencv2/3d/detail/mst.hpp>

namespace opencv_test {
namespace {

using namespace cv;

typedef tuple<size_t /*MSTalgorithm*/,
              size_t /*numNodes*/,
              std::vector<detail::MSTEdge>/*edges*/,
              std::vector<detail::MSTEdge>/*expectedEdges*/
             > MSTParamType;
typedef testing::TestWithParam<MSTParamType> MST;

TEST_P(MST, checkCorrectness)
{
    const size_t algorithm = get<0>(GetParam());
    const size_t numNodes = get<1>(GetParam());
    const std::vector<detail::MSTEdge>& edges = get<2>(GetParam());
    const std::vector<detail::MSTEdge>& expectedEdges = get<3>(GetParam());

    std::vector<detail::MSTEdge> mstEdges;

    switch (algorithm) {
        case 0: /* Prim */
            // Select first node for root
            mstEdges = detail::buildMSTPrim(numNodes, edges, 0);
            break;

        case 1: /* Kruskal*/
            mstEdges = detail::buildMSTKruskal(numNodes, edges);
            break;

        default:
            FAIL() << "Unknown selected MST algorithm: " << algorithm;
    }

    EXPECT_EQ(mstEdges.size(), expectedEdges.size());
    for (const auto& edge : expectedEdges)
    {
        auto it = std::find_if(mstEdges.begin(), mstEdges.end(), [&edge](const detail::MSTEdge& e) {
            return (e.source == edge.source && e.target == edge.target) ||
                    (e.source == edge.target && e.target == edge.source);
        });
        EXPECT_TRUE(it != mstEdges.end()) << "Missing expected edge: "
            << edge.source << " -> " << edge.target;
    }
}

const MSTParamType mst_graphs[] =
{
    // Small Graph
    MSTParamType(0, 4,
        {
            {0, 1, 1.0}, {0, 2, 2.0}, {1, 2, 1.5}, {1, 3, 2.5}, {2, 3, 1.0}
        },
        {
            {0, 1, 1.0}, {1, 2, 1.5}, {2, 3, 1.0}
        }
    ),

    MSTParamType(1, 4,
        {
            {0, 1, 1.0}, {0, 2, 2.0}, {1, 2, 1.5}, {1, 3, 2.5}, {2, 3, 1.0}
        },
        {
            {0, 1, 1.0}, {1, 2, 1.5}, {2, 3, 1.0}
        }
    ),

    // Disconnected Graph
    MSTParamType(0, 4,
        {
            {0, 1, 1.0}, {2, 3, 2.0}
        },
        {
            {0, 1, 1.0}
        }
    ),

    MSTParamType(1, 4,
        {
            {0, 1, 1.0}, {2, 3, 2.0}
        },
        {
            {0, 1, 1.0}, {2, 3, 2.0}
        }
    ),

    // Fully Disconnected
    MSTParamType(0, 6,
        {

        },
        {

        }
    ),

    MSTParamType(1, 6,
        {

        },
        {

        }
    ),

    // 2 Nodes, 1 Edge
    MSTParamType(0, 2,
        {
            {0, 1, 42.0}
        },
        {
            {0, 1, 42.0}
        }
    ),

    MSTParamType(1, 2,
        {
            {0, 1, 42.0}
        },
        {
            {0, 1, 42.0}
        }
    ),

    // Dense graph (clique)
    MSTParamType(0, 4,
        {
            {0, 1, 1.0}, {0, 2, 2.0}, {0, 3, 3.0},
            {1, 2, 1.5}, {1, 3, 2.5}, {2, 3, 1.0}
        },
        {
            {0, 1, 1.0}, {2, 3, 1.0}, {1, 2, 1.5}
        }
    ),

    MSTParamType(1, 4,
        {
            {0, 1, 1.0}, {0, 2, 2.0}, {0, 3, 3.0},
            {1, 2, 1.5}, {1, 3, 2.5}, {2, 3, 1.0}
        },
        {
            {0, 1, 1.0}, {2, 3, 1.0}, {1, 2, 1.5}
        }
    ),

    // Sparse
    MSTParamType(0, 4,
        {
            {0, 1, 1.0}, {1, 2, 2.0}, {1, 3, 3.0}
        },
        {
            {0, 1, 1.0}, {1, 2, 2.0}, {1, 3, 3.0}
        }
    ),

    MSTParamType(1, 4,
        {
            {0, 1, 1.0}, {1, 2, 2.0}, {1, 3, 3.0}
        },
        {
            {0, 1, 1.0}, {1, 2, 2.0}, {1, 3, 3.0}
        }
    ),

    // Weight Floating point check
    MSTParamType(0, 3,
        {
            {0, 1, 1.000001}, {1, 2, 1.000002}, {0, 2, 1.000003}
        },
        {
            {0, 1, 1.000001}, {1, 2, 1.000002}
        }
    ),

    MSTParamType(1, 3,
        {
            {0, 1, 1.000001}, {1, 2, 1.000002}, {0, 2, 1.000003}
        },
        {
            {0, 1, 1.000001}, {1, 2, 1.000002}
        }
    ),

    // 0 or ~0 weight valuess
    MSTParamType(0, 3,
        {
            {0, 1, 0.0}, {1, 2, 1e-9}, {0, 2, 1.0}
        },
        {
            {0, 1, 0.0}, {1, 2, 1e-9}
        }
    ),

    MSTParamType(0, 3,
        {
            {0, 1, 0.0}, {1, 2, 1e-9}, {0, 2, 1.0}
        },
        {
            {0, 1, 0.0}, {1, 2, 1e-9}
        }
    ),

    // Duplicate edges (picks the one with the smallest weight)
    MSTParamType(0, 3,
        {
            {0, 1, 3.0}, {0, 1, 1.0}, {1, 2, 2.0}
        },
        {
            {0, 1, 1.0}, {1, 2, 2.0}
        }
    ),

    MSTParamType(1, 3,
        {
            {0, 1, 3.0}, {0, 1, 1.0}, {1, 2, 2.0}
        },
        {
            {0, 1, 1.0}, {1, 2, 2.0}
        }
    ),

    // Negative weights
    MSTParamType(0, 3,
        {
            {0, 1, -1.0}, {1, 2, -2.0}, {0, 2, -3.0}
        },
        {
            {1, 2, -2.0}, {0, 2, -3.0}
        }
    ),

    MSTParamType(1, 3,
        {
            {0, 1, -1.0}, {1, 2, -2.0}, {0, 2, -3.0}
        },
        {
            {0, 2, -3.0}, {1, 2, -2.0}
        }
    ),
};

inline static std::string MST_name_printer(const testing::TestParamInfo<MST::ParamType>& info)
{
    std::ostringstream os;
    const auto& algorithm = get<0>(info.param);
    const auto& numNodes = get<1>(info.param);
    const auto& edges = get<2>(info.param);
    const auto& expectedEdges = get<3>(info.param);

    os << "TestCase_" << info.index << "_";
    switch (algorithm)
    {
    case 0: os << "Prim"; break;
    case 1: os << "Kruskal"; break;
    default: os << "Unknown algorithm"; break;
    }
    os << "_Nodes_" << numNodes;
    os << "_Edges_" << edges.size();
    os << "_ExpectedEdges_" << expectedEdges.size();

    return os.str();
}

INSTANTIATE_TEST_CASE_P(/**/, MST, testing::ValuesIn(mst_graphs), MST_name_printer);

TEST(MSTstress, LargeGraph)
{
    const size_t numNodes = 100000;

    std::vector<detail::MSTEdge> edges;

    for (size_t i = 0; i < numNodes - 1; ++i)
        edges.push_back({i, i + 1, static_cast<double>(i + 1)});

    // Add extra edges for complexity
    for (size_t i = 0; i < numNodes - 10; i += 10)
        edges.push_back({i, i + 10,  static_cast<double>(i)});
    for (size_t i = 0; i + 20 < numNodes; i += 5)
        edges.push_back({i, i + 20, static_cast<double>(i + 1)});
    for (size_t i = 0; i + 30 < numNodes; i += 3)
        edges.push_back({i, i + 30, static_cast<double>(i % 50 + 1)});
    for (size_t i = 50; i < numNodes; i += 10)
        edges.push_back({i, i - 25, static_cast<double>(i % 100 + 2)});

    std::vector<detail::MSTEdge> primMST = detail::buildMSTPrim(numNodes, edges, 0);
    std::vector<detail::MSTEdge> kruskalMST = detail::buildMSTKruskal(numNodes, edges);

    EXPECT_EQ(primMST.size(), numNodes - 1);
    EXPECT_EQ(kruskalMST.size(), numNodes - 1);
}

}} // namespace
