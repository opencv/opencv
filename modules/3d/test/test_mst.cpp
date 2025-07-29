// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"
#include <opencv2/3d/mst.hpp>

namespace opencv_test {
namespace {

using namespace cv;

typedef tuple<MSTAlgorithm /*MSTalgorithm*/,
              int /*numNodes*/,
              std::vector<MSTEdge>/*edges*/,
              std::vector<MSTEdge>/*expectedEdges*/
             > MSTParamType;
typedef testing::TestWithParam<MSTParamType> MST;

TEST_P(MST, checkCorrectness)
{
    const int algorithm = get<0>(GetParam());
    const int numNodes = get<1>(GetParam());
    const std::vector<MSTEdge>& edges = get<2>(GetParam());
    const std::vector<MSTEdge>& expectedEdges = get<3>(GetParam());

    std::vector<MSTEdge> mstEdges;
    bool result = false;

    switch (algorithm) {
        case MST_PRIM:
            // Select first node for root
            result = buildMST(numNodes, edges, mstEdges, MST_PRIM, 0);
            break;

        case MST_KRUSKAL:
            result = buildMST(numNodes, edges, mstEdges, MST_KRUSKAL, 0);
            break;

        default:
            FAIL() << "Unknown selected MST algorithm: " << algorithm;
    }

    EXPECT_TRUE(result);
    EXPECT_EQ(mstEdges.size(), expectedEdges.size());
    for (const auto& edge : expectedEdges)
    {
        auto it = std::find_if(mstEdges.begin(), mstEdges.end(), [&edge](const MSTEdge& e) {
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
    MSTParamType(MST_PRIM, 4,
        {
            {0, 1, 1.0}, {0, 2, 2.0}, {1, 2, 1.5}, {1, 3, 2.5}, {2, 3, 1.0}
        },
        {
            {0, 1, 1.0}, {1, 2, 1.5}, {2, 3, 1.0}
        }
    ),

    MSTParamType(MST_KRUSKAL, 4,
        {
            {0, 1, 1.0}, {0, 2, 2.0}, {1, 2, 1.5}, {1, 3, 2.5}, {2, 3, 1.0}
        },
        {
            {0, 1, 1.0}, {1, 2, 1.5}, {2, 3, 1.0}
        }
    ),

    // 2 Nodes, 1 Edge
    MSTParamType(MST_PRIM, 2,
        {
            {0, 1, 42.0}
        },
        {
            {0, 1, 42.0}
        }
    ),

    MSTParamType(MST_KRUSKAL, 2,
        {
            {0, 1, 42.0}
        },
        {
            {0, 1, 42.0}
        }
    ),

    // Dense graph (clique)
    MSTParamType(MST_PRIM, 4,
        {
            {0, 1, 1.0}, {0, 2, 2.0}, {0, 3, 3.0},
            {1, 2, 1.5}, {1, 3, 2.5}, {2, 3, 1.0}
        },
        {
            {0, 1, 1.0}, {2, 3, 1.0}, {1, 2, 1.5}
        }
    ),

    MSTParamType(MST_KRUSKAL, 4,
        {
            {0, 1, 1.0}, {0, 2, 2.0}, {0, 3, 3.0},
            {1, 2, 1.5}, {1, 3, 2.5}, {2, 3, 1.0}
        },
        {
            {0, 1, 1.0}, {2, 3, 1.0}, {1, 2, 1.5}
        }
    ),

    // Sparse
    MSTParamType(MST_PRIM, 4,
        {
            {0, 1, 1.0}, {1, 2, 2.0}, {1, 3, 3.0}
        },
        {
            {0, 1, 1.0}, {1, 2, 2.0}, {1, 3, 3.0}
        }
    ),

    MSTParamType(MST_KRUSKAL, 4,
        {
            {0, 1, 1.0}, {1, 2, 2.0}, {1, 3, 3.0}
        },
        {
            {0, 1, 1.0}, {1, 2, 2.0}, {1, 3, 3.0}
        }
    ),

    // Weight Floating point check
    MSTParamType(MST_PRIM, 3,
        {
            {0, 1, 1.000001}, {1, 2, 1.000002}, {0, 2, 1.000003}
        },
        {
            {0, 1, 1.000001}, {1, 2, 1.000002}
        }
    ),

    MSTParamType(MST_KRUSKAL, 3,
        {
            {0, 1, 1.000001}, {1, 2, 1.000002}, {0, 2, 1.000003}
        },
        {
            {0, 1, 1.000001}, {1, 2, 1.000002}
        }
    ),

    // 0 or ~0 weight valuess
    MSTParamType(MST_PRIM, 3,
        {
            {0, 1, 0.0}, {1, 2, 1e-9}, {0, 2, 1.0}
        },
        {
            {0, 1, 0.0}, {1, 2, 1e-9}
        }
    ),

    MSTParamType(MST_KRUSKAL, 3,
        {
            {0, 1, 0.0}, {1, 2, 1e-9}, {0, 2, 1.0}
        },
        {
            {0, 1, 0.0}, {1, 2, 1e-9}
        }
    ),

    // Duplicate edges (picks the one with the smallest weight)
    MSTParamType(MST_PRIM, 3,
        {
            {0, 1, 3.0}, {0, 1, 1.0}, {1, 2, 2.0}
        },
        {
            {0, 1, 1.0}, {1, 2, 2.0}
        }
    ),

    MSTParamType(MST_KRUSKAL, 3,
        {
            {0, 1, 3.0}, {0, 1, 1.0}, {1, 2, 2.0}
        },
        {
            {0, 1, 1.0}, {1, 2, 2.0}
        }
    ),

    // Negative weights
    MSTParamType(MST_PRIM, 3,
        {
            {0, 1, -1.0}, {1, 2, -2.0}, {0, 2, -3.0}
        },
        {
            {1, 2, -2.0}, {0, 2, -3.0}
        }
    ),

    MSTParamType(MST_KRUSKAL, 3,
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
    case MST_PRIM: os << "Prim"; break;
    case MST_KRUSKAL: os << "Kruskal"; break;
    default: os << "Unknown algorithm"; break;
    }
    os << "_Nodes_" << numNodes;
    os << "_Edges_" << edges.size();
    os << "_ExpectedEdges_" << expectedEdges.size();

    return os.str();
}

INSTANTIATE_TEST_CASE_P(/**/, MST, testing::ValuesIn(mst_graphs), MST_name_printer);

TEST(MSTStress, largeGraph)
{
    const int numNodes = 100000;

    std::vector<MSTEdge> edges;

    for (int i = 0; i < numNodes - 1; ++i)
        edges.push_back({i, i + 1, static_cast<double>(i + 1)});

    // Add extra edges for complexity
    for (int i = 0; i < numNodes - 10; i += 10)
        edges.push_back({i, i + 10,  static_cast<double>(i)});
    for (int i = 0; i + 20 < numNodes; i += 5)
        edges.push_back({i, i + 20, static_cast<double>(i + 1)});
    for (int i = 0; i + 30 < numNodes; i += 3)
        edges.push_back({i, i + 30, static_cast<double>(i % 50 + 1)});
    for (int i = 50; i < numNodes; i += 10)
        edges.push_back({i, i - 25, static_cast<double>(i % 100 + 2)});

    std::vector<MSTEdge> primMST, kruskalMST;
    bool resultPrim = buildMST(numNodes, edges, primMST, MST_PRIM, 0);
    bool resultKruskal = buildMST(numNodes, edges, kruskalMST, MST_KRUSKAL, 0);

    EXPECT_TRUE(resultPrim) << "Prim's algorithm failed for large graph ( "
        << numNodes << " nodes & " << edges.size() << " edges)";
    EXPECT_TRUE(resultKruskal) << "Kruskal's algorithm failed for large graph ( "
        << numNodes << " nodes & " << edges.size() << " edges)";
    EXPECT_EQ(primMST.size(), static_cast<size_t>(numNodes - 1))
        << "Prim's MST size incorrect for large graph: expected "
        << (numNodes - 1) << " edges, got " << primMST.size() << ".";
    EXPECT_EQ(kruskalMST.size(), static_cast<size_t>(numNodes - 1))
        << "Kruskal's MST size incorrect for large graph: expected "
        << (numNodes - 1) << " edges, got " << kruskalMST.size() << ".";
}

typedef tuple<int /*numNodes*/,
              std::vector<MSTEdge>/*edges*/
             > MSTNegativeParamType;
typedef testing::TestWithParam<MSTNegativeParamType> MSTNegative;

TEST_P(MSTNegative, disconnectedGraphs)
{
    const int numNodes = get<0>(GetParam());
    const std::vector<MSTEdge>& edges = get<1>(GetParam());

    std::vector<MSTEdge> primMST, kruskalMST;
    bool resultPrim = buildMST(numNodes, edges, primMST, MST_PRIM, 0);
    bool resultKruskal = buildMST(numNodes, edges, kruskalMST, MST_KRUSKAL, 0);

    EXPECT_FALSE(resultPrim) << "Prim's algorithm should fail for disconnected graphs ( "
        << numNodes << " nodes & " << edges.size() << " edges)";
    EXPECT_FALSE(resultKruskal) << "Kruskal's algorithm should fail for disconnected graphs ( "
        << numNodes << " nodes & " << edges.size() << " edges)";
    EXPECT_LT(primMST.size(), static_cast<size_t>(numNodes - 1))
        << "Prim's MST should not contain " << numNodes - 1 << " edges for disconnected graphs";
    EXPECT_LT(kruskalMST.size(), static_cast<size_t>(numNodes - 1))
        << "Kruskal's MST should not contain " << numNodes - 1 << " edges for disconnected graphs";
}

const MSTNegativeParamType mst_negative[] =
{
    // Two disconnected subgraphs

    MSTNegativeParamType(5,
        {
            // Subgraph 1: 0-1
            {0, 1, 1.0},
            // Subgraph 2: 2-3-4
            {2, 3, 2.0}, {3, 4, 3.0}
        }
    ),

    MSTNegativeParamType(8,
        {
            // Subgraph 1: 0-1-2-3
            {0, 1, 1.0}, {1, 2, 2.0}, {2, 3, 3.0},
            // Subgraph 2: 4-5-6-7
            {4, 5, 4.0}, {5, 6, 5.0}, {6, 7, 6.0}
        }
    ),

    MSTNegativeParamType(4,
        {
            // Subgraph 1: 0-1
            {0, 1, 1.0},
            // Subgraph 2: 2-3
            {2, 3, 1.0},
            // self-loop edges (should be ignored by the algorithms)
            {0, 0, 1.0}, {1, 1, 1.0}, {2, 2, 1.0}, {3, 3, 1.0}
        }
    ),

    // Three disconnected components

    MSTNegativeParamType(6,
        {
            // Subgraph 1: 0-1
            {0, 1, 1.0},
            // Subgraph 2: 2-3
            {2, 3, 1.0},
            // Subgraph 3: 4-5
            {4, 5, 1.0}
        }
    ),

    MSTNegativeParamType(9,
        {
            // Subgraph 1: 0-1-2-3
            {0, 1, 1.0}, {1, 2, 1.0}, {2, 3, 1.0},
            // Subgraph 2: 4-5-6
            {4, 5, 1.0}, {5, 6, 1.0},
            // Subgraph 3: 7-8
            {7, 8, 1.0}
        }
    ),

    MSTNegativeParamType(8,
        {
            // Subgraph 1: 0-1-2
            {0, 1, 1.0}, {1, 2, 1.0},
            // Subgraph 2: 3-4-5
            {3, 4, 1.0}, {4, 5, 1.0},
            // Subgraph 3: 6-7
            {6, 7, 1.0}
        }
    ),
    
    // Isolated nodes
    MSTNegativeParamType(4,
        {
            // Subgraph 1: 0-1-2
            {0, 1, 1.0}, {1, 2, 1.0},
            // Isolated node: 3
        }
    ),

    MSTNegativeParamType(5,
        {
            // Subgraph 1: 0-1-2
            {0, 1, 1.0}, {1, 2, 1.0},
            // Isolated nodes: 3, 4
        }
    ),

    MSTNegativeParamType(3,
        {
            // Isolated node: 0, 1, 2 (no edges)
        }
    ),
};

inline static std::string MST_negative_name_printer(const testing::TestParamInfo<MSTNegative::ParamType>& info) {
    std::ostringstream os;
    const auto& numNodes = get<0>(info.param);
    const auto& edges = get<1>(info.param);

    os << "TestCase_" << info.index << "_Nodes_" << numNodes;
    os << "_Edges_" << edges.size();

    return os.str();
}

INSTANTIATE_TEST_CASE_P(/**/, MSTNegative, testing::ValuesIn(mst_negative), MST_negative_name_printer);

}} // namespace
