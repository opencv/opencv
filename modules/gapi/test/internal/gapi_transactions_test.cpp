// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"
#include <ade/graph.hpp>
#include "compiler/transactions.hpp"

namespace opencv_test
{
namespace
{

bool contains(const ade::Graph& graph, const ade::NodeHandle& node)
{
    auto nodes = graph.nodes();
    return nodes.end() != std::find(nodes.begin(), nodes.end(), node);
}

bool connected(const ade::NodeHandle& src_node, const ade::NodeHandle& dst_node)
{
    auto nodes = src_node->outNodes();
    return nodes.end() != std::find(nodes.begin(), nodes.end(), dst_node);
}

struct SimpleGraph
{
    //       ehs[0]      ehs[1]     ehs[2]     ehs[3]
    // nhs[0] -- > nhs[1] --> nhs[2] --> nhs[3] --> nhs[4]

    enum { node_nums = 5 };
    ade::Graph        graph;
    ade::NodeHandle   fused_nh;                     /* For check that fusion  node is connected to the
                                                               inputs of the prod and the outputs of the cons */
    std::array<ade::NodeHandle, node_nums>     nhs;
    std::array<ade::EdgeHandle, node_nums - 1> ehs;
    Change::List changes;

    SimpleGraph()
    {
        nhs[0] = graph.createNode();
        for (int i = 1; i < node_nums; ++i)
        {
            nhs[i    ] = graph.createNode();
            ehs[i - 1] = graph.link(nhs[i - 1], nhs[i]);
        }
    }

    void fuse()
    {
        // nhs[0] --> fused_nh --> nhs[4]

        fused_nh = graph.createNode();
        changes.enqueue<Change::NodeCreated>(fused_nh);
        changes.enqueue<Change::NewLink> (graph, nhs[0],    fused_nh);
        changes.enqueue<Change::DropLink>(graph, nhs[1],    ehs[0]);
        changes.enqueue<Change::NewLink> (graph, fused_nh, nhs[4]);
        changes.enqueue<Change::DropLink>(graph, nhs[3],    ehs[3]);
        changes.enqueue<Change::DropLink>(graph, nhs[1],    ehs[1]);
        changes.enqueue<Change::DropLink>(graph, nhs[2],    ehs[2]);
        changes.enqueue<Change::DropNode>(nhs[1]);
        changes.enqueue<Change::DropNode>(nhs[2]);
        changes.enqueue<Change::DropNode>(nhs[3]);
    }

    void commit()   { changes.commit(graph);   }
    void rollback() { changes.rollback(graph); }

};

struct Transactions: public ::testing::Test, public SimpleGraph {};

} // anonymous namespace

TEST_F(Transactions, NodeCreated_Create)
{
    auto new_nh = graph.createNode();
    Change::NodeCreated node_created(new_nh);

    EXPECT_EQ(6u, static_cast<std::size_t>(graph.nodes().size()));
    EXPECT_TRUE(contains(graph, new_nh));
}

TEST_F(Transactions, NodeCreated_RollBack)
{
    auto new_nh = graph.createNode();
    Change::NodeCreated node_created(new_nh);

    node_created.rollback(graph);

    EXPECT_EQ(5u, static_cast<std::size_t>(graph.nodes().size()));
    EXPECT_FALSE(contains(graph, new_nh));
}

TEST_F(Transactions, NodeCreated_Commit)
{
    auto new_nh = graph.createNode();
    Change::NodeCreated node_created(new_nh);

    node_created.commit(graph);

    EXPECT_EQ(6u, static_cast<std::size_t>(graph.nodes().size()));
    EXPECT_TRUE(contains(graph, new_nh));
}

TEST_F(Transactions, DropLink_Create)
{
    Change::DropLink drop_link(graph, nhs[0], ehs[0]);

    EXPECT_FALSE(connected(nhs[0], nhs[1]));
}

TEST_F(Transactions, DropLink_RollBack)
{
    Change::DropLink drop_link(graph, nhs[0], ehs[0]);

    drop_link.rollback(graph);

    EXPECT_TRUE(connected(nhs[0], nhs[1]));
}

TEST_F(Transactions, DropLink_Commit)
{
    Change::DropLink drop_link(graph, nhs[0], ehs[0]);

    drop_link.commit(graph);

    EXPECT_FALSE(connected(nhs[0], nhs[1]));
}

TEST_F(Transactions, NewLink_Create)
{
    auto new_nh = graph.createNode();
    Change::NewLink new_link(graph, new_nh, nhs[0]);

    EXPECT_TRUE(connected(new_nh, nhs[0]));
}

TEST_F(Transactions, NewLink_RollBack)
{
    auto new_nh = graph.createNode();
    Change::NewLink new_link(graph, new_nh, nhs[0]);

    new_link.rollback(graph);

    EXPECT_FALSE(connected(new_nh, nhs[0]));
}

TEST_F(Transactions, NewLink_Commit)
{
    auto new_nh = graph.createNode();
    Change::NewLink new_link(graph, new_nh, nhs[0]);

    new_link.commit(graph);

    EXPECT_TRUE(connected(new_nh, nhs[0]));
}

TEST_F(Transactions, DropNode_Create)
{
    auto new_nh = graph.createNode();
    Change::DropNode drop_node(new_nh);

    EXPECT_EQ(6u, static_cast<std::size_t>(graph.nodes().size()));
    EXPECT_TRUE(contains(graph, new_nh));
}

TEST_F(Transactions, DropNode_RollBack)
{
    auto new_nh = graph.createNode();
    Change::DropNode drop_node(new_nh);

    drop_node.rollback(graph);

    EXPECT_EQ(6u, static_cast<std::size_t>(graph.nodes().size()));
    EXPECT_TRUE(contains(graph, new_nh));
}

TEST_F(Transactions, DropNode_Commit)
{
    auto new_nh = graph.createNode();
    Change::DropNode drop_node(new_nh);

    drop_node.commit(graph);

    EXPECT_EQ(5u, static_cast<std::size_t>(graph.nodes().size()));
    EXPECT_FALSE(contains(graph, new_nh));
}

TEST_F(Transactions, Fusion_Commit)
{
    namespace C = Change;

    fuse();
    commit();

    EXPECT_EQ(3u, static_cast<std::size_t>(graph.nodes().size()));
    EXPECT_TRUE(connected(nhs[0]   , fused_nh));
    EXPECT_TRUE(connected(fused_nh, nhs[4]));
}

TEST_F(Transactions, Fusion_RollBack)
{
    namespace C = Change;

    fuse();
    rollback();

    EXPECT_EQ(static_cast<std::size_t>(node_nums),
              static_cast<std::size_t>(graph.nodes().size()));
    EXPECT_FALSE(contains(graph, fused_nh));

    for (int i = 0; i < static_cast<int>(node_nums) - 1; ++i)
    {
        EXPECT_TRUE(connected(nhs[i], nhs[i + 1]));
    }
}

} // opencv_test
