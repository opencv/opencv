// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 - 2020 Intel Corporation


#include "../test_precomp.hpp"

#include <ade/graph.hpp>
#include <ade/typed_graph.hpp>

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
    ade::NodeHandle   fused_nh;  // For check that fusion  node is connected to the
                                 // inputs of the prod and the outputs of the cons
    std::array<ade::NodeHandle, node_nums>     nhs;
    std::array<ade::EdgeHandle, node_nums - 1> ehs;
    using Change = ChangeT<>;
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
    fuse();
    commit();

    EXPECT_EQ(3u, static_cast<std::size_t>(graph.nodes().size()));
    EXPECT_TRUE(connected(nhs[0]   , fused_nh));
    EXPECT_TRUE(connected(fused_nh, nhs[4]));
}

TEST_F(Transactions, Fusion_RollBack)
{
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

namespace
{
    struct MetaInt {
        static const char *name() { return "int_meta"; }
        int x;
    };

    struct MetaStr {
        static const char *name() { return "string_meta"; }
        std::string s;
    };
}

TEST(PreservedMeta, TestMetaCopy_Full)
{
    ade::Graph g;
    ade::TypedGraph<MetaInt, MetaStr> tg(g);

    auto src_nh = tg.createNode();
    tg.metadata(src_nh).set(MetaInt{42});
    tg.metadata(src_nh).set(MetaStr{"hi"});

    auto dst_nh = tg.createNode();

    EXPECT_FALSE(tg.metadata(dst_nh).contains<MetaInt>());
    EXPECT_FALSE(tg.metadata(dst_nh).contains<MetaStr>());

    // Here we specify all the meta types we know about the src node
    // Assume Preserved copies its all for us
    Preserved<ade::NodeHandle, MetaInt, MetaStr>(g, src_nh).copyTo(g, dst_nh);

    ASSERT_TRUE(tg.metadata(dst_nh).contains<MetaInt>());
    ASSERT_TRUE(tg.metadata(dst_nh).contains<MetaStr>());

    EXPECT_EQ(42,   tg.metadata(dst_nh).get<MetaInt>().x);
    EXPECT_EQ("hi", tg.metadata(dst_nh).get<MetaStr>().s);
}


TEST(PreservedMeta, TestMetaCopy_Partial_Dst)
{
    ade::Graph g;
    ade::TypedGraph<MetaInt, MetaStr> tg(g);

    auto tmp_nh1 = tg.createNode();
    auto tmp_nh2 = tg.createNode();
    auto src_eh  = tg.link(tmp_nh1, tmp_nh2);

    tg.metadata(src_eh).set(MetaInt{42});
    tg.metadata(src_eh).set(MetaStr{"hi"});

    auto tmp_nh3 = tg.createNode();
    auto tmp_nh4 = tg.createNode();
    auto dst_eh  = tg.link(tmp_nh3, tmp_nh4);

    EXPECT_FALSE(tg.metadata(dst_eh).contains<MetaInt>());
    EXPECT_FALSE(tg.metadata(dst_eh).contains<MetaStr>());

    // Here we specify just a single meta type for the src node
    // Assume Preserved copies only this type and nothing else
    Preserved<ade::EdgeHandle, MetaStr>(g, src_eh).copyTo(g, dst_eh);

    ASSERT_FALSE(tg.metadata(dst_eh).contains<MetaInt>());
    ASSERT_TRUE (tg.metadata(dst_eh).contains<MetaStr>());

    EXPECT_EQ("hi", tg.metadata(dst_eh).get<MetaStr>().s);
}

TEST(PreservedMeta, TestMetaCopy_Partial_Src)
{
    ade::Graph g;
    ade::TypedGraph<MetaInt, MetaStr> tg(g);

    auto src_nh = tg.createNode();
    tg.metadata(src_nh).set(MetaInt{42});

    auto dst_nh = tg.createNode();

    EXPECT_FALSE(tg.metadata(dst_nh).contains<MetaInt>());
    EXPECT_FALSE(tg.metadata(dst_nh).contains<MetaStr>());

    // Here we specify all the meta types we know about the src node
    // but the src node has just one of them.
    // A valid situation, only MetaInt to be copied.
    Preserved<ade::NodeHandle, MetaInt, MetaStr>(g, src_nh).copyTo(g, dst_nh);

    ASSERT_TRUE (tg.metadata(dst_nh).contains<MetaInt>());
    ASSERT_FALSE(tg.metadata(dst_nh).contains<MetaStr>());

    EXPECT_EQ(42, tg.metadata(dst_nh).get<MetaInt>().x);
}

TEST(PreservedMeta, TestMetaCopy_Nothing)
{
    ade::Graph g;
    ade::TypedGraph<MetaInt, MetaStr> tg(g);

    auto src_nh = tg.createNode();
    auto dst_nh = tg.createNode();

    EXPECT_FALSE(tg.metadata(src_nh).contains<MetaInt>());
    EXPECT_FALSE(tg.metadata(src_nh).contains<MetaStr>());

    EXPECT_FALSE(tg.metadata(dst_nh).contains<MetaInt>());
    EXPECT_FALSE(tg.metadata(dst_nh).contains<MetaStr>());

    // Here we specify all the meta types we know about the src node
    // but the src node has none of those. See how it works now
    Preserved<ade::NodeHandle, MetaInt, MetaStr>(g, src_nh).copyTo(g, dst_nh);

    ASSERT_FALSE(tg.metadata(dst_nh).contains<MetaInt>());
    ASSERT_FALSE(tg.metadata(dst_nh).contains<MetaStr>());
}

TEST(PreservedMeta, DropEdge)
{
    ade::Graph g;
    ade::TypedGraph<MetaInt, MetaStr> tg(g);

    auto nh1 = tg.createNode();
    auto nh2 = tg.createNode();
    auto eh  = tg.link(nh1, nh2);

    tg.metadata(eh).set(MetaInt{42});
    tg.metadata(eh).set(MetaStr{"hi"});

    // Drop an edge using the transaction API
    using Change = ChangeT<MetaInt, MetaStr>;
    Change::List changes;
    changes.enqueue<Change::DropLink>(g, nh1, eh);

    EXPECT_EQ(0u,      nh1->outNodes().size());
    EXPECT_EQ(nullptr, eh);

    // Now restore the edge and check if it's meta was restored
    changes.rollback(g);

    ASSERT_EQ(1u,      nh1->outNodes().size());
    eh = *nh1->outEdges().begin();

    ASSERT_TRUE(tg.metadata(eh).contains<MetaInt>());
    ASSERT_TRUE(tg.metadata(eh).contains<MetaStr>());

    EXPECT_EQ(42,   tg.metadata(eh).get<MetaInt>().x);
    EXPECT_EQ("hi", tg.metadata(eh).get<MetaStr>().s);
}

} // opencv_test
