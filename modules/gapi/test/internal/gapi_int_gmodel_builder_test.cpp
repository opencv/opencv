// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#include "../test_precomp.hpp"

#include <ade/util/zip_range.hpp>   // util::indexed

#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/gcommon.hpp>
#include "compiler/gmodelbuilder.hpp"
#include "compiler/gmodel.hpp" // RcDesc, GModel::init

namespace opencv_test
{

namespace test
{

namespace
{
    namespace D = cv::detail;
    cv::GMat unaryOp(cv::GMat m)
    {
        return cv::GCall(cv::GKernel{ "gapi.test.unaryop"
                                    , ""
                                    , nullptr
                                    , { GShape::GMAT }
                                    , { D::OpaqueKind::CV_UNKNOWN }
                                    , { D::HostCtor{cv::util::monostate{}} }
                                    , { D::OpaqueKind::CV_UNKNOWN }
                                    }).pass(m).yield(0);
    }

    cv::GMat binaryOp(cv::GMat m1, cv::GMat m2)
    {
        return cv::GCall(cv::GKernel{ "gapi.test.binaryOp"
                                    , ""
                                    , nullptr
                                    , { GShape::GMAT }
                                    , { D::OpaqueKind::CV_UNKNOWN, D::OpaqueKind::CV_UNKNOWN }
                                    , { D::HostCtor{cv::util::monostate{}} }
                                    , { D::OpaqueKind::CV_UNKNOWN}
                                    }).pass(m1, m2).yield(0);
    }

    std::vector<ade::NodeHandle> collectOperations(const cv::gimpl::GModel::Graph& gr)
    {
        std::vector<ade::NodeHandle> ops;
        for (const auto& nh : gr.nodes())
        {
            if (gr.metadata(nh).get<cv::gimpl::NodeType>().t == cv::gimpl::NodeType::OP)
                ops.push_back(nh);
        }
        return ops;
    }

    ade::NodeHandle inputOf(cv::gimpl::GModel::Graph& gm, ade::NodeHandle nh, std::size_t port)
    {
        for (const auto& eh : nh->inEdges())
        {
            if (gm.metadata(eh).get<cv::gimpl::Input>().port == port)
            {
                return eh->srcNode();
            }
        }
        util::throw_error(std::logic_error("port " + std::to_string(port) + " not found"));
    }
}
}// namespace opencv_test::test

TEST(GModelBuilder, Unroll_TestUnary)
{
    cv::GMat in;
    cv::GMat out = test::unaryOp(in);

    auto unrolled = cv::gimpl::unrollExpr(cv::GIn(in).m_args, cv::GOut(out).m_args);

    EXPECT_EQ(1u, unrolled.all_ops.size());  // There is one operation
    EXPECT_EQ(2u, unrolled.all_data.size()); // And two data objects (in, out)

    // TODO check what the operation is, and so on, and so on
}

TEST(GModelBuilder, Unroll_TestUnaryOfUnary)
{
    cv::GMat in;
    cv::GMat out = test::unaryOp(test::unaryOp(in));

    auto unrolled = cv::gimpl::unrollExpr(cv::GIn(in).m_args, cv::GOut(out).m_args);

    EXPECT_EQ(2u, unrolled.all_ops.size());  // There're two operations
    EXPECT_EQ(3u, unrolled.all_data.size()); // And three data objects (in, out)

    // TODO check what the operation is, and so on, and so on
}

TEST(GModelBuilder, Unroll_Not_All_Protocol_Inputs_Are_Reached)
{
    cv::GMat in1, in2;                                      // in1 -> unaryOp() -> u_op1 -> unaryOp() -> out
    auto u_op1 = test::unaryOp(in1);                        // in2 -> unaryOp() -> u_op2
    auto u_op2 = test::unaryOp(in2);
    auto out   = test::unaryOp(u_op1);

    EXPECT_THROW(cv::gimpl::unrollExpr(cv::GIn(in1, in2).m_args, cv::GOut(out).m_args), std::logic_error);
}

TEST(GModelBuilder, Unroll_Parallel_Path)
{
    cv::GMat in1, in2;                                      // in1 -> unaryOp() -> out1
    auto out1 = test::unaryOp(in1);                         // in2 -> unaryOp() -> out2
    auto out2 = test::unaryOp(in2);

    auto unrolled = cv::gimpl::unrollExpr(cv::GIn(in1, in2).m_args, cv::GOut(out1, out2).m_args);

    EXPECT_EQ(unrolled.all_ops.size(),  2u);
    EXPECT_EQ(unrolled.all_data.size(), 4u);
}

TEST(GModelBuilder, Unroll_WithBranch)
{
    // in -> unaryOp() -> tmp -->unaryOp() -> out1
    //                     `---->unaryOp() -> out2

    GMat in;
    auto tmp = test::unaryOp(in);
    auto out1 = test::unaryOp(tmp);
    auto out2 = test::unaryOp(tmp);

    auto unrolled = cv::gimpl::unrollExpr(cv::GIn(in).m_args, cv::GOut(out1, out2).m_args);

    EXPECT_EQ(unrolled.all_ops.size(),  3u);
    EXPECT_EQ(unrolled.all_data.size(), 4u);
}

TEST(GModelBuilder, Build_Unary)
{
    cv::GMat in;
    cv::GMat out = test::unaryOp(in);

    ade::Graph g;
    cv::gimpl::GModel::Graph gm(g);
    cv::gimpl::GModel::init(gm);
    cv::gimpl::GModelBuilder(g).put(cv::GIn(in).m_args, cv::GOut(out).m_args);

    EXPECT_EQ(3u, static_cast<std::size_t>(g.nodes().size()));    // Generated graph should have three nodes

    // TODO: Check what the nodes are
}

TEST(GModelBuilder, Constant_GScalar)
{
    // in -> addC()-----(GMat)---->mulC()-----(GMat)---->unaryOp()----out
    //         ^                     ^
    //         |                     |
    // 3-------`           c_s-------'

    cv::GMat in;
    cv::GScalar c_s = 5;
    auto out = test::unaryOp((in + 3) * c_s);    // 3 converted to GScalar

    ade::Graph g;
    cv::gimpl::GModel::Graph gm(g);
    cv::gimpl::GModel::init(gm);
    auto proto_slots = cv::gimpl::GModelBuilder(g).put(cv::GIn(in).m_args, cv::GOut(out).m_args);
    cv::gimpl::Protocol p;
    std::tie(p.inputs, p.outputs, p.in_nhs, p.out_nhs) = proto_slots;

    auto in_nh   = p.in_nhs.front();
    auto addC_nh = in_nh->outNodes().front();
    auto mulC_nh = addC_nh->outNodes().front()->outNodes().front();

    ASSERT_TRUE(gm.metadata(addC_nh).get<cv::gimpl::NodeType>().t == cv::gimpl::NodeType::OP);
    ASSERT_TRUE(gm.metadata(mulC_nh).get<cv::gimpl::NodeType>().t == cv::gimpl::NodeType::OP);

    auto s_3 = test::inputOf(gm, addC_nh, 1);
    auto s_5 = test::inputOf(gm, mulC_nh, 1);

    EXPECT_EQ(9u, static_cast<std::size_t>(g.nodes().size()));          // 6 data nodes (1 -input, 1 output, 2 constant, 2 temp) and 3 op nodes
    EXPECT_EQ(2u, static_cast<std::size_t>(addC_nh->inNodes().size())); // in and 3
    EXPECT_EQ(2u, static_cast<std::size_t>(mulC_nh->inNodes().size())); // addC output and c_s
    EXPECT_EQ(3, (util::get<cv::Scalar>(gm.metadata(s_3).get<cv::gimpl::ConstValue>().arg))[0]);
    EXPECT_EQ(5, (util::get<cv::Scalar>(gm.metadata(s_5).get<cv::gimpl::ConstValue>().arg))[0]);
}

TEST(GModelBuilder, Check_Multiple_Outputs)
{
    //            ------------------------------> r
    //            '
    //            '                    -----------> i_out1
    //            '                    '
    // in ----> split3() ---> g ---> integral()
    //            '                    '
    //            '                    -----------> i_out2
    //            '
    //            '---------> b ---> unaryOp() ---> u_out

    cv::GMat in, r, g, b, i_out1, i_out2, u_out;
    std::tie(r, g, b) = cv::gapi::split3(in);
    std::tie(i_out1, i_out2) = cv::gapi::integral(g, 1, 1);
    u_out = test::unaryOp(b);

    ade::Graph gr;
    cv::gimpl::GModel::Graph gm(gr);
    cv::gimpl::GModel::init(gm);
    auto proto_slots = cv::gimpl::GModelBuilder(gr).put(cv::GIn(in).m_args, cv::GOut(r, i_out1, i_out2, u_out).m_args);
    cv::gimpl::Protocol p;
    std::tie(p.inputs, p.outputs, p.in_nhs, p.out_nhs) = proto_slots;

    EXPECT_EQ(4u, static_cast<std::size_t>(p.out_nhs.size()));
    EXPECT_EQ(0u, gm.metadata(p.out_nhs[0]->inEdges().front()).get<cv::gimpl::Output>().port);
    EXPECT_EQ(0u, gm.metadata(p.out_nhs[1]->inEdges().front()).get<cv::gimpl::Output>().port);
    EXPECT_EQ(1u, gm.metadata(p.out_nhs[2]->inEdges().front()).get<cv::gimpl::Output>().port);
    EXPECT_EQ(0u, gm.metadata(p.out_nhs[3]->inEdges().front()).get<cv::gimpl::Output>().port);
    for (const auto it : ade::util::indexed(p.out_nhs))
    {
        const auto& out_nh = ade::util::value(it);

        EXPECT_EQ(cv::gimpl::NodeType::DATA, gm.metadata(out_nh).get<cv::gimpl::NodeType>().t);
        EXPECT_EQ(GShape::GMAT, gm.metadata(out_nh).get<cv::gimpl::Data>().shape);
    }
}

TEST(GModelBuilder, Unused_Outputs)
{
    cv::GMat in;
    auto yuv_p = cv::gapi::split3(in);

    ade::Graph g;
    cv::gimpl::GModel::Graph gm(g);
    cv::gimpl::GModel::init(gm);
    cv::gimpl::GModelBuilder(g).put(cv::GIn(in).m_args, cv::GOut(std::get<0>(yuv_p)).m_args);

    EXPECT_EQ(5u, static_cast<std::size_t>(g.nodes().size()));    // 1 input, 1 operation, 3 outputs
}

TEST(GModelBuilder, Work_With_One_Channel_From_Split3)
{
    cv::GMat in, y, u, v;
    std::tie(y, u, v) = cv::gapi::split3(in);
    auto y_blur = cv::gapi::gaussianBlur(y, cv::Size(3, 3), 1);

    ade::Graph g;
    cv::gimpl::GModel::Graph gm(g);
    cv::gimpl::GModel::init(gm);
    cv::gimpl::GModelBuilder(g).put(cv::GIn(in).m_args, cv::GOut(y_blur).m_args);

    EXPECT_EQ(7u, static_cast<std::size_t>(g.nodes().size())); // 1 input, 2 operation, 3 nodes from split3, 1 output
}

TEST(GModelBuilder, Add_Nodes_To_Unused_Nodes)
{
    cv::GMat in, y, u, v;
    std::tie(y, u, v) = cv::gapi::split3(in);
    auto y_blur = cv::gapi::gaussianBlur(y, cv::Size(3, 3), 1);
    // unused nodes
    auto u_blur = cv::gapi::gaussianBlur(y, cv::Size(3, 3), 1);
    auto v_blur = cv::gapi::gaussianBlur(y, cv::Size(3, 3), 1);

    ade::Graph g;
    cv::gimpl::GModel::Graph gm(g);
    cv::gimpl::GModel::init(gm);
    cv::gimpl::GModelBuilder(g).put(cv::GIn(in).m_args, cv::GOut(y_blur).m_args);

    EXPECT_EQ(7u, static_cast<std::size_t>(g.nodes().size())); // 1 input, 2 operation, 3 nodes from split3, 1 output
}

TEST(GModelBuilder, Unlisted_Inputs)
{
    // in1 -> binaryOp() -> out
    //         ^
    //         |
    // in2 ----'

    cv::GMat in1, in2;
    auto out = test::binaryOp(in1, in2);

    ade::Graph g;
    cv::gimpl::GModel::Graph gm(g);
    cv::gimpl::GModel::init(gm);
    // add required 2 inputs but pass 1
    EXPECT_THROW(cv::gimpl::GModelBuilder(g).put(cv::GIn(in1).m_args, cv::GOut(out).m_args), std::logic_error);
}

TEST(GModelBuilder, Unroll_No_Link_Between_In_And_Out)
{
    // in    -> unaryOp() -> u_op
    // other -> unaryOp() -> out

    cv::GMat in, other;
    auto u_op = test::unaryOp(in);
    auto out  = test::unaryOp(other);

    EXPECT_THROW(cv::gimpl::unrollExpr(cv::GIn(in).m_args, cv::GOut(out).m_args), std::logic_error);
}


TEST(GModel_builder, Check_Binary_Op)
{
    // in1 -> binaryOp() -> out
    //          ^
    //          |
    // in2 -----'

    cv::GMat in1, in2;
    auto out = test::binaryOp(in1, in2);

    ade::Graph g;
    cv::gimpl::GModel::Graph gm(g);
    cv::gimpl::GModel::init(gm);
    auto proto_slots = cv::gimpl::GModelBuilder(g).put(cv::GIn(in1, in2).m_args, cv::GOut(out).m_args);

    cv::gimpl::Protocol p;
    std::tie(p.inputs, p.outputs, p.in_nhs, p.out_nhs) = proto_slots;
    auto ops = test::collectOperations(g);

    EXPECT_EQ(1u, ops.size());
    EXPECT_EQ("gapi.test.binaryOp", gm.metadata(ops.front()).get<cv::gimpl::Op>().k.name);
    EXPECT_EQ(2u, static_cast<std::size_t>(ops.front()->inEdges().size()));
    EXPECT_EQ(1u, static_cast<std::size_t>(ops.front()->outEdges().size()));
    EXPECT_EQ(1u, static_cast<std::size_t>(ops.front()->outNodes().size()));
}

TEST(GModelBuilder, Add_Operation_With_Two_Out_One_Time)
{
    // in -> integral() --> out_b1 -> unaryOp() -> out1
    //            |
    //            '-------> out_b2 -> unaryOp() -> out2

    cv::GMat in, out_b1, out_b2;
    std::tie(out_b1, out_b2) = cv::gapi::integral(in, 1, 1);
    auto out1 = test::unaryOp(out_b1);
    auto out2 = test::unaryOp(out_b1);

    ade::Graph g;
    cv::gimpl::GModel::Graph gm(g);
    cv::gimpl::GModel::init(gm);
    auto proto_slots = cv::gimpl::GModelBuilder(g).put(cv::GIn(in).m_args, cv::GOut(out1, out2).m_args);

    auto ops = test::collectOperations(gm);

    cv::gimpl::Protocol p;
    std::tie(p.inputs, p.outputs, p.in_nhs, p.out_nhs) = proto_slots;
    auto integral_nh = p.in_nhs.front()->outNodes().front();

    EXPECT_EQ(3u, ops.size());
    EXPECT_EQ("org.opencv.core.matrixop.integral", gm.metadata(integral_nh).get<cv::gimpl::Op>().k.name);
    EXPECT_EQ(1u, static_cast<std::size_t>(integral_nh->inEdges().size()));
    EXPECT_EQ(2u, static_cast<std::size_t>(integral_nh->outEdges().size()));
    EXPECT_EQ(2u, static_cast<std::size_t>(integral_nh->outNodes().size()));
}
TEST(GModelBuilder, Add_Operation_With_One_Out_One_Time)
{
    // in1 -> binaryOp() -> b_out -> unaryOp() -> out1
    //            ^           |
    //            |           |
    // in2 -------            '----> unaryOp() -> out2

    cv::GMat in1, in2;
    auto b_out = test::binaryOp(in1, in2);
    auto out1 = test::unaryOp(b_out);
    auto out2 = test::unaryOp(b_out);

    ade::Graph g;
    cv::gimpl::GModel::Graph gm(g);
    cv::gimpl::GModel::init(gm);
    auto proto_slots = cv::gimpl::GModelBuilder(g).put(cv::GIn(in1, in2).m_args, cv::GOut(out1, out2).m_args);
    cv::gimpl::Protocol p;
    std::tie(p.inputs, p.outputs, p.in_nhs, p.out_nhs) = proto_slots;
    cv::gimpl::GModel::Graph gr(g);
    auto binaryOp_nh = p.in_nhs.front()->outNodes().front();

    EXPECT_EQ(2u, static_cast<std::size_t>(binaryOp_nh->inEdges().size()));
    EXPECT_EQ(1u, static_cast<std::size_t>(binaryOp_nh->outEdges().size()));
    EXPECT_EQ(8u, static_cast<std::size_t>(g.nodes().size()));
}
} // namespace opencv_test
