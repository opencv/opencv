// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"
#include "compiler/transactions.hpp"

#include "gapi_mock_kernels.hpp"

#include "compiler/gmodel.hpp"
#include "compiler/gislandmodel.hpp"
#include "compiler/gcompiler.hpp"

namespace opencv_test
{

TEST(IslandFusion, TwoOps_OneIsland)
{
    namespace J = Jupiter; // see mock_kernels.cpp

    // Define a computation:
    //
    //    (in) -> J::Foo1 -> (tmp0) -> J::Foo2 -> (out)
    //          :                               :
    //          :          "island0"            :
    //          :<----------------------------->:

    cv::GMat in;
    cv::GMat tmp0 = I::Foo::on(in);
    cv::GMat out  = I::Foo::on(tmp0);
    cv::GComputation cc(in, out);

    // Prepare compilation parameters manually
    const auto in_meta = cv::GMetaArg(cv::GMatDesc{CV_8U,1,cv::gapi::own::Size(32,32)});
    const auto pkg     = cv::gapi::kernels<J::Foo>();

    // Directly instantiate G-API graph compiler and run partial compilation
    cv::gimpl::GCompiler compiler(cc, {in_meta}, cv::compile_args(pkg));
    cv::gimpl::GCompiler::GPtr graph = compiler.generateGraph();
    compiler.runPasses(*graph);

    // Inspect the graph and verify the islands configuration
    cv::gimpl::GModel::ConstGraph gm(*graph);

    auto in_nh  = cv::gimpl::GModel::dataNodeOf(gm, in);
    auto tmp_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp0);
    auto out_nh = cv::gimpl::GModel::dataNodeOf(gm, out);

    // in/out mats shouldn't be assigned to any Island
    EXPECT_FALSE(gm.metadata(in_nh ).contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(out_nh).contains<cv::gimpl::Island>());

    // Since tmp is surrounded by two J kernels, tmp should be assigned
    // to island J
    EXPECT_TRUE(gm.metadata(tmp_nh).contains<cv::gimpl::Island>());
}

TEST(IslandFusion, TwoOps_TwoIslands)
{
    namespace J = Jupiter; // see mock_kernels.cpp
    namespace S = Saturn;  // see mock_kernels.cpp

    // Define a computation:
    //
    //    (in) -> J::Foo --> (tmp0) -> S::Bar --> (out)
    //          :          :        ->          :
    //          :          :         :          :
    //          :<-------->:         :<-------->:

    cv::GMat in;
    cv::GMat tmp0 = I::Foo::on(in);
    cv::GMat out  = I::Bar::on(tmp0, tmp0);
    cv::GComputation cc(in, out);

    // Prepare compilation parameters manually
    const auto in_meta = cv::GMetaArg(cv::GMatDesc{CV_8U,1,cv::gapi::own::Size(32,32)});
    const auto pkg     = cv::gapi::kernels<J::Foo, S::Bar>();

    // Directly instantiate G-API graph compiler and run partial compilation
    cv::gimpl::GCompiler compiler(cc, {in_meta}, cv::compile_args(pkg));
    cv::gimpl::GCompiler::GPtr graph = compiler.generateGraph();
    compiler.runPasses(*graph);

    // Inspect the graph and verify the islands configuration
    cv::gimpl::GModel::ConstGraph gm(*graph);

    auto in_nh  = cv::gimpl::GModel::dataNodeOf(gm, in);
    auto tmp_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp0);
    auto out_nh = cv::gimpl::GModel::dataNodeOf(gm, out);

    // in/tmp/out mats shouldn't be assigned to any Island
    EXPECT_FALSE(gm.metadata(in_nh ).contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(out_nh).contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(tmp_nh).contains<cv::gimpl::Island>());

    auto isl_model = gm.metadata().get<cv::gimpl::IslandModel>().model;
    cv::gimpl::GIslandModel::ConstGraph gim(*isl_model);

    // There should be two islands in the GIslandModel
    const auto is_island = [&](ade::NodeHandle nh) {
        return (cv::gimpl::NodeKind::ISLAND
                == gim.metadata(nh).get<cv::gimpl::NodeKind>().k);
    };
    const std::size_t num_isl = std::count_if(gim.nodes().begin(),
                                              gim.nodes().end(),
                                              is_island);
    EXPECT_EQ(2u, num_isl);

    auto isl_foo_nh  = cv::gimpl::GIslandModel::producerOf(gim, tmp_nh);
    auto isl_bar_nh  = cv::gimpl::GIslandModel::producerOf(gim, out_nh);
    ASSERT_NE(nullptr, isl_foo_nh);
    ASSERT_NE(nullptr, isl_bar_nh);

    // Islands should be different
    auto isl_foo_obj = gim.metadata(isl_foo_nh).get<cv::gimpl::FusedIsland>().object;
    auto isl_bar_obj = gim.metadata(isl_bar_nh).get<cv::gimpl::FusedIsland>().object;
    EXPECT_FALSE(isl_foo_obj == isl_bar_obj);
}

TEST(IslandFusion, ConsumerHasTwoInputs)
{
    namespace J = Jupiter; // see mock_kernels.cpp

    // Define a computation:     island
    //            ............................
    //    (in0) ->:J::Foo -> (tmp) -> S::Bar :--> (out)
    //            :....................^.....:
    //                                 |
    //    (in1) -----------------------`
    //

    // Check that island is build correctly, when consumer has two inputs

    GMat in[2];
    GMat tmp = I::Foo::on(in[0]);
    GMat out = I::Bar::on(tmp, in[1]);

    cv::GComputation cc(cv::GIn(in[0], in[1]), cv::GOut(out));

    // Prepare compilation parameters manually
    cv::GMetaArgs in_metas = {GMetaArg(cv::GMatDesc{CV_8U,1,cv::gapi::own::Size(32,32)}),
                              GMetaArg(cv::GMatDesc{CV_8U,1,cv::gapi::own::Size(32,32)})};
    const auto pkg = cv::gapi::kernels<J::Foo, J::Bar>();

    // Directly instantiate G-API graph compiler and run partial compilation
    cv::gimpl::GCompiler compiler(cc, std::move(in_metas), cv::compile_args(pkg));
    cv::gimpl::GCompiler::GPtr graph = compiler.generateGraph();
    compiler.runPasses(*graph);

    cv::gimpl::GModel::ConstGraph gm(*graph);

    auto in0_nh = cv::gimpl::GModel::dataNodeOf(gm, in[0]);
    auto in1_nh = cv::gimpl::GModel::dataNodeOf(gm, in[1]);
    auto tmp_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp);
    auto out_nh = cv::gimpl::GModel::dataNodeOf(gm, out);

    EXPECT_FALSE(gm.metadata(in0_nh ).contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(in1_nh ).contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(out_nh).contains<cv::gimpl::Island>());
    EXPECT_TRUE(gm.metadata(tmp_nh).contains<cv::gimpl::Island>());

    auto isl_model = gm.metadata().get<cv::gimpl::IslandModel>().model;
    cv::gimpl::GIslandModel::ConstGraph gim(*isl_model);

    const auto is_island = [&](ade::NodeHandle nh) {
        return (cv::gimpl::NodeKind::ISLAND
                == gim.metadata(nh).get<cv::gimpl::NodeKind>().k);
    };
    const std::size_t num_isl = std::count_if(gim.nodes().begin(),
                                              gim.nodes().end(),
                                              is_island);
    EXPECT_EQ(1u, num_isl);

    auto isl_nh  = cv::gimpl::GIslandModel::producerOf(gim, out_nh);
    auto isl_obj = gim.metadata(isl_nh).get<cv::gimpl::FusedIsland>().object;

    EXPECT_TRUE(ade::util::contains(isl_obj->contents(), tmp_nh));

    EXPECT_EQ(2u, static_cast<std::size_t>(isl_nh->inNodes().size()));
    EXPECT_EQ(1u, static_cast<std::size_t>(isl_nh->outNodes().size()));
}

TEST(IslandFusion, DataNodeUsedDifferentBackend)
{
    // Define a computation:
    //
    //           internal isl            isl0
    //             ...........................
    //    (in1) -> :J::Foo--> (tmp) -> J::Foo: --> (out0)
    //             :............|............:
    //                          |     ........
    //                          `---->:S::Baz: --> (out1)
    //                                :......:

    // Check that the node was not dropped out of the island
    // because it is used by the kernel from another backend

    namespace J = Jupiter;
    namespace S = Saturn;

    cv::GMat in, tmp, out0;
    cv::GScalar out1;
    tmp  = I::Foo::on(in);
    out0 = I::Foo::on(tmp);
    out1 = I::Baz::on(tmp);

    cv::GComputation cc(cv::GIn(in), cv::GOut(out0, out1));

    // Prepare compilation parameters manually
    const auto in_meta = cv::GMetaArg(cv::GMatDesc{CV_8U,1,cv::gapi::own::Size(32,32)});
    const auto pkg     = cv::gapi::kernels<J::Foo, S::Baz>();

    // Directly instantiate G-API graph compiler and run partial compilation
    cv::gimpl::GCompiler compiler(cc, {in_meta}, cv::compile_args(pkg));
    cv::gimpl::GCompiler::GPtr graph = compiler.generateGraph();
    compiler.runPasses(*graph);

    // Inspect the graph and verify the islands configuration
    cv::gimpl::GModel::ConstGraph gm(*graph);

    auto in_nh   = cv::gimpl::GModel::dataNodeOf(gm, in);
    auto tmp_nh  = cv::gimpl::GModel::dataNodeOf(gm, tmp);
    auto out0_nh = cv::gimpl::GModel::dataNodeOf(gm, out0);
    auto out1_nh = cv::gimpl::GModel::dataNodeOf(gm, out1);

    EXPECT_TRUE(gm.metadata(tmp_nh).contains<cv::gimpl::Island>());

    auto isl_model = gm.metadata().get<cv::gimpl::IslandModel>().model;
    cv::gimpl::GIslandModel::ConstGraph gim(*isl_model);

    auto isl_nh  = cv::gimpl::GIslandModel::producerOf(gim, tmp_nh);
    auto isl_obj = gim.metadata(isl_nh).get<cv::gimpl::FusedIsland>().object;

    EXPECT_TRUE(ade::util::contains(isl_obj->contents(), tmp_nh));

    EXPECT_EQ(2u, static_cast<std::size_t>(isl_nh->outNodes().size()));
    EXPECT_EQ(7u, static_cast<std::size_t>(gm.nodes().size()));
    EXPECT_EQ(6u, static_cast<std::size_t>(gim.nodes().size()));
}

TEST(IslandFusion, LoopBetweenDifferentBackends)
{
    // Define a computation:
    //
    //
    //            .............................
    //    (in) -> :J::Baz -> (tmp0) -> J::Quux: -> (out0)
    //      |     :............|..........^....
    //      |     ........     |          |         ........
    //      `---->:S::Foo:     `----------|-------->:S::Qux:-> (out1)
    //            :....|.:                |         :....^.:
    //                 |                  |              |
    //                 `-------------- (tmp1) -----------`

    // Kernels S::Foo and S::Qux cannot merge, because there will be a cycle between islands

    namespace J = Jupiter;
    namespace S = Saturn;

    cv::GScalar tmp0;
    cv::GMat in, tmp1, out0, out1;

    tmp0 = I::Baz::on(in);
    tmp1 = I::Foo::on(in);
    out1 = I::Qux::on(tmp1, tmp0);
    out0 = I::Quux::on(tmp0, tmp1);

    cv::GComputation cc(cv::GIn(in), cv::GOut(out1, out0));

    // Prepare compilation parameters manually
    const auto in_meta = cv::GMetaArg(cv::GMatDesc{CV_8U,1,cv::gapi::own::Size(32,32)});
    const auto pkg     = cv::gapi::kernels<J::Baz, J::Quux, S::Foo, S::Qux>();

    // Directly instantiate G-API graph compiler and run partial compilation
    cv::gimpl::GCompiler compiler(cc, {in_meta}, cv::compile_args(pkg));
    cv::gimpl::GCompiler::GPtr graph = compiler.generateGraph();
    compiler.runPasses(*graph);

    cv::gimpl::GModel::ConstGraph gm(*graph);
    auto isl_model = gm.metadata().get<cv::gimpl::IslandModel>().model;
    cv::gimpl::GIslandModel::ConstGraph gim(*isl_model);

    auto in_nh   = cv::gimpl::GModel::dataNodeOf(gm, in);
    auto tmp0_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp0);
    auto tmp1_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp1);
    auto out0_nh = cv::gimpl::GModel::dataNodeOf(gm, out0);
    auto out1_nh = cv::gimpl::GModel::dataNodeOf(gm, out1);

    EXPECT_FALSE(gm.metadata(in_nh ).contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(out0_nh).contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(out1_nh).contains<cv::gimpl::Island>());
    // The node does not belong to the island so as not to form a cycle
    EXPECT_FALSE(gm.metadata(tmp1_nh).contains<cv::gimpl::Island>());

    EXPECT_TRUE(gm.metadata(tmp0_nh).contains<cv::gimpl::Island>());

    // There should be three islands in the GIslandModel
    const auto is_island = [&](ade::NodeHandle nh) {
        return (cv::gimpl::NodeKind::ISLAND
                == gim.metadata(nh).get<cv::gimpl::NodeKind>().k);
    };
    const std::size_t num_isl = std::count_if(gim.nodes().begin(),
                                              gim.nodes().end(),
                                              is_island);
    EXPECT_EQ(3u, num_isl);
}

TEST(IslandsFusion, PartionOverlapUserIsland)
{
    // Define a computation:
    //
    //           internal isl            isl0
    //             ........            ........
    //    (in0) -> :J::Foo:--> (tmp) ->:S::Bar: --> (out)
    //             :......:            :......:
    //                                    ^
    //                                    |
    //    (in1) --------------------------`

    // Check that internal islands does't overlap user island

    namespace J = Jupiter;
    namespace S = Saturn;

    GMat in[2];
    GMat tmp = I::Foo::on(in[0]);
    GMat out = I::Bar::on(tmp, in[1]);

    cv::gapi::island("isl0", cv::GIn(tmp, in[1]), cv::GOut(out));
    cv::GComputation cc(cv::GIn(in[0], in[1]), cv::GOut(out));

    // Prepare compilation parameters manually
    cv::GMetaArgs in_metas = {GMetaArg(cv::GMatDesc{CV_8U,1,cv::gapi::own::Size(32,32)}),
                              GMetaArg(cv::GMatDesc{CV_8U,1,cv::gapi::own::Size(32,32)})};
    const auto pkg = cv::gapi::kernels<J::Foo, J::Bar>();

    // Directly instantiate G-API graph compiler and run partial compilation
    cv::gimpl::GCompiler compiler(cc, std::move(in_metas), cv::compile_args(pkg));
    cv::gimpl::GCompiler::GPtr graph = compiler.generateGraph();
    compiler.runPasses(*graph);

    cv::gimpl::GModel::ConstGraph gm(*graph);
    auto isl_model = gm.metadata().get<cv::gimpl::IslandModel>().model;
    cv::gimpl::GIslandModel::ConstGraph gim(*isl_model);

    auto in0_nh = cv::gimpl::GModel::dataNodeOf(gm, in[0]);
    auto in1_nh = cv::gimpl::GModel::dataNodeOf(gm, in[1]);
    auto tmp_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp);
    auto out_nh = cv::gimpl::GModel::dataNodeOf(gm, out);

    auto foo_nh  = cv::gimpl::GIslandModel::producerOf(gim, tmp_nh);
    auto foo_obj = gim.metadata(foo_nh).get<cv::gimpl::FusedIsland>().object;

    auto bar_nh  = cv::gimpl::GIslandModel::producerOf(gim, out_nh);
    auto bar_obj = gim.metadata(bar_nh).get<cv::gimpl::FusedIsland>().object;

    EXPECT_FALSE(gm.metadata(in0_nh ).contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(in1_nh ).contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(out_nh).contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(tmp_nh).contains<cv::gimpl::Island>());
    EXPECT_FALSE(foo_obj->is_user_specified());
    EXPECT_TRUE(bar_obj->is_user_specified());
}

TEST(IslandsFusion, DISABLED_IslandContainsDifferentBackends)
{
    // Define a computation:
    //
    //                       isl0
    //             ............................
    //    (in0) -> :J::Foo:--> (tmp) -> S::Bar: --> (out)
    //             :..........................:
    //                                    ^
    //                                    |
    //    (in1) --------------------------`

    // Try create island contains different backends

    namespace J = Jupiter;
    namespace S = Saturn;

    GMat in[2];
    GMat tmp = I::Foo::on(in[0]);
    GMat out = I::Bar::on(tmp, in[1]);

    cv::gapi::island("isl0", cv::GIn(in[0], in[1]), cv::GOut(out));
    cv::GComputation cc(cv::GIn(in[0], in[1]), cv::GOut(out));

    // Prepare compilation parameters manually
    cv::GMetaArgs in_metas = {GMetaArg(cv::GMatDesc{CV_8U,1,cv::gapi::own::Size(32,32)}),
                              GMetaArg(cv::GMatDesc{CV_8U,1,cv::gapi::own::Size(32,32)})};
    const auto pkg = cv::gapi::kernels<J::Foo, S::Bar>();

    // Directly instantiate G-API graph compiler and run partial compilation
    cv::gimpl::GCompiler compiler(cc, std::move(in_metas), cv::compile_args(pkg));
    cv::gimpl::GCompiler::GPtr graph = compiler.generateGraph();
    EXPECT_ANY_THROW(compiler.runPasses(*graph));
}

TEST(IslandFusion, WithLoop)
{
    namespace J = Jupiter; // see mock_kernels.cpp

    // Define a computation:
    //
    //    (in) -> J::Foo --> (tmp0) -> J::Foo --> (tmp1) -> J::Qux -> (out)
    //                            :                        ^
    //                            '--> J::Baz --> (scl0) --'
    //
    // The whole thing should be merged to a single island
    // There's a cycle warning if Foo/Foo/Qux are merged first
    // Then this island both produces data for Baz and consumes data
    // from Baz. This is a cycle and it should be avoided by the merging code.
    //
    cv::GMat    in;
    cv::GMat    tmp0 = I::Foo::on(in);
    cv::GMat    tmp1 = I::Foo::on(tmp0);
    cv::GScalar scl0 = I::Baz::on(tmp0);
    cv::GMat    out  = I::Qux::on(tmp1, scl0);
    cv::GComputation cc(in, out);

    // Prepare compilation parameters manually
    const auto in_meta = cv::GMetaArg(cv::GMatDesc{CV_8U,1,cv::gapi::own::Size(32,32)});
    const auto pkg     = cv::gapi::kernels<J::Foo, J::Baz, J::Qux>();

    // Directly instantiate G-API graph compiler and run partial compilation
    cv::gimpl::GCompiler compiler(cc, {in_meta}, cv::compile_args(pkg));
    cv::gimpl::GCompiler::GPtr graph = compiler.generateGraph();
    compiler.runPasses(*graph);

    // Inspect the graph and verify the islands configuration
    cv::gimpl::GModel::ConstGraph gm(*graph);

    auto in_nh   = cv::gimpl::GModel::dataNodeOf(gm, in);
    auto tmp0_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp0);
    auto tmp1_nh = cv::gimpl::GModel::dataNodeOf(gm, tmp1);
    auto scl0_nh = cv::gimpl::GModel::dataNodeOf(gm, scl0);
    auto out_nh  = cv::gimpl::GModel::dataNodeOf(gm, out);

    // in/out mats shouldn't be assigned to any Island
    EXPECT_FALSE(gm.metadata(in_nh ).contains<cv::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(out_nh).contains<cv::gimpl::Island>());

    // tmp0/tmp1/scl should be assigned to island
    EXPECT_TRUE(gm.metadata(tmp0_nh).contains<cv::gimpl::Island>());
    EXPECT_TRUE(gm.metadata(tmp1_nh).contains<cv::gimpl::Island>());
    EXPECT_TRUE(gm.metadata(scl0_nh).contains<cv::gimpl::Island>());

    // Check that there's a single island object and it contains all
    // that data object handles

    cv::gimpl::GModel::ConstGraph cg(*graph);
    auto isl_model = cg.metadata().get<cv::gimpl::IslandModel>().model;
    cv::gimpl::GIslandModel::ConstGraph gim(*isl_model);

    const auto is_island = [&](ade::NodeHandle nh) {
        return (cv::gimpl::NodeKind::ISLAND
                == gim.metadata(nh).get<cv::gimpl::NodeKind>().k);
    };
    const std::size_t num_isl = std::count_if(gim.nodes().begin(),
                                              gim.nodes().end(),
                                              is_island);
    EXPECT_EQ(1u, num_isl);

    auto isl_nh  = cv::gimpl::GIslandModel::producerOf(gim, out_nh);
    auto isl_obj = gim.metadata(isl_nh).get<cv::gimpl::FusedIsland>().object;
    EXPECT_TRUE(ade::util::contains(isl_obj->contents(), tmp0_nh));
    EXPECT_TRUE(ade::util::contains(isl_obj->contents(), tmp1_nh));
    EXPECT_TRUE(ade::util::contains(isl_obj->contents(), scl0_nh));
}

TEST(IslandFusion, Regression_ShouldFuseAll)
{
    // Initially the merge procedure didn't work as expected and
    // stopped fusion even if it could be continued (e.g. full
    // GModel graph could be fused into a single GIsland node).
    // Example of this is custom RGB 2 YUV pipeline as shown below:

    cv::GMat r, g, b;
    cv::GMat y = 0.299f*r + 0.587f*g + 0.114f*b;
    cv::GMat u = 0.492f*(b - y);
    cv::GMat v = 0.877f*(r - y);

    cv::GComputation customCvt({r, g, b}, {y, u, v});

    const auto in_meta = cv::GMetaArg(cv::GMatDesc{CV_8U,1,cv::Size(32,32)});

    // Directly instantiate G-API graph compiler and run partial compilation
    cv::gimpl::GCompiler compiler(customCvt, {in_meta,in_meta,in_meta}, cv::compile_args());
    cv::gimpl::GCompiler::GPtr graph = compiler.generateGraph();
    compiler.runPasses(*graph);

    cv::gimpl::GModel::ConstGraph cg(*graph);
    auto isl_model = cg.metadata().get<cv::gimpl::IslandModel>().model;
    cv::gimpl::GIslandModel::ConstGraph gim(*isl_model);

    std::vector<ade::NodeHandle> data_nhs;
    std::vector<ade::NodeHandle> isl_nhs;
    for (auto &&nh : gim.nodes())
    {
        if (gim.metadata(nh).contains<cv::gimpl::FusedIsland>())
            isl_nhs.push_back(std::move(nh));
        else if (gim.metadata(nh).contains<cv::gimpl::DataSlot>())
            data_nhs.push_back(std::move(nh));
        else FAIL() << "GIslandModel node with unexpected metadata type";
    }

    EXPECT_EQ(6u, data_nhs.size()); // 3 input nodes + 3 output nodes
    EXPECT_EQ(1u, isl_nhs.size());  // 1 island
}

// FIXME: add more tests on mixed (hetero) graphs
// ADE-222, ADE-223

// FIXME: add test on combination of user-specified island
// which should be heterogeneous (based on kernel availability)
// but as we don't support this, compilation should fail

// FIXME: add tests on automatic inferred islands which are
// connected via 1) gmat 2) gscalar 3) garray,
// check the case with executor
// check the case when this 1/2/3 interim object is also gcomputation output

} // namespace opencv_test
