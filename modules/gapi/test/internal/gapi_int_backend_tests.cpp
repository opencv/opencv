// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"
#include "../gapi_mock_kernels.hpp"

#include "compiler/gmodel.hpp"
#include "compiler/gcompiler.hpp"

namespace opencv_test {

namespace {

struct MockMeta
{
    static const char* name() { return "MockMeta"; }
};

class GMockBackendImpl final: public cv::gapi::GBackend::Priv
{
    virtual void unpackKernel(ade::Graph            &,
                              const ade::NodeHandle &,
                              const cv::GKernelImpl &) override
    {
        // Do nothing here
    }

    virtual EPtr compile(const ade::Graph &,
                         const cv::GCompileArgs &,
                         const std::vector<ade::NodeHandle> &) const override
    {
        // Do nothing here as well
        return {};
    }

    virtual void addBackendPasses(ade::ExecutionEngineSetupContext &ectx) override
    {
        ectx.addPass("transform", "set_mock_meta", [](ade::passes::PassContext &ctx) {
                ade::TypedGraph<MockMeta> me(ctx.graph);
                for (const auto &nh : me.nodes())
                {
                    me.metadata(nh).set(MockMeta{});
                }
            });
    }
};

static cv::gapi::GBackend mock_backend(std::make_shared<GMockBackendImpl>());

GAPI_OCV_KERNEL(MockFoo, I::Foo)
{
    static void run(const cv::Mat &, cv::Mat &) { /*Do nothing*/ }
    static cv::gapi::GBackend backend() { return mock_backend; } // FIXME: Must be removed
};

} // anonymous namespace

TEST(GBackend, CustomPassesExecuted)
{
    cv::GMat in;
    cv::GMat out = I::Foo::on(in);
    cv::GComputation c(in, out);

    // Prepare compilation parameters manually
    const auto in_meta = cv::GMetaArg(cv::GMatDesc{CV_8U,1,cv::Size(32,32)});
    const auto pkg     = cv::gapi::kernels<MockFoo>();

    // Directly instantiate G-API graph compiler and run partial compilation
    cv::gimpl::GCompiler compiler(c, {in_meta}, cv::compile_args(pkg));
    cv::gimpl::GCompiler::GPtr graph = compiler.generateGraph();
    compiler.runPasses(*graph);

    // Inspect the graph and verify the metadata written by Mock backend
    ade::TypedGraph<MockMeta> me(*graph);
    EXPECT_LT(0u, static_cast<std::size_t>(me.nodes().size()));
    for (const auto &nh : me.nodes())
    {
        EXPECT_TRUE(me.metadata(nh).contains<MockMeta>());
    }
}

} // namespace opencv_test
