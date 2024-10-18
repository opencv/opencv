// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#include "precomp.hpp"

#include <vector>
#include <stack>
#include <unordered_map>

#include <ade/util/algorithm.hpp>      // any_of
#include <ade/util/zip_range.hpp>      // zip_range, indexed

#include <ade/graph.hpp>
#include <ade/passes/check_cycles.hpp>

#include "api/gcomputation_priv.hpp"
#include "api/gnode_priv.hpp"   // FIXME: why it is here?
#include "api/gproto_priv.hpp"  // FIXME: why it is here?
#include "api/gcall_priv.hpp"   // FIXME: why it is here?

#include "api/gbackend_priv.hpp" // Backend basic API (newInstance, etc)

#include "compiler/gmodel.hpp"
#include "compiler/gmodelbuilder.hpp"
#include "compiler/gcompiler.hpp"
#include "compiler/gcompiled_priv.hpp"
#include "compiler/gstreaming_priv.hpp"
#include "compiler/passes/passes.hpp"
#include "compiler/passes/pattern_matching.hpp"

#include "executor/gexecutor.hpp"
#include "executor/gthreadedexecutor.hpp"
#include "executor/gstreamingexecutor.hpp"
#include "backends/common/gbackend.hpp"
#include "backends/common/gmetabackend.hpp"
#include "backends/streaming/gstreamingbackend.hpp" // cv::gimpl::streaming::kernels()

// <FIXME:>
#if !defined(GAPI_STANDALONE)
#include <opencv2/gapi/cpu/core.hpp>    // Also directly refer to Core,
#include <opencv2/gapi/cpu/imgproc.hpp> // ...Imgproc
#include <opencv2/gapi/cpu/video.hpp>   // ...and Video kernel implementations
#include <opencv2/gapi/render/render.hpp>   // render::ocv::backend()
#endif // !defined(GAPI_STANDALONE)
// </FIXME:>

#include <opencv2/gapi/gcompoundkernel.hpp> // compound::backend()

#include "logger.hpp"

namespace
{
    cv::GKernelPackage getKernelPackage(cv::GCompileArgs &args)
    {
        auto withAuxKernels = [](const cv::GKernelPackage& pkg) {
            cv::GKernelPackage aux_pkg;
            for (const auto &b : pkg.backends()) {
                aux_pkg = cv::gapi::combine(aux_pkg, b.priv().auxiliaryKernels());
            }
            // Always include built-in meta<> and copy implementation
            return cv::gapi::combine(pkg,
                                     aux_pkg,
                                     cv::gimpl::meta::kernels(),
                                     cv::gimpl::streaming::kernels());
        };

        auto has_use_only = cv::gapi::getCompileArg<cv::gapi::use_only>(args);
        if (has_use_only)
            return withAuxKernels(has_use_only.value().pkg);

        static auto ocv_pkg =
#if !defined(GAPI_STANDALONE)
            cv::gapi::combine(cv::gapi::core::cpu::kernels(),
                              cv::gapi::imgproc::cpu::kernels(),
                              cv::gapi::video::cpu::kernels(),
                              cv::gapi::render::ocv::kernels(),
                              cv::gapi::streaming::kernels());
#else
            cv::GKernelPackage();
#endif // !defined(GAPI_STANDALONE)

        auto user_pkg = cv::gapi::getCompileArg<cv::GKernelPackage>(args);
        auto user_pkg_with_aux = withAuxKernels(user_pkg.value_or(cv::GKernelPackage{}));
        return cv::gapi::combine(ocv_pkg, user_pkg_with_aux);
    }

    cv::gapi::GNetPackage getNetworkPackage(cv::GCompileArgs &args)
    {
        return cv::gapi::getCompileArg<cv::gapi::GNetPackage>(args)
            .value_or(cv::gapi::GNetPackage{});
    }

    cv::util::optional<std::string> getGraphDumpDirectory(cv::GCompileArgs& args)
    {
        auto dump_info = cv::gapi::getCompileArg<cv::graph_dump_path>(args);
        if (!dump_info.has_value())
        {
            const std::string path = cv::utils::getConfigurationParameterString("GRAPH_DUMP_PATH");
            return !path.empty()
                ? cv::util::make_optional(path)
                : cv::util::optional<std::string>();
        }
        else
        {
            return cv::util::make_optional(dump_info.value().m_dump_path);
        }
    }

    template<typename C>
    cv::GKernelPackage auxKernelsFrom(const C& c) {
        cv::GKernelPackage result;
        for (const auto &b : c) {
            result = cv::gapi::combine(result, b.priv().auxiliaryKernels());
        }
        return result;
    }

    using adeGraphs = std::vector<std::unique_ptr<ade::Graph>>;

    // Creates ADE graphs (patterns and substitutes) from pkg's transformations
    void makeTransformationGraphs(const cv::GKernelPackage& pkg,
                                  adeGraphs& patterns,
                                  adeGraphs& substitutes) {
        const auto& transforms = pkg.get_transformations();
        const auto size = transforms.size();
        if (0u == size) return;

        // pre-generate all required graphs
        patterns.resize(size);
        substitutes.resize(size);
        for (auto it : ade::util::zip(ade::util::toRange(transforms),
                                      ade::util::toRange(patterns),
                                      ade::util::toRange(substitutes))) {
            const auto& t = std::get<0>(it);
            auto&       p = std::get<1>(it);
            auto&       s = std::get<2>(it);
            p = cv::gimpl::GCompiler::makeGraph(t.pattern().priv());
            s = cv::gimpl::GCompiler::makeGraph(t.substitute().priv());
        }
    }

    void checkTransformations(const cv::GKernelPackage& pkg,
                              const adeGraphs& patterns,
                              const adeGraphs& substitutes) {
        const auto& transforms = pkg.get_transformations();
        const auto size = transforms.size();
        if (0u == size) return;

        GAPI_Assert(size == patterns.size());
        GAPI_Assert(size == substitutes.size());

        const auto empty = [] (const cv::gimpl::SubgraphMatch& m) {
            return m.inputDataNodes.empty() && m.startOpNodes.empty()
                && m.finishOpNodes.empty() && m.outputDataNodes.empty()
                && m.inputTestDataNodes.empty() && m.outputTestDataNodes.empty();
        };

        // **FIXME**: verify other types of endless loops. now, only checking if pattern exists in
        //            substitute within __the same__ transformation
        for (size_t i = 0; i < size; ++i) {
            const auto& p = patterns[i];
            const auto& s = substitutes[i];

            auto matchInSubstitute = cv::gimpl::findMatches(*p, *s);
            if (!empty(matchInSubstitute)) {
                std::stringstream ss;
                ss << "Error: (in transformation with description: '"
                    << transforms[i].description
                    << "') pattern is detected inside substitute";
                throw std::runtime_error(ss.str());
            }
        }
    }
} // anonymous namespace


// GCompiler implementation ////////////////////////////////////////////////////

cv::gimpl::GCompiler::GCompiler(const cv::GComputation &c,
                                GMetaArgs              &&metas,
                                GCompileArgs           &&args)
    : m_c(c), m_metas(std::move(metas)), m_args(std::move(args))
{
    using namespace std::placeholders;

    auto kernels_to_use  = getKernelPackage(m_args);
    auto networks_to_use = getNetworkPackage(m_args);
    std::unordered_set<cv::gapi::GBackend> all_backends;
    const auto take = [&](std::vector<cv::gapi::GBackend> &&v) {
        all_backends.insert(v.begin(), v.end());
    };
    take(kernels_to_use.backends());
    take(networks_to_use.backends());

    m_all_kernels = cv::gapi::combine(kernels_to_use,
                                      auxKernelsFrom(all_backends));
    // NB: The expectation in the line above is that
    // NN backends (present here via network package) always add their
    // inference kernels via auxiliary...()

    // sanity check transformations
    {
        adeGraphs patterns, substitutes;
        // FIXME: returning vectors of unique_ptrs from makeTransformationGraphs results in
        //        compile error (at least) on GCC 9 with usage of copy-ctor of std::unique_ptr, so
        //        using initialization by lvalue reference instead
        makeTransformationGraphs(m_all_kernels, patterns, substitutes);
        checkTransformations(m_all_kernels, patterns, substitutes);

        // NB: saving generated patterns to m_all_patterns to be used later in passes
        m_all_patterns = std::move(patterns);
    }

    auto dump_path       = getGraphDumpDirectory(m_args);

    m_e.addPassStage("init");
    m_e.addPass("init", "check_cycles",  ade::passes::CheckCycles());
    m_e.addPass("init", "apply_transformations",
                std::bind(passes::applyTransformations, _1, std::cref(m_all_kernels),
                    std::cref(m_all_patterns)));  // Note: and re-using patterns here
    m_e.addPass("init", "expand_kernels",
                std::bind(passes::expandKernels, _1,
                          m_all_kernels)); // NB: package is copied
    m_e.addPass("init", "topo_sort",     ade::passes::TopologicalSort());
    m_e.addPass("init", "init_islands",  passes::initIslands);
    m_e.addPass("init", "check_islands", passes::checkIslands);
    // TODO:
    // - Check basic graph validity (i.e., all inputs are connected)
    // - Complex dependencies (i.e. parent-child) unrolling
    // - etc, etc, etc

    // Remove GCompoundBackend to avoid calling setupBackend() with it in the list
    m_all_kernels.remove(cv::gapi::compound::backend());

    m_e.addPassStage("kernels");
    m_e.addPass("kernels", "bind_net_params",
                std::bind(passes::bindNetParams, _1,
                          networks_to_use));
    m_e.addPass("kernels", "resolve_kernels",
                std::bind(passes::resolveKernels, _1,
                          std::ref(m_all_kernels)));  // NB: and not copied here
                                                      // (no compound backend present here)
    m_e.addPass("kernels", "check_islands_content", passes::checkIslandsContent);

    // Special stage for intrinsics handling
    m_e.addPassStage("intrin");
    m_e.addPass("intrin", "desync",         passes::intrinDesync);
    m_e.addPass("intrin", "finalizeIntrin", passes::intrinFinalize);

    //Input metas may be empty when a graph is compiled for streaming
    m_e.addPassStage("meta");
    if (!m_metas.empty())
    {
        m_e.addPass("meta", "initialize",   std::bind(passes::initMeta, _1, std::ref(m_metas)));
        m_e.addPass("meta", "propagate",    std::bind(passes::inferMeta, _1, false));
        m_e.addPass("meta", "finalize",     passes::storeResultingMeta);
        // moved to another stage, FIXME: two dumps?
        //    m_e.addPass("meta", "dump_dot",     passes::dumpDotStdout);
    }
    // Special stage for backend-specific transformations
    // FIXME: document passes hierarchy and order for backend developers
    m_e.addPassStage("transform");

    m_e.addPassStage("exec");
    m_e.addPass("exec", "fuse_islands",     passes::fuseIslands);
    m_e.addPass("exec", "sync_islands",     passes::syncIslandTags);

    // FIXME: Since a set of passes is shared between
    // GCompiled/GStreamingCompiled, this pass is added here unconditionally
    // (even if it is not actually required to produce a GCompiled).
    // FIXME: add a better way to do that!
    m_e.addPass("exec", "add_streaming",    passes::addStreaming);

    // Note: Must be called after addStreaming as addStreaming pass
    // can possibly add new nodes to the IslandModel
    m_e.addPass("exec", "sort_islands",     passes::topoSortIslands);

    if (dump_path.has_value())
    {
        m_e.addPass("exec", "dump_dot", std::bind(passes::dumpGraph, _1,
                                                  dump_path.value()));
    }

    // FIXME: This should be called for "ActiveBackends" only (see metadata).
    // However, ActiveBackends are known only after passes are actually executed.
    // At these stage, they are not executed yet.
    ade::ExecutionEngineSetupContext ectx(m_e);
    auto backends = m_all_kernels.backends();
    for (auto &b : backends)
    {
        b.priv().addBackendPasses(ectx);
        if (!m_metas.empty())
        {
            b.priv().addMetaSensitiveBackendPasses(ectx);
        }
    }
}

void cv::gimpl::GCompiler::validateInputMeta()
{
    // FIXME: implement testing/accessor methods at the Priv's API level?
    if (!util::holds_alternative<GComputation::Priv::Expr>(m_c.priv().m_shape))
    {
        GAPI_LOG_WARNING(NULL, "Metadata validation is not implemented yet for"
                               " deserialized graphs!");
        return;
    }
    const auto &c_expr = util::get<cv::GComputation::Priv::Expr>(m_c.priv().m_shape);
    if (m_metas.size() != c_expr.m_ins.size())
    {
        util::throw_error(std::logic_error
                    ("COMPILE: GComputation interface / metadata mismatch! "
                     "(expected " + std::to_string(c_expr.m_ins.size()) + ", "
                     "got " + std::to_string(m_metas.size()) + " meta arguments)"));
    }

    const auto meta_matches = [](const GMetaArg &meta, const GProtoArg &proto) {
        switch (proto.index())
        {
        // FIXME: Auto-generate methods like this from traits:
        case GProtoArg::index_of<cv::GMat>():
        case GProtoArg::index_of<cv::GMatP>():
            return util::holds_alternative<cv::GMatDesc>(meta);

        case GProtoArg::index_of<cv::GFrame>():
            return util::holds_alternative<cv::GFrameDesc>(meta);

        case GProtoArg::index_of<cv::GScalar>():
            return util::holds_alternative<cv::GScalarDesc>(meta);

        case GProtoArg::index_of<cv::detail::GArrayU>():
            return util::holds_alternative<cv::GArrayDesc>(meta);

        case GProtoArg::index_of<cv::detail::GOpaqueU>():
            return util::holds_alternative<cv::GOpaqueDesc>(meta);

        default:
            GAPI_Error("InternalError");
        }
        return false; // should never happen
    };

    GAPI_LOG_DEBUG(nullptr, "Total count: " << m_metas.size());
    for (const auto meta_arg_idx : ade::util::indexed(ade::util::zip(m_metas, c_expr.m_ins)))
    {
        const auto &meta  = std::get<0>(ade::util::value(meta_arg_idx));
        const auto &proto = std::get<1>(ade::util::value(meta_arg_idx));

        const auto index = ade::util::index(meta_arg_idx);
        GAPI_LOG_DEBUG(nullptr, "Process index: " << index);

        // check types validity
        if (!meta_matches(meta, proto))
        {
            util::throw_error(std::logic_error
                        ("GComputation object type / metadata descriptor mismatch "
                         "(argument " + std::to_string(index) + ")"));
            // FIXME: report what we've got and what we've expected
        }

        // check value consistency
        gimpl::proto::validate_input_meta_arg(meta); //may throw
    }
    // All checks are ok
}

void cv::gimpl::GCompiler::validateOutProtoArgs()
{
    // FIXME: implement testing/accessor methods at the Priv's API level?
    if (!util::holds_alternative<GComputation::Priv::Expr>(m_c.priv().m_shape))
    {
        GAPI_LOG_WARNING(NULL, "Output parameter validation is not implemented yet for"
                               " deserialized graphs!");
        return;
    }
    const auto &c_expr = util::get<cv::GComputation::Priv::Expr>(m_c.priv().m_shape);
    for (const auto out_pos : ade::util::indexed(c_expr.m_outs))
    {
        const auto &node = proto::origin_of(ade::util::value(out_pos)).node;
        if (node.shape() != cv::GNode::NodeShape::CALL)
        {
            auto pos = ade::util::index(out_pos);
            util::throw_error(std::logic_error
                        ("Computation output " + std::to_string(pos) +
                         " is not a result of any operation"));
        }
    }
}

cv::gimpl::GCompiler::GPtr cv::gimpl::GCompiler::generateGraph()
{
    if (!m_metas.empty())
    {
        // Metadata may be empty if we're compiling our graph for streaming
        validateInputMeta();
    }
    validateOutProtoArgs();
    auto g = makeGraph(m_c.priv());
    if (!m_metas.empty())
    {
        GModel::Graph(*g).metadata().set(OriginalInputMeta{m_metas});
    }
    // FIXME: remove m_args, remove GCompileArgs from backends' method signatures,
    // rework backends to access GCompileArgs from graph metadata
    GModel::Graph(*g).metadata().set(CompileArgs{m_args});
    return g;
}

void cv::gimpl::GCompiler::runPasses(ade::Graph &g)
{
    m_e.runPasses(g);
    GAPI_LOG_INFO(NULL, "All compiler passes are successful");
}

void cv::gimpl::GCompiler::compileIslands(ade::Graph &g)
{
    compileIslands(g, m_args);
}

void cv::gimpl::GCompiler::compileIslands(ade::Graph &g, const cv::GCompileArgs &args)
{
    GModel::Graph gm(g);
    std::shared_ptr<ade::Graph> gptr(gm.metadata().get<IslandModel>().model);
    GIslandModel::Graph gim(*gptr);

    GIslandModel::compileIslands(gim, g, args);
}

cv::GCompiled cv::gimpl::GCompiler::produceCompiled(GPtr &&pg)
{
    // This is the final compilation step. Here:
    // - An instance of GExecutor is created. Depending on the platform,
    //   build configuration, etc, a GExecutor may be:
    //   - a naive single-thread graph interpreter;
    //   - a std::thread-based thing
    //   - a TBB-based thing, etc.
    // - All this stuff is wrapped into a GCompiled object and returned
    //   to user.

    // Note: this happens in the last pass ("compile_islands"):
    // - Each GIsland of GIslandModel instantiates its own,
    //   backend-specific executable object
    //   - Every backend gets a subgraph to execute, and builds
    //     an execution plan for it (backend-specific execution)
    // ...before call to produceCompiled();

    GModel::ConstGraph cgr(*pg);
    const auto &outMetas = GModel::ConstGraph(*pg).metadata()
        .get<OutputMeta>().outMeta;
    // FIXME: select which executor will be actually used,
    // make GExecutor abstract.

    auto use_threaded_exec = cv::gapi::getCompileArg<cv::use_threaded_executor>(m_args);
    std::unique_ptr<GAbstractExecutor> pE;
    if (use_threaded_exec) {
        const auto num_threads = use_threaded_exec.value().num_threads;
        GAPI_LOG_INFO(NULL, "Threaded executor with " << num_threads << " thread(s) will be used");
        pE.reset(new GThreadedExecutor(num_threads, std::move(pg)));
    } else {
        pE.reset(new GExecutor(std::move(pg)));
    }
    GCompiled compiled;
    compiled.priv().setup(m_metas, outMetas, std::move(pE));

    return compiled;
}

cv::GStreamingCompiled cv::gimpl::GCompiler::produceStreamingCompiled(GPtr &&pg)
{
    GStreamingCompiled compiled;
    GMetaArgs outMetas;

    // FIXME: the whole below construct is ugly, need to revise
    // how G*Compiled learns about its meta.
    if (!m_metas.empty())
    {
        outMetas = GModel::ConstGraph(*pg).metadata().get<OutputMeta>().outMeta;
    }

    GModel::ConstGraph cgr(*pg);

    std::unique_ptr<GStreamingExecutor> pE(new GStreamingExecutor(std::move(pg),
                                                                  m_args));
    if (!m_metas.empty() && !outMetas.empty())
    {
        compiled.priv().setup(m_metas, outMetas, std::move(pE));
    }
    else if (m_metas.empty() && outMetas.empty())
    {
        // Otherwise, set it up with executor object only
        compiled.priv().setup(std::move(pE));
    }
    else GAPI_Error("Impossible happened -- please report a bug");
    return compiled;
}

cv::GCompiled cv::gimpl::GCompiler::compile()
{
    std::unique_ptr<ade::Graph> pG = generateGraph();
    runPasses(*pG);
    compileIslands(*pG);
    return produceCompiled(std::move(pG));
}

cv::GStreamingCompiled cv::gimpl::GCompiler::compileStreaming()
{
    // FIXME: self-note to DM: now keep these compile()/compileStreaming() in sync!
    std::unique_ptr<ade::Graph> pG = generateGraph();
    GModel::Graph(*pG).metadata().set(Streaming{});
    runPasses(*pG);
    if (!m_metas.empty())
    {
        // If the metadata has been passed, compile our islands!
        compileIslands(*pG);
    }
    return produceStreamingCompiled(std::move(pG));
}

void cv::gimpl::GCompiler::runMetaPasses(ade::Graph &g, const cv::GMetaArgs &metas)
{
    auto pass_ctx = ade::passes::PassContext{g};
    cv::gimpl::passes::initMeta(pass_ctx, metas);
    cv::gimpl::passes::inferMeta(pass_ctx, true);
    cv::gimpl::passes::storeResultingMeta(pass_ctx);

    // Also run meta-sensitive backend-specific passes, if there's any.
    // FIXME: This may be hazardous if our backend are not very robust
    // in their passes -- how can we guarantee correct functioning in the
    // future?
    ade::ExecutionEngine engine;
    engine.addPassStage("exec"); // FIXME: Need a better decision on how we replicate
                                 // our main compiler stages here.
    ade::ExecutionEngineSetupContext ectx(engine);

    // NB: &&b or &b doesn't work here since "backends" is a set. Nevermind
    for (auto b : GModel::Graph(g).metadata().get<ActiveBackends>().backends)
    {
        b.priv().addMetaSensitiveBackendPasses(ectx);
    }
    engine.runPasses(g);
}

// Creates ADE graph from input/output proto args OR from its
// deserialized form
cv::gimpl::GCompiler::GPtr cv::gimpl::GCompiler::makeGraph(const cv::GComputation::Priv &priv) {
    std::unique_ptr<ade::Graph> pG(new ade::Graph);
    ade::Graph& g = *pG;

    if (cv::util::holds_alternative<cv::GComputation::Priv::Expr>(priv.m_shape)) {
        auto c_expr = cv::util::get<cv::GComputation::Priv::Expr>(priv.m_shape);
        cv::gimpl::GModel::Graph gm(g);
        cv::gimpl::GModel::init(gm);
        cv::gimpl::GModelBuilder builder(g);
        auto proto_slots = builder.put(c_expr.m_ins, c_expr.m_outs);

        // Store Computation's protocol in metadata
        cv::gimpl::Protocol p;
        std::tie(p.inputs, p.outputs, p.in_nhs, p.out_nhs) = proto_slots;
        gm.metadata().set(p);
    } else if (cv::util::holds_alternative<cv::GComputation::Priv::Dump>(priv.m_shape)) {
        auto c_dump = cv::util::get<cv::GComputation::Priv::Dump>(priv.m_shape);
        cv::gapi::s11n::reconstruct(c_dump, g);
    }
    return pG;
}
