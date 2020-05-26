// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#include "precomp.hpp"

#include <ade/util/algorithm.hpp>

#include <ade/util/range.hpp>
#include <ade/util/zip_range.hpp>
#include <ade/util/chain_range.hpp>

#include <ade/typed_graph.hpp>

#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/util/any.hpp>
#include <opencv2/gapi/gtype_traits.hpp>

#include "compiler/gobjref.hpp"
#include "compiler/gmodel.hpp"

#include "gs11nbackend.hpp"

#include "api/gbackend_priv.hpp" // FIXME: Make it part of Backend SDK!

#include "backends/common/serialization.hpp"
#include "logger.hpp"

using namespace cv::gimpl;

// FIXME: Is there a way to take a typed graph (our GModel),
// and create a new typed graph _ATOP_ of that (by extending with a couple of
// new types?).
// Alternatively, is there a way to compose types graphs?
//
// If not, we need to introduce that!
using GS11NModel = ade::TypedGraph
    < opencv_test::s11n::impl::Unit
    , cv::gimpl::Protocol
    >;

// FIXME: Same issue with Typed and ConstTyped
using GConstGS11NModel = ade::ConstTypedGraph
    < opencv_test::s11n::impl::Unit
    , cv::gimpl::Protocol
    >;

namespace
{
    class GS11NBackendImpl final: public cv::gapi::GBackend::Priv
    {
        virtual void unpackKernel(ade::Graph            &graph,
                                  const ade::NodeHandle &op_node,
                                  const cv::GKernelImpl &impl) override
        {
            GS11NModel gm(graph);
            auto s11n_impl = cv::util::any_cast<opencv_test::GS11NKernel>(impl.opaque);
            gm.metadata(op_node).set(opencv_test::s11n::impl::Unit{s11n_impl});
        }

        virtual EPtr compile(const ade::Graph &graph,
                             const cv::GCompileArgs &,
                             const std::vector<ade::NodeHandle> &nodes) const override
        {
            //cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_INFO);
            cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
            //Graph serialization - dump - read - de-serialization path
            //Dump graph file.
            cv::gimpl::s11n::SerializationStream out_stream;
            cv::gimpl::s11n::serialize(out_stream, graph, nodes);
            CV_LOG_INFO(NULL, "out_stream.dump_storage size in bytes ..." << out_stream.getSize());

            std::ofstream dump_file;
            dump_file.open("my_graph.bin", std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
            if (dump_file.is_open())
            {
                dump_file.seekp(0, std::ofstream::beg);
                dump_file.write((const char*)out_stream.getData(), out_stream.getSize());
            }
            dump_file.close();
            ////////////////////////////////////////////////////////////
            //Restore graph from file.
            std::ifstream new_dump_file ("my_graph.bin", std::ifstream::in | std::ofstream::binary);
            auto gp = std::make_shared<ade::Graph>();
            if (new_dump_file.is_open())
            {
                new_dump_file.seekg(0, new_dump_file.end);
                std::streampos length = new_dump_file.tellg();
                CV_LOG_INFO(NULL, "new_dump_file size in bytes ..." << (int)length);
                new_dump_file.seekg(0, new_dump_file.beg);
                char * buffer = new char[(int)length];
                new_dump_file.read(buffer, length);
                cv::gimpl::s11n::DeSerializationStream in_stream =
                    cv::gimpl::s11n::DeSerializationStream(buffer, length);
                delete[] buffer;
                in_stream >> *gp;
            }
            new_dump_file.close();

            auto& g_s = *gp.get();

            //Use CPU serialization kernels package to test
            cv::gapi::GKernelPackage s11n_kernels = opencv_test::s11n::kernels();
            //Compiler pass one more time
            auto pass_ctx = ade::passes::PassContext{g_s};
            cv::gimpl::passes::resolveKernels(pass_ctx, s11n_kernels);

            std::vector<ade::NodeHandle> nh(gp->nodes().begin(), gp->nodes().end());
            return EPtr{new opencv_test::s11n::impl::GS11NExecutable(g_s, std::move(nh), gp)};
        }
   };
}

cv::gapi::GBackend opencv_test::s11n::impl::backend()
{
    static cv::gapi::GBackend this_backend(std::make_shared<GS11NBackendImpl>());
    return this_backend;
}

// GS11NExecutable implementation //////////////////////////////////////////////
opencv_test::s11n::impl::GS11NExecutable::GS11NExecutable(const ade::Graph &g,
                                          const std::vector<ade::NodeHandle> &nodes,
                                          std::shared_ptr<ade::Graph> gp)
    : m_g(g), m_gm(m_g), m_gp(gp)
{
    //const auto s = s11n::serialize(m_gm, nodes);
    //s11n::deserialize(s);
    //s11n::printGSerialized(s);

    // FIXME: reuse code from GModelBuilder/GModel!
    // ObjectCounter?? (But seems we need existing mapping by shape+id)



    // Convert list of operations (which is topologically sorted already)
    // into an execution script.
    for (auto &nh : nodes)
    {
        switch (m_gm.metadata(nh).get<NodeType>().t)
        {
        case NodeType::OP: m_script.push_back({nh, GModel::collectOutputMeta(m_gm, nh)}); break;
        case NodeType::DATA:
        {
            m_dataNodes.push_back(nh);
            const auto &desc = m_gm.metadata(nh).get<Data>();
            if (desc.storage == Data::Storage::CONST_VAL)
            {
                auto rc = RcDesc{desc.rc, desc.shape, desc.ctor};
                magazine::bindInArg(m_res, rc, m_gm.metadata(nh).get<ConstValue>().arg);
            }
            //preallocate internal Mats in advance
            if (desc.storage == Data::Storage::INTERNAL && desc.shape == cv::GShape::GMAT)
            {
                const auto mat_desc = cv::util::get<cv::GMatDesc>(desc.meta);
                auto& mat = m_res.slot<cv::Mat>()[desc.rc];
                createMat(mat_desc, mat);
            }
            break;
        }
        default: cv::util::throw_error(std::logic_error("Unsupported NodeType type"));
        }
    }
}

// FIXME: Document what it does
cv::GArg opencv_test::s11n::impl::GS11NExecutable::packArg(const cv::GArg &arg)
{
    // No API placeholders allowed at this point
    // FIXME: this check has to be done somewhere in compilation stage.
    GAPI_Assert(   arg.kind != cv::detail::ArgKind::GMAT
                && arg.kind != cv::detail::ArgKind::GSCALAR
                && arg.kind != cv::detail::ArgKind::GARRAY
                && arg.kind != cv::detail::ArgKind::GOPAQUE
    );

    if (arg.kind != cv::detail::ArgKind::GOBJREF)
    {
        // All other cases - pass as-is, with no transformations to GArg contents.
        return arg;
    }
    GAPI_Assert(arg.kind == cv::detail::ArgKind::GOBJREF);

    // Wrap associated CPU object (either host or an internal one)
    // FIXME: object can be moved out!!! GExecutor faced that.
    const cv::gimpl::RcDesc &ref = arg.get<cv::gimpl::RcDesc>();
    switch (ref.shape)
    {
    case cv::GShape::GMAT:    return cv::GArg(m_res.slot<cv::Mat>()   [ref.id]);
    case cv::GShape::GSCALAR: return cv::GArg(m_res.slot<cv::Scalar>()[ref.id]);
    // Note: .at() is intentional for GArray and GOpaque as objects MUST be already there
    //   (and constructed by either bindIn/Out or resetInternal)
    case cv::GShape::GARRAY:  return cv::GArg(m_res.slot<cv::detail::VectorRef>().at(ref.id));
    case cv::GShape::GOPAQUE: return cv::GArg(m_res.slot<cv::detail::OpaqueRef>().at(ref.id));
    default:
        cv::util::throw_error(std::logic_error("Unsupported GShape type"));
        break;
    }
}

void opencv_test::s11n::impl::GS11NExecutable::run(std::vector<InObj>  &&input_objs,
                                    std::vector<OutObj> &&output_objs)
{
    // Update resources with run-time information - what this Island
    // has received from user (or from another Island, or mix...)
    // FIXME: Check input/output objects against GIsland protocol

    for (auto& it : input_objs)   magazine::bindInArg (m_res, it.first, it.second);
    for (auto& it : output_objs)  magazine::bindOutArg(m_res, it.first, it.second);

    // Initialize (reset) internal data nodes with user structures
    // before processing a frame (no need to do it for external data structures)
    GModel::ConstGraph gm(m_g);
    for (auto nh : m_dataNodes)
    {
        const auto &desc = gm.metadata(nh).get<Data>();

        if (   desc.storage == Data::Storage::INTERNAL
            && !cv::util::holds_alternative<cv::util::monostate>(desc.ctor))
        {
            // FIXME: Note that compile-time constant data objects (like
            // a value-initialized GArray<T>) also satisfy this condition
            // and should be excluded, but now we just don't support it
            magazine::resetInternalData(m_res, desc);
        }
    }

    // OpenCV backend execution is not a rocket science at all.
    // Simply invoke our kernels in the proper order.
    GConstGS11NModel gcm(m_g);
    for (auto &op_info : m_script)
    {
        const auto &op = m_gm.metadata(op_info.nh).get<Op>();

        // Obtain our real execution unit
        // TODO: Should kernels be copyable?
        GS11NKernel k = gcm.metadata(op_info.nh).get<Unit>().k;

        // Initialize kernel's execution context:
        // - Input parameters
        GS11NContext context;
        context.m_args.reserve(op.args.size());

        using namespace std::placeholders;
        ade::util::transform(op.args,
                             std::back_inserter(context.m_args),
                             std::bind(&GS11NExecutable::packArg, this, _1)
        );

        // - Output parameters.
        // FIXME: pre-allocate internal Mats, etc, according to the known meta
        for (const auto &out_it : ade::util::indexed(op.outs))
        {
            // FIXME: Can the same GArg type resolution mechanism be reused here?
            const auto out_port  = ade::util::index(out_it);
            const auto out_desc  = ade::util::value(out_it);
            context.m_results[out_port] = magazine::getObjPtr(m_res, out_desc);
        }

        // Now trigger the executable unit
        k.apply(context);

        //As Kernels are forbidden to allocate memory for (Mat) outputs,
        //this code seems redundant, at least for Mats
        //FIXME: unify with cv::detail::ensure_out_mats_not_reallocated
        //FIXME: when it's done, remove can_describe(const GMetaArg&, const GRunArgP&)
        //and descr_of(const cv::GRunArgP &argp)
        for (const auto &out_it : ade::util::indexed(op_info.expected_out_metas))
        {
            const auto out_index      = ade::util::index(out_it);
            const auto expected_meta  = ade::util::value(out_it);

            if (!can_describe(expected_meta, context.m_results[out_index]))
            {
                const auto out_meta = descr_of(context.m_results[out_index]);
                cv::util::throw_error
                    (std::logic_error
                     ("Output meta doesn't "
                      "coincide with the generated meta\n"
                      "Expected: " + ade::util::to_string(expected_meta) + "\n"
                      "Actual  : " + ade::util::to_string(out_meta)));
            }
        }
    } // for(m_script)

    for (auto &it : output_objs) magazine::writeBack(m_res, it.first, it.second);
}
