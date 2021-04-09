// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <ade/util/zip_range.hpp> // zip_range, indexed

#include <opencv2/gapi/util/throw.hpp> // throw_error
#include <opencv2/gapi/python/python.hpp>

#include "api/gbackend_priv.hpp"
#include "backends/common/gbackend.hpp"

cv::gapi::python::GPythonKernel::GPythonKernel(cv::gapi::python::Impl run)
    : m_run(run)
{
}

cv::GRunArgs cv::gapi::python::GPythonKernel::operator()(const cv::gapi::python::GPythonContext& ctx)
{
    return m_run(ctx);
}

cv::gapi::python::GPythonFunctor::GPythonFunctor(const char* id,
                                                 const cv::gapi::python::GPythonFunctor::Meta &meta,
                                                 const cv::gapi::python::Impl& impl)
    : gapi::GFunctor(id), impl_{GPythonKernel{impl}, meta}
{
}

cv::GKernelImpl cv::gapi::python::GPythonFunctor::impl() const
{
    return impl_;
}

cv::gapi::GBackend cv::gapi::python::GPythonFunctor::backend() const
{
    return cv::gapi::python::backend();
}

namespace {

struct PythonUnit
{
    static const char *name() { return "PythonUnit"; }
    cv::gapi::python::GPythonKernel kernel;
};

using PythonModel = ade::TypedGraph
    < cv::gimpl::Op
    , PythonUnit
    >;

using ConstPythonModel = ade::ConstTypedGraph
    < cv::gimpl::Op
    , PythonUnit
    >;

class GPythonExecutable final: public cv::gimpl::GIslandExecutable
{
    virtual void run(std::vector<InObj>  &&,
                     std::vector<OutObj> &&) override;

    virtual bool allocatesOutputs() const override { return true; }
    // Return an empty RMat since we will reuse the input.
    // There is no need to allocate and copy 4k image here.
    virtual cv::RMat allocate(const cv::GMatDesc&) const override { return {}; }

    virtual bool canReshape() const override { return true; }
    virtual void reshape(ade::Graph&, const cv::GCompileArgs&) override {
        // Do nothing here
    }

public:
    GPythonExecutable(const ade::Graph                   &,
                      const std::vector<ade::NodeHandle> &);

    const ade::Graph& m_g;
    cv::gimpl::GModel::ConstGraph m_gm;
    cv::gapi::python::GPythonKernel m_kernel;
    ade::NodeHandle m_op;

    cv::GTypesInfo m_out_info;
    cv::GMetaArgs  m_in_metas;
    cv::gimpl::Mag m_res;
};

static cv::GArg packArg(cv::gimpl::Mag& m_res, const cv::GArg &arg)
{
    // No API placeholders allowed at this point
    // FIXME: this check has to be done somewhere in compilation stage.
    GAPI_Assert(   arg.kind != cv::detail::ArgKind::GMAT
                && arg.kind != cv::detail::ArgKind::GSCALAR
                && arg.kind != cv::detail::ArgKind::GARRAY
                && arg.kind != cv::detail::ArgKind::GOPAQUE
                && arg.kind != cv::detail::ArgKind::GFRAME);

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
    case cv::GShape::GFRAME:  return cv::GArg(m_res.slot<cv::MediaFrame>().at(ref.id));
    default:
        cv::util::throw_error(std::logic_error("Unsupported GShape type"));
        break;
    }
}

static void writeBack(cv::GRunArg& arg, cv::GRunArgP& out)
{
    switch (arg.index())
    {
        case cv::GRunArg::index_of<cv::Mat>():
        {
            auto& rmat = *cv::util::get<cv::RMat*>(out);
            rmat = cv::make_rmat<cv::gimpl::RMatAdapter>(cv::util::get<cv::Mat>(arg));
            break;
        }
        case cv::GRunArg::index_of<cv::Scalar>():
        {
            *cv::util::get<cv::Scalar*>(out) = cv::util::get<cv::Scalar>(arg);
            break;
        }
        case cv::GRunArg::index_of<cv::detail::OpaqueRef>():
        {
            auto& oref = cv::util::get<cv::detail::OpaqueRef>(arg);
            cv::util::get<cv::detail::OpaqueRef>(out).mov(oref);
            break;
        }
        case cv::GRunArg::index_of<cv::detail::VectorRef>():
        {
            auto& vref = cv::util::get<cv::detail::VectorRef>(arg);
            cv::util::get<cv::detail::VectorRef>(out).mov(vref);
            break;
        }
        default:
            GAPI_Assert(false && "Unsupported output type");
    }
}

void GPythonExecutable::run(std::vector<InObj>  &&input_objs,
                            std::vector<OutObj> &&output_objs)
{
    const auto &op = m_gm.metadata(m_op).get<cv::gimpl::Op>();
    for (auto& it : input_objs) cv::gimpl::magazine::bindInArg(m_res, it.first, it.second);

    using namespace std::placeholders;
    cv::GArgs inputs;
    ade::util::transform(op.args,
                         std::back_inserter(inputs),
                         std::bind(&packArg, std::ref(m_res), _1));


    cv::gapi::python::GPythonContext ctx{inputs, m_in_metas, m_out_info};
    auto outs = m_kernel(ctx);

    for (auto&& it : ade::util::zip(outs, output_objs))
    {
        writeBack(std::get<0>(it), std::get<1>(it).second);
    }
}

class GPythonBackendImpl final: public cv::gapi::GBackend::Priv
{
    virtual void unpackKernel(ade::Graph            &graph,
            const ade::NodeHandle &op_node,
            const cv::GKernelImpl &impl) override
    {
        PythonModel gm(graph);
        const auto &kernel  = cv::util::any_cast<cv::gapi::python::GPythonKernel>(impl.opaque);
        gm.metadata(op_node).set(PythonUnit{kernel});
    }

    virtual EPtr compile(const ade::Graph &graph,
                         const cv::GCompileArgs &,
                         const std::vector<ade::NodeHandle> &nodes) const override
    {
        return EPtr{new GPythonExecutable(graph, nodes)};
    }

    virtual bool controlsMerge() const override
    {
        return true;
    }

    virtual bool allowsMerge(const cv::gimpl::GIslandModel::Graph &,
                             const ade::NodeHandle &,
                             const ade::NodeHandle &,
                             const ade::NodeHandle &) const override
    {
        return false;
    }
};

GPythonExecutable::GPythonExecutable(const ade::Graph& g,
                                     const std::vector<ade::NodeHandle>& nodes)
    : m_g(g), m_gm(m_g)
{
    using namespace cv::gimpl;
    const auto is_op = [this](const ade::NodeHandle &nh)
    {
        return m_gm.metadata(nh).get<NodeType>().t == NodeType::OP;
    };

    auto it = std::find_if(nodes.begin(), nodes.end(), is_op);
    GAPI_Assert(it != nodes.end() && "No operators found for this island?!");

    ConstPythonModel cag(m_g);

    m_op = *it;
    m_kernel = cag.metadata(m_op).get<PythonUnit>().kernel;

    // Ensure this the only op in the graph
    if (std::any_of(it+1, nodes.end(), is_op))
    {
        cv::util::throw_error
            (std::logic_error
             ("Internal error: Python subgraph has multiple operations"));
    }

    m_out_info.reserve(m_op->outEdges().size());
    for (const auto &e : m_op->outEdges())
    {
        const auto& out_data = m_gm.metadata(e->dstNode()).get<cv::gimpl::Data>();
        m_out_info.push_back(cv::GTypeInfo{out_data.shape, out_data.kind, out_data.ctor});
    }

    const auto& op = m_gm.metadata(m_op).get<cv::gimpl::Op>();
    m_in_metas.resize(op.args.size());
    GAPI_Assert(m_op->inEdges().size() > 0);
    for (const auto &in_eh : m_op->inEdges())
    {
        const auto& input_port = m_gm.metadata(in_eh).get<Input>().port;
        const auto& input_nh   = in_eh->srcNode();
        const auto& input_meta = m_gm.metadata(input_nh).get<Data>().meta;
        m_in_metas.at(input_port) = input_meta;
    }
}

} // anonymous namespace

cv::gapi::GBackend cv::gapi::python::backend()
{
    static cv::gapi::GBackend this_backend(std::make_shared<GPythonBackendImpl>());
    return this_backend;
}
