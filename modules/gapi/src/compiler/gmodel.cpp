// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include <string>
#include <sstream> // used in GModel::log


#include <ade/util/zip_range.hpp>   // util::indexed
#include <ade/util/checked_cast.hpp>

#include "opencv2/gapi/gproto.hpp"
#include "api/gnode_priv.hpp"
#include "compiler/gobjref.hpp"
#include "compiler/gmodel.hpp"

namespace cv { namespace gimpl {

ade::NodeHandle GModel::mkOpNode(GModel::Graph &g, const GKernel &k, const std::vector<GArg> &args, const std::string &island)
{
    ade::NodeHandle op_h = g.createNode();
    g.metadata(op_h).set(NodeType{NodeType::OP});
    //These extra empty {} are to please GCC (-Wmissing-field-initializers)
    g.metadata(op_h).set(Op{k, args, {}, {}, {}});
    if (!island.empty())
        g.metadata(op_h).set(Island{island});
    return op_h;
}

ade::NodeHandle GModel::mkDataNode(GModel::Graph &g, const GOrigin& origin)
{
    ade::NodeHandle op_h = g.createNode();
    const auto id = g.metadata().get<DataObjectCounter>().GetNewId(origin.shape);
    g.metadata(op_h).set(NodeType{NodeType::DATA});

    GMetaArg meta;
    Data::Storage storage = Data::Storage::INTERNAL; // By default, all objects are marked INTERNAL

    if (origin.node.shape() == GNode::NodeShape::CONST_BOUNDED)
    {
        auto value = value_of(origin);
        meta       = descr_of(value);
        storage    = Data::Storage::CONST;
        g.metadata(op_h).set(ConstValue{value});
    }
    g.metadata(op_h).set(Data{origin.shape, id, meta, origin.ctor, storage});
    return op_h;
}

void GModel::linkIn(Graph &g, ade::NodeHandle opH, ade::NodeHandle objH, std::size_t in_port)
{
    // Check if input is already connected
    for (const auto& in_e : opH->inEdges())
    {
        GAPI_Assert(g.metadata(in_e).get<Input>().port != in_port);
    }

    auto &op = g.metadata(opH).get<Op>();
    auto &gm = g.metadata(objH).get<Data>();

     // FIXME: check validity using kernel prototype
    GAPI_Assert(in_port < op.args.size());

    ade::EdgeHandle eh = g.link(objH, opH);
    g.metadata(eh).set(Input{in_port});

    // Replace an API object with a REF (G* -> GOBJREF)
    op.args[in_port] = cv::GArg(RcDesc{gm.rc, gm.shape, {}});
}

void GModel::linkOut(Graph &g, ade::NodeHandle opH, ade::NodeHandle objH, std::size_t out_port)
{
    // FIXME: check validity using kernel prototype

    // Check if output is already connected
    for (const auto& out_e : opH->outEdges())
    {
        GAPI_Assert(g.metadata(out_e).get<Output>().port != out_port);
    }

    auto &op = g.metadata(opH).get<Op>();
    auto &gm = g.metadata(objH).get<Data>();

    GAPI_Assert(objH->inNodes().size() == 0u);

    ade::EdgeHandle eh = g.link(opH, objH);
    g.metadata(eh).set(Output{out_port});

    // TODO: outs must be allocated according to kernel protocol!
    const auto storage_with_port = ade::util::checked_cast<std::size_t>(out_port+1);
    const auto min_out_size = std::max(op.outs.size(), storage_with_port);
    op.outs.resize(min_out_size, RcDesc{-1,GShape::GMAT,{}}); // FIXME: Invalid shape instead?
    op.outs[out_port] = RcDesc{gm.rc, gm.shape, {}};
}

std::vector<ade::NodeHandle> GModel::orderedInputs(Graph &g, ade::NodeHandle nh)
{
    std::vector<ade::NodeHandle> sorted_in_nhs(nh->inEdges().size());
    for (const auto& in_eh : nh->inEdges())
    {
        const auto port = g.metadata(in_eh).get<cv::gimpl::Input>().port;
        GAPI_Assert(port < sorted_in_nhs.size());
        sorted_in_nhs[port] = in_eh->srcNode();
    }
    return sorted_in_nhs;
}

std::vector<ade::NodeHandle> GModel::orderedOutputs(Graph &g, ade::NodeHandle nh)
{
    std::vector<ade::NodeHandle> sorted_out_nhs(nh->outEdges().size());
    for (const auto& out_eh : nh->outEdges())
    {
        const auto port = g.metadata(out_eh).get<cv::gimpl::Output>().port;
        GAPI_Assert(port < sorted_out_nhs.size());
        sorted_out_nhs[port] = out_eh->dstNode();
    }
    return sorted_out_nhs;
}

void GModel::init(Graph& g)
{
    g.metadata().set(DataObjectCounter());
}

void GModel::log(Graph &g, ade::NodeHandle nh, std::string &&msg, ade::NodeHandle updater)
{
    std::string s = std::move(msg);
    if (updater != nullptr)
    {
        std::stringstream fmt;
        fmt << " (via " << updater << ")";
        s += fmt.str();
    }

    if (g.metadata(nh).contains<Journal>())
    {
        g.metadata(nh).get<Journal>().messages.push_back(s);
    }
    else
    {
        g.metadata(nh).set(Journal{{s}});
    }
}

// FIXME:
// Unify with GModel::log(.. ade::NodeHandle ..)
void GModel::log(Graph &g, ade::EdgeHandle eh, std::string &&msg, ade::NodeHandle updater)
{
    std::string s = std::move(msg);
    if (updater != nullptr)
    {
        std::stringstream fmt;
        fmt << " (via " << updater << ")";
        s += fmt.str();
    }

    if (g.metadata(eh).contains<Journal>())
    {
        g.metadata(eh).get<Journal>().messages.push_back(s);
    }
    else
    {
        g.metadata(eh).set(Journal{{s}});
    }
}

ade::NodeHandle GModel::detail::dataNodeOf(const ConstGraph &g, const GOrigin &origin)
{
    // FIXME: Does it still work with graph transformations, e.g. redirectWriter()??
    return g.metadata().get<Layout>().object_nodes.at(origin);
}

void GModel::redirectReaders(Graph &g, ade::NodeHandle from, ade::NodeHandle to)
{
    std::vector<ade::EdgeHandle> ehh(from->outEdges().begin(), from->outEdges().end());
    for (auto e : ehh)
    {
        auto dst = e->dstNode();
        auto input = g.metadata(e).get<Input>();
        g.erase(e);
        linkIn(g, dst, to, input.port);
    }
}

void GModel::redirectWriter(Graph &g, ade::NodeHandle from, ade::NodeHandle to)
{
    GAPI_Assert(from->inEdges().size() == 1);
    auto e = from->inEdges().front();
    auto op = e->srcNode();
    auto output = g.metadata(e).get<Output>();
    g.erase(e);
    linkOut(g, op, to, output.port);
}

GMetaArgs GModel::collectInputMeta(GModel::ConstGraph cg, ade::NodeHandle node)
{
    GAPI_Assert(cg.metadata(node).get<NodeType>().t == NodeType::OP);
    GMetaArgs in_meta_args(cg.metadata(node).get<Op>().args.size());

    for (const auto &e : node->inEdges())
    {
        const auto& in_data = cg.metadata(e->srcNode()).get<Data>();
        in_meta_args[cg.metadata(e).get<Input>().port] = in_data.meta;
    }

    return in_meta_args;
}


ade::EdgeHandle GModel::getInEdgeByPort(const GModel::ConstGraph& cg,
                                        const ade::NodeHandle&    nh,
                                              std::size_t         in_port)
{
    auto inEdges = nh->inEdges();
    const auto& edge = ade::util::find_if(inEdges, [&](ade::EdgeHandle eh) {
        return cg.metadata(eh).get<Input>().port == in_port;
    });
    GAPI_Assert(edge != inEdges.end());
    return *edge;
}

GMetaArgs GModel::collectOutputMeta(GModel::ConstGraph cg, ade::NodeHandle node)
{
    GAPI_Assert(cg.metadata(node).get<NodeType>().t == NodeType::OP);
    GMetaArgs out_meta_args(cg.metadata(node).get<Op>().outs.size());

    for (const auto &e : node->outEdges())
    {
        const auto& out_data = cg.metadata(e->dstNode()).get<Data>();
        out_meta_args[cg.metadata(e).get<Output>().port] = out_data.meta;
    }

    return out_meta_args;
}

bool GModel::isActive(const GModel::Graph &cg, const cv::gapi::GBackend &backend)
{
    return ade::util::contains(cg.metadata().get<ActiveBackends>().backends,
                               backend);
}

}} // cv::gimpl
