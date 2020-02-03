// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#include <backends/cpu/serialization.hpp>

#include <opencv2/gapi/gtype_traits.hpp>

#include "compiler/passes/passes.hpp" // dump_dot

#include <ade/util/zip_range.hpp>

namespace cv {
namespace gimpl {
namespace serialization {
//namespace {

// FIXME? make a method of GSerialized?
void putData(GSerialized& s, const GModel::ConstGraph& cg, const ade::NodeHandle nh)
{
    const auto gdata = cg.metadata(nh).get<gimpl::Data>();
    Data d{RcDesc{gdata.shape, gdata.rc}, gdata.meta};

    auto dataInSerialized = ade::util::find(s.m_datas, d);
    // FIXME:
    // put meta check here
    if (s.m_datas.end() == dataInSerialized)
    {
        s.m_datas.push_back(d);
    }
}

void putOp(GSerialized& s, const GModel::ConstGraph& cg, const ade::NodeHandle nh)
{
    const auto& op = cg.metadata(nh).get<gimpl::Op>();

    serialization::Op sop{Kernel{op.k.name, op.k.tag}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}};
    sop.args.resize(op.args.size());
    sop.outs.resize(op.outs.size());

    for(size_t i=0; i < op.args.size(); ++i)
    {
        if(op.args[i].kind == detail::ArgKind::GOBJREF)
        {
            const gimpl::RcDesc &rc = op.args[i].get<gimpl::RcDesc>();
            RcDesc src = {rc.shape, rc.id};
            sop.args[i] = GArg(src);
        }
        else if(op.args[i].kind == detail::ArgKind::OPAQUE)
        {
            switch (op.args[i].opaque_kind)
            {
            case detail::OpaqueKind::INT:
                std::cout << "putOp    int " << op.args[i].get<int>() << std::endl;
                sop.opaque_ints.push_back(op.args[i].get<int>());
                break;
            case detail::OpaqueKind::DOUBLE:
                std::cout << "putOp    double " << op.args[i].get<double>() << std::endl;
                sop.opaque_doubles.push_back(op.args[i].get<double>());
                break;
            case detail::OpaqueKind::CV_SIZE:
                std::cout << "putOp    cv::Size " << op.args[i].get<cv::Size>().width << "x" << op.args[i].get<cv::Size>().height << std::endl;
                sop.opaque_cvsizes.push_back(op.args[i].get<cv::Size>());
                break;
            case detail::OpaqueKind::BOOL:
                std::cout << "putOp    bool " << op.args[i].get<bool>() << std::endl;
                sop.opaque_bools.push_back(op.args[i].get<bool>());
                break;
            case detail::OpaqueKind::CV_SCALAR:
                std::cout << "putOp    cv::Scalar " << op.args[i].get<cv::Scalar>()[0] << " "
                          << op.args[i].get<cv::Scalar>()[1] << " "
                          << op.args[i].get<cv::Scalar>()[2] << " "
                          << op.args[i].get<cv::Scalar>()[3] << " "
                          << std::endl;
                sop.opaque_cvscalars.push_back(op.args[i].get<cv::Scalar>());
                break;
            case detail::OpaqueKind::CV_POINT:
                std::cout << "putOp    cv::Point " << op.args[i].get<cv::Point>().x << " "
                          << op.args[i].get<cv::Point>().y << " "
                          << std::endl;
                sop.opaque_cvpoints.push_back(op.args[i].get<cv::Point>());
                break;
            case detail::OpaqueKind::CV_MAT:
                std::cout << "putOp    cv::Mat " << op.args[i].get<cv::Mat>().rows << " "
                          << op.args[i].get<cv::Mat>().cols << " "
                          << op.args[i].get<cv::Mat>().type() << " "
                          << std::endl;
                sop.opaque_cvmats.push_back(op.args[i].get<cv::Mat>());
                break;
            case detail::OpaqueKind::CV_RECT:
                std::cout << "putOp    cv::Rect " << op.args[i].get<cv::Rect>().x << " "
                          << op.args[i].get<cv::Rect>().y << " "
                          << op.args[i].get<cv::Rect>().width << " "
                          << op.args[i].get<cv::Rect>().height << " "
                          << std::endl;
                sop.opaque_cvrects.push_back(op.args[i].get<cv::Rect>());
                break;               break;

            default:
                std::cout << "putOp    OpaqueKind::UNSUPPORTED" << std::endl;
            }
            sop.args[i].opaque_kind = op.args[i].opaque_kind;
            sop.args[i] = op.args[i];
        }
        else
        {
            util::throw_error(std::logic_error("Unexpected ArgKind: expected GOBJREF or OPAQUE"));
        }
    }

    for (size_t i = 0; i < op.outs.size(); i++)
    {
        const auto rc = op.outs[i];
        RcDesc src{rc.shape, rc.id};
        sop.outs[i] = src;
    }

    for (const auto &in_nh : nh->inNodes())
    {
        putData(s, cg, in_nh);
    }

    for (const auto &out_nh : nh->outNodes())
    {
        putData(s, cg, out_nh);
    }

    s.m_ops.push_back(sop);
}

void printOp(const Op& op)
{
    std::cout << "Op" << std::endl;
    std::cout << "  Kernel" << std::endl;
    std::cout << "    " << op.k.name << std::endl;
    if (!op.k.tag.empty())
    {
        std::cout << op.k.tag << std::endl;
    }

    std::cout << "  Args" << std::endl;
    for (const auto& arg : op.args)
    {
        if(arg.kind == detail::ArgKind::GOBJREF)
        {
            const auto& rc = arg.get<RcDesc>();
            std::cout << "    rc.shape " << (int)rc.shape << ", rc.id " << rc.id  << std::endl;
        }
        else if(arg.kind == detail::ArgKind::OPAQUE)
        {
            switch (arg.opaque_kind)
            {
            case detail::OpaqueKind::INT:
                std::cout << "    int " << arg.get<int>() << std::endl;
                break;
            case detail::OpaqueKind::DOUBLE:
                std::cout << "    double " << arg.get<double>() << std::endl;
                break;
            case detail::OpaqueKind::CV_SIZE:
                std::cout << "    cv::Size " << arg.get<cv::Size>().width << "x" << arg.get<cv::Size>().height << std::endl;
                break;
            case detail::OpaqueKind::BOOL:
                std::cout << "    bool " << arg.get<bool>() << std::endl;
                break;
            case detail::OpaqueKind::CV_SCALAR:
                std::cout << "    cv::Scalar " << arg.get<cv::Scalar>()[0] << " "
                          << arg.get<cv::Scalar>()[1] << " "
                          << arg.get<cv::Scalar>()[2] << " "
                          << arg.get<cv::Scalar>()[3] << " "
                          << std::endl;
                break;
            case detail::OpaqueKind::CV_POINT:
                std::cout << "    cv::Point " << arg.get<cv::Point>().x << " "
                          << arg.get<cv::Point>().y << " "
                          << std::endl;
                break;
            case detail::OpaqueKind::CV_MAT:
                std::cout << "    cv::Mat " << arg.get<cv::Mat>().rows << " "
                          << arg.get<cv::Mat>().cols << " "
                          << arg.get<cv::Mat>().type() << " "
                          << std::endl;
                break;
            case detail::OpaqueKind::CV_RECT:
                std::cout << "     cv::Rect " << arg.get<cv::Rect>().x << " "
                          << arg.get<cv::Rect>().y << " "
                          << arg.get<cv::Rect>().width << " "
                          << arg.get<cv::Rect>().height << " "
                          << std::endl;
                break;
            default:
                std::cout << "    OpaqueKind::UNSUPPORTED" << std::endl;
            }
        }
        else
        {
            util::throw_error(std::logic_error("Unexpected ArgKind: expected GOBJREF or OPAQUE"));
        }
    }
    std::cout << "  Outs" << std::endl;
    for (const auto& out : op.outs)
    {
        std::cout << "    rc.shape " << (int)out.shape << ", rc.id " << out.id  << std::endl;
    }
}

void printData(const Data& data)
{
    std::cout << "Data" << std::endl;
    std::cout << "  rc.shape " << (int)data.rc.shape << ", rc.id = " << data.rc.id << std::endl;
    std::cout << "  " << data.meta << std::endl;
}

void printGSerialized(const GSerialized s)
{
    for (const auto& op : s.m_ops)
    {
        printOp(op);
    }

    for (const auto& data : s.m_datas)
    {
        printData(data);
    }
}

void mkDataNode(ade::Graph& g, const Data& data)
{
    auto nh = g.createNode();
    GModel::Graph gm(g);

    gm.metadata(nh).set(NodeType{NodeType::DATA});

    HostCtor ctor{};
    // internal?
    auto storage = gimpl::Data::Storage::INTERNAL;
    gm.metadata(nh).set(gimpl::Data{data.rc.shape, data.rc.id, data.meta, ctor, storage});
}

void mkOpNode(ade::Graph& g, const Op& op)
{
    auto nh = g.createNode();
    GModel::Graph gm(g);

    gm.metadata(nh).set(NodeType{NodeType::OP});

    std::vector<gimpl::RcDesc> outs(op.outs.size());
    for (size_t i = 0; i < outs.size(); i++)
    {
        outs[i] = gimpl::RcDesc{op.outs[i].id, op.outs[i].shape, {}};
    }

    GArgs args(op.args.size());
    size_t i_int = 0;
    size_t i_double = 0;
    size_t i_size = 0;
    size_t i_bool = 0;
    size_t i_scalar = 0;
    size_t i_point = 0;
    size_t i_mat = 0;
    size_t i_rect = 0;
    for (size_t i = 0; i < args.size(); i++)
    {
        if(op.args[i].kind == detail::ArgKind::GOBJREF)
        {
            const auto rc = op.args[i].get<serialization::RcDesc>();
            args[i] = GArg(gimpl::RcDesc{rc.id, rc.shape, {}});
        }
        else if(op.args[i].kind == detail::ArgKind::OPAQUE)
        {
            switch (op.args[i].opaque_kind)
            {
            case detail::OpaqueKind::INT:
            {
                auto opaque_int = op.opaque_ints[i_int]; i_int++;
                args[i] = GArg(opaque_int);
                std::cout << "mkOpNode    int " << args[i].get<int>() << std::endl;
                break;
            }
            case detail::OpaqueKind::DOUBLE:
            {
                auto opaque_double = op.opaque_doubles[i_double]; i_double++;
                args[i] = GArg(opaque_double);
                std::cout << "mkOpNode    double " << args[i].get<double>() << std::endl;
                break;
            }
            case detail::OpaqueKind::CV_SIZE:
            {
                auto opaque_cvsize = op.opaque_cvsizes[i_size]; i_size++;
                args[i] = GArg(opaque_cvsize);
                std::cout << "mkOpNode    cv::Size " << args[i].get<cv::Size>().width << "x" << args[i].get<cv::Size>().height << std::endl;
                break;
            }
            case detail::OpaqueKind::BOOL:
            {
                auto opaque_bool = op.opaque_bools[i_bool]; i_bool++;
                args[i] = GArg(opaque_bool);
                std::cout << "mkOpNode    bool " << args[i].get<bool>() << std::endl;
                break;
            }
            case detail::OpaqueKind::CV_SCALAR:
            {
                auto opaque_cvscalar = op.opaque_cvscalars[i_scalar]; i_scalar++;
                args[i] = GArg(opaque_cvscalar);
                std::cout << "mkOpNode    cv::Scalar " << args[i].get<cv::Scalar>()[0] << " "
                          << args[i].get<cv::Scalar>()[1] << " "
                          << args[i].get<cv::Scalar>()[2] << " "
                          << args[i].get<cv::Scalar>()[3] << " "
                          << std::endl;
                break;
            }
            case detail::OpaqueKind::CV_POINT:
            {
                auto opaque_cvpoint = op.opaque_cvpoints[i_point]; i_point++;
                args[i] = GArg(opaque_cvpoint);
                std::cout << "mkOpNode    cv::Point " << args[i].get<cv::Point>().x << " "
                          << args[i].get<cv::Point>().y << " "
                          << std::endl;
                break;
            }
            case detail::OpaqueKind::CV_MAT:
            {
                auto opaque_cvmat = op.opaque_cvmats[i_mat]; i_mat++;
                args[i] = GArg(opaque_cvmat);
                std::cout << "mkOpNode    cv::Mat " << args[i].get<cv::Mat>().rows << " "
                          << args[i].get<cv::Mat>().cols << " "
                          << args[i].get<cv::Mat>().type() << " "
                          << std::endl;
                break;
            }
            case detail::OpaqueKind::CV_RECT:
            {
                auto opaque_cvrect = op.opaque_cvrects[i_rect]; i_rect++;
                args[i] = GArg(opaque_cvrect);
                std::cout << "mkOpNode     cv::Rect " << args[i].get<cv::Rect>().x << " "
                          << args[i].get<cv::Rect>().y << " "
                          << args[i].get<cv::Rect>().width << " "
                          << args[i].get<cv::Rect>().height << " "
                          << std::endl;
                break;
            }
            default:
                std::cout << "    OpaqueKind::UNSUPPORTED" << std::endl;
            }
        }
        else
        {
            util::throw_error(std::logic_error("Unexpected ArgKind: expected GOBJREF or OPAQUE"));
        }
    }

    //auto cpu_impl = cv::util::any_cast<cv::GCPUKernel>(op.k.Kernel.impl.opaque);
    //gm.metadata(nh).set(cv::gimpl::Unit{cpu_impl});
    gm.metadata(nh).set(gimpl::Op{cv::GKernel{op.k.name, op.k.tag, {},{}}, std::move(args), std::move(outs), {}});
}

std::vector<ade::NodeHandle> linkNodes(ade::Graph& g)
{
    GModel::Graph gm(g);
    GModel::ConstGraph cgm(g);

    using nodeMap = std::unordered_map<int, ade::NodeHandle>;
    std::unordered_map<cv::GShape, nodeMap> dataNodes;
    std::vector<ade::NodeHandle> nodes;

    for (const auto& nh : g.nodes())
    {
        if (cgm.metadata(nh).get<NodeType>().t == NodeType::DATA)
        {
            auto d = cgm.metadata(nh).get<gimpl::Data>();
            dataNodes[d.shape][d.rc] = nh;
        }
        nodes.push_back(nh);
    }

    for (const auto& nh : g.nodes())
    {
        if (cgm.metadata(nh).get<NodeType>().t == NodeType::OP)
        {
            const auto& op = cgm.metadata(nh).get<gimpl::Op>();

            for (const auto& in : ade::util::indexed(op.args))
            {
                const auto& arg = ade::util::value(in);
                if (arg.kind == detail::ArgKind::GOBJREF)
                {
                    const auto idx = ade::util::index(in);
                    const auto rc = arg.get<gimpl::RcDesc>();

                    const auto& inDataNode = dataNodes[rc.shape][rc.id];
                    const auto& e = g.link(inDataNode, nh);
                    gm.metadata(e).set(Input{idx});
                }
            }

            for (const auto& out : ade::util::indexed(op.outs))
            {
                const auto idx = ade::util::index(out);
                const auto rc = ade::util::value(out);

                const auto& outDataNode = dataNodes[rc.shape][rc.id];
                const auto& e = g.link(nh, outDataNode);
                gm.metadata(e).set(Output{idx});
            }
        }
    }
    return nodes;
}
//} // anonymous namespace

GSerialized serialize(const gimpl::GModel::ConstGraph& cg, const std::vector<ade::NodeHandle>& nodes)
{
    GSerialized s;
    for (auto &nh : nodes)
    {
        switch (cg.metadata(nh).get<NodeType>().t)
        {
        case NodeType::OP:   putOp  (s, cg, nh); break;
        case NodeType::DATA: putData(s, cg, nh); break;
        default: util::throw_error(std::logic_error("Unknown NodeType"));
        }
    }
    return s;
}

void deserialize(const serialization::GSerialized& s)
{
    printGSerialized(s);

    // FIXME: reuse code from GModelBuilder/GModel!
    // ObjectCounter?? (But seems we need existing mapping by shape+id)

    ade::Graph g;

    for (const auto& data : s.m_datas)
    {
        mkDataNode(g, data);
    }

    for (const auto& op : s.m_ops)
    {
        mkOpNode(g, op);
    }

    linkNodes(g);

//  FIXME:
//  Handle IslandModel!
//    std::shared_ptr<ade::Graph> ig;
//    GModel::Graph gm(g);
//    gm.metadata().set(gimpl::IslandModel{std::move(ig)});

    auto pass_ctx = ade::passes::PassContext{g};
    ade::passes::TopologicalSort{}(pass_ctx);
    gimpl::passes::dumpDotToFile(pass_ctx, "graph.dot");
}

} // namespace serialization
} // namespace gimpl
} // namespace cv

