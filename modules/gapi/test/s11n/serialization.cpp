// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "serialization.hpp"

#include <opencv2/gapi/gtype_traits.hpp>

#include "compiler/passes/passes.hpp" // dump_dot

#include <ade/util/zip_range.hpp>

#ifdef _WIN32
#include <winsock.h>
#else
#include <netinet/in.h>
//#include <arpa/inet.h>
#endif


namespace cv {
namespace gimpl {
namespace s11n {
//namespace {

// Moved here from serialization.hpp
void deserialize(const GSerialized& gs);
void mkDataNode(ade::Graph& g, const Data& data);
void mkOpNode(ade::Graph& g, const Op& op);
std::vector<ade::NodeHandle> linkNodes(ade::Graph& g);
void putData(GSerialized& s, const GModel::ConstGraph& cg, const ade::NodeHandle nh);
void putOp(GSerialized& s, const GModel::ConstGraph& cg, const ade::NodeHandle nh);
void printOp(const Op& op);
void printData(const Data& data);
void printGSerialized(const GSerialized s);
void cleanGSerializedOps(GSerialized &s);
void cleanupGSerializedDatas(GSerialized &s);


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

    s11n::Op sop{Kernel{op.k.name, op.k.tag}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}};
    sop.outs.resize(op.outs.size());

    for(size_t i=0; i < op.args.size(); ++i)
    {
        if(op.args[i].kind == detail::ArgKind::GOBJREF)
        {
            const gimpl::RcDesc &rc = op.args[i].get<gimpl::RcDesc>();
            RcDesc src = {rc.shape, rc.id};
            sop.kind.push_back((int)detail::ArgKind::GOBJREF);
            sop.opaque_kind.push_back((int)detail::OpaqueKind::CV_UNKNOWN);
            sop.ins.push_back(src);
        }
        else if(op.args[i].kind == detail::ArgKind::OPAQUE_VAL)
        {
            sop.kind.push_back((int)detail::ArgKind::OPAQUE_VAL);
            sop.opaque_kind.push_back((int)op.args[i].opaque_kind);
            switch (op.args[i].opaque_kind)
            {
            case detail::OpaqueKind::CV_INT:
                CV_LOG_INFO(NULL, "putOp    int " << op.args[i].get<int>());
                sop.opaque_ints.push_back(op.args[i].get<int>());
                break;
            case detail::OpaqueKind::CV_DOUBLE:
                CV_LOG_INFO(NULL, "putOp    double " << op.args[i].get<double>());
                sop.opaque_doubles.push_back(op.args[i].get<double>());
                break;
            case detail::OpaqueKind::CV_SIZE:
                CV_LOG_INFO(NULL, "putOp    cv::Size " << op.args[i].get<cv::Size>().width << "x"
                    << op.args[i].get<cv::Size>().height);
                sop.opaque_cvsizes.push_back(op.args[i].get<cv::Size>());
                break;
            case detail::OpaqueKind::CV_BOOL:
                CV_LOG_INFO(NULL, "putOp    bool " << op.args[i].get<bool>());
                sop.opaque_bools.push_back(op.args[i].get<bool>());
                break;
            case detail::OpaqueKind::CV_SCALAR:
                CV_LOG_INFO(NULL, "putOp    cv::Scalar " << op.args[i].get<cv::Scalar>()[0] << " "
                    << op.args[i].get<cv::Scalar>()[1] << " "
                    << op.args[i].get<cv::Scalar>()[2] << " "
                    << op.args[i].get<cv::Scalar>()[3] << " ");
                sop.opaque_cvscalars.push_back(op.args[i].get<cv::Scalar>());
                break;
            case detail::OpaqueKind::CV_POINT:
                CV_LOG_INFO(NULL, "putOp    cv::Point " << op.args[i].get<cv::Point>().x << " "
                    << op.args[i].get<cv::Point>().y << " ");
                sop.opaque_cvpoints.push_back(op.args[i].get<cv::Point>());
                break;
            case detail::OpaqueKind::CV_MAT:
                CV_LOG_INFO(NULL, "putOp    cv::Mat " << op.args[i].get<cv::Mat>().rows << " "
                    << op.args[i].get<cv::Mat>().cols << " "
                    << op.args[i].get<cv::Mat>().type() << " ");
                sop.opaque_cvmats.push_back(op.args[i].get<cv::Mat>());
                break;
            case detail::OpaqueKind::CV_RECT:
                CV_LOG_INFO(NULL, "putOp    cv::Rect " << op.args[i].get<cv::Rect>().x << " "
                    << op.args[i].get<cv::Rect>().y << " "
                    << op.args[i].get<cv::Rect>().width << " "
                    << op.args[i].get<cv::Rect>().height << " ");
                sop.opaque_cvrects.push_back(op.args[i].get<cv::Rect>());
                break;
            default:
                CV_LOG_WARNING(NULL, "putOp    OpaqueKind::UNSUPPORTED");
            }
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
    std::cout << "printOp Op" << std::endl;
    std::cout << "printOp  Kernel" << std::endl;
    std::cout << "    " << op.k.name << std::endl;
    if (!op.k.tag.empty())
    {
        std::cout << op.k.tag << std::endl;
    }

    std::cout << "printOp  Args" << std::endl;
    size_t i_int = 0; size_t i_double = 0; size_t i_size = 0; size_t i_bool = 0;
    size_t i_scalar = 0; size_t i_point = 0; size_t i_mat = 0; size_t i_rect = 0;
    size_t i_objref = 0; size_t i_opaque = 0;
    for (const auto& kind : op.kind)
    {
        if(kind == (int)detail::ArgKind::GOBJREF)
        {
            const auto& rc = op.ins[i_objref]; i_objref++;
            std::cout << "printOp    rc.shape " << (int)rc.shape << ", rc.id " << rc.id  << std::endl;
        }
        else if(kind == (int)detail::ArgKind::OPAQUE_VAL)
        {
            switch ((detail::OpaqueKind)op.opaque_kind[i_opaque])
            {
            case detail::OpaqueKind::CV_INT:
                std::cout << "printOp    int " << op.opaque_ints[i_int] << std::endl;
                i_int++;
                break;
            case detail::OpaqueKind::CV_DOUBLE:
                std::cout << "printOp    double " << op.opaque_doubles[i_double] << std::endl;
                i_double++;
                break;
            case detail::OpaqueKind::CV_SIZE:
                std::cout << "printOp    cv::Size " << op.opaque_cvsizes[i_size].width << "x"
                          << op.opaque_cvsizes[i_size].height << std::endl;
                i_size++;
                break;
            case detail::OpaqueKind::CV_BOOL:
                std::cout << "printOp    bool " << op.opaque_bools[i_bool] << std::endl;
                i_bool++;
                break;
            case detail::OpaqueKind::CV_SCALAR:
                std::cout << "printOp    cv::Scalar " << op.opaque_cvscalars[i_scalar][0] << " "
                          << op.opaque_cvscalars[i_scalar][1] << " "
                          << op.opaque_cvscalars[i_scalar][2] << " "
                          << op.opaque_cvscalars[i_scalar][3] << " "
                          << std::endl;
                i_scalar++;
                break;
            case detail::OpaqueKind::CV_POINT:
                std::cout << "printOp    cv::Point " << op.opaque_cvpoints[i_point].x << " "
                          << op.opaque_cvpoints[i_point].y << " "
                          << std::endl;
                i_point++;
                break;
            case detail::OpaqueKind::CV_MAT:
                std::cout << "printOp    cv::Mat " << op.opaque_cvmats[i_mat].rows << " "
                          << op.opaque_cvmats[i_mat].cols << " "
                          << op.opaque_cvmats[i_mat].type() << " "
                          << std::endl;
                i_mat++;
                break;
            case detail::OpaqueKind::CV_RECT:
                std::cout << "printOp     cv::Rect " << op.opaque_cvrects[i_rect].x << " "
                          << op.opaque_cvrects[i_rect].y << " "
                          << op.opaque_cvrects[i_rect].width << " "
                          << op.opaque_cvrects[i_rect].height << " "
                          << std::endl;
                i_rect++;
                break;
            default:
                std::cout << "printOp    OpaqueKind::UNSUPPORTED" << std::endl;
            }
        }
        else
        {
            util::throw_error(std::logic_error("Unexpected ArgKind: expected GOBJREF or OPAQUE"));
        }
        i_opaque++;
    }
    std::cout << "printOp  Outs" << std::endl;
    for (const auto& out : op.outs)
    {
        std::cout << "printOp    rc.shape " << (int)out.shape << ", rc.id " << out.id  << std::endl;
    }
}

void printData(const Data& data)
{
    std::cout << "printData Data" << std::endl;
    std::cout << "printData  rc.shape " << (int)data.rc.shape << ", rc.id = " << data.rc.id << std::endl;
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

void cleanGSerializedOps(GSerialized &s)
{

//    std::cout << "cleanGSerializedOps" <<  std::endl;
    for (uint i = 0; i < s.m_ops.size();  i++)
    {
        //Kernel k
        std::fill(s.m_ops[i].k.name.begin(), s.m_ops[i].k.name.end(), 0);
        s.m_ops[i].k.name.clear();
        std::fill(s.m_ops[i].k.tag.begin(), s.m_ops[i].k.tag.end(), 0);
        s.m_ops[i].k.tag.clear();

        //std::vector<int>   kind;
        std::fill(s.m_ops[i].kind.begin(), s.m_ops[i].kind.end(), 0);
        s.m_ops[i].kind.clear();

        //std::vector<int>   opaque_kind;
        std::fill(s.m_ops[i].opaque_kind.begin(), s.m_ops[i].opaque_kind.end(), 0);
        s.m_ops[i].opaque_kind.clear();

        RcDesc zero_desc; zero_desc.id = 0; zero_desc.shape = GShape::GOPAQUE;
        //std::vector<RcDesc> outs;
        std::fill(s.m_ops[i].outs.begin(), s.m_ops[i].outs.end(), zero_desc);
        s.m_ops[i].outs.clear();

        //std::vector<RcDesc> ins;
        std::fill(s.m_ops[i].ins.begin(), s.m_ops[i].ins.end(), zero_desc);
        s.m_ops[i].ins.clear();

        //opaque args
        //std::vector<int> opaque_ints;
        std::fill(s.m_ops[i].opaque_ints.begin(), s.m_ops[i].opaque_ints.end(), 0);
        s.m_ops[i].opaque_ints.clear();

        //std::vector<double> opaque_doubles;
        std::fill(s.m_ops[i].opaque_doubles.begin(), s.m_ops[i].opaque_doubles.end(), 0.0);
        s.m_ops[i].opaque_doubles.clear();

        //std::vector<cv::Size> opaque_cvsizes;
        std::fill(s.m_ops[i].opaque_cvsizes.begin(), s.m_ops[i].opaque_cvsizes.end(), cv::Size(0,0));
        s.m_ops[i].opaque_cvsizes.clear();

        //std::vector<bool> opaque_bools;
        std::fill(s.m_ops[i].opaque_bools.begin(), s.m_ops[i].opaque_bools.end(), false);
        s.m_ops[i].opaque_bools.clear();

        //std::vector<cv::Scalar> opaque_cvscalars;
        std::fill(s.m_ops[i].opaque_cvscalars.begin(), s.m_ops[i].opaque_cvscalars.end(), cv::Scalar(0,0,0,0));
        s.m_ops[i].opaque_cvscalars.clear();

        //std::vector<cv::Point> opaque_cvpoints;
        std::fill(s.m_ops[i].opaque_cvpoints.begin(), s.m_ops[i].opaque_cvpoints.end(), cv::Point(0,0));
        s.m_ops[i].opaque_cvpoints.clear();

        //std::vector<cv::Mat> opaque_cvmats;
        s.m_ops[i].opaque_cvmats.clear();

        //std::vector<cv::Rect> opaque_cvrects;
        std::fill(s.m_ops[i].opaque_cvrects.begin(), s.m_ops[i].opaque_cvrects.end(), cv::Rect(0, 0, 0, 0));
        s.m_ops[i].opaque_cvrects.clear();
    }
    s.m_ops.clear();
}

void cleanupGSerializedDatas(GSerialized &s)
{
    //std::cout << "cleanupGSerializedDatas" <<  std::endl;
    for (uint i = 0; i < s.m_datas.size(); i++)
    {
        //RcDesc rc;
        s.m_datas[i].rc.id = 0;
        s.m_datas[i].rc.shape = cv::GShape::GMAT;
        //GMetaArg meta;
        s.m_datas[i].meta = util::monostate();
    }
    s.m_datas.clear();
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

    GArgs args(op.kind.size());
    size_t i_int = 0; size_t i_double = 0; size_t i_size = 0; size_t i_bool = 0;
    size_t i_scalar = 0; size_t i_point = 0; size_t i_mat = 0; size_t i_rect = 0;
    size_t i_objref = 0;
    for (size_t i = 0; i < args.size(); i++)
    {
        if(op.kind[i] == (int)detail::ArgKind::GOBJREF)
        {
            auto rc = op.ins[i_objref]; i_objref++;
            args[i] = GArg(gimpl::RcDesc{rc.id, rc.shape, {}});
        }
        else if(op.kind[i] == (int)detail::ArgKind::OPAQUE_VAL)
        {
            switch ((detail::OpaqueKind)op.opaque_kind[i])
            {
            case detail::OpaqueKind::CV_INT:
            {
                auto opaque_int = op.opaque_ints[i_int]; i_int++;
                args[i] = GArg(opaque_int);
                CV_LOG_INFO(NULL, "mkOpNode    int " << args[i].get<int>());
                break;
            }
            case detail::OpaqueKind::CV_DOUBLE:
            {
                auto opaque_double = op.opaque_doubles[i_double]; i_double++;
                args[i] = GArg(opaque_double);
                CV_LOG_INFO(NULL, "mkOpNode    double " << args[i].get<double>());
                break;
            }
            case detail::OpaqueKind::CV_SIZE:
            {
                auto opaque_cvsize = op.opaque_cvsizes[i_size]; i_size++;
                args[i] = GArg(opaque_cvsize);
                CV_LOG_INFO(NULL, "mkOpNode    cv::Size " << args[i].get<cv::Size>().width << "x"
                    << args[i].get<cv::Size>().height);
                break;
            }
            case detail::OpaqueKind::CV_BOOL:
            {
                bool opaque_bool = op.opaque_bools[i_bool]; i_bool++;
                args[i] = GArg(opaque_bool);
                CV_LOG_INFO(NULL, "mkOpNode    bool " << args[i].get<bool>());
                break;
            }
            case detail::OpaqueKind::CV_SCALAR:
            {
                auto opaque_cvscalar = op.opaque_cvscalars[i_scalar]; i_scalar++;
                args[i] = GArg(opaque_cvscalar);
                CV_LOG_INFO(NULL, "mkOpNode    cv::Scalar " << args[i].get<cv::Scalar>()[0] << " "
                    << args[i].get<cv::Scalar>()[1] << " "
                    << args[i].get<cv::Scalar>()[2] << " "
                    << args[i].get<cv::Scalar>()[3] << " ");
                break;
            }
            case detail::OpaqueKind::CV_POINT:
            {
                auto opaque_cvpoint = op.opaque_cvpoints[i_point]; i_point++;
                args[i] = GArg(opaque_cvpoint);
                CV_LOG_INFO(NULL, "mkOpNode    cv::Point " << args[i].get<cv::Point>().x << " "
                              << args[i].get<cv::Point>().y << " ");
                break;
            }
            case detail::OpaqueKind::CV_MAT:
            {
                auto opaque_cvmat = op.opaque_cvmats[i_mat]; i_mat++;
                args[i] = GArg(opaque_cvmat);
                CV_LOG_INFO(NULL, "mkOpNode    cv::Mat " << args[i].get<cv::Mat>().rows << " "
                    << args[i].get<cv::Mat>().cols << " "
                    << args[i].get<cv::Mat>().type() << " ");
                break;
            }
            case detail::OpaqueKind::CV_RECT:
            {
                auto opaque_cvrect = op.opaque_cvrects[i_rect]; i_rect++;
                args[i] = GArg(opaque_cvrect);
                CV_LOG_INFO(NULL, "mkOpNode    cv::Rect " << args[i].get<cv::Rect>().x << " "
                    << args[i].get<cv::Rect>().y << " "
                    << args[i].get<cv::Rect>().width << " "
                    << args[i].get<cv::Rect>().height << " ");
                break;
            }
            default:
                CV_LOG_WARNING(NULL, "mkOpNode    OpaqueKind::UNSUPPORTED");
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

void deserialize(const s11n::GSerialized& s)
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

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//Graph dump operators

// Basic types /////////////////////////////////////////////////////////////////

I::OStream& operator<< (I::OStream& os, bool atom) {
    os.put(atom ? 1 : 0);
    return os;
}
I::IStream& operator>> (I::IStream& is, bool& atom) {
    atom = is.getUInt32() == 0 ? false : true;
    return is;
}

I::OStream& operator<< (I::OStream& os, char atom) {
    os.put(atom);
    return os;
}
I::IStream& operator>> (I::IStream& is, char &atom) {
    atom = static_cast<char>(is.getUInt32());
    return is;
}

I::OStream& operator<< (I::OStream& os, uint32_t atom) {
    os.put(atom);
    return os;
}
I::IStream& operator>> (I::IStream& is, uint32_t &atom) {
    atom = is.getUInt32();
    return is;
}

I::OStream& operator<< (I::OStream& os, int atom) {
    os.put(atom);
    return os;
}
I::IStream& operator>> (I::IStream& is, int& atom) {
    atom = is.getUInt32();
    return is;
}

I::OStream& operator<< (I::OStream& os, std::size_t atom) {
    os.put(static_cast<uint32_t>(atom));  // FIXME: type truncated??
    return os;
}
I::IStream& operator>> (I::IStream& is, std::size_t& atom) {
    atom = is.getUInt32();                // FIXME: type truncated??
    return is;
}

I::OStream& operator<< (I::OStream& os, float atom) {
    uint32_t element_tmp = 0u;
    memcpy(&element_tmp, &atom, sizeof(uint32_t));
    os << element_tmp;
    return os;
}
I::IStream& operator>> (I::IStream& is, float& atom) {
    uint32_t element_tmp = 0u;
    is >> element_tmp;
    memcpy(&atom, &element_tmp, sizeof(uint32_t));
    return is;
}

I::OStream& operator<< (I::OStream& os, double atom) {
    uint32_t element_tmp[2] = {0u};
    memcpy(&element_tmp, &atom, 2 * sizeof(uint32_t));
    os << element_tmp[0];
    os << element_tmp[1];
    return os;
}
I::IStream& operator>> (I::IStream& is, double& atom) {
    uint32_t element_tmp[2] = {0u};
    is >> element_tmp[0];
    is >> element_tmp[1];
    memcpy(&atom, &element_tmp, 2 * sizeof(uint32_t));
    return is;
}


I::OStream& operator<< (I::OStream& os, const std::string &str) {
    os << static_cast<std::size_t>(str.size()); // N.B. Put type explicitly
    for (auto c : str) os << c;
    return os;
}
I::IStream& operator>> (I::IStream& is, std::string& str) {
    std::size_t sz = 0u;
    is >> sz;
    if (sz == 0u) {
        str.clear();
    } else {
        str.resize(sz);
        for (auto &&i : ade::util::iota(sz)) { is >> str[i]; }
    }
    return is;
}

// OpenCV types ////////////////////////////////////////////////////////////////

I::OStream& operator<< (I::OStream& os, const cv::Point &pt) {
    return os << pt.x << pt.y;
}
I::IStream& operator>> (I::IStream& is, cv::Point& pt) {
    return is >> pt.x >> pt.y;
}

I::OStream& operator<< (I::OStream& os, const cv::Size &sz) {
    return os << sz.width << sz.height;
}
I::IStream& operator>> (I::IStream& is, cv::Size& sz) {
    return is >> sz.width >> sz.height;
}

I::OStream& operator<< (I::OStream& os, const cv::Rect &rc) {
    return os << rc.x << rc.y << rc.width << rc.height;
}
I::IStream& operator>> (I::IStream& is, cv::Rect& rc) {
    return is >> rc.x >> rc.y >> rc.width >> rc.height;
}

I::OStream& operator<< (I::OStream& os, const cv::Scalar &s) {
    return os << s.val[0] << s.val[1] << s.val[2] << s.val[3];
}
I::IStream& operator>> (I::IStream& is, cv::Scalar& s) {
    return is >> s.val[0] >> s.val[1] >> s.val[2] >> s.val[3];
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// FIXME: this needs to be reworked
I::OStream& operator<< (I::OStream& os, const cv::Mat &m) {
    size_t matSizeInBytes = m.rows * m.step[0];
    int mat_type = m.type();
    os << m.cols;
    os << m.rows;
    os << mat_type;
    os << (uint)m.step[0];

    if (matSizeInBytes != 0) {
        size_t numAtoms = matSizeInBytes % sizeof(uint) ==
            0 ? (matSizeInBytes / sizeof(uint)) : (matSizeInBytes / sizeof(uint)) + 1;
        uint* hton_buff = (uint*)malloc(numAtoms * sizeof(uint));
        memcpy(hton_buff, m.data, matSizeInBytes);
        for (uint a = 0; a < numAtoms; a++) {
            os << hton_buff[a];
        }
        free(hton_buff);
    }
    return os;
}
I::IStream& operator>> (I::IStream& is, cv::Mat& m) {
    int rows, cols, type, step;
    size_t matSizeInBytes;
    is >> cols;
    is >> rows;
    is >> type;
    is >> step;
    matSizeInBytes = rows*step;
    if (matSizeInBytes != 0) {
        void *mat_data = malloc(matSizeInBytes);

        size_t numAtoms = matSizeInBytes % sizeof(uint) ==
            0 ? (matSizeInBytes / sizeof(uint)) : (matSizeInBytes / sizeof(uint)) + 1;
        uint* ntoh_buff = (uint*)malloc(numAtoms * sizeof(uint));
        for (uint a = 0; a < numAtoms; a++) {
            is >> ntoh_buff[a];
        }
        memcpy(mat_data, ntoh_buff, matSizeInBytes);
        free(ntoh_buff);

        cv::Mat tmp_mat = cv::Mat(rows, cols, type, mat_data, step);
        tmp_mat.copyTo(m);
        free(mat_data);
    }
    return is;
}
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


// G-API types /////////////////////////////////////////////////////////////////

// Stubs (empty types)

I::OStream& operator<< (I::OStream& os, cv::util::monostate  ) {return os;}
I::IStream& operator>> (I::IStream& is, cv::util::monostate &) {return is;}

I::OStream& operator<< (I::OStream& os, const cv::GScalarDesc &) {return os;}
I::IStream& operator>> (I::IStream& is,       cv::GScalarDesc &) {return is;}

I::OStream& operator<< (I::OStream& os, const cv::GOpaqueDesc &) {return os;}
I::IStream& operator>> (I::IStream& is,       cv::GOpaqueDesc &) {return is;}

I::OStream& operator<< (I::OStream& os, const cv::GArrayDesc &) {return os;}
I::IStream& operator>> (I::IStream& is,       cv::GArrayDesc &) {return is;}

// Enums and structures

namespace {
template<typename E> I::OStream& put_enum(I::OStream& os, E e) {
    return os << static_cast<int>(e);
}
template<typename E> I::IStream& get_enum(I::IStream& is, E &e) {
    int x{}; is >> x; e = static_cast<E>(x);
    return is;
}
} // anonymous namespace

I::OStream& operator<< (I::OStream& os, cv::GShape  sh) {
    return put_enum(os, sh);
}
I::IStream& operator>> (I::IStream& is, cv::GShape &sh) {
    return get_enum<cv::GShape>(is, sh);
}
I::OStream& operator<< (I::OStream& os, cv::detail::ArgKind  k) {
    return put_enum(os, k);
}
I::IStream& operator>> (I::IStream& is, cv::detail::ArgKind &k) {
    return get_enum<cv::detail::ArgKind>(is, k);
}
I::OStream& operator<< (I::OStream& os, cv::detail::OpaqueKind  k) {
    return put_enum(os, k);
}
I::IStream& operator>> (I::IStream& is, cv::detail::OpaqueKind &k) {
    return get_enum<cv::detail::OpaqueKind>(is, k);
}
I::OStream& operator<< (I::OStream& os, cv::gimpl::Data::Storage s) {
    return put_enum(os, s);
}
I::IStream& operator>> (I::IStream& is, cv::gimpl::Data::Storage &s) {
    return get_enum<cv::gimpl::Data::Storage>(is, s);
}


I::OStream& operator<< (I::OStream& os, const cv::GArg &arg) {
    // Only GOBJREF and OPAQUE_VAL kinds can be serialized/deserialized
    GAPI_Assert(   arg.kind == cv::detail::ArgKind::OPAQUE_VAL
                || arg.kind == cv::detail::ArgKind::GOBJREF);
    GAPI_Assert(arg.opaque_kind != cv::detail::OpaqueKind::CV_UNKNOWN);

    os << arg.kind << arg.opaque_kind;
    if (arg.kind == cv::detail::ArgKind::GOBJREF) {
        os << arg.get<cv::gimpl::RcDesc>();
    } else {
        GAPI_Assert(arg.kind == cv::detail::ArgKind::OPAQUE_VAL);
        switch (arg.opaque_kind) {
        case cv::detail::OpaqueKind::CV_BOOL:   os << arg.get<bool>();       break;
        case cv::detail::OpaqueKind::CV_INT:    os << arg.get<int>();        break;
        case cv::detail::OpaqueKind::CV_DOUBLE: os << arg.get<double>();     break;
        case cv::detail::OpaqueKind::CV_POINT:  os << arg.get<cv::Point>();  break;
        case cv::detail::OpaqueKind::CV_SIZE:   os << arg.get<cv::Size>();   break;
        case cv::detail::OpaqueKind::CV_RECT:   os << arg.get<cv::Rect>();   break;
        case cv::detail::OpaqueKind::CV_SCALAR: os << arg.get<cv::Scalar>(); break;
        case cv::detail::OpaqueKind::CV_MAT:    os << arg.get<cv::Mat>();    break;
        default: GAPI_Assert(false && "GArg: Unsupported (unknown?) opaque value type");
        }
    }
    return os;
}
I::IStream& operator>> (I::IStream& is, cv::GArg &arg) {
    is >> arg.kind >> arg.opaque_kind;

    // Only GOBJREF and OPAQUE_VAL kinds can be serialized/deserialized
    GAPI_Assert(   arg.kind == cv::detail::ArgKind::OPAQUE_VAL
                || arg.kind == cv::detail::ArgKind::GOBJREF);
    GAPI_Assert(arg.opaque_kind != cv::detail::OpaqueKind::CV_UNKNOWN);

    if (arg.kind == cv::detail::ArgKind::GOBJREF) {
        cv::gimpl::RcDesc rc;
        is >> rc;
        arg = std::move(GArg(rc));
    } else {
        GAPI_Assert(arg.kind == cv::detail::ArgKind::OPAQUE_VAL);
        switch (arg.opaque_kind) {
#define HANDLE_CASE(E,T) case cv::detail::OpaqueKind::CV_##E:           \
            { T t{}; is >> t; arg = std::move(cv::GArg(t)); } break
            HANDLE_CASE(BOOL   , bool);
            HANDLE_CASE(INT    , int);
            HANDLE_CASE(DOUBLE , double);
            HANDLE_CASE(POINT  , cv::Point);
            HANDLE_CASE(SIZE   , cv::Size);
            HANDLE_CASE(RECT   , cv::Rect);
            HANDLE_CASE(SCALAR , cv::Scalar);
            HANDLE_CASE(MAT    , cv::Mat);
#undef HANDLE_CASE
        default: GAPI_Assert(false && "GArg: Unsupported (unknown?) opaque value type");
        }
    }
    return is;
}


I::OStream& operator<< (I::OStream& os, const cv::GKernel &k) {
    return os << k.name << k.tag << k.outShapes;
}
I::IStream& operator>> (I::IStream& is, cv::GKernel &k) {
    return is >> const_cast<std::string&>(k.name)
              >> const_cast<std::string&>(k.tag)
              >> const_cast<cv::GShapes&>(k.outShapes);
}


I::OStream& operator<< (I::OStream& os, const cv::GMatDesc &d) {
    return os << d.depth << d.chan << d.size << d.planar << d.dims;
}
I::IStream& operator>> (I::IStream& is, cv::GMatDesc &d) {
    return is >> d.depth >> d.chan >> d.size >> d.planar >> d.dims;
}


I::OStream& operator<< (I::OStream& os, const cv::gimpl::RcDesc &rc) {
    // FIXME: HostCtor is not serialized!
    return os << rc.id << rc.shape;
}
I::IStream& operator>> (I::IStream& is, cv::gimpl::RcDesc &rc) {
    // FIXME: HostCtor is not deserialized!
    return is >> rc.id >> rc.shape;
}


I::OStream& operator<< (I::OStream& os, const cv::gimpl::Op &op) {
    return os << op.k << op.args << op.outs;
}
I::IStream& operator>> (I::IStream& is, cv::gimpl::Op &op) {
    return is >> op.k >> op.args >> op.outs;
}


I::OStream& operator<< (I::OStream& os, const cv::gimpl::Data &d) {
    // FIXME: HostCtor is not stored here!!
    // FIXME: Storage may be incorrect for subgraph-to-graph process
    return os << d.shape << d.rc << d.meta << d.storage;
}
I::IStream& operator>> (I::IStream& is, cv::gimpl::Data &d) {
    // FIXME: HostCtor is not stored here!!
    // FIXME: Storage may be incorrect for subgraph-to-graph process
    return is >> d.shape >> d.rc >> d.meta >> d.storage;
}


// Legacy //////////////////////////////////////////////////////////////////////

I::OStream& operator<< (I::OStream& os, const Kernel &k)
{
    CV_LOG_INFO(NULL, "k.name  " << k.name.c_str());
    CV_LOG_INFO(NULL, "k.tag  " << k.tag.c_str());
    os << k.name;
    os << k.tag;
    return os;
}


I::OStream& operator<< (I::OStream& os, const RcDesc &desc)
{
    os << desc.id;
    os << (int)desc.shape;
    return os;
}




I::OStream& operator<< (I::OStream& os, const Data &data)
{
    os << data.rc;

    //GMetaArg meta;
    switch (data.rc.shape)
    {
    case cv::GShape::GMAT:
    {
        GMatDesc mat_desc = util::get<GMatDesc>(data.meta);
        os << mat_desc;

        int bool_val;
        bool_val = mat_desc.planar ? 1 : 0;
        os << bool_val;

        uint dims_size = (uint)mat_desc.dims.size();
        os << dims_size;
        for (uint j = 0; j < dims_size; j++)
        {
            os << (uint)mat_desc.dims.data()[j];
        }
        break;
    }
    case cv::GShape::GSCALAR:
    {
        //std::cout << "dumpGSerializedDatas  GSCALAR " << std::endl;
        //GScalarDesc scalar_desc = util::get<GScalarDesc>(data.meta);
        break;
    }
    case cv::GShape::GARRAY:
    {   //std::cout << "dumpGSerializedDatas  GARRAY " << std::endl;
        //GArrayDesc array_desc = util::get<GArrayDesc>(data.meta);
        break;
    }
    case cv::GShape::GOPAQUE:
    {
        //std::cout << "dumpGSerializedDatas  GOPAQUE " << std::endl;
        //GOpaqueDesc opaque_desc = util::get<GOpaqueDesc>(data.meta);
        break;
    }
    default:
        //std::cout << "dumpGSerializedDatas  unsupported" << std::endl;
        break;
    }

    return os;
}

I::OStream& operator<< (I::OStream& os, const Op &op)
{
    //Kernel k
    os << op.k;

    //std::vector<int>   kind;
    os << op.kind;

    //std::vector<int>   opaque_kind;
    os << op.opaque_kind;

    //std::vector<RcDesc> outs;
    os << op.outs;

    //std::vector<RcDesc> ins;
    os << op.ins;

    //opaque args
    //std::vector<int> opaque_ints;
    os << op.opaque_ints;

    //std::vector<double> opaque_doubles;
    os << op.opaque_doubles;

    //std::vector<cv::Size> opaque_cvsizes;
    os << op.opaque_cvsizes;

    //std::vector<bool> opaque_bools;
    os << op.opaque_bools;

    //std::vector<cv::Scalar> opaque_cvscalars;
    os << op.opaque_cvscalars;

    //std::vector<cv::Point> opaque_cvpoints;
    os << op.opaque_cvpoints;

    //std::vector<cv::Mat> opaque_cvmats;
    os << op.opaque_cvmats;

    //std::vector<cv::Rect> opaque_cvrects;
    os << op.opaque_cvrects;

    return os;
}

void dumpGSerialized(const GSerialized s, I::OStream &ofs_serialized)
{
    ofs_serialized << s.m_ops;
    ofs_serialized << s.m_datas;
}

//Graph restore operators
I::IStream& operator>> (I::IStream& is, Kernel& k)
{
    is >> k.name;
    is >> k.tag;
    CV_LOG_INFO(NULL, "k.name  " << k.name.c_str());
    CV_LOG_INFO(NULL, "k.tag  " << k.tag.c_str());
    return is;
}



I::IStream& operator>> (I::IStream& is, RcDesc& desc)
{
    //is.read((char*)&desc,  sizeof(RcDesc));
    uint atom;
    is >> desc.id;
    is >> atom; desc.shape = (cv::GShape)atom;
    return is;
}

I::IStream& operator>> (I::IStream& is, Data& data)
{
    is >> data.rc;

    //GMetaArg meta;
    switch (data.rc.shape)
    {
    case cv::GShape::GMAT:
    {
        GMatDesc mat_desc;
        is >> mat_desc;

        int bool_val;
        is >> bool_val;
        mat_desc.planar = bool_val == 1 ? true : false;

        uint dims_size;
        is >> dims_size;
        std::vector<int> dims(dims_size);
        for (uint j = 0; j < dims_size; j++)
        {
            is >> dims.data()[j];
        }

        mat_desc.dims = dims;
        data.meta = GMatDesc(mat_desc.depth, mat_desc.chan, mat_desc.size, mat_desc.planar);
        break;
    }
    case cv::GShape::GSCALAR:
    {
        data.meta = GScalarDesc();
        break;
    }
    case cv::GShape::GARRAY:
    {
        data.meta = GArrayDesc();
        break;
    }
    case cv::GShape::GOPAQUE:
    {
        data.meta = GOpaqueDesc();
        break;
    }
    default:
        //std::cout << "dumpGSerializedDatas  unsupported" << std::endl;
        break;
    }

    return is;
}

I::IStream& operator>> (I::IStream& is, Op& op)
{
    //Kernel k
    is >> op.k;

    //std::vector<int>   kind;
    is >> op.kind;

    //std::vector<int>   opaque_kind;
    is >> op.opaque_kind;

    //std::vector<RcDesc> outs;
    is >> op.outs;

    //std::vector<RcDesc> ins;
    is >> op.ins;

    //opaque args
    //std::vector<int> opaque_ints;
    is >> op.opaque_ints;

    //std::vector<double> opaque_doubles;
    is >> op.opaque_doubles;

    //std::vector<cv::Size> opaque_cvsizes;
    is >> op.opaque_cvsizes;

    //std::vector<bool> opaque_bools;
    is >> op.opaque_bools;

    //std::vector<cv::Scalar> opaque_cvscalars;
    is >> op.opaque_cvscalars;

    //std::vector<cv::Point> opaque_cvpoints;
    is >> op.opaque_cvpoints;

    //std::vector<cv::Mat> opaque_cvmats;
    is >> op.opaque_cvmats;

    //std::vector<cv::Rect> opaque_cvrects;
    is >> op.opaque_cvrects;

    return is;
}

void readGSerialized(GSerialized &s, I::IStream &serialized_data)
{
    //cleanGSerializedOps(s);
    serialized_data >> s.m_ops;
    //cleanupGSerializedDatas(s);
    serialized_data >> s.m_datas;
}

std::vector<ade::NodeHandle> reconstructGModel(ade::Graph &g, const GSerialized &s)
{
    for (const auto& data : s.m_datas)
    {
        cv::gimpl::s11n::mkDataNode(g, data);
    }

    for (const auto& op : s.m_ops)
    {
        cv::gimpl::s11n::mkOpNode(g, op);
    }

    // FIXME: ???
    std::vector<ade::NodeHandle> nh  = cv::gimpl::s11n::linkNodes(g);
    CV_LOG_INFO(NULL, "nh Size " << nh.size());
    return nh;
}

/////////////////////////////////////////////////////////////////////////////////////

char* SerializationStream::getData() {
    return (char*)m_dump_storage.data();
}

size_t SerializationStream::getSize() {
    return (size_t)(m_dump_storage.size()*sizeof(uint));
};

void SerializationStream::putAtom(uint new_atom) {
    m_dump_storage.push_back(new_atom);
};

void SerializationStream::put(uint32_t v) {
    putAtom(htonl(v));
}

DeSerializationStream::DeSerializationStream(char* data, size_t sz) {
    uint* uint_data = (uint*)data;
    size_t uint_size = sz / sizeof(uint);
    for (size_t i = 0; i < uint_size; i++) {
        m_dump_storage.push_back(uint_data[i]);
    }
}

char* DeSerializationStream::getData() {
    return (char*)m_dump_storage.data();
}

size_t DeSerializationStream::getSize() {
    return (size_t)(m_dump_storage.size() * sizeof(uint));
}

void DeSerializationStream::putAtom(uint& new_atom) {
    m_dump_storage.push_back(new_atom);
}

uint DeSerializationStream::getAtom() {
    uint next_atom = m_dump_storage.data()[m_storage_index++];
    return next_atom;
};

uint32_t DeSerializationStream::getUInt32() {
    return ntohl(getAtom());
}

} // namespace s11n
} // namespace gimpl
} // namespace cv
