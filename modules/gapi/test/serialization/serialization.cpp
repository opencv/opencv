// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#include "serialization.hpp"

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

    serialization::Op sop{Kernel{op.k.name, op.k.tag}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}};
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

/////////////////////////////////////////////////////////////////////////////////////
//Graph dump operators
template<typename _Tp> static inline
SerializationStream&  operator << (SerializationStream& os, const _Tp &value)
{
    os << value;
    return os;
}

template<typename _Tp> static inline
SerializationStream&  operator << (SerializationStream& os, const std::vector<_Tp> &values)
{
    os << (uint)values.size();
    for (auto & element : values)
    {
        os << element;
    }
    return os;
}


SerializationStream& operator << (SerializationStream& os, const Kernel &k)
{
    CV_LOG_INFO(NULL, "k.name  " << k.name.c_str());
    CV_LOG_INFO(NULL, "k.tag  " << k.tag.c_str());
    os << k.name;
    os << k.tag;
    return os;
}

SerializationStream& operator << (SerializationStream& os, const std::string &str)
{
    uint str_size = (uint)str.size();
    os << str_size;
    size_t numAtoms = str_size % sizeof(uint) ==
        0 ? (str_size / sizeof(uint)) : (str_size / sizeof(uint)) + 1;
    uint* hton_buff = (uint*)malloc(numAtoms * sizeof(uint));
    memcpy(hton_buff, str.c_str(), str_size);
    for (uint a = 0; a < numAtoms; a++)
    {
        os << hton_buff[a];
    }
    free(hton_buff);
    return os;
}

SerializationStream& operator << (SerializationStream& os, const std::vector<int> &ints)
{
    os << (uint)ints.size();
    for (auto & element : ints)
    {
        os << element;
    }
    return os;
}

SerializationStream& operator << (SerializationStream& os, const std::vector<RcDesc> &descs)
{
    os << (uint)descs.size();
    for (auto & element : descs)
    {
        os << element;
    }
    return os;
}

SerializationStream& operator << (SerializationStream& os, const RcDesc &desc)
{
    os << desc.id;
    os << (int)desc.shape;
    return os;
}

SerializationStream& operator << (SerializationStream& os, const std::vector<double> &doubles)
{
    os << (uint)doubles.size();
    for (auto & element : doubles)
    {
        os << element;
    }
    return os;
}

SerializationStream&  operator << (SerializationStream& os, const double &double_val)
{
    uint element_tmp[2];
    memcpy(&element_tmp, &double_val, 2 * sizeof(uint));
    os << element_tmp[0];
    os << element_tmp[1];
    return os;
}

SerializationStream& operator << (SerializationStream& os, const std::vector<cv::Size> &cvsizes)
{
    os << (uint)cvsizes.size();
    for (auto & element : cvsizes)
    {
        os << element;
    }
    return os;
}

SerializationStream& operator << (SerializationStream& os, const cv::Size &cvsize)
{
    os << cvsize.width;
    os << cvsize.height;
    return os;
}

SerializationStream& operator << (SerializationStream& os, const std::vector<bool> &bools)
{
    uint bools_size = (uint)bools.size();
    os << bools_size;
    //std::cout << "bool vector " << bools_size<< std::endl;
    for (uint j = 0; j < bools_size; j++)
    {
        int bool_val_int;
        bool_val_int = bools[j] ? 1 : 0;
        os << bool_val_int;
        //os << bools[j];
    }
    return os;
}

SerializationStream&  operator << (SerializationStream& os, const bool &bool_val)
{
    int bool_val_int;
    bool_val_int = bool_val ? 1 : 0;
    os << bool_val_int;
    return os;
}

SerializationStream& operator << (SerializationStream& os, const std::vector<cv::Scalar> &cvscalars)
{
    os << (uint)cvscalars.size();
    for (auto & element : cvscalars)
    {
       os << element;
    }
    //os << cvscalars;
    return os;
}

SerializationStream& operator << (SerializationStream& os, const cv::Scalar &cvscalar)
{
    uint element_tmp[2];
    for (uint i = 0; i < 4; i++)
    {
        memcpy(&element_tmp, &cvscalar.val[i], 2 * sizeof(uint));
        os << element_tmp[0];
        os << element_tmp[1];
    }
    return os;
}

SerializationStream& operator << (SerializationStream& os, const std::vector<cv::Point> &cvpoints)
{
    os << (uint)cvpoints.size();
    for (auto & element : cvpoints)
    {
        os << element;
    }
    return os;
}

SerializationStream& operator << (SerializationStream& os, const cv::Point &cvpoint)
{
    os << cvpoint.x;
    os << cvpoint.y;
    return os;
}

SerializationStream& operator << (SerializationStream& os, const cv::GMatDesc &cvmatdesc)
{
    os << cvmatdesc.depth;
    os << cvmatdesc.chan;
    os << cvmatdesc.size.width;
    os << cvmatdesc.size.height;
    return os;
}

SerializationStream& operator << (SerializationStream& os, const std::vector<cv::Mat> &cvmats)
{
    os << (uint)cvmats.size();
    for (auto & element : cvmats)
    {
        os << element;
    }
    return os;
}

SerializationStream& operator << (SerializationStream& os, const cv::Mat &cvmat)
{
    size_t matSizeInBytes = cvmat.rows * cvmat.step[0];
    int mat_type = cvmat.type();
    os << cvmat.cols;
    os << cvmat.rows;
    os << mat_type;
    os << (uint)cvmat.step[0];

    if (matSizeInBytes != 0)
    {
        size_t numAtoms = matSizeInBytes % sizeof(uint) ==
            0 ? (matSizeInBytes / sizeof(uint)) : (matSizeInBytes / sizeof(uint)) + 1;
        uint* hton_buff = (uint*)malloc(numAtoms * sizeof(uint));
        memcpy(hton_buff, cvmat.data, matSizeInBytes);
        for (uint a = 0; a < numAtoms; a++)
        {
            os << hton_buff[a];
        }
        free(hton_buff);
    }
    return os;
}

SerializationStream& operator << (SerializationStream& os, const std::vector<cv::Rect> &cvrects)
{

    os << (uint)cvrects.size();
    for (auto & element : cvrects)
    {
        os << element;
    }
    return os;
}

SerializationStream& operator << (SerializationStream& os, const cv::Rect &cvrect)
{
    os << cvrect.x;
    os << cvrect.y;
    os << cvrect.width;
    os << cvrect.height;
    return os;
}

SerializationStream& operator << (SerializationStream& os, const std::vector<Data> &datas)
{
    os << (uint)datas.size();
    for (const auto& data : datas)
    {
        //Data
        os << data;
    }
    return os;
}

SerializationStream& operator << (SerializationStream& os, const Data &data)
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

SerializationStream& operator << (SerializationStream& os, const Op &op)
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

SerializationStream& operator << (SerializationStream& os, const std::vector<Op> &ops)
{
    //dumpAtomS(os, (uint)ops.size());
    os << (uint)ops.size();
    for (const auto& op : ops)
    {
        //Op
        os << op;
    }
    return os;
}

SerializationStream& operator << (SerializationStream& os, const int &atom)
{
    uint atom_htonl = htonl(atom);
    os.putAtom(atom_htonl);
    return os;
}

SerializationStream& operator << (SerializationStream& os, const uint &atom)
{
    uint atom_htonl = htonl(atom);
    os.putAtom(atom_htonl);
    return os;
}

void dumpGSerialized(const GSerialized s, SerializationStream &ofs_serialized)
{
    ofs_serialized << s.m_ops;
    ofs_serialized << s.m_datas;
}

//Graph restore operators
template<typename _Tp> static inline
DeSerializationStream& operator >> (DeSerializationStream& is, /*const*/ _Tp& value)
{
    value >> is;
    return is;
}

DeSerializationStream& operator >> (DeSerializationStream& is, Kernel& k)
{
    is >> k.name;
    is >> k.tag;
    CV_LOG_INFO(NULL, "k.name  " << k.name.c_str());
    CV_LOG_INFO(NULL, "k.tag  " << k.tag.c_str());
    return is;
}

DeSerializationStream& operator >> (DeSerializationStream& is, std::string& str)
{
    uint str_size;
    is >> str_size;
    str.resize(str_size);
    size_t numAtoms = str_size % sizeof(uint) ==
        0 ? (str_size / sizeof(uint)) : (str_size / sizeof(uint)) + 1;
    uint* ntoh_buff = (uint*)malloc(numAtoms * sizeof(uint));
    for (uint a = 0; a < numAtoms; a++)
    {
        is >> ntoh_buff[a];
    }
    memcpy((char*)str.c_str(), ntoh_buff, str_size);
    return is;
}

DeSerializationStream& operator >> (DeSerializationStream& is, std::vector<int>& ints)
{
    uint ints_size;
    is >> ints_size;
    ints.resize(ints_size);
    for (auto & element : ints)
    {
        is >> element;
    }
    return is;
}

DeSerializationStream& operator >> (DeSerializationStream& is, std::vector<RcDesc>& descs)
{
    uint descs_size;
    is >> descs_size;
    descs.resize(descs_size);
    for (auto & element : descs)
    {
        is >> element;
    }
    return is;
}

DeSerializationStream& operator >> (DeSerializationStream& is, RcDesc& desc)
{
    //is.read((char*)&desc,  sizeof(RcDesc));
    uint atom;
    is >> desc.id;
    is >> atom; desc.shape = (cv::GShape)atom;
    return is;
}

DeSerializationStream& operator >> (DeSerializationStream& is, std::vector<double>& doubles)
{
    uint doubles_size;
    is >> doubles_size;
    doubles.resize(doubles_size);
    for (auto & element : doubles)
    {
        is >> element;
    }
    return is;
}

DeSerializationStream& operator >> (DeSerializationStream& is, double& double_val)
{

    uint element_tmp[2];
    is >> element_tmp[0];
    is >> element_tmp[1];
    memcpy(&double_val, &element_tmp, 2 * sizeof(uint));
    return is;
}

DeSerializationStream& operator >> (DeSerializationStream& is, std::vector<cv::Size>& cvsizes)
{
    uint cvsizes_size;
    is >> cvsizes_size;
    cvsizes.resize(cvsizes_size);
    for (auto & element : cvsizes)
    {
        is >> element;
    }
    return is;
}

DeSerializationStream& operator >> (DeSerializationStream& is, cv::Size& cvsize)
{
    is >> cvsize.width;
    is >> cvsize.height;
    return is;
}

DeSerializationStream& operator >> (DeSerializationStream& is, std::vector<bool>& bools)
{
    uint bools_size;
    is >> bools_size;
    bools.resize(bools_size);

    uint bool_val;
    for (uint j = 0; j < bools_size; j++)
    {
        is >> bool_val;
        bools[j] = bool_val == 1 ? true : false;
        //is >> bools[j];
    }
    return is;
}

DeSerializationStream& operator >> (DeSerializationStream& is, bool& bool_val)
{
    uint bool_val_uint;
    is >> bool_val_uint;
    bool_val = bool_val_uint == 1 ? true : false;
    return is;
}

DeSerializationStream& operator >> (DeSerializationStream& is, std::vector<cv::Scalar>& cvscalars)
{
    uint cvscalars_size;
    is >> cvscalars_size;
    cvscalars.resize(cvscalars_size);
    for (auto & element : cvscalars)
    {
        is >> element;
    }
    return is;
}

DeSerializationStream& operator >> (DeSerializationStream& is, cv::Scalar& cvscalar)
{

    uint element_tmp[2];
    for (uint i = 0; i < 4; i++)
    {
        is >> element_tmp[0];
        is >> element_tmp[1];
        memcpy(&cvscalar.val[i], &element_tmp, sizeof(double));
    }
    return is;
}

DeSerializationStream& operator >> (DeSerializationStream& is, std::vector<cv::Point>& cvpoints)
{
    uint cvpoints_size;
    is >> cvpoints_size;
    cvpoints.resize(cvpoints_size);
    for (auto & element : cvpoints)
    {
        is >> element;
    }
    return is;
}

DeSerializationStream& operator >> (DeSerializationStream& is, cv::Point& cvpoint)
{
    is >> cvpoint.x;
    is >> cvpoint.y;
    return is;
}

DeSerializationStream& operator >> (DeSerializationStream& is, cv::GMatDesc& cvmatdesc)
{
    is >> cvmatdesc.depth;
    is >> cvmatdesc.chan;
    is >> cvmatdesc.size.width;
    is >> cvmatdesc.size.height;
    return is;
}

DeSerializationStream& operator >> (DeSerializationStream& is, cv::Mat& cvmat)
{
    int rows, cols, type, step;
    size_t matSizeInBytes;
    is >> cols;
    is >> rows;
    is >> type;
    is >> step;
    matSizeInBytes = rows*step;
    if (matSizeInBytes != 0)
    {
        void *mat_data = malloc(matSizeInBytes);

        size_t numAtoms = matSizeInBytes % sizeof(uint) ==
            0 ? (matSizeInBytes / sizeof(uint)) : (matSizeInBytes / sizeof(uint)) + 1;
        uint* ntoh_buff = (uint*)malloc(numAtoms * sizeof(uint));
        for (uint a = 0; a < numAtoms; a++)
        {
            is >> ntoh_buff[a];
        }
        memcpy(mat_data, ntoh_buff, matSizeInBytes);
        free(ntoh_buff);

        cv::Mat tmp_mat = cv::Mat(rows, cols, type, mat_data, step);
        tmp_mat.copyTo(cvmat);
        free(mat_data);
    }
    return is;
}

DeSerializationStream& operator >> (DeSerializationStream& is, std::vector<cv::Mat>& cvmats)
{
    uint cvmats_size;
    is >> cvmats_size;
    cvmats.resize(cvmats_size);
    for (auto & element : cvmats)
    {
        is >> element;
    }
    return is;
}

DeSerializationStream& operator >> (DeSerializationStream& is, std::vector<cv::Rect>& cvrects)
{
    uint cvrects_size;
    is >> cvrects_size;
    cvrects.resize(cvrects_size);
    for (auto & element : cvrects)
    {
        is >>element;
    }
    return is;
}

DeSerializationStream& operator >> (DeSerializationStream& is, cv::Rect& cvrect)
{
    is >> cvrect.x;
    is >> cvrect.y;
    is >> cvrect.width;
    is >> cvrect.height;
    return is;
}

DeSerializationStream& operator >> (DeSerializationStream& is, std::vector<Data>& datas)
{
    uint datas_size;
    is >> datas_size;
    datas.resize(datas_size);
    for (uint i = 0; i < datas_size; i++)
    {
        //Data
        is >> datas[i];
    }
    return is;
}

DeSerializationStream& operator >> (DeSerializationStream& is, Data& data)
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

DeSerializationStream& operator >> (DeSerializationStream& is, Op& op)
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

DeSerializationStream& operator >> (DeSerializationStream& is, std::vector<Op>& ops)
{
    uint ops_size;// = restoreAtomS(is);
    is >> ops_size;
    ops.resize(ops_size);
    for (uint i = 0; i < ops_size; i++)
    {
        //Op
        is >> ops[i];
    }
    return is;
}

DeSerializationStream& operator >> (DeSerializationStream& is, int& atom)
{
    atom = ntohl(is.getAtom());
    return is;
}

DeSerializationStream& operator >> (DeSerializationStream& is, uint& atom)
{
    atom = ntohl(is.getAtom());
    return is;
}

void readGSerialized(GSerialized &s, DeSerializationStream &serialized_data)
{
    //cleanGSerializedOps(s);
    serialized_data >> s.m_ops;
    //cleanupGSerializedDatas(s);
    serialized_data >> s.m_datas;
}

/////////////////////////////////////////////////////////////////////////////////////

} // namespace serialization
} // namespace gimpl
} // namespace cv
