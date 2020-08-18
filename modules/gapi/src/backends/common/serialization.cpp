// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include <set> // set
#include <map> // map
#include <ade/util/zip_range.hpp> // indexed

#define NOMINMAX

#ifdef _WIN32
#include <winsock.h>      // htonl, ntohl
#else
#include <netinet/in.h>   // htonl, ntohl
#endif

#include <opencv2/gapi/gtype_traits.hpp>

#include "backends/common/serialization.hpp"

namespace cv {
namespace gimpl {
namespace s11n {
namespace {

void putData(GSerialized& s, const GModel::ConstGraph& cg, const ade::NodeHandle &nh) {
    const auto gdata = cg.metadata(nh).get<gimpl::Data>();
    const auto it = ade::util::find_if(s.m_datas, [&gdata](const cv::gimpl::Data &cd) {
            return cd.rc == gdata.rc && cd.shape == gdata.shape;
        });
    if (s.m_datas.end() == it) {
        s.m_datas.push_back(gdata);
    }
}

void putOp(GSerialized& s, const GModel::ConstGraph& cg, const ade::NodeHandle &nh) {
    const auto& op = cg.metadata(nh).get<gimpl::Op>();
    for (const auto &in_nh  : nh->inNodes())  { putData(s, cg, in_nh);  }
    for (const auto &out_nh : nh->outNodes()) { putData(s, cg, out_nh); }
    s.m_ops.push_back(op);
}

void mkDataNode(ade::Graph& g, const cv::gimpl::Data& data) {
    GModel::Graph gm(g);
    auto nh = gm.createNode();
    gm.metadata(nh).set(NodeType{NodeType::DATA});
    gm.metadata(nh).set(data);
}

void mkOpNode(ade::Graph& g, const cv::gimpl::Op& op) {
    GModel::Graph gm(g);
    auto nh = gm.createNode();
    gm.metadata(nh).set(NodeType{NodeType::OP});
    gm.metadata(nh).set(op);
}

void linkNodes(ade::Graph& g) {
    std::map<cv::gimpl::RcDesc, ade::NodeHandle> dataNodes;
    GModel::Graph gm(g);

    for (const auto& nh : g.nodes()) {
        if (gm.metadata(nh).get<NodeType>().t == NodeType::DATA) {
            const auto &d = gm.metadata(nh).get<gimpl::Data>();
            const auto rc = cv::gimpl::RcDesc{d.rc, d.shape, d.ctor};
            dataNodes[rc] = nh;
        }
    }

    for (const auto& nh : g.nodes()) {
        if (gm.metadata(nh).get<NodeType>().t == NodeType::OP) {
            const auto& op = gm.metadata(nh).get<gimpl::Op>();
            for (const auto& in : ade::util::indexed(op.args)) {
                const auto& arg = ade::util::value(in);
                if (arg.kind == cv::detail::ArgKind::GOBJREF) {
                    const auto idx = ade::util::index(in);
                    const auto rc  = arg.get<gimpl::RcDesc>();
                    const auto& in_nh = dataNodes.at(rc);
                    const auto& in_eh = g.link(in_nh, nh);
                    gm.metadata(in_eh).set(Input{idx});
                }
            }

            for (const auto& out : ade::util::indexed(op.outs)) {
                const auto idx = ade::util::index(out);
                const auto rc  = ade::util::value(out);
                const auto& out_nh = dataNodes.at(rc);
                const auto& out_eh = g.link(nh, out_nh);
                gm.metadata(out_eh).set(Output{idx});
            }
        }
    }
}

void relinkProto(ade::Graph& g) {
    // identify which node handles map to the protocol
    // input/output object in the reconstructed graph
    using S = std::set<cv::gimpl::RcDesc>;                  // FIXME: use ...
    using M = std::map<cv::gimpl::RcDesc, ade::NodeHandle>; // FIXME: unordered!

    cv::gimpl::GModel::Graph gm(g);
    auto &proto = gm.metadata().get<Protocol>();

    const S set_in(proto.inputs.begin(), proto.inputs.end());
    const S set_out(proto.outputs.begin(), proto.outputs.end());
    M map_in, map_out;

    // Associate the protocol node handles with their resource identifiers
    for (auto &&nh : gm.nodes()) {
        if (gm.metadata(nh).get<cv::gimpl::NodeType>().t == cv::gimpl::NodeType::DATA) {
            const auto &d = gm.metadata(nh).get<cv::gimpl::Data>();
            const auto rc = cv::gimpl::RcDesc{d.rc, d.shape, d.ctor};
            if (set_in.count(rc) > 0) {
                GAPI_DbgAssert(set_out.count(rc) == 0);
                map_in[rc] = nh;
            } else if (set_out.count(rc) > 0) {
                GAPI_DbgAssert(set_in.count(rc) == 0);
                map_out[rc] = nh;
            }
        }
    }

    // Reconstruct the protocol vectors, ordered
    proto.in_nhs.reserve(proto.inputs.size());
    proto.in_nhs.clear();
    proto.out_nhs.reserve(proto.outputs.size());
    proto.out_nhs.clear();
    for (auto &rc : proto.inputs)  { proto.in_nhs .push_back(map_in .at(rc)); }
    for (auto &rc : proto.outputs) { proto.out_nhs.push_back(map_out.at(rc)); }
}

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Graph dump operators

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

namespace
{

#if !defined(GAPI_STANDALONE)
template<typename T>
    void write_plain(I::OStream &os, const T *arr, std::size_t sz) {
        for (auto &&it : ade::util::iota(sz)) os << arr[it];
}
template<typename T>
    void read_plain(I::IStream &is, T *arr, std::size_t sz) {
        for (auto &&it : ade::util::iota(sz)) is >> arr[it];
}
template<typename T>
void write_mat_data(I::OStream &os, const cv::Mat &m) {
    // Write every row individually (handles the case when Mat is a view)
    for (auto &&r : ade::util::iota(m.rows)) {
        write_plain(os, m.ptr<T>(r), m.cols*m.channels());
    }
}
template<typename T>
void read_mat_data(I::IStream &is, cv::Mat &m) {
    // Write every row individually (handles the case when Mat is aligned)
    for (auto &&r : ade::util::iota(m.rows)) {
        read_plain(is, m.ptr<T>(r), m.cols*m.channels());
    }
}
#else
void write_plain(I::OStream &os, const uchar *arr, std::size_t sz) {
    for (auto &&it : ade::util::iota(sz)) os << arr[it];
}
void read_plain(I::IStream &is, uchar *arr, std::size_t sz) {
    for (auto &&it : ade::util::iota(sz)) is >> arr[it];
}
template<typename T>
void write_mat_data(I::OStream &os, const cv::Mat &m) {
    // Write every row individually (handles the case when Mat is a view)
    for (auto &&r : ade::util::iota(m.rows)) {
        write_plain(os, m.ptr(r), m.cols*m.channels()*sizeof(T));
    }
}
template<typename T>
void read_mat_data(I::IStream &is, cv::Mat &m) {
    // Write every row individually (handles the case when Mat is aligned)
    for (auto &&r : ade::util::iota(m.rows)) {
        read_plain(is, m.ptr(r), m.cols*m.channels()*sizeof(T));
    }
}
#endif
} // namespace

I::OStream& operator<< (I::OStream& os, const cv::Mat &m) {
#if !defined(GAPI_STANDALONE)
    GAPI_Assert(m.size.dims() == 2 && "Only 2D images are supported now");
#else
    GAPI_Assert(m.dims.size() == 2 && "Only 2D images are supported now");
#endif
    os << m.rows << m.cols << m.type();
    switch (m.depth()) {
    case CV_8U:  write_mat_data< uint8_t>(os, m); break;
    case CV_8S:  write_mat_data<    char>(os, m); break;
    case CV_16U: write_mat_data<uint16_t>(os, m); break;
    case CV_16S: write_mat_data< int16_t>(os, m); break;
    case CV_32S: write_mat_data< int32_t>(os, m); break;
    case CV_32F: write_mat_data<   float>(os, m); break;
    case CV_64F: write_mat_data<  double>(os, m); break;
    default: GAPI_Assert(false && "Unsupported Mat depth");
    }
    return os;
}
I::IStream& operator>> (I::IStream& is, cv::Mat& m) {
    int rows = -1, cols = -1, type = 0;
    is >> rows >> cols >> type;
    m.create(cv::Size(cols, rows), type);
    switch (m.depth()) {
    case CV_8U:  read_mat_data< uint8_t>(is, m); break;
    case CV_8S:  read_mat_data<    char>(is, m); break;
    case CV_16U: read_mat_data<uint16_t>(is, m); break;
    case CV_16S: read_mat_data< int16_t>(is, m); break;
    case CV_32S: read_mat_data< int32_t>(is, m); break;
    case CV_32F: read_mat_data<   float>(is, m); break;
    case CV_64F: read_mat_data<  double>(is, m); break;
    default: GAPI_Assert(false && "Unsupported Mat depth");
    }
    return is;
}

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

#if !defined(GAPI_STANDALONE)
I::OStream& operator<< (I::OStream& os, const cv::UMat &)
{
    GAPI_Assert(false && "Serialization: Unsupported << for UMat");
    return os;
}
I::IStream& operator >> (I::IStream& is, cv::UMat &)
{
    GAPI_Assert(false && "Serialization: Unsupported >> for UMat");
    return is;
}
#endif // !defined(GAPI_STANDALONE)

I::OStream& operator<< (I::OStream& os, const cv::gapi::wip::IStreamSource::Ptr &)
{
    GAPI_Assert(false && "Serialization: Unsupported << for IStreamSource::Ptr");
    return os;
}
I::IStream& operator >> (I::IStream& is, cv::gapi::wip::IStreamSource::Ptr &)
{
    GAPI_Assert("Serialization: Unsupported >> for IStreamSource::Ptr");
    return is;
}

namespace
{
    template <typename T>
    bool opaque_serialized(I::OStream& os, const cv::detail::OpaqueRef& ref, const cv::detail::OpaqueKind kind)
    {
        if (ref.holds<T>())
        {
            os << kind << ref.rref<T>();
            return true;
        }
        return false;
    }

    template <typename T>
    bool array_serialized(I::OStream& os, const cv::detail::VectorRef& ref, const cv::detail::OpaqueKind kind)
    {
        if (ref.holds<T>())
        {
            os << kind;
            auto vec = ref.rref<T>();
            // FIXME: add size to GArrayDesc
            os << static_cast<int>(vec.size());
            for (const auto& v : vec)
                os << v;
            return true;
        }
        return false;
    }

    template <typename T>
    void deserialize_array(I::IStream& is, cv::detail::VectorRef& ref, int sz)
    {
        ref.reset<T>();
        ref.wref<T>() = std::vector<T>(sz);
        auto& vec = ref.wref<T>();
        for (typename std::vector<T>::reference v : vec)
            is >> v;
    }
} // anonymous namespace

I::OStream& operator<< (I::OStream& os, const cv::detail::VectorRef& ref)
{
    if (array_serialized<bool>(os, ref, cv::detail::OpaqueKind::CV_BOOL)) return os;
    else if (array_serialized<int>(os, ref, cv::detail::OpaqueKind::CV_INT)) return os;
    else if (array_serialized<double>(os, ref, cv::detail::OpaqueKind::CV_DOUBLE)) return os;
    else if (array_serialized<cv::Point>(os, ref, cv::detail::OpaqueKind::CV_POINT)) return os;
    else if (array_serialized<cv::Size>(os, ref, cv::detail::OpaqueKind::CV_SIZE)) return os;
    else if (array_serialized<cv::Rect>(os, ref, cv::detail::OpaqueKind::CV_RECT)) return os;
    else if (array_serialized<cv::Mat>(os, ref, cv::detail::OpaqueKind::CV_MAT)) return os;
    else if (array_serialized<cv::Scalar>(os, ref, cv::detail::OpaqueKind::CV_SCALAR)) return os;
    else GAPI_Assert(false && "Unsupported GArray type for serialization");
    return os;
}
I::IStream& operator >> (I::IStream& is, cv::detail::VectorRef& ref)
{
    cv::detail::OpaqueKind type;
    int sz;
    is >> type >> sz;
    switch (type)
    {
        case cv::detail::OpaqueKind::CV_BOOL   : { deserialize_array<bool>(is, ref, sz); break; }
        case cv::detail::OpaqueKind::CV_INT    : { deserialize_array<int>(is, ref, sz); break; }
        case cv::detail::OpaqueKind::CV_DOUBLE : { deserialize_array<double>(is, ref, sz); break; }
        case cv::detail::OpaqueKind::CV_POINT  : { deserialize_array<cv::Point>(is, ref, sz); break; }
        case cv::detail::OpaqueKind::CV_SIZE   : { deserialize_array<cv::Size>(is, ref, sz); break; }
        case cv::detail::OpaqueKind::CV_RECT   : { deserialize_array<cv::Rect>(is, ref, sz); break; }
        case cv::detail::OpaqueKind::CV_MAT    : { deserialize_array<cv::Mat>(is, ref, sz); break; }
        case cv::detail::OpaqueKind::CV_SCALAR : { deserialize_array<cv::Scalar>(is, ref, sz); break; }
        default:
        {
            GAPI_Assert(false && "Unsupported GArray type for deserialization");
            break;
        }
    }
    return is;
}

I::OStream& operator<< (I::OStream& os, const cv::detail::OpaqueRef& ref)
{
    if (opaque_serialized<bool>(os, ref, cv::detail::OpaqueKind::CV_BOOL)) return os;
    else if (opaque_serialized<int>(os, ref, cv::detail::OpaqueKind::CV_INT)) return os;
    else if (opaque_serialized<double>(os, ref, cv::detail::OpaqueKind::CV_DOUBLE)) return os;
    else if (opaque_serialized<cv::Point>(os, ref, cv::detail::OpaqueKind::CV_POINT)) return os;
    else if (opaque_serialized<cv::Size>(os, ref, cv::detail::OpaqueKind::CV_SIZE)) return os;
    else if (opaque_serialized<cv::Rect>(os, ref, cv::detail::OpaqueKind::CV_RECT)) return os;
    else GAPI_Assert(false && "Unsupported GOpaque type for serialization");
    return os;
}
I::IStream& operator >> (I::IStream& is, cv::detail::OpaqueRef& ref)
{
    cv::detail::OpaqueKind type;
    is >> type;
    switch (type)
    {
        case cv::detail::OpaqueKind::CV_BOOL   : { ref.reset<bool>();      is >> ref.wref<bool>(); break; }
        case cv::detail::OpaqueKind::CV_INT    : { ref.reset<int>();       is >> ref.wref<int>(); break; }
        case cv::detail::OpaqueKind::CV_DOUBLE : { ref.reset<double>();    is >> ref.wref<double>(); break; }
        case cv::detail::OpaqueKind::CV_POINT  : { ref.reset<cv::Point>(); is >> ref.wref<cv::Point>(); break; }
        case cv::detail::OpaqueKind::CV_SIZE   : { ref.reset<cv::Size>();  is >> ref.wref<cv::Size>(); break; }
        case cv::detail::OpaqueKind::CV_RECT   : { ref.reset<cv::Rect>();  is >> ref.wref<cv::Rect>(); break; }
        default:
            GAPI_Assert(false && "Unsupported GOpaque type for deserialization");
            break;
    }
    return is;
}
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

    os << arg.kind << arg.opaque_kind;
    if (arg.kind == cv::detail::ArgKind::GOBJREF) {
        os << arg.get<cv::gimpl::RcDesc>();
    } else {
        GAPI_Assert(arg.kind == cv::detail::ArgKind::OPAQUE_VAL);
        GAPI_Assert(arg.opaque_kind != cv::detail::OpaqueKind::CV_UNKNOWN);
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

    if (arg.kind == cv::detail::ArgKind::GOBJREF) {
        cv::gimpl::RcDesc rc;
        is >> rc;
        arg = (GArg(rc));
    } else {
        GAPI_Assert(arg.kind == cv::detail::ArgKind::OPAQUE_VAL);
        GAPI_Assert(arg.opaque_kind != cv::detail::OpaqueKind::CV_UNKNOWN);
        switch (arg.opaque_kind) {
#define HANDLE_CASE(E,T) case cv::detail::OpaqueKind::CV_##E:           \
            { T t{}; is >> t; arg = (cv::GArg(t)); } break
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
    return os << d.shape << d.rc << d.meta << d.storage << d.kind;
}

namespace
{
    template <typename T>
    void initCtorArr(cv::gimpl::Data &d)
    {
        static std::function<void(cv::detail::VectorRef&)> arr_ctor = [](cv::detail::VectorRef& ref){ref.reset<T>();};
        d.ctor = arr_ctor;
    }
    template <typename T>
    void initCtorOpaq(cv::gimpl::Data &d)
    {
        static std::function<void(cv::detail::OpaqueRef&)> op_ctor = [](cv::detail::OpaqueRef& ref){ref.reset<T>();};
        d.ctor = op_ctor;
    }
} // anonymous namespace

I::IStream& operator>> (I::IStream& is, cv::gimpl::Data &d) {
    // FIXME: HostCtor is not stored here!!
    // FIXME: Storage may be incorrect for subgraph-to-graph process
    is >> d.shape >> d.rc >> d.meta >> d.storage >> d.kind;
    if (d.shape == cv::GShape::GARRAY)
    {
        switch (d.kind)
        {
            case cv::detail::OpaqueKind::CV_BOOL:   initCtorArr<bool>(d); break;
            case cv::detail::OpaqueKind::CV_INT:    initCtorArr<int>(d); break;
            case cv::detail::OpaqueKind::CV_DOUBLE: initCtorArr<double>(d); break;
            case cv::detail::OpaqueKind::CV_POINT:  initCtorArr<cv::Point>(d); break;
            case cv::detail::OpaqueKind::CV_SIZE:   initCtorArr<cv::Size>(d); break;
            case cv::detail::OpaqueKind::CV_RECT:   initCtorArr<cv::Rect>(d); break;
            case cv::detail::OpaqueKind::CV_SCALAR: initCtorArr<cv::Scalar>(d); break;
            case cv::detail::OpaqueKind::CV_MAT:    initCtorArr<cv::Mat>(d); break;
            default : GAPI_Assert(false && "Unsupported GArray type for deserialization");
        }
    }
    else if (d.shape == cv::GShape::GOPAQUE)
    {
        switch (d.kind)
        {
            case cv::detail::OpaqueKind::CV_BOOL:   initCtorOpaq<bool>(d); break;
            case cv::detail::OpaqueKind::CV_INT:    initCtorOpaq<int>(d); break;
            case cv::detail::OpaqueKind::CV_DOUBLE: initCtorOpaq<double>(d); break;
            case cv::detail::OpaqueKind::CV_POINT:  initCtorOpaq<cv::Point>(d); break;
            case cv::detail::OpaqueKind::CV_SIZE:   initCtorOpaq<cv::Size>(d); break;
            case cv::detail::OpaqueKind::CV_RECT:   initCtorOpaq<cv::Rect>(d); break;
            default : GAPI_Assert(false && "Unsupported GOpaque type for deserialization");
        }
    }
    return is;
}


I::OStream& operator<< (I::OStream& os, const cv::gimpl::DataObjectCounter &c) {
    return os << c.m_next_data_id;
}
I::IStream& operator>> (I::IStream& is,       cv::gimpl::DataObjectCounter &c) {
    return is >> c.m_next_data_id;
}


I::OStream& operator<< (I::OStream& os, const cv::gimpl::Protocol &p) {
    // NB: in_nhs/out_nhs are not written!
    return os << p.inputs << p.outputs;
}
I::IStream& operator>> (I::IStream& is,       cv::gimpl::Protocol &p) {
    // NB: in_nhs/out_nhs are reconstructed at a later phase
    return is >> p.inputs >> p.outputs;
}


void serialize( I::OStream& os
              , const ade::Graph &g
              , const std::vector<ade::NodeHandle> &nodes) {
    cv::gimpl::GModel::ConstGraph cg(g);
    serialize(os, g, cg.metadata().get<cv::gimpl::Protocol>(), nodes);
}

void serialize( I::OStream& os
              , const ade::Graph &g
              , const cv::gimpl::Protocol &p
              , const std::vector<ade::NodeHandle> &nodes) {
    cv::gimpl::GModel::ConstGraph cg(g);
    GSerialized s;
    for (auto &nh : nodes) {
        switch (cg.metadata(nh).get<NodeType>().t)
        {
        case NodeType::OP:   putOp  (s, cg, nh); break;
        case NodeType::DATA: putData(s, cg, nh); break;
        default: util::throw_error(std::logic_error("Unknown NodeType"));
        }
    }
    s.m_counter = cg.metadata().get<cv::gimpl::DataObjectCounter>();
    s.m_proto   = p;
    os << s.m_ops << s.m_datas << s.m_counter << s.m_proto;
}

GSerialized deserialize(I::IStream &is) {
    GSerialized s;
    is >> s.m_ops >> s.m_datas >> s.m_counter >> s.m_proto;
    return s;
}

void reconstruct(const GSerialized &s, ade::Graph &g) {
    GAPI_Assert(g.nodes().empty());
    for (const auto& d  : s.m_datas) cv::gimpl::s11n::mkDataNode(g, d);
    for (const auto& op : s.m_ops)   cv::gimpl::s11n::mkOpNode(g, op);
    cv::gimpl::s11n::linkNodes(g);

    cv::gimpl::GModel::Graph gm(g);
    gm.metadata().set(s.m_counter);
    gm.metadata().set(s.m_proto);
    cv::gimpl::s11n::relinkProto(g);
    gm.metadata().set(cv::gimpl::Deserialized{});
}

////////////////////////////////////////////////////////////////////////////////
// Streams /////////////////////////////////////////////////////////////////////

const std::vector<char>& ByteMemoryOutStream::data() const {
    return m_storage;
}
I::OStream& ByteMemoryOutStream::operator<< (uint32_t atom) {
    m_storage.push_back(0xFF & (atom));
    m_storage.push_back(0xFF & (atom >> 8));
    m_storage.push_back(0xFF & (atom >> 16));
    m_storage.push_back(0xFF & (atom >> 24));
    return *this;
}
I::OStream& ByteMemoryOutStream::operator<< (bool atom) {
    m_storage.push_back(atom ? 1 : 0);
    return *this;
}
I::OStream& ByteMemoryOutStream::operator<< (std::vector<bool>::reference atom) {
    m_storage.push_back(atom ? 1 : 0);
    return *this;
}
I::OStream& ByteMemoryOutStream::operator<< (char atom) {
    m_storage.push_back(atom);
    return *this;
}
I::OStream& ByteMemoryOutStream::operator<< (unsigned char atom) {
    return *this << static_cast<char>(atom);
}
I::OStream& ByteMemoryOutStream::operator<< (short atom) {
    static_assert(sizeof(short) == 2, "Expecting sizeof(short) == 2");
    m_storage.push_back(0xFF & (atom));
    m_storage.push_back(0xFF & (atom >> 8));
    return *this;
}
I::OStream& ByteMemoryOutStream::operator<< (unsigned short atom) {
    return *this << static_cast<short>(atom);
}
I::OStream& ByteMemoryOutStream::operator<< (int atom) {
    static_assert(sizeof(int) == 4, "Expecting sizeof(int) == 4");
    return *this << static_cast<uint32_t>(atom);
}
//I::OStream& ByteMemoryOutStream::operator<< (std::size_t atom) {
//    // NB: type truncated!
//    return *this << static_cast<uint32_t>(atom);
//}
I::OStream& ByteMemoryOutStream::operator<< (float atom) {
    static_assert(sizeof(float) == 4, "Expecting sizeof(float) == 4");
    uint32_t tmp = 0u;
    memcpy(&tmp, &atom, sizeof(float));
    return *this << static_cast<uint32_t>(htonl(tmp));
}
I::OStream& ByteMemoryOutStream::operator<< (double atom) {
    static_assert(sizeof(double) == 8, "Expecting sizeof(double) == 8");
    uint32_t tmp[2] = {0u};
    memcpy(tmp, &atom, sizeof(double));
    *this << static_cast<uint32_t>(htonl(tmp[0]));
    *this << static_cast<uint32_t>(htonl(tmp[1]));
    return *this;
}
I::OStream& ByteMemoryOutStream::operator<< (const std::string &str) {
    //*this << static_cast<std::size_t>(str.size()); // N.B. Put type explicitly
    *this << static_cast<uint32_t>(str.size()); // N.B. Put type explicitly
    for (auto c : str) *this << c;
    return *this;
}

ByteMemoryInStream::ByteMemoryInStream(const std::vector<char> &data)
    : m_storage(data) {
}
I::IStream& ByteMemoryInStream::operator>> (uint32_t &atom) {
    check(sizeof(uint32_t));
    uint8_t x[4];
    x[0] = static_cast<uint8_t>(m_storage[m_idx++]);
    x[1] = static_cast<uint8_t>(m_storage[m_idx++]);
    x[2] = static_cast<uint8_t>(m_storage[m_idx++]);
    x[3] = static_cast<uint8_t>(m_storage[m_idx++]);
    atom = ((x[0]) | (x[1] << 8) | (x[2] << 16) | (x[3] << 24));
    return *this;
}
I::IStream& ByteMemoryInStream::operator>> (bool& atom) {
    check(sizeof(char));
    atom = (m_storage[m_idx++] == 0) ? false : true;
    return *this;
}
I::IStream& ByteMemoryInStream::operator>> (std::vector<bool>::reference atom) {
    check(sizeof(char));
    atom = (m_storage[m_idx++] == 0) ? false : true;
    return *this;
}
I::IStream& ByteMemoryInStream::operator>> (char &atom) {
    check(sizeof(char));
    atom = m_storage[m_idx++];
    return *this;
}
I::IStream& ByteMemoryInStream::operator>> (unsigned char &atom) {
    char c{};
    *this >> c;
    atom = static_cast<unsigned char>(c);
    return *this;
}
I::IStream& ByteMemoryInStream::operator>> (short &atom) {
    static_assert(sizeof(short) == 2, "Expecting sizeof(short) == 2");
    check(sizeof(short));
    uint8_t x[2];
    x[0] = static_cast<uint8_t>(m_storage[m_idx++]);
    x[1] = static_cast<uint8_t>(m_storage[m_idx++]);
    atom = ((x[0]) | (x[1] << 8));
    return *this;
}
I::IStream& ByteMemoryInStream::operator>> (unsigned short &atom) {
    short s{};
    *this >> s;
    atom = static_cast<unsigned short>(s);
    return *this;
}
I::IStream& ByteMemoryInStream::operator>> (int& atom) {
    static_assert(sizeof(int) == 4, "Expecting sizeof(int) == 4");
    atom = static_cast<int>(getU32());
    return *this;
}
//I::IStream& ByteMemoryInStream::operator>> (std::size_t& atom) {
//    // NB. Type was truncated!
//    atom = static_cast<std::size_t>(getU32());
//    return *this;
//}
I::IStream& ByteMemoryInStream::operator>> (float& atom) {
    static_assert(sizeof(float) == 4, "Expecting sizeof(float) == 4");
    uint32_t tmp = ntohl(getU32());
    memcpy(&atom, &tmp, sizeof(float));
    return *this;
}
I::IStream& ByteMemoryInStream::operator>> (double& atom) {
    static_assert(sizeof(double) == 8, "Expecting sizeof(double) == 8");
    uint32_t tmp[2] = {ntohl(getU32()), ntohl(getU32())};
    memcpy(&atom, tmp, sizeof(double));
    return *this;
}
I::IStream& ByteMemoryInStream::operator>> (std::string& str) {
    //std::size_t sz = 0u;
    uint32_t sz = 0u;
    *this >> sz;
    if (sz == 0u) {
        str.clear();
    } else {
        str.resize(sz);
        for (auto &&i : ade::util::iota(sz)) { *this >> str[i]; }
    }
    return *this;
}

GAPI_EXPORTS void serialize(I::OStream& os, const cv::GMetaArgs &ma) {
    os << ma;
}
GAPI_EXPORTS void serialize(I::OStream& os, const cv::GRunArgs &ra) {
    os << ra;
}
GAPI_EXPORTS GMetaArgs meta_args_deserialize(I::IStream& is) {
    GMetaArgs s;
    is >> s;
    return s;
}
GAPI_EXPORTS GRunArgs run_args_deserialize(I::IStream& is) {
    GRunArgs s;
    is >> s;
    return s;
}


} // namespace s11n
} // namespace gimpl
} // namespace cv
