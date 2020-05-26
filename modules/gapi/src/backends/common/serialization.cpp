// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include <map> // map
#include <ade/util/zip_range.hpp> // indexed

#include <opencv2/gapi/gtype_traits.hpp>

#include "backends/common/serialization.hpp"

#ifdef _WIN32
#include <winsock.h>      // htonl, ntohl
#else
#include <netinet/in.h>   // htonl, ntohl
#endif

namespace cv {
namespace gimpl {
namespace s11n {
namespace {

struct GSerialized {
    std::vector<cv::gimpl::Op> m_ops;
    std::vector<cv::gimpl::Data> m_datas;
};

void mkDataNode(ade::Graph& g, const cv::gimpl::Data& data);
void mkOpNode(ade::Graph& g, const cv::gimpl::Op& op);
void linkNodes(ade::Graph& g);
void putData(GSerialized& s, const GModel::ConstGraph& cg, const ade::NodeHandle &nh);
void putOp(GSerialized& s, const GModel::ConstGraph& cg, const ade::NodeHandle &nh);


// FIXME? make a method of GSerialized?
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
            const auto rc = cv::gimpl::RcDesc{d.rc, d.shape};
            dataNodes[rc] = nh;
        }
    }

    for (const auto& nh : g.nodes()) {
        if (gm.metadata(nh).get<NodeType>().t == NodeType::OP) {
            const auto& op = gm.metadata(nh).get<gimpl::Op>();
            for (const auto& in : ade::util::indexed(op.args)) {
                const auto& arg = ade::util::value(in);
                if (arg.kind == detail::ArgKind::GOBJREF) {
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
} // anonymous namespace

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
        arg = std::move(GArg(rc));
    } else {
        GAPI_Assert(arg.kind == cv::detail::ArgKind::OPAQUE_VAL);
        GAPI_Assert(arg.opaque_kind != cv::detail::OpaqueKind::CV_UNKNOWN);
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

I::OStream& serialize ( I::OStream& os
                      , const ade::Graph &g
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
    return os << s.m_ops << s.m_datas;
}
I::IStream& operator>> (I::IStream& is, ade::Graph &g) {
    GSerialized s;
    is >> s.m_ops >> s.m_datas;

    for (const auto& d  : s.m_datas) cv::gimpl::s11n::mkDataNode(g, d);
    for (const auto& op : s.m_ops)   cv::gimpl::s11n::mkOpNode(g, op);
    cv::gimpl::s11n::linkNodes(g);

    return is;
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
