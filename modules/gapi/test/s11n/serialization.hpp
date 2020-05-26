// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include <iostream>
#include <fstream>
#include <string.h>

#include <ade/util/iota_range.hpp>

#include "compiler/gmodel.hpp"
#include "logger.hpp"

namespace cv {
namespace gimpl {
namespace s11n {

struct GSerialized
{
    // Need to monitor ins/outs of the graph?
    // Remove m_?
    std::vector<cv::gimpl::Op> m_ops;
    std::vector<cv::gimpl::Data> m_datas;
};

GSerialized serialize(const gimpl::GModel::ConstGraph& m_gm, const std::vector<ade::NodeHandle>& nodes);

// Stream interfaces, so far temporary
namespace I {
    struct OStream {
        virtual void put(uint32_t) = 0;
        virtual ~OStream() = default;
    };

    struct IStream {
        virtual uint getUInt32() = 0;
        virtual ~IStream() = default;
    };
} // namespace I

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// S11N operators

// Basic types /////////////////////////////////////////////////////////////////

I::OStream& operator<< (I::OStream& os, bool  bool_val);
I::IStream& operator>> (I::IStream& is, bool &bool_val);

I::OStream& operator<< (I::OStream& os, char  atom);
I::IStream& operator>> (I::IStream& is, char &atom);

I::OStream& operator<< (I::OStream& os, uint32_t  atom);
I::IStream& operator>> (I::IStream& is, uint32_t &atom);

I::OStream& operator<< (I::OStream& os, int  atom);
I::IStream& operator>> (I::IStream& is, int &atom);

I::OStream& operator<< (I::OStream& os, float  atom);
I::IStream& operator>> (I::IStream& is, float &atom);

I::OStream& operator<< (I::OStream& os, double  atom);
I::IStream& operator>> (I::IStream& is, double &atom);

I::OStream& operator<< (I::OStream& os, std::size_t  atom);
I::IStream& operator>> (I::IStream& is, std::size_t &atom);

I::OStream& operator<< (I::OStream& os, const std::string &str);
I::IStream& operator>> (I::IStream& is,       std::string &str);

// OpenCV types ////////////////////////////////////////////////////////////////

I::OStream& operator<< (I::OStream& os, const cv::Point &pt);
I::IStream& operator>> (I::IStream& is,       cv::Point &pt);

I::OStream& operator<< (I::OStream& os, const cv::Size &sz);
I::IStream& operator>> (I::IStream& is,       cv::Size &sz);

I::OStream& operator<< (I::OStream& os, const cv::Rect &rc);
I::IStream& operator>> (I::IStream& is,       cv::Rect &rc);

I::OStream& operator<< (I::OStream& os, const cv::Scalar &s);
I::IStream& operator>> (I::IStream& is,       cv::Scalar &s);

I::OStream& operator<< (I::OStream& os, const cv::Mat &m);
I::IStream& operator>> (I::IStream& is,       cv::Mat &m);


// G-API types /////////////////////////////////////////////////////////////////

I::OStream& operator<< (I::OStream& os, cv::util::monostate  );
I::IStream& operator>> (I::IStream& is, cv::util::monostate &);

I::OStream& operator<< (I::OStream& os, cv::GShape  shape);
I::IStream& operator>> (I::IStream& is, cv::GShape &shape);

I::OStream& operator<< (I::OStream& os, cv::detail::ArgKind  k);
I::IStream& operator>> (I::IStream& is, cv::detail::ArgKind &k);

I::OStream& operator<< (I::OStream& os, cv::detail::OpaqueKind  k);
I::IStream& operator>> (I::IStream& is, cv::detail::OpaqueKind &k);

I::OStream& operator<< (I::OStream& os, cv::gimpl::Data::Storage  s);
I::IStream& operator>> (I::IStream& is, cv::gimpl::Data::Storage &s);

I::OStream& operator<< (I::OStream& os, const cv::GArg &arg);
I::IStream& operator>> (I::IStream& is,       cv::GArg &arg);

I::OStream& operator<< (I::OStream& os, const cv::GKernel &k);
I::IStream& operator>> (I::IStream& is,       cv::GKernel &k);

I::OStream& operator<< (I::OStream& os, const cv::GMatDesc &d);
I::IStream& operator>> (I::IStream& is,       cv::GMatDesc &d);

I::OStream& operator<< (I::OStream& os, const cv::GScalarDesc &);
I::IStream& operator>> (I::IStream& is,       cv::GScalarDesc &);

I::OStream& operator<< (I::OStream& os, const cv::GOpaqueDesc &);
I::IStream& operator>> (I::IStream& is,       cv::GOpaqueDesc &);

I::OStream& operator<< (I::OStream& os, const cv::GArrayDesc &);
I::IStream& operator>> (I::IStream& is,       cv::GArrayDesc &);

I::OStream& operator<< (I::OStream& os, const cv::gimpl::RcDesc &rc);
I::IStream& operator>> (I::IStream& is,       cv::gimpl::RcDesc &rc);

I::OStream& operator<< (I::OStream& os, const cv::gimpl::Op &op);
I::IStream& operator>> (I::IStream& is,       cv::gimpl::Op &op);

I::OStream& operator<< (I::OStream& os, const cv::gimpl::Data &op);
I::IStream& operator>> (I::IStream& is,       cv::gimpl::Data &op);

// Legacy //////////////////////////////////////////////////////////////////////


void dumpGSerialized(const GSerialized s, I::OStream &ofs_serialized);
void readGSerialized(GSerialized &s, I::IStream &ifs_serialized);
std::vector<ade::NodeHandle> reconstructGModel(ade::Graph &g, const GSerialized &s);

// Generic: vector serialization ///////////////////////////////////////////////
template<typename T>
I::OStream& operator<< (I::OStream& os, const std::vector<T> &ts) {
    const std::size_t sz = ts.size(); // explicitly specify type
    os << sz;
    for (auto &&v : ts) os << v;
    return os;
}
template<typename T>
I::IStream& operator>> (I::IStream& is, std::vector<T> &ts) {
    std::size_t sz = 0u;
    is >> sz;
    if (sz == 0u) { ts.clear();
    } else {
        ts.resize(sz);
        for (auto &&i : ade::util::iota(sz)) is >> ts[i];
    }
    return is;
}

// Generic: variant serialization //////////////////////////////////////////////
// namespace detail { // FIXME: breaks old code
template<typename V>
I::OStream& put_v(I::OStream&, const V&, std::size_t) {
    GAPI_Assert(false && "variant>>: requested index is invalid");
};
template<typename V, typename X, typename... Xs>
I::OStream& put_v(I::OStream& os, const V& v, std::size_t x) {
    return (x == 0u)
        ? os << cv::util::get<X>(v)
        : put_v<V, Xs...>(os, v, x-1);
}
template<typename V>
I::IStream& get_v(I::IStream&, V&, std::size_t, std::size_t) {
    GAPI_Assert(false && "variant<<: requested index is invalid");
}
template<typename V, typename X, typename... Xs>
I::IStream& get_v(I::IStream& is, V& v, std::size_t i, std::size_t gi) {
    if (i == gi) {
        X x{};
        is >> x;
        v = std::move(x);
        return is;
    } else return get_v<V, Xs...>(is, v, i+1, gi);
}
// } // namespace detail FIXME: breaks old code

template<typename... Ts>
I::OStream& operator<< (I::OStream& os, const cv::util::variant<Ts...> &v) {
    os << v.index();
    return put_v<cv::util::variant<Ts...>, Ts...>(os, v, v.index());
}
template<typename... Ts>
I::IStream& operator>> (I::IStream& is, cv::util::variant<Ts...> &v) {
    int idx = -1;
    is >> idx;
    GAPI_Assert(idx >= 0 && idx < sizeof...(Ts));
    return get_v<cv::util::variant<Ts...>, Ts...>(is, v, 0u, idx);
}

// FIXME: Basic Stream implementaions //////////////////////////////////////////

// Basic (dummy) stream implementations.
class SerializationStream final: public I::OStream {
    std::vector<uint> m_dump_storage{};

public:
    SerializationStream() = default;
    char* getData();
    size_t getSize();
    void putAtom(uint new_atom);

    // Implement OStream interface
    virtual void put(uint32_t) override;
};

class DeSerializationStream final: public I::IStream {
    std::vector<uint> m_dump_storage{};
    size_t m_storage_index = 0;

public:
    DeSerializationStream(char* data, size_t sz);
    char* getData();
    size_t getSize();
    void putAtom(uint& new_atom);
    uint getAtom();

    // Implement IStream interface
    virtual uint32_t getUInt32() override;
};


} // namespace s11n
} // namespace gimpl
} // namespace cv
