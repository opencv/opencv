// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include <iostream>
#include <fstream>
#include <string.h>

#include <ade/util/iota_range.hpp> // used in the vector<</>>

#include "compiler/gmodel.hpp"

namespace cv {
namespace gimpl {
namespace s11n {

////////////////////////////////////////////////////////////////////////////////
// Stream interfaces, so far temporary
namespace I {
    struct GAPI_EXPORTS OStream {
        virtual void put(uint32_t) = 0;
        virtual ~OStream() = default;
    };

    struct GAPI_EXPORTS IStream {
        virtual uint getUInt32() = 0;
        virtual ~IStream() = default;
    };
} // namespace I

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// S11N operators

// Basic types /////////////////////////////////////////////////////////////////

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, bool  bool_val);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is, bool &bool_val);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, char  atom);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is, char &atom);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, uint32_t  atom);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is, uint32_t &atom);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, int  atom);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is, int &atom);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, float  atom);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is, float &atom);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, double  atom);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is, double &atom);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, std::size_t  atom);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is, std::size_t &atom);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const std::string &str);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       std::string &str);

// OpenCV types ////////////////////////////////////////////////////////////////

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::Point &pt);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::Point &pt);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::Size &sz);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::Size &sz);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::Rect &rc);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::Rect &rc);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::Scalar &s);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::Scalar &s);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::Mat &m);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::Mat &m);


// G-API types /////////////////////////////////////////////////////////////////

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, cv::util::monostate  );
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is, cv::util::monostate &);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, cv::GShape  shape);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is, cv::GShape &shape);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, cv::detail::ArgKind  k);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is, cv::detail::ArgKind &k);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, cv::detail::OpaqueKind  k);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is, cv::detail::OpaqueKind &k);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, cv::gimpl::Data::Storage  s);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is, cv::gimpl::Data::Storage &s);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::GArg &arg);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::GArg &arg);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::GKernel &k);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::GKernel &k);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::GMatDesc &d);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::GMatDesc &d);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::GScalarDesc &);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::GScalarDesc &);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::GOpaqueDesc &);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::GOpaqueDesc &);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::GArrayDesc &);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::GArrayDesc &);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::gimpl::RcDesc &rc);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::gimpl::RcDesc &rc);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::gimpl::Op &op);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::gimpl::Op &op);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::gimpl::Data &op);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::gimpl::Data &op);

GAPI_EXPORTS I::OStream& serialize( I::OStream& os
                                  , const ade::Graph &g
                                  , const std::vector<ade::NodeHandle> &nodes);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       ade::Graph &g);

// Legacy //////////////////////////////////////////////////////////////////////


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
class GAPI_EXPORTS SerializationStream final: public I::OStream {
    std::vector<uint> m_dump_storage;

public:
    SerializationStream() = default;
    char* getData();
    size_t getSize();
    void putAtom(uint new_atom);

    // Implement OStream interface
    virtual void put(uint32_t) override;
};

class GAPI_EXPORTS DeSerializationStream final: public I::IStream {
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
