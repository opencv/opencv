#ifndef OPENCV_GAPI_COMMON_SERIALIZATION_HPP
#define OPENCV_GAPI_COMMON_SERIALIZATION_HPP

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

#if (defined _WIN32 || defined _WIN64) && defined _MSC_VER
#pragma warning(disable: 4702)
#endif

namespace cv {
namespace gimpl {
namespace s11n {

struct GSerialized {
    std::vector<cv::gimpl::Op> m_ops;
    std::vector<cv::gimpl::Data> m_datas;
    cv::gimpl::DataObjectCounter m_counter;
    cv::gimpl::Protocol m_proto;
};

////////////////////////////////////////////////////////////////////////////////
// Stream interfaces, so far temporary
namespace I {
    struct GAPI_EXPORTS OStream {
        virtual ~OStream() = default;

        // Define the native support for basic C++ types at the API level:
        virtual OStream& operator<< (bool) = 0;
        virtual OStream& operator<< (char) = 0;
        virtual OStream& operator<< (unsigned char) = 0;
        virtual OStream& operator<< (short) = 0;
        virtual OStream& operator<< (unsigned short) = 0;
        virtual OStream& operator<< (int) = 0;
        //virtual OStream& operator<< (std::size_t) = 0;
        virtual OStream& operator<< (uint32_t) = 0;
        virtual OStream& operator<< (float) = 0;
        virtual OStream& operator<< (double) = 0;
        virtual OStream& operator<< (const std::string&) = 0;
    };

    struct GAPI_EXPORTS IStream {
        virtual ~IStream() = default;

        virtual IStream& operator>> (bool &) = 0;
        virtual IStream& operator>> (char &) = 0;
        virtual IStream& operator>> (unsigned char &) = 0;
        virtual IStream& operator>> (short &) = 0;
        virtual IStream& operator>> (unsigned short &) = 0;
        virtual IStream& operator>> (int &) = 0;
        virtual IStream& operator>> (float &) = 0;
        virtual IStream& operator>> (double &) = 0;
        //virtual IStream& operator>> (std::size_t &) = 0;
        virtual IStream& operator >> (uint32_t &) = 0;
        virtual IStream& operator>> (std::string &) = 0;
    };
} // namespace I

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// S11N operators
// Note: operators for basic types are defined in IStream/OStream

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

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::gimpl::DataObjectCounter &c);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::gimpl::DataObjectCounter &c);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::gimpl::Protocol &p);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::gimpl::Protocol &p);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::GArg &arg);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::GArg &arg);

//Forward declaration
//I::OStream& operator<< (I::OStream& os, const cv::GRunArg &arg);
//I::IStream& operator>> (I::IStream& is, cv::GRunArg &arg);


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

#if !defined(GAPI_STANDALONE)
GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::UMat &);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::UMat &);
#endif // !defined(GAPI_STANDALONE)

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::gapi::wip::IStreamSource::Ptr &);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::gapi::wip::IStreamSource::Ptr &);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::detail::VectorRef &);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::detail::VectorRef &);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::detail::OpaqueRef &);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::detail::OpaqueRef &);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::gimpl::RcDesc &rc);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::gimpl::RcDesc &rc);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::gimpl::Op &op);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::gimpl::Op &op);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::gimpl::Data &op);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::gimpl::Data &op);

// The top-level serialization routine.
// Note it is just a single function which takes a GModel and a list of nodes
// and writes the data to the stream (recursively)
GAPI_EXPORTS void serialize( I::OStream& os
                           , const ade::Graph &g
                           , const std::vector<ade::NodeHandle> &nodes);

// The top-level deserialization routineS.
// Unfortunately the deserialization is a two-step process:
// 1. First we decode a stream into some intermediate representation
//     (called "GSerialized");
// 2. Then we produce an ade::Graph from this intermediate representation.
//
// An ade::Graph can't be produced from the stream immediately
// since every GCompiled object has its own unique ade::Graph, so
// we can't do it once and for all since every compilation process
// is individual and _is_ altering the ade::Graph state (structure and metadata).
// At the same time, we can't hold the reference to "is" within the GComputation
// forever since this input stream may be associated with an external resource
// and have side effects.
//
// Summarizing, the `deserialize()` happens *once per GComputation* immediately
// during the cv::gapi::deserialize<GComputation>(), and `reconstruct()` happens
// on every compilation process issued for this GComputation.
GAPI_EXPORTS GSerialized deserialize(I::IStream& is);
GAPI_EXPORTS void reconstruct(const GSerialized &s, ade::Graph &g);

// Legacy //////////////////////////////////////////////////////////////////////
// Generic: unordered_map serialization ////////////////////////////////////////
template<typename K, typename V>
I::OStream& operator<< (I::OStream& os, const std::unordered_map<K, V> &m) {
    //const std::size_t sz = m.size(); // explicitly specify type
    const uint32_t sz = (uint32_t)m.size(); // explicitly specify type
    os << sz;
    for (auto &&it : m) os << it.first << it.second;
    return os;
}
template<typename K, typename V>
I::IStream& operator>> (I::IStream& is, std::unordered_map<K, V> &m) {
    m.clear();
    //std::size_t sz = 0u;
    uint32_t sz = 0u;
    is >> sz;
    if (sz != 0u) {
        for (auto &&i : ade::util::iota(sz)) {
            (void) i;
            K k{};
            V v{};
            is >> k >> v;
            m.insert({k,v});
        }
        GAPI_Assert(sz == m.size());
    }
    return is;
}

// Generic: variant serialization //////////////////////////////////////////////
namespace detail { // FIXME: breaks old code
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
} // namespace detail FIXME: breaks old code

template<typename... Ts>
I::OStream& operator<< (I::OStream& os, const cv::util::variant<Ts...> &v) {
    os << (uint32_t)v.index();
    return detail::put_v<cv::util::variant<Ts...>, Ts...>(os, v, v.index());
}
template<typename... Ts>
I::IStream& operator>> (I::IStream& is, cv::util::variant<Ts...> &v) {
    int idx = -1;
    is >> idx;
    GAPI_Assert(idx >= 0 && idx < (int)sizeof...(Ts));
    return detail::get_v<cv::util::variant<Ts...>, Ts...>(is, v, 0u, idx);
}

// Generic: vector serialization ///////////////////////////////////////////////
// Moved here to fix CLang issues https://clang.llvm.org/compatibility.html
// Unqualified lookup in templates
template<typename T>
I::OStream& operator<< (I::OStream& os, const std::vector<T> &ts) {
    //const std::size_t sz = ts.size(); // explicitly specify type
    const uint32_t sz = (uint32_t)ts.size(); // explicitly specify type
    os << sz;
    for (auto &&v : ts) os << v;
    return os;
}
template<typename T>
I::IStream& operator >> (I::IStream& is, std::vector<T> &ts) {
    //std::size_t sz = 0u;
    uint32_t sz = 0u;
    is >> sz;
    if (sz == 0u) {
        ts.clear();
    }
    else {
        ts.resize(sz);
        for (auto &&i : ade::util::iota(sz)) is >> ts[i];
    }
    return is;
}

// FIXME: Basic Stream implementaions //////////////////////////////////////////

// Basic in-memory stream implementations.
class GAPI_EXPORTS ByteMemoryOutStream final: public I::OStream {
    std::vector<char> m_storage;

    //virtual I::OStream& operator << (uint32_t) override;
    //virtual I::OStream& operator<< (uint32_t) final;
public:
    const std::vector<char>& data() const;

    virtual I::OStream& operator<< (bool) override;
    virtual I::OStream& operator<< (char) override;
    virtual I::OStream& operator<< (unsigned char) override;
    virtual I::OStream& operator<< (short) override;
    virtual I::OStream& operator<< (unsigned short) override;
    virtual I::OStream& operator<< (int) override;
    //virtual I::OStream& operator<< (std::size_t) override;
    virtual I::OStream& operator<< (float) override;
    virtual I::OStream& operator<< (double) override;
    virtual I::OStream& operator<< (const std::string&) override;
    virtual I::OStream& operator<< (uint32_t) override;
};

class GAPI_EXPORTS ByteMemoryInStream final: public I::IStream {
    const std::vector<char>& m_storage;
    size_t m_idx = 0u;

    void check(std::size_t n) { (void) n; GAPI_DbgAssert(m_idx+n-1 < m_storage.size()); }
    uint32_t getU32() { uint32_t v{}; *this >> v; return v; };

    //virtual I::IStream& operator>> (uint32_t &) final;

public:
    explicit ByteMemoryInStream(const std::vector<char> &data);

    virtual I::IStream& operator>> (bool &) override;
    virtual I::IStream& operator>> (char &) override;
    virtual I::IStream& operator>> (unsigned char &) override;
    virtual I::IStream& operator>> (short &) override;
    virtual I::IStream& operator>> (unsigned short &) override;
    virtual I::IStream& operator>> (int &) override;
    virtual I::IStream& operator>> (float &) override;
    virtual I::IStream& operator>> (double &) override;
    //virtual I::IStream& operator>> (std::size_t &) override;
    virtual I::IStream& operator >> (uint32_t &) override;
    virtual I::IStream& operator>> (std::string &) override;
};

GAPI_EXPORTS void serialize(I::OStream& os, const cv::GMetaArgs &ma);
GAPI_EXPORTS void serialize(I::OStream& os, const cv::GRunArgs &ra);
GAPI_EXPORTS GMetaArgs meta_args_deserialize(I::IStream& is);
GAPI_EXPORTS GRunArgs run_args_deserialize(I::IStream& is);

} // namespace s11n
} // namespace gimpl
} // namespace cv

#endif // OPENCV_GAPI_COMMON_SERIALIZATION_HPP
