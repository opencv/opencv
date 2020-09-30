// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_S11N_HPP
#define OPENCV_GAPI_S11N_HPP

#include <vector>
#include <map>
#include <unordered_map>
#include <opencv2/gapi/gcomputation.hpp>
#include <opencv2/gapi/render/render_types.hpp>
#include <opencv2/gapi/util/variant.hpp>
#include <opencv2/gapi/gkernel.hpp>

namespace cv {
namespace gapi {

namespace detail {
    GAPI_EXPORTS cv::GComputation getGraph(const std::vector<char> &p);
} // namespace detail

namespace detail {
    GAPI_EXPORTS cv::GMetaArgs getMetaArgs(const std::vector<char> &p);
} // namespace detail

namespace detail {
    GAPI_EXPORTS cv::GRunArgs getRunArgs(const std::vector<char> &p);
} // namespace detail

GAPI_EXPORTS std::vector<char> serialize(const cv::GComputation &c);
//namespace{

template<typename T> static inline
T deserialize(const std::vector<char> &p);

//} //ananymous namespace

GAPI_EXPORTS std::vector<char> serialize(const cv::GMetaArgs&);
GAPI_EXPORTS std::vector<char> serialize(const cv::GRunArgs&);

template<> inline
cv::GComputation deserialize(const std::vector<char> &p) {
    return detail::getGraph(p);
}

template<> inline
cv::GMetaArgs deserialize(const std::vector<char> &p) {
    return detail::getMetaArgs(p);
}

template<> inline
cv::GRunArgs deserialize(const std::vector<char> &p) {
    return detail::getRunArgs(p);
}

} // namespace gapi
} // namespace cv

// FIXME: forward declaration
// FIXME moved here due to clang lookup issue: https://clang.llvm.org/compatibility.html
namespace cv {
namespace gimpl {
    struct RcDesc;
    struct Op;
    struct Data;
} // namespace gimpl
} // namespace cv

namespace cv {
namespace gapi {
namespace s11n {
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
        virtual OStream& operator<< (uint32_t) = 0;
        virtual OStream& operator<< (uint64_t) = 0;
        virtual OStream& operator<< (float) = 0;
        virtual OStream& operator<< (double) = 0;
        virtual OStream& operator<< (const std::string&) = 0;
    };

    struct GAPI_EXPORTS IStream {
        virtual ~IStream() = default;

        virtual IStream& operator>> (bool &) = 0;
        virtual IStream& operator>> (std::vector<bool>::reference) = 0;
        virtual IStream& operator>> (char &) = 0;
        virtual IStream& operator>> (unsigned char &) = 0;
        virtual IStream& operator>> (short &) = 0;
        virtual IStream& operator>> (unsigned short &) = 0;
        virtual IStream& operator>> (int &) = 0;
        virtual IStream& operator>> (float &) = 0;
        virtual IStream& operator>> (double &) = 0;
        virtual IStream& operator >> (uint32_t &) = 0;
        virtual IStream& operator >> (uint64_t &) = 0;
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

    GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::gapi::wip::draw::Text &t);
    GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::gapi::wip::draw::Text &t);

    GAPI_EXPORTS I::OStream& operator<< (I::OStream&, const cv::gapi::wip::draw::FText &);
    GAPI_EXPORTS I::IStream& operator>> (I::IStream&,       cv::gapi::wip::draw::FText &);

    GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::gapi::wip::draw::Circle &c);
    GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::gapi::wip::draw::Circle &c);

    GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::gapi::wip::draw::Rect &r);
    GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::gapi::wip::draw::Rect &r);

    GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::gapi::wip::draw::Image &i);
    GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::gapi::wip::draw::Image &i);

    GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::gapi::wip::draw::Mosaic &m);
    GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::gapi::wip::draw::Mosaic &m);

    GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::gapi::wip::draw::Poly &p);
    GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::gapi::wip::draw::Poly &p);

    GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::gapi::wip::draw::Line &l);
    GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::gapi::wip::draw::Line &l);

    // G-API types /////////////////////////////////////////////////////////////////

    GAPI_EXPORTS I::IStream& operator<< (I::OStream& os, const cv::GCompileArg &arg);

    GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, cv::util::monostate  );
    GAPI_EXPORTS I::IStream& operator>> (I::IStream& is, cv::util::monostate &);

    GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, cv::GShape  shape);
    GAPI_EXPORTS I::IStream& operator>> (I::IStream& is, cv::GShape &shape);

    GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, cv::detail::ArgKind  k);
    GAPI_EXPORTS I::IStream& operator>> (I::IStream& is, cv::detail::ArgKind &k);

    GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, cv::detail::OpaqueKind  k);
    GAPI_EXPORTS I::IStream& operator>> (I::IStream& is, cv::detail::OpaqueKind &k);

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

    // FIXME: moved here due to clang lookup issue (see the comment above)
    GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::gimpl::RcDesc &rc);
    GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::gimpl::RcDesc &rc);

    GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::gimpl::Op &op);
    GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::gimpl::Op &op);

    GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::gimpl::Data &op);
    GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::gimpl::Data &op);


    // Generic: variant serialization //////////////////////////////////////////////
    namespace detail {
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
    } // namespace detail

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

    // Generic: stl structures serialization //////////////////////////////////////////////
    template<typename K, typename V>
    GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const std::map<K, V> &m) {
        const uint32_t sz = static_cast<uint32_t>(m.size());
        os << sz;
        for (const auto& it : m) os << it.first << it.second;
        return os;
    }
    template<typename K, typename V>
    GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const std::unordered_map<K, V> &m) {
        const uint32_t sz = static_cast<uint32_t>(m.size());
        os << sz;
        for (auto &&it : m) os << it.first << it.second;
        return os;
    }
    template<typename T>
    GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const std::vector<T> &ts) {
        const uint32_t sz = static_cast<uint32_t>(ts.size());
        os << sz;
        for (auto &&v : ts) os << v;
        return os;
    }

    template<typename K, typename V>
    GAPI_EXPORTS I::IStream& operator>> (I::IStream& is, std::map<K, V> &m) {
        m.clear();
        uint32_t sz = 0u;
        is >> sz;
        for (std::size_t i = 0; i < sz; ++i) {
            K k{};
            V v{};
            is >> k >> v;
            m[k] = v;
        }
        return is;
    }
    template<typename K, typename V>
    GAPI_EXPORTS I::IStream& operator>> (I::IStream& is, std::unordered_map<K, V> &m) {
        m.clear();
        uint32_t sz = 0u;
        is >> sz;
        for (std::size_t i = 0; i < sz; ++i) {
            K k{};
            V v{};
            is >> k >> v;
            m[k] = v;
        }
        return is;
    }
    template<typename T>
    GAPI_EXPORTS I::IStream& operator>> (I::IStream& is, std::vector<T> &ts) {
        uint32_t sz = 0u;
        is >> sz;
        if (sz == 0u) {
            ts.clear();
        }
        else {
            ts.resize(sz);
            for (std::size_t i = 0; i < sz; ++i) is >> ts[i];
        }
        return is;
    }

namespace detail {
    // Will be used along with default types if possible in specific cases (compile args, etc)
    // Note: actual implementation is defined by user
    template<typename T>
    struct GAPI_EXPORTS S11N;
} // namespace detail
} // namespace s11n
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_S11N_HPP
