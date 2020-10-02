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
#include <opencv2/gapi/gcommon.hpp>

namespace cv {
namespace gapi {

namespace s11n {
    GAPI_EXPORTS std::unique_ptr<I::IStream> getInStream(const std::vector<char> &p);
} // namespace s11n

namespace detail {
    GAPI_EXPORTS cv::GComputation getGraph(const std::vector<char> &p);

    GAPI_EXPORTS cv::GMetaArgs getMetaArgs(const std::vector<char> &p);

    GAPI_EXPORTS cv::GRunArgs getRunArgs(const std::vector<char> &p);

    template<typename T>
    GAPI_EXPORTS GCompileArg deserialize_arg(cv::gapi::s11n::I::IStream& is, const std::string &tag);

    template<typename T1, typename T2, typename... Types>
    GAPI_EXPORTS GCompileArg deserialize_arg(cv::gapi::s11n::I::IStream& is, const std::string &tag);

    template<typename... Types>
    GAPI_EXPORTS cv::GCompileArgs getCompileArgs(const std::vector<char> &p) {
        std::unique_ptr<cv::gapi::s11n::I::IStream> pIs = cv::gapi::s11n::getInStream(p);
        cv::gapi::s11n::I::IStream& is = *pIs.get();
        cv::GCompileArgs args;
        std::size_t sz = 0u;
        is >> sz;
        for (int i = 0; i < sz; ++i) {
            std::string tag;
            is >> tag;
            args.push_back(cv::gapi::detail::deserialize_arg<Types...>(is, tag)); // may be defined here.
        }

        return args;
    }
} // namespace detail

GAPI_EXPORTS std::vector<char> serialize(const cv::GComputation &c);
//namespace{

template<typename T> static inline
T deserialize(const std::vector<char> &p);

//} //ananymous namespace

GAPI_EXPORTS std::vector<char> serialize(const cv::GCompileArgs&);
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

template<typename T, typename... Types> inline
typename std::enable_if<std::is_same<T, GCompileArgs>::value, GCompileArgs>::
type deserialize(const std::vector<char> &p) {
    return detail::getCompileArgs<Types...>(p);
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

    GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::GCompileArg &arg);

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

namespace detail
{
// Note: actual implementation is defined by user
template<typename T>
struct GAPI_EXPORTS S11N;

//Add implementation for default types via SFINAE!!
//typename std::enable_if<can_be_serialized_by_framework<T>::value, T>::type
// template<typename T>
// struct GAPI_EXPORTS S11N { serialize, deserialize };

template<typename T> struct wrap_serialize<T, cv::GCompileArg>
{
    static std::function<void(gapi::s11n::I::OStream& os, const util::any& arg)> serialize;

private:
    template<typename> using sfinae_true = std::true_type;

    template<typename Q = T>
    static auto try_call_serialize(gapi::s11n::I::OStream& os, const util::any& arg, int)
        -> sfinae_true<decltype(S11N<Q>::serialize(os, util::any_cast<Q>(arg)), void())>
    {
        S11N<Q>::serialize(os, util::any_cast<Q>(arg));
        return sfinae_true<void>{};
    }

    template<typename Q = T>
    static auto try_call_serialize(gapi::s11n::I::OStream& os, const util::any& arg, long)
        -> sfinae_true<decltype(os << std::declval<const typename std::add_lvalue_reference<Q>::type>(), void())>
    {
        os << util::any_cast<Q>(arg);
        return sfinae_true<void>{};
    }

    template<typename Q = T>
    static std::false_type try_call_serialize(gapi::s11n::I::OStream &os, const util::any& arg, ...);

    static void call_serialize(gapi::s11n::I::OStream& os, const util::any& arg)
    {
        try_call_serialize<T>(os, arg, 0);
    }
};

template<typename T>
std::function<void(gapi::s11n::I::OStream& os, const util::any& arg)>
wrap_serialize<T, cv::GCompileArg>::serialize =
        decltype(try_call_serialize(std::declval<
                                        typename std::add_lvalue_reference<gapi::s11n::I::OStream>::type>(),
                                    std::declval<typename std::add_lvalue_reference<util::any>::type>(),
                                    int()))::value ? &call_serialize : nullptr;


template<typename T> struct wrap_deserialize
{
private:
    template<typename Q = T>
    static auto call_deserialize(gapi::s11n::I::IStream& is, int)
        -> decltype(S11N<Q>::deserialize(is), S11N<Q>::deserialize(is))
    {
        return S11N<Q>::deserialize(is); // returning reference to temporary?
    }

    // FIXME: Add trait for basic types?
    template<typename Q = T>
    static auto call_deserialize(gapi::s11n::I::IStream& is, long)
        -> decltype(std::declval<typename std::add_lvalue_reference<Q>::type>() << is, std::declval<typename std::decay<Q>::type>())
    {
        return T() << is;
    }

    template<typename Q = T>
    static cv::GCompileArg call_deserialize(gapi::s11n::I::IStream&, ...)
    {
        return GCompileArg { };
    }

public:
    static auto deserialize(gapi::s11n::I::IStream& is)
        -> decltype(call_deserialize<T>(is, 0))
    {
        return call_deserialize<T>(is, 0);
    }
};
} // namespace detail
} // namespace s11n

namespace detail
{
template<typename T>
GAPI_EXPORTS GCompileArg deserialize_arg(cv::gapi::s11n::I::IStream& is, const std::string &tag) {
    if (tag == cv::detail::CompileArgTag<T>::tag()) {
        return GCompileArg { cv::gapi::s11n::detail::wrap_deserialize<T>::deserialize(is) };
    }
    else {
        throw std::logic_error("No CompileArgTag specialization for some of custom types!");
    } 
}
template<typename T1, typename T2, typename... Types>
GAPI_EXPORTS GCompileArg deserialize_arg(cv::gapi::s11n::I::IStream& is, const std::string &tag) {
    if (tag == cv::detail::CompileArgTag<T1>::tag()) {
        return GCompileArg { cv::gapi::s11n::detail::wrap_deserialize<T1>::deserialize(is) };
    }
    else {
        return deserialize_arg<T2, Types...>(is, tag);
    } 
}
} // namespace detail

} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_S11N_HPP
