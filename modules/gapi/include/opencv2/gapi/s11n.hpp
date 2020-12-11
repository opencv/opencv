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
#include <opencv2/gapi/s11n/base.hpp>
#include <opencv2/gapi/gcomputation.hpp>
#include <opencv2/gapi/rmat.hpp>

namespace cv {
namespace gapi {

namespace detail {
    GAPI_EXPORTS cv::GComputation getGraph(const std::vector<char> &p);

    GAPI_EXPORTS cv::GMetaArgs getMetaArgs(const std::vector<char> &p);

    GAPI_EXPORTS cv::GRunArgs getRunArgs(const std::vector<char> &p);

    template<typename... Types>
    cv::GCompileArgs getCompileArgs(const std::vector<char> &p);

    template<typename RMatAdapterType>
    cv::GRunArgs getRunArgsWithRMats(const std::vector<char> &p);
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

template<typename T, typename RMatAdapterType> inline
typename std::enable_if<std::is_same<T, GRunArgs>::value, GRunArgs>::
type deserialize(const std::vector<char> &p) {
    return detail::getRunArgsWithRMats<RMatAdapterType>(p);
}
} // namespace gapi
} // namespace cv

namespace cv {
namespace gapi {
namespace s11n {
struct GAPI_EXPORTS IOStream {
    virtual ~IOStream() = default;
    // Define the native support for basic C++ types at the API level:
    virtual IOStream& operator<< (bool) = 0;
    virtual IOStream& operator<< (char) = 0;
    virtual IOStream& operator<< (unsigned char) = 0;
    virtual IOStream& operator<< (short) = 0;
    virtual IOStream& operator<< (unsigned short) = 0;
    virtual IOStream& operator<< (int) = 0;
    virtual IOStream& operator<< (uint32_t) = 0;
    virtual IOStream& operator<< (uint64_t) = 0;
    virtual IOStream& operator<< (float) = 0;
    virtual IOStream& operator<< (double) = 0;
    virtual IOStream& operator<< (const std::string&) = 0;
};

struct GAPI_EXPORTS IIStream {
    virtual ~IIStream() = default;
    virtual IIStream& operator>> (bool &) = 0;
    virtual IIStream& operator>> (std::vector<bool>::reference) = 0;
    virtual IIStream& operator>> (char &) = 0;
    virtual IIStream& operator>> (unsigned char &) = 0;
    virtual IIStream& operator>> (short &) = 0;
    virtual IIStream& operator>> (unsigned short &) = 0;
    virtual IIStream& operator>> (int &) = 0;
    virtual IIStream& operator>> (float &) = 0;
    virtual IIStream& operator>> (double &) = 0;
    virtual IIStream& operator >> (uint32_t &) = 0;
    virtual IIStream& operator >> (uint64_t &) = 0;
    virtual IIStream& operator>> (std::string &) = 0;
};

namespace detail {
GAPI_EXPORTS std::unique_ptr<IIStream> getInStream(const std::vector<char> &p);
} // namespace detail

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// S11N operators
// Note: operators for basic types are defined in IIStream/IOStream

// OpenCV types ////////////////////////////////////////////////////////////////

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::Point &pt);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::Point &pt);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::Point2f &pt);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::Point2f &pt);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::Size &sz);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::Size &sz);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::Rect &rc);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::Rect &rc);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::Scalar &s);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::Scalar &s);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::Mat &m);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::Mat &m);

// FIXME: for GRunArgs serailization
#if !defined(GAPI_STANDALONE)
GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::UMat &);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::UMat &);
#endif // !defined(GAPI_STANDALONE)

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::RMat &r);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::RMat &r);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::gapi::wip::IStreamSource::Ptr &);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::gapi::wip::IStreamSource::Ptr &);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::detail::VectorRef &);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::detail::VectorRef &);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::detail::OpaqueRef &);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::detail::OpaqueRef &);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::MediaFrame &);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::MediaFrame &);

// Generic STL types ////////////////////////////////////////////////////////////////
template<typename K, typename V>
IOStream& operator<< (IOStream& os, const std::map<K, V> &m) {
    const uint32_t sz = static_cast<uint32_t>(m.size());
    os << sz;
    for (const auto& it : m) os << it.first << it.second;
    return os;
}
template<typename K, typename V>
IIStream& operator>> (IIStream& is, std::map<K, V> &m) {
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
IOStream& operator<< (IOStream& os, const std::unordered_map<K, V> &m) {
    const uint32_t sz = static_cast<uint32_t>(m.size());
    os << sz;
    for (auto &&it : m) os << it.first << it.second;
    return os;
}
template<typename K, typename V>
IIStream& operator>> (IIStream& is, std::unordered_map<K, V> &m) {
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
IOStream& operator<< (IOStream& os, const std::vector<T> &ts) {
    const uint32_t sz = static_cast<uint32_t>(ts.size());
    os << sz;
    for (auto &&v : ts) os << v;
    return os;
}
template<typename T>
IIStream& operator>> (IIStream& is, std::vector<T> &ts) {
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

// Generic: variant serialization
namespace detail {
template<typename V>
IOStream& put_v(IOStream&, const V&, std::size_t) {
    GAPI_Assert(false && "variant>>: requested index is invalid");
};
template<typename V, typename X, typename... Xs>
IOStream& put_v(IOStream& os, const V& v, std::size_t x) {
    return (x == 0u)
        ? os << cv::util::get<X>(v)
        : put_v<V, Xs...>(os, v, x-1);
}
template<typename V>
IIStream& get_v(IIStream&, V&, std::size_t, std::size_t) {
    GAPI_Assert(false && "variant<<: requested index is invalid");
}
template<typename V, typename X, typename... Xs>
IIStream& get_v(IIStream& is, V& v, std::size_t i, std::size_t gi) {
    if (i == gi) {
        X x{};
        is >> x;
        v = V{std::move(x)};
        return is;
    } else return get_v<V, Xs...>(is, v, i+1, gi);
}
} // namespace detail

template<typename... Ts>
IOStream& operator<< (IOStream& os, const cv::util::variant<Ts...> &v) {
    os << static_cast<uint32_t>(v.index());
    return detail::put_v<cv::util::variant<Ts...>, Ts...>(os, v, v.index());
}
template<typename... Ts>
IIStream& operator>> (IIStream& is, cv::util::variant<Ts...> &v) {
    int idx = -1;
    is >> idx;
    GAPI_Assert(idx >= 0 && idx < (int)sizeof...(Ts));
    return detail::get_v<cv::util::variant<Ts...>, Ts...>(is, v, 0u, idx);
}

// FIXME: consider a better solution
template<typename... Ts>
void getRunArgByIdx (IIStream& is, cv::util::variant<Ts...> &v, uint32_t idx) {
    is = detail::get_v<cv::util::variant<Ts...>, Ts...>(is, v, 0u, idx);
}
} // namespace s11n

namespace detail
{
template<typename T> struct try_deserialize_comparg;

template<> struct try_deserialize_comparg<std::tuple<>> {
static cv::util::optional<GCompileArg> exec(const std::string&, cv::gapi::s11n::IIStream&) {
        return { };
    }
};

template<typename T, typename... Types>
struct try_deserialize_comparg<std::tuple<T, Types...>> {
static cv::util::optional<GCompileArg> exec(const std::string& tag, cv::gapi::s11n::IIStream& is) {
    if (tag == cv::detail::CompileArgTag<T>::tag()) {
        static_assert(cv::gapi::s11n::detail::has_S11N_spec<T>::value,
            "cv::gapi::deserialize<GCompileArgs, Types...> expects Types to have S11N "
            "specializations with deserialization callbacks!");
        return cv::util::optional<GCompileArg>(
            GCompileArg { cv::gapi::s11n::detail::S11N<T>::deserialize(is) });
    }
    return try_deserialize_comparg<std::tuple<Types...>>::exec(tag, is);
}
};

template<typename T> struct deserialize_runarg;

template<typename RMatAdapterType>
struct deserialize_runarg {
static GRunArg exec(cv::gapi::s11n::IIStream& is, uint32_t idx) {
    if (idx == GRunArg::index_of<RMat>()) {
        auto ptr = std::make_shared<RMatAdapterType>();
        ptr->deserialize(is);
        return GRunArg { RMat(std::move(ptr)) };
    } else { // non-RMat arg - use default deserialization
        GRunArg arg;
        getRunArgByIdx(is, arg, idx);
        return arg;
    }
}
};

template<typename... Types>
inline cv::util::optional<GCompileArg> tryDeserializeCompArg(const std::string& tag,
                                                             const std::vector<char>& sArg) {
    std::unique_ptr<cv::gapi::s11n::IIStream> pArgIs = cv::gapi::s11n::detail::getInStream(sArg);
    return try_deserialize_comparg<std::tuple<Types...>>::exec(tag, *pArgIs);
}

template<typename... Types>
cv::GCompileArgs getCompileArgs(const std::vector<char> &sArgs) {
    cv::GCompileArgs args;

    std::unique_ptr<cv::gapi::s11n::IIStream> pIs = cv::gapi::s11n::detail::getInStream(sArgs);
    cv::gapi::s11n::IIStream& is = *pIs;

    uint32_t sz = 0;
    is >> sz;
    for (uint32_t i = 0; i < sz; ++i) {
        std::string tag;
        is >> tag;

        std::vector<char> sArg;
        is >> sArg;

        cv::util::optional<GCompileArg> dArg =
            cv::gapi::detail::tryDeserializeCompArg<Types...>(tag, sArg);

        if (dArg.has_value())
        {
            args.push_back(dArg.value());
        }
    }

    return args;
}

template<typename RMatAdapterType>
cv::GRunArgs getRunArgsWithRMats(const std::vector<char> &p) {
    std::unique_ptr<cv::gapi::s11n::IIStream> pIs = cv::gapi::s11n::detail::getInStream(p);
    cv::gapi::s11n::IIStream& is = *pIs;
    cv::GRunArgs args;

    uint32_t sz = 0;
    is >> sz;
    for (uint32_t i = 0; i < sz; ++i) {
        uint32_t idx = 0;
        is >> idx;
        args.push_back(cv::gapi::detail::deserialize_runarg<RMatAdapterType>::exec(is, idx));
    }

    return args;
}
} // namespace detail
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_S11N_HPP
