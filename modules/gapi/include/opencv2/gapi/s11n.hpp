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
#include <opencv2/gapi/gcommon.hpp>

namespace cv {
namespace gapi {

namespace detail {
    GAPI_EXPORTS cv::GComputation getGraph(const std::vector<char> &p);

    GAPI_EXPORTS cv::GMetaArgs getMetaArgs(const std::vector<char> &p);

    GAPI_EXPORTS cv::GRunArgs getRunArgs(const std::vector<char> &p);

    template<typename T>
    GAPI_EXPORTS GCompileArg deserialize_arg(cv::gapi::s11n::IIStream& is, const std::string &tag);

    template<typename T1, typename T2, typename... Types>
    GAPI_EXPORTS GCompileArg deserialize_arg(cv::gapi::s11n::IIStream& is, const std::string &tag);

    template<typename... Types>
    GAPI_EXPORTS cv::GCompileArgs getCompileArgs(const std::vector<char> &p);
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

GAPI_EXPORTS std::unique_ptr<IIStream> getInStream(const std::vector<char> &p);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// S11N operators
// Note: operators for basic types are defined in IIStream/IOStream

// OpenCV types ////////////////////////////////////////////////////////////////

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::Point &pt);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::Point &pt);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::Size &sz);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::Size &sz);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::Rect &rc);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::Rect &rc);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::Scalar &s);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::Scalar &s);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::Mat &m);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::Mat &m);

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

namespace detail
{
// Note: actual implementation is defined by user
template<typename T>
struct GAPI_EXPORTS S11N;


template<typename T> struct wrap_serialize<T, cv::GCompileArg> {
    static std::function<void(gapi::s11n::IOStream& os, const util::any& arg)> serialize;

private:
    template<typename> using sfinae_true = std::true_type;

    template<typename Q = T>
    static auto try_call_serialize(gapi::s11n::IOStream& os, const util::any& arg, int)
        -> sfinae_true<decltype(S11N<Q>::serialize(os, util::any_cast<Q>(arg)), void())> {

        S11N<Q>::serialize(os, util::any_cast<Q>(arg));
        return sfinae_true<void>{};
    }

    // FIXME: Add compile-time check that this can't be called for custom types.
    //        Such behaviour can alternatively be implemented in S11N<T> with std::enable_if
    //        contraining T to be types defined by framework.
    template<typename Q = T>
    static auto try_call_serialize(gapi::s11n::IOStream& os, const util::any& arg, long)
        -> sfinae_true<decltype(os << std::declval<const typename std::add_lvalue_reference<Q>::type>(), void())> {

        os << util::any_cast<Q>(arg);
        return sfinae_true<void>{};
    }

    template<typename Q = T>
    static std::false_type try_call_serialize(gapi::s11n::IOStream &os, const util::any& arg, ...);

    static void call_serialize(gapi::s11n::IOStream& os, const util::any& arg) {
        try_call_serialize<T>(os, arg, 0);
    }
};

template<typename T>
std::function<void(gapi::s11n::IOStream& os, const util::any& arg)>
wrap_serialize<T, cv::GCompileArg>::serialize =
        decltype(try_call_serialize(
                    std::declval<typename std::add_lvalue_reference<gapi::s11n::IOStream>::type>(),
                    std::declval<typename std::add_lvalue_reference<util::any>::type>(),
                    int()))::value
        ? &call_serialize : nullptr;


template<typename T> struct wrap_deserialize
{
private:
    template<typename Q = T>
    static auto call_deserialize(gapi::s11n::IIStream& is, int)
        -> decltype(S11N<Q>::deserialize(is)) {
        return S11N<Q>::deserialize(is);
    }

    // FIXME: Add compile-time check that this can't be called for custom types.
    //        Such behaviour can alternatively be implemented in S11N<T> with std::enable_if
    //        contraining T to be types defined by framework.
    template<typename Q = T>
    static auto call_deserialize(gapi::s11n::IIStream& is, long)
        -> decltype(is >> std::declval<typename std::add_lvalue_reference<Q>::type>(),
                    Q()) {
        Q obj;
        is >> obj;
        return obj;
    }

    template<typename Q = T>
    static cv::GCompileArg call_deserialize(gapi::s11n::IIStream&, ...) {
        return GCompileArg { };
    }

public:
    static auto deserialize(gapi::s11n::IIStream& is)
        -> decltype(call_deserialize<T>(is, 0)) {
        return call_deserialize<T>(is, 0);
    }
};
} // namespace detail
} // namespace s11n

namespace detail
{
template<typename T>
GAPI_EXPORTS GCompileArg deserialize_arg(cv::gapi::s11n::IIStream& is, const std::string& tag) {
    if (tag == cv::detail::CompileArgTag<T>::tag()) {
        return GCompileArg { cv::gapi::s11n::detail::wrap_deserialize<T>::deserialize(is) };
    }
    else {
        throw std::logic_error("No CompileArgTag<> specialization for some of passed types!");
    }
}

template<typename T1, typename T2, typename... Types>
GAPI_EXPORTS GCompileArg deserialize_arg(cv::gapi::s11n::IIStream& is, const std::string& tag) {
    if (tag == cv::detail::CompileArgTag<T1>::tag()) {
        return GCompileArg { cv::gapi::s11n::detail::wrap_deserialize<T1>::deserialize(is) };
    }
    else {
        return deserialize_arg<T2, Types...>(is, tag);
    }
}

template<typename... Types>
GAPI_EXPORTS cv::GCompileArgs getCompileArgs(const std::vector<char> &p) {
    std::unique_ptr<cv::gapi::s11n::IIStream> pIs = cv::gapi::s11n::getInStream(p);
    cv::gapi::s11n::IIStream& is = *pIs.get();
    cv::GCompileArgs args;

    uint32_t sz;
    is >> sz;
    for (uint32_t i = 0; i < sz; ++i) {
        std::string tag;
        is >> tag;
        args.push_back(cv::gapi::detail::deserialize_arg<Types...>(is, tag));
    }

    return args;
}
} // namespace detail
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_S11N_HPP
