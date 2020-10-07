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

namespace cv {
namespace gapi {

namespace detail {
    GAPI_EXPORTS cv::GComputation getGraph(const std::vector<char> &p);

    GAPI_EXPORTS cv::GMetaArgs getMetaArgs(const std::vector<char> &p);

    GAPI_EXPORTS cv::GRunArgs getRunArgs(const std::vector<char> &p);

    template<typename... Types>
    cv::GCompileArgs getCompileArgs(const std::vector<char> &p);
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
} // namespace s11n

namespace detail
{
template<typename T> struct deserialize_arg;

template<> struct deserialize_arg<std::tuple<>> {
static GCompileArg exec(cv::gapi::s11n::IIStream&, const std::string&) {
        throw std::logic_error("Passed arg can't be deserialized!");
    }
};

template<typename T, typename... Types>
struct deserialize_arg<std::tuple<T, Types...>> {
static GCompileArg exec(cv::gapi::s11n::IIStream& is, const std::string& tag) {
    if (tag == cv::detail::CompileArgTag<T>::tag()) {
        return GCompileArg {
            cv::gapi::s11n::detail::S11N<T>::deserialize(is)
        };
    }

    return deserialize_arg<std::tuple<Types...>>::exec(is, tag);
}
};

template<typename... Types>
cv::GCompileArgs getCompileArgs(const std::vector<char> &p) {
    std::unique_ptr<cv::gapi::s11n::IIStream> pIs = cv::gapi::s11n::detail::getInStream(p);
    cv::gapi::s11n::IIStream& is = *pIs;
    cv::GCompileArgs args;

    uint32_t sz = 0;
    is >> sz;
    for (uint32_t i = 0; i < sz; ++i) {
        std::string tag;
        is >> tag;
        args.push_back(cv::gapi::detail::deserialize_arg<std::tuple<Types...>>::exec(is, tag));
    }

    return args;
}
} // namespace detail
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_S11N_HPP
