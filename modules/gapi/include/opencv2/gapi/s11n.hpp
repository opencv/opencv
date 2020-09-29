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

    // Generic stl structures
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

    // Generic stl structures
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
    struct GAPI_EXPORTS S11N {
        static void serialize(I::OStream &os, const T &p) {}
        static T deserialize(I::IStream &is) { T t; return t; }
        static constexpr const bool isSupported = false; // type T can be serialized
    };
} // namespace detail
} // namespace s11n
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_S11N_HPP
