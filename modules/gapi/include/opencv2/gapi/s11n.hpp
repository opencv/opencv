// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_S11N_HPP
#define OPENCV_GAPI_S11N_HPP

#include <vector>
#include <opencv2/gapi/gcomputation.hpp>

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
namespace gimpl {
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

namespace detail {
    // Will be used along with default types if possible in specific cases (compile args, etc)
    // Note: header only, implementation is defined by user
    template<typename T>
    struct GAPI_EXPORTS S11N {
        static void serialize(I::OStream &os, const T &p);
        static T deserialize(I::IStream &is);
        static bool isSupported() { return false; } // supports serialization for type T
    };
} // namespace detail
} // namespace s11n
} // namespace gimpl
} // namespace cv

#endif // OPENCV_GAPI_S11N_HPP
