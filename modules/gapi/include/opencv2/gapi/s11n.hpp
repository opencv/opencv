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

namespace detail {
template<typename T> struct S11N;

template<typename T> struct wrap_serialize
{
    template<typename Q=T>
    static auto serialize_impl(I::OStream &os, util::any arg, int)
        -> decltype(os << std::declval<Q>(), void())
    {
        os << util::get<T>(arg);
    }

    template<typename Q=T>
    static auto serialize_impl(I::OStream &os, util::any arg, long)
        -> decltype(S11N<Q>::serialize(os, std::declval<Q>()), void())
    {
        S11N<Q>::serialize(os, util::get<T>(arg));
    }

    template<typename Q=T>
    static void serialize_impl(I::OStream &os, util::any arg, long long)
    {
        throw std::runtime_error("No serialization available for this type!");
    }

    static void serialize(I::OStream &os, util::any arg)
    {
        serialize_impl(os, arg, 0);
    }
};

} // namespace detail
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_S11N_HPP
