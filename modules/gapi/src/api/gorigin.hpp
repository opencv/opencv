// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_GORIGIN_HPP
#define OPENCV_GAPI_GORIGIN_HPP

#include <set>   // set
#include <map>   // map

#include <opencv2/gapi/util/variant.hpp>   // variant
#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/opencv_includes.hpp>

#include "compiler/gobjref.hpp"
#include "api/gnode.hpp"

namespace cv
{

// TODO namespace gimpl?

struct GOrigin
{
    static constexpr const std::size_t INVALID_PORT = std::numeric_limits<std::size_t>::max();

    GOrigin(GShape s,
            const GNode& n,
            std::size_t p = INVALID_PORT,
            const gimpl::HostCtor h = {});
    GOrigin(GShape s, gimpl::ConstVal value);

    const GShape          shape;           // Shape of a produced object
    const GNode           node;            // a GNode which produces an object
    const gimpl::ConstVal value;           // Node can have initial constant value, now only scalar is supported
    const std::size_t     port;            // GNode's output number; FIXME: "= max_size" in C++14
    gimpl::HostCtor       ctor;            // FIXME: replace with an interface?
};

namespace detail
{
    struct GOriginCmp
    {
        bool operator() (const GOrigin &lhs, const GOrigin &rhs) const;
    };
} // namespace cv::details

// TODO introduce a hash on GOrigin and define this via unordered_ ?
using GOriginSet = std::set<GOrigin, detail::GOriginCmp>;
template<typename T> using GOriginMap = std::map<GOrigin, T, detail::GOriginCmp>;

} // namespace cv

#endif // OPENCV_GAPI_GORIGIN_HPP
