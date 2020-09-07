// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_GOBJREF_HPP
#define OPENCV_GAPI_GOBJREF_HPP

#include "opencv2/gapi/util/variant.hpp"
#include "opencv2/gapi/garg.hpp"

namespace cv
{

namespace gimpl
{
    // Union type for various user-defined type constructors (GArray<T>, GOpaque<T>, etc)
    // FIXME: Replace construct-only API with a more generic one
    //    (probably with bits of introspection)
    // Not required for non-user-defined types (GMat, GScalar, etc)
    using HostCtor = util::variant
    < util::monostate
    , detail::ConstructVec
    , detail::ConstructOpaque
    >;

    using ConstVal = util::variant
    < util::monostate
    , cv::Scalar
    >;

    struct RcDesc
    {
        int      id;      // id is unique but local to shape
        GShape   shape;   // pair <id,shape> IS the unique ID
        HostCtor ctor;    // FIXME: is it really used here? Or in <Data>?

        bool operator==(const RcDesc &rhs) const
        {
            // FIXME: ctor is not checked (should be?)
            return id == rhs.id && shape == rhs.shape;
        }

        bool operator< (const RcDesc &rhs) const
        {
            return (id == rhs.id) ? shape < rhs.shape : id < rhs.id;
        }
    };
} // gimpl

namespace detail
{
    template<> struct GTypeTraits<cv::gimpl::RcDesc>
    {
        static constexpr const ArgKind kind = ArgKind::GOBJREF;
    };
}

} // cv

#endif // OPENCV_GAPI_GOBJREF_HPP
