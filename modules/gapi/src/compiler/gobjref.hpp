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
    // HostCtor was there, but then moved to public
    // Redeclare here to avoid changing tons of code
    using HostCtor = cv::detail::HostCtor;

    using ConstVal = util::variant
    < util::monostate
    , cv::Scalar
    , cv::detail::VectorRef
    , cv::Mat
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
