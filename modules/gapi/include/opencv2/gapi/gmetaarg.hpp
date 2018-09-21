// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_GMETAARG_HPP
#define OPENCV_GAPI_GMETAARG_HPP

#include <vector>
#include <type_traits>

#include "opencv2/gapi/util/util.hpp"
#include "opencv2/gapi/util/variant.hpp"

#include "opencv2/gapi/gmat.hpp"
#include "opencv2/gapi/gscalar.hpp"
#include "opencv2/gapi/garray.hpp"

namespace cv
{
// FIXME: Rename to GMeta?
// FIXME: user shouldn't deal with it - put to detail?
// GMetaArg is an union type over descriptions of G-types which can serve as
// GComputation's in/output slots.
//
// GMetaArg objects are passed as arguments to GComputation::compile()
// to specify which data a compiled computation should be specialized on.
// For manual compile(), user must supply this metadata, in case of apply()
// this metadata is taken from arguments computation should operate on.
//
// The first type (monostate) is equal to "uninitialized"/"unresolved" meta.
using GMetaArg = util::variant
    < util::monostate
    , GMatDesc
    , GScalarDesc
    , GArrayDesc
    >;
std::ostream& operator<<(std::ostream& os, const GMetaArg &);

using GMetaArgs = std::vector<GMetaArg>;

namespace detail
{
    // These traits are used by GComputation::compile()

    // FIXME: is_constructible<T> doesn't work as variant doesn't do any SFINAE
    // in its current template constructor

    template<typename T> struct is_meta_descr    : std::false_type {};
    template<> struct is_meta_descr<GMatDesc>    : std::true_type {};
    template<> struct is_meta_descr<GScalarDesc> : std::true_type {};
    template<> struct is_meta_descr<GArrayDesc>  : std::true_type {};

    template<typename... Ts>
    using are_meta_descrs = all_satisfy<is_meta_descr, Ts...>;

    template<typename... Ts>
    using are_meta_descrs_but_last = all_satisfy<is_meta_descr, typename all_but_last<Ts...>::type>;

} // namespace detail

} // namespace cv

#endif // OPENCV_GAPI_GMETAARG_HPP
