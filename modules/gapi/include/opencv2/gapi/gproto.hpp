// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_GPROTO_HPP
#define OPENCV_GAPI_GPROTO_HPP

#include <type_traits>
#include <vector>
#include <ostream>

#include "opencv2/gapi/util/variant.hpp"

#include "opencv2/gapi/gmat.hpp"
#include "opencv2/gapi/gscalar.hpp"
#include "opencv2/gapi/garray.hpp"
#include "opencv2/gapi/garg.hpp"
#include "opencv2/gapi/gmetaarg.hpp"

namespace cv {

// FIXME: user shouldn't deal with it - put to detail?
// GProtoArg is an union type over G-types which can serve as
// GComputation's in/output slots. In other words, GProtoArg
// wraps any type which can serve as G-API exchange type.
//
// In Runtime, GProtoArgs are substituted with appropriate GRunArgs.
//
// GProtoArg objects are constructed in-place when user describes
// (captures) computations, user doesn't interact with these types
// directly.
using GProtoArg = util::variant
    < GMat
    , GScalar
    , detail::GArrayU // instead of GArray<T>
    >;

using GProtoArgs = std::vector<GProtoArg>;

namespace detail
{
template<typename... Ts> inline GProtoArgs packArgs(Ts... args)
{
    return GProtoArgs{ GProtoArg(wrap_gapi_helper<Ts>::wrap(args))... };
}

}

template<class Tag>
struct GIOProtoArgs
{
public:
    explicit GIOProtoArgs(const GProtoArgs& args) : m_args(args) {}
    explicit GIOProtoArgs(GProtoArgs &&args)      : m_args(std::move(args)) {}

    GProtoArgs m_args;
};

struct In_Tag{};
struct Out_Tag{};

using GProtoInputArgs  = GIOProtoArgs<In_Tag>;
using GProtoOutputArgs = GIOProtoArgs<Out_Tag>;

// Perfect forwarding
template<typename... Ts> inline GProtoInputArgs GIn(Ts&&... ts)
{
    return GProtoInputArgs(detail::packArgs(std::forward<Ts>(ts)...));
}

template<typename... Ts> inline GProtoOutputArgs GOut(Ts&&... ts)
{
    return GProtoOutputArgs(detail::packArgs(std::forward<Ts>(ts)...));
}

// Extract run-time arguments from node origin
// Can be used to extract constant values associated with G-objects
// (like GScalar) at graph construction time
GRunArg value_of(const GOrigin &origin);

// Transform run-time computation arguments into a collection of metadata
// extracted from that arguments
GMetaArg  GAPI_EXPORTS descr_of(const GRunArg  &arg );
GMetaArgs GAPI_EXPORTS descr_of(const GRunArgs &args);

// Transform run-time operation result argument into metadata extracted from that argument
// Used to compare the metadata, which generated at compile time with the metadata result operation in run time
GMetaArg  GAPI_EXPORTS descr_of(const GRunArgP& argp);


} // namespace cv

#endif // OPENCV_GAPI_GPROTO_HPP
