// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#ifndef OPENCV_GAPI_GTYPE_TRAITS_HPP
#define OPENCV_GAPI_GTYPE_TRAITS_HPP

#include <vector>
#include <type_traits>

#include <opencv2/gapi/gmat.hpp>
#include <opencv2/gapi/gscalar.hpp>
#include <opencv2/gapi/garray.hpp>
#include <opencv2/gapi/gopaque.hpp>
#include <opencv2/gapi/gframe.hpp>
#include <opencv2/gapi/streaming/source.hpp>
#include <opencv2/gapi/media.hpp>
#include <opencv2/gapi/gcommon.hpp>

namespace cv
{
namespace detail
{
    // FIXME: These traits and enum and possible numerous switch(kind)
    // block may be replaced with a special Handler<T> object or with
    // a double dispatch
    enum class ArgKind: int
    {
        OPAQUE_VAL,   // Unknown, generic, opaque-to-GAPI data type - STATIC
                      // Note: OPAQUE is sometimes defined in Win sys headers
#if !defined(OPAQUE) && !defined(CV_DOXYGEN)
        OPAQUE = OPAQUE_VAL,  // deprecated value used for compatibility, use OPAQUE_VAL instead
#endif
        GOBJREF,      // <internal> reference to object
        GMAT,         // a cv::GMat
        GMATP,        // a cv::GMatP
        GFRAME,       // a cv::GFrame
        GSCALAR,      // a cv::GScalar
        GARRAY,       // a cv::GArrayU  (note - exactly GArrayU,  not GArray<T>!)
        GOPAQUE,      // a cv::GOpaqueU (note - exactly GOpaqueU, not GOpaque<T>!)
    };

    template<typename T>
    constexpr const char* meta_to_string() noexcept;
    template<>
    constexpr const char* meta_to_string<cv::GMatDesc>() noexcept { return "GMatDesc"; }
    template<>
    constexpr const char* meta_to_string<cv::GScalarDesc>() noexcept { return "GScalarDesc"; }
    template<>
    constexpr const char* meta_to_string<cv::GArrayDesc>() noexcept { return "GArrayDesc"; }
    template<>
    constexpr const char* meta_to_string<cv::GOpaqueDesc>() noexcept { return "GOpaqueDesc"; }
    template<>
    constexpr const char* meta_to_string<cv::GFrameDesc>() noexcept { return "GFrameDesc";}

    // Describe G-API types (G-types) with traits.  Mostly used by
    // cv::GArg to store meta information about types passed into
    // operation arguments. Please note that cv::GComputation is
    // defined on GProtoArgs, not GArgs!
    template<typename T> struct GTypeTraits;
    template<typename T> struct GTypeTraits
    {
        static constexpr const ArgKind kind = ArgKind::OPAQUE_VAL;
        static constexpr const OpaqueKind op_kind = OpaqueKind::CV_UNKNOWN;
    };
    template<>           struct GTypeTraits<cv::GMat>
    {
        static constexpr const ArgKind kind = ArgKind::GMAT;
        static constexpr const GShape shape = GShape::GMAT;
        static constexpr const OpaqueKind op_kind = OpaqueKind::CV_UNKNOWN;
    };
    template<>           struct GTypeTraits<cv::GMatP>
    {
        static constexpr const ArgKind kind = ArgKind::GMATP;
        static constexpr const GShape shape = GShape::GMAT;
        static constexpr const OpaqueKind op_kind = OpaqueKind::CV_UNKNOWN;
    };
    template<>           struct GTypeTraits<cv::GFrame>
    {
        static constexpr const ArgKind kind = ArgKind::GFRAME;
        static constexpr const GShape shape = GShape::GFRAME;
        static constexpr const OpaqueKind op_kind = OpaqueKind::CV_UNKNOWN;
    };
    template<>           struct GTypeTraits<cv::GScalar>
    {
        static constexpr const ArgKind kind = ArgKind::GSCALAR;
        static constexpr const GShape shape = GShape::GSCALAR;
        static constexpr const OpaqueKind op_kind = OpaqueKind::CV_UNKNOWN;
    };
    template<class T> struct GTypeTraits<cv::GArray<T> >
    {
        static constexpr const ArgKind kind = ArgKind::GARRAY;
        static constexpr const GShape shape = GShape::GARRAY;
        static constexpr const OpaqueKind op_kind = GOpaqueTraits<T>::kind;
        using host_type  = std::vector<T>;
        using strip_type = cv::detail::VectorRef;
        static cv::detail::GArrayU   wrap_value(const cv::GArray<T>  &t) { return t.strip();}
        static cv::detail::VectorRef wrap_in   (const std::vector<T> &t) { return detail::VectorRef(t); }
        static cv::detail::VectorRef wrap_out  (      std::vector<T> &t) { return detail::VectorRef(t); }
    };
    template<class T> struct GTypeTraits<cv::GOpaque<T> >
    {
        static constexpr const ArgKind kind = ArgKind::GOPAQUE;
        static constexpr const GShape shape = GShape::GOPAQUE;
        static constexpr const OpaqueKind op_kind = GOpaqueTraits<T>::kind;
        using host_type  = T;
        using strip_type = cv::detail::OpaqueRef;
        static cv::detail::GOpaqueU  wrap_value(const cv::GOpaque<T>  &t) { return t.strip();}
        static cv::detail::OpaqueRef wrap_in   (const T &t) { return detail::OpaqueRef(t); }
        static cv::detail::OpaqueRef wrap_out  (      T &t) { return detail::OpaqueRef(t); }
    };

    // Tests if Trait for type T requires extra marshalling ("custom wrap") or not.
    // If Traits<T> has wrap_value() defined, it does.
    template<class T> struct has_custom_wrap
    {
        template<class,class> class check;
        template<typename C> static std::true_type  test(check<C, decltype(&GTypeTraits<C>::wrap_value)> *);
        template<typename C> static std::false_type test(...);
        using type = decltype(test<T>(nullptr));
        static const constexpr bool value = std::is_same<std::true_type, decltype(test<T>(nullptr))>::value;
    };

    // Resolve a Host type back to its associated G-Type.
    // FIXME: Probably it can be avoided
    // FIXME: GMatP is not present here.
    // (Actually these traits is used only to check
    // if associated G-type has custom wrap functions
    // and GMat behavior is correct for GMatP)
    template<typename T> struct GTypeOf;
#if !defined(GAPI_STANDALONE)
    template<>           struct GTypeOf<cv::UMat>              { using type = cv::GMat;      };
#endif // !defined(GAPI_STANDALONE)
    template<>           struct GTypeOf<cv::Mat>               { using type = cv::GMat;      };
    template<>           struct GTypeOf<cv::RMat>              { using type = cv::GMat;      };
    template<>           struct GTypeOf<cv::Scalar>            { using type = cv::GScalar;   };
    template<typename U> struct GTypeOf<std::vector<U> >       { using type = cv::GArray<U>; };
    template<typename U> struct GTypeOf                        { using type = cv::GOpaque<U>;};
    template<>           struct GTypeOf<cv::MediaFrame>        { using type = cv::GFrame;    };
    // FIXME: This is not quite correct since IStreamSource may produce not only Mat but also Scalar
    // and vector data. TODO: Extend the type dispatching on these types too.
    template<>           struct GTypeOf<cv::gapi::wip::IStreamSource::Ptr> { using type = cv::GMat;};
    template<class T> using g_type_of_t = typename GTypeOf<T>::type;

    // Marshalling helper for G-types and its Host types. Helps G-API
    // to store G types in internal generic containers for further
    // processing. Implements the following callbacks:
    //
    // * wrap() - converts user-facing G-type into an internal one
    //   for internal storage.
    //   Used when G-API operation is instantiated (G<Kernel>::on(),
    //   etc) during expressing a pipeline. Mostly returns input
    //   value "as is" except the case when G-type is a template. For
    //   template G-classes, calls custom wrap() from Traits.
    //   The value returned by wrap() is then wrapped into GArg() and
    //   stored in G-API metadata.
    //
    //   Example:
    //   - cv::GMat arguments are passed as-is.
    //   - integers, pointers, STL containers, user types are passed as-is.
    //   - cv::GArray<T> is converted to cv::GArrayU.
    //
    // * wrap_in() / wrap_out() - convert Host type associated with
    //   G-type to internal representation type.
    //
    //   - For "simple" (non-template) G-types, returns value as-is.
    //     Example: cv::GMat has host type cv::Mat, when user passes a
    //              cv::Mat, system stores it internally as cv::Mat.
    //
    //   - For "complex" (template) G-types, utilizes custom
    //     wrap_in()/wrap_out() as described in Traits.
    //     Example: cv::GArray<T> has host type std::vector<T>, when
    //              user passes a std::vector<T>, system stores it
    //              internally as VectorRef (with <T> stripped away).
    template<typename T, class Custom = void> struct WrapValue
    {
        static auto wrap(const T& t) ->
            typename std::remove_reference<T>::type
        {
            return static_cast<typename std::remove_reference<T>::type>(t);
        }

        template<typename U> static U  wrap_in (const U &u) { return  u;  }
        template<typename U> static U* wrap_out(U &u)       { return &u;  }
    };
    template<typename T> struct WrapValue<T, typename std::enable_if<has_custom_wrap<T>::value>::type>
    {
        static auto wrap(const T& t) -> decltype(GTypeTraits<T>::wrap_value(t))
        {
            return GTypeTraits<T>::wrap_value(t);
        }
        template<typename U> static auto wrap_in (const U &u) -> typename GTypeTraits<T>::strip_type
        {
            return GTypeTraits<T>::wrap_in(u);
        }
        template<typename U> static auto wrap_out(U &u) -> typename GTypeTraits<T>::strip_type
        {
            return GTypeTraits<T>::wrap_out(u);
        }
    };

    template<typename T> using wrap_gapi_helper = WrapValue<typename std::decay<T>::type>;
    template<typename T> using wrap_host_helper = WrapValue<typename std::decay<g_type_of_t<T> >::type>;

// Union type for various user-defined type constructors (GArray<T>,
// GOpaque<T>, etc)
//
// TODO: Replace construct-only API with a more generic one (probably
//    with bits of introspection)
//
// Not required for non-user-defined types (GMat, GScalar, etc)
using HostCtor = util::variant
    < util::monostate
    , detail::ConstructVec
    , detail::ConstructOpaque
    >;

template<typename T> struct GObtainCtor {
    static HostCtor get() { return HostCtor{}; }
};
template<typename T> struct GObtainCtor<GArray<T> > {
    static HostCtor get() { return HostCtor{ConstructVec{&GArray<T>::VCtor}}; };
};
template<typename T> struct GObtainCtor<GOpaque<T> > {
    static HostCtor get() { return HostCtor{ConstructOpaque{&GOpaque<T>::Ctor}}; };
};
} // namespace detail
} // namespace cv

#endif // OPENCV_GAPI_GTYPE_TRAITS_HPP
