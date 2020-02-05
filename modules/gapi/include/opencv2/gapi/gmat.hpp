// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_GMAT_HPP
#define OPENCV_GAPI_GMAT_HPP

#include <ostream>
#include <memory>                 // std::shared_ptr

#include <opencv2/gapi/opencv_includes.hpp>
#include <opencv2/gapi/gcommon.hpp> // GShape

#include <opencv2/gapi/own/types.hpp> // cv::gapi::own::Size
#include <opencv2/gapi/own/convert.hpp> // to_own
#include <opencv2/gapi/own/assert.hpp>

// TODO GAPI_EXPORTS or so
namespace cv
{
// Forward declaration; GNode and GOrigin are an internal
// (user-inaccessible) classes.
class GNode;
struct GOrigin;

/** \addtogroup gapi_data_objects
 * @{
 *
 * @brief G-API data objects used to build G-API expressions.
 *
 * These objects do not own any particular data (except compile-time
 * associated values like with cv::GScalar) and are used to construct
 * graphs.
 *
 * Every graph in G-API starts and ends with data objects.
 *
 * Once constructed and compiled, G-API operates with regular host-side
 * data instead. Refer to the below table to find the mapping between
 * G-API and regular data types.
 *
 *    G-API data type    | I/O data type
 *    ------------------ | -------------
 *    cv::GMat           | cv::Mat
 *    cv::GScalar        | cv::Scalar
 *    `cv::GArray<T>`    | std::vector<T>
 *    `cv::GOpaque<T>`   | T
 */
class GAPI_EXPORTS GMat
{
public:
    GMat();                                 // Empty constructor
    GMat(const GNode &n, std::size_t out);  // Operation result constructor

    GOrigin& priv();                        // Internal use only
    const GOrigin& priv()  const;           // Internal use only

private:
    std::shared_ptr<GOrigin> m_priv;
};

class GAPI_EXPORTS GMatP : public GMat
{
public:
    using GMat::GMat;
};

namespace gapi { namespace own {
    class Mat;
}}//gapi::own

/** @} */

/**
 * \addtogroup gapi_meta_args
 * @{
 */
struct GAPI_EXPORTS GMatDesc
{
    // FIXME: Default initializers in C++14
    int depth;
    int chan;
    cv::gapi::own::Size size; // NB.: no multi-dimensional cases covered yet
    bool planar;
    std::vector<int> dims; // FIXME: Maybe it's real questionable to have it here

    GMatDesc(int d, int c, cv::gapi::own::Size s, bool p = false)
        : depth(d), chan(c), size(s), planar(p) {}

    GMatDesc(int d, const std::vector<int> &dd)
        : depth(d), chan(-1), size{-1,-1}, planar(false), dims(dd) {}

    GMatDesc(int d, std::vector<int> &&dd)
        : depth(d), chan(-1), size{-1,-1}, planar(false), dims(std::move(dd)) {}

    GMatDesc() : GMatDesc(-1, -1, {-1,-1}) {}

    inline bool operator== (const GMatDesc &rhs) const
    {
        return    depth  == rhs.depth
               && chan   == rhs.chan
               && size   == rhs.size
               && planar == rhs.planar
               && dims   == rhs.dims;
    }

    inline bool operator!= (const GMatDesc &rhs) const
    {
        return !(*this == rhs);
    }

    bool isND() const { return !dims.empty(); }

    // Checks if the passed mat can be described by this descriptor
    // (it handles the case when
    // 1-channel mat can be reinterpreted as is (1-channel mat)
    // and as a 3-channel planar mat with height divided by 3)
    bool canDescribe(const cv::gapi::own::Mat& mat) const;

    // Meta combinator: return a new GMatDesc which differs in size by delta
    // (all other fields are taken unchanged from this GMatDesc)
    // FIXME: a better name?
    GMatDesc withSizeDelta(cv::gapi::own::Size delta) const
    {
        GMatDesc desc(*this);
        desc.size += delta;
        return desc;
    }
#if !defined(GAPI_STANDALONE)
    GMatDesc withSizeDelta(cv::Size delta) const
    {
        return withSizeDelta(to_own(delta));
    }

    GMatDesc withSize(cv::Size sz) const
    {
        return withSize(to_own(sz));
    }

    bool canDescribe(const cv::Mat& mat) const;
#endif // !defined(GAPI_STANDALONE)
    // Meta combinator: return a new GMatDesc which differs in size by delta
    // (all other fields are taken unchanged from this GMatDesc)
    //
    // This is an overload.
    GMatDesc withSizeDelta(int dx, int dy) const
    {
        return withSizeDelta(cv::gapi::own::Size{dx,dy});
    }

    GMatDesc withSize(cv::gapi::own::Size sz) const
    {
        GMatDesc desc(*this);
        desc.size = sz;
        return desc;
    }

    // Meta combinator: return a new GMatDesc with specified data depth.
    // (all other fields are taken unchanged from this GMatDesc)
    GMatDesc withDepth(int ddepth) const
    {
        GAPI_Assert(CV_MAT_CN(ddepth) == 1 || ddepth == -1);
        GMatDesc desc(*this);
        if (ddepth != -1) desc.depth = ddepth;
        return desc;
    }

    // Meta combinator: return a new GMatDesc with specified data depth
    // and number of channels.
    // (all other fields are taken unchanged from this GMatDesc)
    GMatDesc withType(int ddepth, int dchan) const
    {
        GAPI_Assert(CV_MAT_CN(ddepth) == 1 || ddepth == -1);
        GMatDesc desc = withDepth(ddepth);
        desc.chan = dchan;
        return desc;
    }

    // Meta combinator: return a new GMatDesc with planar flag set
    // (no size changes are performed, only channel interpretation is changed
    // (interleaved -> planar)
    GMatDesc asPlanar() const
    {
        GAPI_Assert(planar == false);
        GMatDesc desc(*this);
        desc.planar = true;
        return desc;
    }

    // Meta combinator: return a new GMatDesc
    // reinterpreting 1-channel input as planar image
    // (size height is divided by plane number)
    GMatDesc asPlanar(int planes) const
    {
        GAPI_Assert(planar == false);
        GAPI_Assert(chan == 1);
        GAPI_Assert(planes > 1);
        GAPI_Assert(size.height % planes == 0);
        GMatDesc desc(*this);
        desc.size.height /=  planes;
        desc.chan = planes;
        return desc.asPlanar();
    }

    // Meta combinator: return a new GMatDesc with planar flag set to false
    // (no size changes are performed, only channel interpretation is changed
    // (planar -> interleaved)
    GMatDesc asInterleaved() const
    {
        GAPI_Assert(planar == true);
        GMatDesc desc(*this);
        desc.planar = false;
        return desc;
    }
};

static inline GMatDesc empty_gmat_desc() { return GMatDesc{-1,-1,{-1,-1}}; }

#if !defined(GAPI_STANDALONE)
class Mat;
GAPI_EXPORTS GMatDesc descr_of(const cv::Mat &mat);
GAPI_EXPORTS GMatDesc descr_of(const cv::UMat &mat);
#endif // !defined(GAPI_STANDALONE)

/** @} */

// FIXME: WHY??? WHY it is under different namespace?
namespace gapi { namespace own {
    GAPI_EXPORTS GMatDesc descr_of(const Mat &mat);
}}//gapi::own

GAPI_EXPORTS std::ostream& operator<<(std::ostream& os, const cv::GMatDesc &desc);

} // namespace cv

#endif // OPENCV_GAPI_GMAT_HPP
