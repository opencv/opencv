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

#include "opencv2/gapi/own/types.hpp" // cv::gapi::own::Size
#include "opencv2/gapi/own/convert.hpp" // to_own
#include "opencv2/gapi/own/assert.hpp"

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
 * @brief Data-representing objects which can be used to build G-API
 * expressions.
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

    inline bool operator== (const GMatDesc &rhs) const
    {
        return depth == rhs.depth && chan == rhs.chan && size == rhs.size;
    }

    inline bool operator!= (const GMatDesc &rhs) const
    {
        return !(*this == rhs);
    }

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
};

static inline GMatDesc empty_gmat_desc() { return GMatDesc{-1,-1,{-1,-1}}; }

#if !defined(GAPI_STANDALONE)
class Mat;
GAPI_EXPORTS GMatDesc descr_of(const cv::Mat &mat);
GAPI_EXPORTS GMatDesc descr_of(const cv::UMat &mat);
#endif // !defined(GAPI_STANDALONE)

/** @} */

namespace gapi { namespace own {
    class Mat;
    GAPI_EXPORTS GMatDesc descr_of(const Mat &mat);
}}//gapi::own

GAPI_EXPORTS std::ostream& operator<<(std::ostream& os, const cv::GMatDesc &desc);

} // namespace cv

#endif // OPENCV_GAPI_GMAT_HPP
