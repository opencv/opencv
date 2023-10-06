// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#ifndef OPENCV_GAPI_GMAT_HPP
#define OPENCV_GAPI_GMAT_HPP

#include <ostream>
#include <memory>                 // std::shared_ptr

#include <opencv2/gapi/opencv_includes.hpp>
#include <opencv2/gapi/gcommon.hpp> // GShape

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
 * associated values like with cv::GScalar or `cv::GArray<T>`) and are
 * used only to construct graphs.
 *
 * Every graph in G-API starts and ends with data objects.
 *
 * Once constructed and compiled, G-API operates with regular host-side
 * data instead. Refer to the below table to find the mapping between
 * G-API and regular data types when passing input and output data
 * structures to G-API:
 *
 *    G-API data type    | I/O data type
 *    ------------------ | -------------
 *    cv::GMat           | cv::Mat, cv::UMat, cv::RMat
 *    cv::GScalar        | cv::Scalar
 *    `cv::GArray<T>`    | std::vector<T>
 *    `cv::GOpaque<T>`   | T
 *    cv::GFrame         | cv::MediaFrame
 */

/**
 * @brief GMat class represents image or tensor data in the
 * graph.
 *
 * GMat doesn't store any data itself, instead it describes a
 * functional relationship between operations consuming and producing
 * GMat objects.
 *
 * GMat is a virtual counterpart of Mat and UMat, but it
 * doesn't mean G-API use Mat or UMat objects internally to represent
 * GMat objects -- the internal data representation may be
 * backend-specific or optimized out at all.
 *
 * @sa Mat, GMatDesc
 */
class GAPI_EXPORTS_W_SIMPLE GMat
{
public:
    /**
     * @brief Constructs an empty GMat
     *
     * Normally, empty G-API data objects denote a starting point of
     * the graph. When an empty GMat is assigned to a result of some
     * operation, it obtains a functional link to this operation (and
     * is not empty anymore).
     */
    GAPI_WRAP GMat();                       // Empty constructor

    /// @private
    GMat(const GNode &n, std::size_t out);  // Operation result constructor
    /// @private
    GOrigin& priv();                        // Internal use only
    /// @private
    const GOrigin& priv()  const;           // Internal use only

private:
    std::shared_ptr<GOrigin> m_priv;
};

class GAPI_EXPORTS GMatP : public GMat
{
public:
    using GMat::GMat;
};

class RMat;

/** @} */

/**
 * \addtogroup gapi_meta_args
 * @{
 */
struct GAPI_EXPORTS_W_SIMPLE GMatDesc
{
    // FIXME: Default initializers in C++14
    GAPI_PROP int depth;
    GAPI_PROP int chan;
    GAPI_PROP cv::Size size; // NB.: no multi-dimensional cases covered yet
    GAPI_PROP bool planar;
    GAPI_PROP std::vector<int> dims; // FIXME: Maybe it's real questionable to have it here

    GAPI_WRAP GMatDesc(int d, int c, cv::Size s, bool p = false)
        : depth(d), chan(c), size(s), planar(p) {}

    GAPI_WRAP GMatDesc(int d, const std::vector<int> &dd)
        : depth(d), chan(-1), size{-1,-1}, planar(false), dims(dd) {}

    GAPI_WRAP GMatDesc(int d, std::vector<int> &&dd)
        : depth(d), chan(-1), size{-1,-1}, planar(false), dims(std::move(dd)) {}

    GAPI_WRAP GMatDesc() : GMatDesc(-1, -1, {-1,-1}) {}

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
    bool canDescribe(const cv::Mat& mat) const;

    bool canDescribe(const cv::RMat& mat) const;

    // Meta combinator: return a new GMatDesc which differs in size by delta
    // (all other fields are taken unchanged from this GMatDesc)
    // FIXME: a better name?
    GAPI_WRAP GMatDesc withSizeDelta(cv::Size delta) const
    {
        GMatDesc desc(*this);
        desc.size += delta;
        return desc;
    }
    // Meta combinator: return a new GMatDesc which differs in size by delta
    // (all other fields are taken unchanged from this GMatDesc)
    //
    // This is an overload.
    GAPI_WRAP GMatDesc withSizeDelta(int dx, int dy) const
    {
        return withSizeDelta(cv::Size{dx,dy});
    }

    GAPI_WRAP GMatDesc withSize(cv::Size sz) const
    {
        GMatDesc desc(*this);
        desc.size = sz;
        return desc;
    }

    // Meta combinator: return a new GMatDesc with specified data depth.
    // (all other fields are taken unchanged from this GMatDesc)
    GAPI_WRAP GMatDesc withDepth(int ddepth) const
    {
        GAPI_Assert(CV_MAT_CN(ddepth) == 1 || ddepth == -1);
        GMatDesc desc(*this);
        if (ddepth != -1) desc.depth = ddepth;
        return desc;
    }

    // Meta combinator: return a new GMatDesc with specified data depth
    // and number of channels.
    // (all other fields are taken unchanged from this GMatDesc)
    GAPI_WRAP GMatDesc withType(int ddepth, int dchan) const
    {
        GAPI_Assert(CV_MAT_CN(ddepth) == 1 || ddepth == -1);
        GMatDesc desc = withDepth(ddepth);
        desc.chan = dchan;
        return desc;
    }

    // Meta combinator: return a new GMatDesc with planar flag set
    // (no size changes are performed, only channel interpretation is changed
    // (interleaved -> planar)
    GAPI_WRAP GMatDesc asPlanar() const
    {
        GAPI_Assert(planar == false);
        GMatDesc desc(*this);
        desc.planar = true;
        return desc;
    }

    // Meta combinator: return a new GMatDesc
    // reinterpreting 1-channel input as planar image
    // (size height is divided by plane number)
    GAPI_WRAP GMatDesc asPlanar(int planes) const
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
    GAPI_WRAP GMatDesc asInterleaved() const
    {
        GAPI_Assert(planar == true);
        GMatDesc desc(*this);
        desc.planar = false;
        return desc;
    }
};

static inline GMatDesc empty_gmat_desc() { return GMatDesc{-1,-1,{-1,-1}}; }

namespace gapi { namespace detail {
/** Checks GMatDesc fields if the passed matrix is a set of n-dimentional points.
@param in GMatDesc to check.
@param n expected dimensionality.
@return the amount of points. In case input matrix can't be described as vector of points
of expected dimensionality, returns -1.
 */
int checkVector(const GMatDesc& in, const size_t n);

/** @overload

Checks GMatDesc fields if the passed matrix can be described as a set of points of any
dimensionality.

@return array of two elements in form of std::vector<int>: the amount of points
and their calculated dimensionality. In case input matrix can't be described as vector of points,
returns {-1, -1}.
 */
std::vector<int> checkVector(const GMatDesc& in);
}} // namespace gapi::detail

#if !defined(GAPI_STANDALONE)
GAPI_EXPORTS GMatDesc descr_of(const cv::UMat &mat);
#endif // !defined(GAPI_STANDALONE)

//Fwd declarations
namespace gapi { namespace own {
    class Mat;
    GAPI_EXPORTS GMatDesc descr_of(const Mat &mat);
}}//gapi::own

GAPI_EXPORTS GMatDesc descr_of(const RMat &mat);

#if !defined(GAPI_STANDALONE)
GAPI_EXPORTS GMatDesc descr_of(const cv::Mat &mat);
#else
using gapi::own::descr_of;
#endif

/** @} */

GAPI_EXPORTS std::ostream& operator<<(std::ostream& os, const cv::GMatDesc &desc);

} // namespace cv

#endif // OPENCV_GAPI_GMAT_HPP
