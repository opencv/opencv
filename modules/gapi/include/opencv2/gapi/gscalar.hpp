// This file is part of OpenCV project.

// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_GSCALAR_HPP
#define OPENCV_GAPI_GSCALAR_HPP

#include <ostream>

#include <opencv2/gapi/opencv_includes.hpp>
#include <opencv2/gapi/gcommon.hpp> // GShape
#include <opencv2/gapi/util/optional.hpp>

namespace cv
{
// Forward declaration; GNode and GOrigin are an internal
// (user-inaccessible) classes.
class GNode;
struct GOrigin;

/** \addtogroup gapi_data_objects
 * @{
 */
/**
 * @brief GScalar class represents cv::Scalar data in the graph.
 *
 * GScalar may be associated with a cv::Scalar value, which becomes
 * its constant value bound in graph compile time. cv::GScalar describes a
 * functional relationship between operations consuming and producing
 * GScalar objects.
 *
 * GScalar is a virtual counterpart of cv::Scalar, which is usually used
 * to represent the GScalar data in G-API during the execution.
 *
 * @sa Scalar
 */
class GAPI_EXPORTS_W_SIMPLE GScalar
{
public:
    /**
     * @brief Constructs an empty GScalar
     *
     * Normally, empty G-API data objects denote a starting point of
     * the graph. When an empty GScalar is assigned to a result of some
     * operation, it obtains a functional link to this operation (and
     * is not empty anymore).
     */
    GAPI_WRAP GScalar();

    /**
     * @brief Constructs a value-initialized GScalar
     *
     * In contrast with GMat (which can be either an explicit graph input
     * or a result of some operation), GScalars may have their values
     * be associated at graph construction time. It is useful when
     * some operation has a GScalar input which doesn't change during
     * the program execution, and is set only once. In this case,
     * there is no need to declare such GScalar as a graph input.
     *
     * @note The value of GScalar may be overwritten by assigning some
     * other GScalar to the object using `operator=` -- on the
     * assignment, the old GScalar value is discarded.
     *
     * @param s a cv::Scalar value to associate with this GScalar object.
     */
    GAPI_WRAP
    explicit GScalar(const cv::Scalar& s);

    /**
     * @overload
     * @brief Constructs a value-initialized GScalar
     *
     * @param s a cv::Scalar value to associate with this GScalar object.
     */
    explicit GScalar(cv::Scalar&& s);       // Constant value move-constructor from cv::Scalar

    /**
     * @overload
     * @brief Constructs a value-initialized GScalar
     *
     * @param v0 A `double` value to associate with this GScalar. Note
     *  that only the first component of a four-component cv::Scalar is
     *  set to this value, with others remain zeros.
     *
     * This constructor overload is not marked `explicit` and can be
     * used in G-API expression code like this:
     *
     * @snippet samples/cpp/tutorial_code/gapi/doc_snippets/api_ref_snippets.cpp gscalar_implicit
     *
     * Here operator+(GMat,GScalar) is used to wrap cv::gapi::addC()
     * and a value-initialized GScalar is created on the fly.
     *
     * @overload
     */
    GScalar(double v0);                                // Constant value constructor from double

    /// @private
    GScalar(const GNode &n, std::size_t out);          // Operation result constructor
    /// @private
    GOrigin& priv();                                   // Internal use only
    /// @private
    const GOrigin& priv()  const;                      // Internal use only

private:
    std::shared_ptr<GOrigin> m_priv;
};

/** @} */

/**
 * \addtogroup gapi_meta_args
 * @{
 */
struct GAPI_EXPORTS_W_SIMPLE GScalarDesc
{
    // NB.: right now it is empty

    inline bool operator== (const GScalarDesc &) const
    {
        return true; // NB: implement this method if GScalar meta appears
    }

    inline bool operator!= (const GScalarDesc &rhs) const
    {
        return !(*this == rhs);
    }
};

GAPI_EXPORTS_W inline GScalarDesc empty_scalar_desc() { return GScalarDesc(); }

GAPI_EXPORTS GScalarDesc descr_of(const cv::Scalar &scalar);

std::ostream& operator<<(std::ostream& os, const cv::GScalarDesc &desc);

} // namespace cv

#endif // OPENCV_GAPI_GSCALAR_HPP
