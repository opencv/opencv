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
#include "opencv2/gapi/own/scalar.hpp"

namespace cv
{
// Forward declaration; GNode and GOrigin are an internal
// (user-inaccessible) classes.
class GNode;
struct GOrigin;

/** \addtogroup gapi_data_objects
 * @{
 */

class GAPI_EXPORTS GScalar
{
public:
    GScalar();                                         // Empty constructor
    explicit GScalar(const cv::gapi::own::Scalar& s);  // Constant value constructor from cv::gapi::own::Scalar
    explicit GScalar(cv::gapi::own::Scalar&& s);       // Constant value move-constructor from cv::gapi::own::Scalar
#if !defined(GAPI_STANDALONE)
    explicit GScalar(const cv::Scalar& s);             // Constant value constructor from cv::Scalar
#endif  // !defined(GAPI_STANDALONE)
    GScalar(double v0);                                // Constant value constructor from double
    GScalar(const GNode &n, std::size_t out);          // Operation result constructor

    GOrigin& priv();                                   // Internal use only
    const GOrigin& priv()  const;                      // Internal use only

private:
    std::shared_ptr<GOrigin> m_priv;
};

/** @} */

/**
 * \addtogroup gapi_meta_args
 * @{
 */
struct GScalarDesc
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

static inline GScalarDesc empty_scalar_desc() { return GScalarDesc(); }

#if !defined(GAPI_STANDALONE)
GAPI_EXPORTS GScalarDesc descr_of(const cv::Scalar            &scalar);
#endif // !defined(GAPI_STANDALONE)
/** @} */

GAPI_EXPORTS GScalarDesc descr_of(const cv::gapi::own::Scalar &scalar);

std::ostream& operator<<(std::ostream& os, const cv::GScalarDesc &desc);

} // namespace cv

#endif // OPENCV_GAPI_GSCALAR_HPP
