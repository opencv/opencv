// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#ifndef OPENCV_GAPI_GFRAME_HPP
#define OPENCV_GAPI_GFRAME_HPP

#include <ostream>
#include <memory>                 // std::shared_ptr

#include <opencv2/gapi/opencv_includes.hpp>
#include <opencv2/gapi/gcommon.hpp> // GShape

#include <opencv2/gapi/gmat.hpp>
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
 */
class GAPI_EXPORTS_W_SIMPLE GFrame
{
public:
    GAPI_WRAP GFrame();                       // Empty constructor
    GFrame(const GNode &n, std::size_t out);  // Operation result constructor

    GOrigin& priv();                        // Internal use only
    const GOrigin& priv()  const;           // Internal use only

private:
    std::shared_ptr<GOrigin> m_priv;
};
/** @} */

enum class MediaFormat: int
{
    BGR = 0,
    NV12,
};

/**
 * \addtogroup gapi_meta_args
 * @{
 */
struct GAPI_EXPORTS GFrameDesc
{
    MediaFormat fmt;
    cv::Size size;

    bool operator== (const GFrameDesc &) const;
};
static inline GFrameDesc empty_gframe_desc() { return GFrameDesc{}; }
/** @} */

GAPI_EXPORTS std::ostream& operator<<(std::ostream& os, const cv::GFrameDesc &desc);

} // namespace cv

#endif // OPENCV_GAPI_GFRAME_HPP
