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
/**
 * @brief GFrame class represents an image or media frame in the graph.
 *
 * GFrame doesn't store any data itself, instead it describes a
 * functional relationship between operations consuming and producing
 * GFrame objects.
 *
 * GFrame is introduced to handle various media formats (e.g., NV12 or
 * I420) under the same type. Various image formats may differ in the
 * number of planes (e.g. two for NV12, three for I420) and the pixel
 * layout inside. GFrame type allows to handle these media formats in
 * the graph uniformly -- the graph structure will not change if the
 * media format changes, e.g. a different camera or decoder is used
 * with the same graph. G-API provides a number of operations which
 * operate directly on GFrame, like `infer<>()` or
 * renderFrame(); these operations are expected to handle different
 * media formats inside. There is also a number of accessor
 * operations like BGR(), Y(), UV() -- these operations provide
 * access to frame's data in the familiar cv::GMat form, which can be
 * used with the majority of the existing G-API operations. These
 * accessor functions may perform color space conversion on the fly if
 * the image format of the GFrame they are applied to differs from the
 * operation's semantic (e.g. the BGR() accessor is called on an NV12
 * image frame).
 *
 * GFrame is a virtual counterpart of cv::MediaFrame.
 *
 * @sa cv::MediaFrame, cv::GFrameDesc, BGR(), Y(), UV(), infer<>().
 */
class GAPI_EXPORTS_W_SIMPLE GFrame
{
public:
    /**
     * @brief Constructs an empty GFrame
     *
     * Normally, empty G-API data objects denote a starting point of
     * the graph. When an empty GFrame is assigned to a result of some
     * operation, it obtains a functional link to this operation (and
     * is not empty anymore).
     */
    GAPI_WRAP GFrame();                      // Empty constructor

    /// @private
    GFrame(const GNode &n, std::size_t out); // Operation result constructor
    /// @private
    GOrigin& priv();                         // Internal use only
    /// @private
    const GOrigin& priv()  const;            // Internal use only

private:
    std::shared_ptr<GOrigin> m_priv;
};
/** @} */

enum class MediaFormat: int
{
    BGR = 0,
    NV12,
    GRAY,
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

class MediaFrame;
GAPI_EXPORTS GFrameDesc descr_of(const MediaFrame &frame);

GAPI_EXPORTS std::ostream& operator<<(std::ostream& os, const cv::GFrameDesc &desc);

} // namespace cv

#endif // OPENCV_GAPI_GFRAME_HPP
