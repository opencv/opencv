// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_RENDER_HPP
#define OPENCV_GAPI_RENDER_HPP

#include <string>
#include <vector>

#include <opencv2/gapi/opencv_includes.hpp>
#include <opencv2/gapi/util/variant.hpp>
#include <opencv2/gapi/own/exports.hpp>
#include <opencv2/gapi/own/scalar.hpp>

namespace cv
{
namespace gapi
{
namespace wip
{
namespace draw
{

/** @brief Parameters for drawing a text string.

This structure is passed to cv::gapi::wip::draw::render function.

@param text Text string to be drawn.
@param org Bottom-left corner of the text string in the image.
@param ff Font type, see #HersheyFonts.
@param fs Font scale factor that is multiplied by the font-specific base size.
@param color Text color.
@param thick Thickness of the lines used to draw a text.
@param lt Line type. See #LineTypes
@param bottom_left_origin When true, the image data origin is at the bottom-left corner. Otherwise,
it is at the top-left corner.
 */
struct Text
{
    std::string text;
    cv::Point   org;

    int         ff;
    double      fs;
    cv::Scalar  color;
    int         thick;
    int         lt;
    bool        bottom_left_origin;
};

/** @brief Parameters for drawing a simple, thick, or filled up-right rectangle.

This structure is passed to cv::gapi::wip::draw::render function.

@param rect Coordinates of the rectangle
@param color Rectangle color or brightness (grayscale image).
@param thick Thickness of lines that make up the rectangle. Negative values, like #FILLED,
mean that the function has to draw a filled rectangle.
@param lt Type of the line. See #LineTypes
@param shift Number of fractional bits in the point coordinates.
 */
struct Rect
{
    cv::Rect   rect;
    cv::Scalar color;
    int        thick;
    int        lt;
    int        shift;
};

using Prim  = util::variant<Text, Rect>;
using Prims = std::vector<Prim>;

/** @brief Parameters for drawing a simple, thick, or filled up-right rectangle.

These parameters is passed to cv::gapi::wip::draw::render function.

@param rect Coordinates of the rectangle
@param color Rectangle color or brightness (grayscale image).
@param thick Thickness of lines that make up the rectangle. Negative values, like #FILLED,
mean that the function has to draw a filled rectangle.
@param lt Type of the line. See #LineTypes
@param shift Number of fractional bits in the point coordinates.
 */
GAPI_EXPORTS void render(cv::Mat& bgr, const Prims& prims);
GAPI_EXPORTS void render(cv::Mat& y_plane, cv::Mat& uv_plane , const Prims& prims);

} // namespace draw
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_RENDER_HPP
