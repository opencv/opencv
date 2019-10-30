// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019 Intel Corporation


#ifndef OPENCV_GAPI_RENDER_HPP
#define OPENCV_GAPI_RENDER_HPP

#include <string>
#include <vector>

#include <opencv2/gapi.hpp>

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

/**
 * A structure allows using freetype library for text rendering
 */
struct use_freetype
{
    /*@{*/
    std::string path; //!< The path to font file (.ttf)
    /*@{*/
};

/**
 * A structure to represent parameters for drawing a text string.
 */
struct Text
{
    /*@{*/
    std::string text;               //!< The text string to be drawn
    cv::Point   org;                //!< The bottom-left corner of the text string in the image
    int         ff;                 //!< The font type, see #HersheyFonts
    double      fs;                 //!< The font scale factor that is multiplied by the font-specific base size
    cv::Scalar  color;              //!< The text color
    int         thick;              //!< The thickness of the lines used to draw a text
    int         lt;                 //!< The line type. See #LineTypes
    bool        bottom_left_origin; //!< When true, the image data origin is at the bottom-left corner. Otherwise, it is at the top-left corner
    /*@{*/
};

/**
 * A structure to represent parameters for drawing a rectangle
 */
struct Rect
{
    cv::Rect   rect;  //!< Coordinates of the rectangle
    cv::Scalar color; //!< The rectangle color or brightness (grayscale image)
    int        thick; //!< The thickness of lines that make up the rectangle. Negative values, like #FILLED, mean that the function has to draw a filled rectangle
    int        lt;    //!< The type of the line. See #LineTypes
    int        shift; //!< The number of fractional bits in the point coordinates
};

/**
 * A structure to represent parameters for drawing a circle
 */
struct Circle
{
    cv::Point  center; //!< The center of the circle
    int        radius; //!< The radius of the circle
    cv::Scalar color;  //!< The color of the  circle
    int        thick;  //!< The thickness of the circle outline, if positive. Negative values, like #FILLED, mean that a filled circle is to be drawn
    int        lt;     //!< The Type of the circle boundary. See #LineTypes
    int        shift;  //!< The Number of fractional bits in the coordinates of the center and in the radius value
};

/**
 * A structure to represent parameters for drawing a line
 */
struct Line
{
    cv::Point  pt1;    //!< The first point of the line segment
    cv::Point  pt2;    //!< The second point of the line segment
    cv::Scalar color;  //!< The line color
    int        thick;  //!< The thickness of line
    int        lt;     //!< The Type of the line. See #LineTypes
    int        shift;  //!< The number of fractional bits in the point coordinates
};

/**
 * A structure to represent parameters for drawing a mosaic
 */
struct Mosaic
{
    cv::Rect   mos;    //!< Coordinates of the mosaic
    int        cellSz; //!< Cell size (same for X, Y). Note: mos size must be multiple of cell size
    int        decim;  //!< Decimation (0 stands for no decimation)
};

/**
 * A structure to represent parameters for drawing an image
 */
struct Image
{
    cv::Point org;   //!< The bottom-left corner of the image
    cv::Mat   img;   //!< Image to draw
    cv::Mat   alpha; //!< Alpha channel for image to draw (same size and number of channels)
};

/**
 * A structure to represent parameters for drawing a polygon
 */
struct Poly
{
    std::vector<cv::Point> points;  //!< Points to connect
    cv::Scalar             color;   //!< The line color
    int                    thick;   //!< The thickness of line
    int                    lt;      //!< The Type of the line. See #LineTypes
    int                    shift;   //!< The number of fractional bits in the point coordinate
};

using Prim  = util::variant
    < Text
    , Rect
    , Circle
    , Line
    , Mosaic
    , Image
    , Poly
    >;

using Prims     = std::vector<Prim>;
using GMat2     = std::tuple<cv::GMat,cv::GMat>;
using GMatDesc2 = std::tuple<cv::GMatDesc,cv::GMatDesc>;

/** @brief The function renders on the input image passed drawing primitivies

@param bgr input image: 8-bit unsigned 3-channel image @ref CV_8UC3.
@param prims vector of drawing primitivies
@param args graph compile time parameters
*/
void GAPI_EXPORTS render(cv::Mat& bgr,
                         const Prims& prims,
                         cv::GCompileArgs&& args = {});

/** @brief The function renders on two NV12 planes passed drawing primitivies

@param y_plane input image: 8-bit unsigned 1-channel image @ref CV_8UC1.
@param uv_plane input image: 8-bit unsigned 2-channel image @ref CV_8UC2.
@param prims vector of drawing primitivies
@param args graph compile time parameters
*/
void GAPI_EXPORTS render(cv::Mat& y_plane,
                         cv::Mat& uv_plane,
                         const Prims& prims,
                         cv::GCompileArgs&& args = {});

G_TYPED_KERNEL_M(GRenderNV12, <GMat2(cv::GMat,cv::GMat,cv::GArray<wip::draw::Prim>)>, "org.opencv.render.nv12")
{
     static GMatDesc2 outMeta(GMatDesc y_plane, GMatDesc uv_plane, GArrayDesc)
     {
         return std::make_tuple(y_plane, uv_plane);
     }
};

G_TYPED_KERNEL(GRenderBGR, <cv::GMat(cv::GMat,cv::GArray<wip::draw::Prim>)>, "org.opencv.render.bgr")
{
     static GMatDesc outMeta(GMatDesc bgr, GArrayDesc)
     {
         return bgr;
     }
};

/** @brief Renders on 3 channels input

Output image must be 8-bit unsigned planar 3-channel image

@param src input image: 8-bit unsigned 3-channel image @ref CV_8UC3
@param prims draw primitives
*/
GAPI_EXPORTS GMat render3ch(const GMat& src, const GArray<Prim>& prims);

/** @brief Renders on two planes

Output y image must be 8-bit unsigned planar 1-channel image @ref CV_8UC1
uv image must be 8-bit unsigned planar 2-channel image @ref CV_8UC2

@param y  input image: 8-bit unsigned 1-channel image @ref CV_8UC1
@param uv input image: 8-bit unsigned 2-channel image @ref CV_8UC2
@param prims draw primitives
*/
GAPI_EXPORTS GMat2 renderNV12(const GMat& y,
                              const GMat& uv,
                              const GArray<Prim>& prims);

} // namespace draw
} // namespace wip

namespace render
{
namespace ocv
{
    GAPI_EXPORTS cv::gapi::GKernelPackage kernels();

} // namespace ocv
} // namespace render
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_RENDER_HPP
