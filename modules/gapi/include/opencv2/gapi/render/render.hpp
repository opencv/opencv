// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019 Intel Corporation


#ifndef OPENCV_GAPI_RENDER_HPP
#define OPENCV_GAPI_RENDER_HPP

#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/gapi.hpp>

#include <opencv2/gapi/opencv_includes.hpp>
#include <opencv2/gapi/util/variant.hpp>
#include <opencv2/gapi/own/exports.hpp>
#include <opencv2/gapi/own/scalar.hpp>


/** \defgroup gapi_draw G-API Drawing and composition functionality
 *  @{
 *
 *  @brief Functions for in-graph drawing.
 *
 *  @note This is a Work in Progress functionality and APIs may
 *  change in the future releases.
 *
 *  G-API can do some in-graph drawing with a generic operations and a
 *  set of [rendering primitives](@ref gapi_draw_prims).
 *  In contrast with traditional OpenCV, in G-API user need to form a
 *  *rendering list* of primitives to draw. This list can be built
 *  manually or generated within a graph. This list is passed to
 *  [special operations or functions](@ref gapi_draw_api) where all
 *  primitives are interpreted and applied to the image.
 *
 *  For example, in a complex pipeline a list of detected objects
 *  can be translated in-graph to a list of cv::gapi::wip::draw::Rect
 *  primitives to highlight those with bounding boxes, or a list of
 *  detected faces can be translated in-graph to a list of
 *  cv::gapi::wip::draw::Mosaic primitives to hide sensitive content
 *  or protect privacy.
 *
 *  Like any other operations, rendering in G-API can be reimplemented
 *  by different backends. Currently only an OpenCV-based backend is
 *  available.
 *
 *  In addition to the graph-level operations, there are also regular
 *  (immediate) OpenCV-like functions are available -- see
 *  cv::gapi::wip::draw::render(). These functions are just wrappers
 *  over regular G-API and build the rendering graphs on the fly, so
 *  take compilation arguments as parameters.
 *
 *  Currently this API is more machine-oriented than human-oriented.
 *  The main purpose is to translate a set of domain-specific objects
 *  to a list of primitives to draw. For example, in order to generate
 *  a picture like this:
 *
 *  ![](modules/gapi/doc/pics/render_example.png)
 *
 *  Rendering list needs to be generated as follows:
 *
 *  @include modules/gapi/samples/draw_example.cpp
 *
 *  @defgroup gapi_draw_prims Drawing primitives
 *  @defgroup gapi_draw_api Drawing operations and functions
 *  @}
 */

namespace cv
{
namespace gapi
{
namespace wip
{
namespace draw
{

/**
 * @brief This structure specifies which FreeType font to use by FText primitives.
 */
struct freetype_font
{
    /*@{*/
    std::string path; //!< The path to the font file (.ttf)
    /*@{*/
};

//! @addtogroup gapi_draw_prims
//! @{
/**
 * @brief This structure represents a text string to draw.
 *
 * Parameters match cv::putText().
 */
struct Text
{
    /**
     * @brief Text constructor
     *
     * @param text_               The text string to be drawn
     * @param org_                The bottom-left corner of the text string in the image
     * @param ff_                 The font type, see #HersheyFonts
     * @param fs_                 The font scale factor that is multiplied by the font-specific base size
     * @param color_              The text color
     * @param thick_              The thickness of the lines used to draw a text
     * @param lt_                 The line type. See #LineTypes
     * @param bottom_left_origin_ When true, the image data origin is at the bottom-left corner. Otherwise, it is at the top-left corner
     */
    Text(const std::string& text_,
         const cv::Point& org_,
         int ff_,
         double fs_,
         const cv::Scalar& color_,
         int thick_ = 1,
         int lt_ = cv::LINE_8,
         bool bottom_left_origin_ = false) :
        text(text_), org(org_), ff(ff_), fs(fs_),
        color(color_), thick(thick_), lt(lt_), bottom_left_origin(bottom_left_origin_)
    {
    }

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
 * @brief This structure represents a text string to draw using
 * FreeType renderer.
 *
 * If OpenCV is built without FreeType support, this primitive will
 * fail at the execution stage.
 */
struct FText
{
    /**
     * @brief FText constructor
     *
     * @param text_ The text string to be drawn
     * @param org_  The bottom-left corner of the text string in the image
     * @param fh_   The height of text
     * @param color_ The text color
     */
    FText(const std::wstring& text_,
          const cv::Point& org_,
          int fh_,
          const cv::Scalar& color_) :
        text(text_), org(org_), fh(fh_), color(color_)
    {
    }

    /*@{*/
    std::wstring text;              //!< The text string to be drawn
    cv::Point    org;               //!< The bottom-left corner of the text string in the image
    int          fh;                //!< The height of text
    cv::Scalar   color;             //!< The text color
    /*@{*/
};

/**
 * @brief This structure represents a rectangle to draw.
 *
 * Parameters match cv::rectangle().
 */
struct Rect
{
    /**
     * @brief Rect constructor
     *
     * @param rect_   Coordinates of the rectangle
     * @param color_  The bottom-left corner of the text string in the image
     * @param thick_  The thickness of lines that make up the rectangle. Negative values, like #FILLED, mean that the function has to draw a filled rectangle
     * @param lt_     The type of the line. See #LineTypes
     * @param shift_  The number of fractional bits in the point coordinates
     */
    Rect(const cv::Rect& rect_,
         const cv::Scalar& color_,
         int thick_ = 1,
         int lt_ = cv::LINE_8,
         int shift_ = 0) :
        rect(rect_), color(color_), thick(thick_), lt(lt_), shift(shift_)
    {
    }

    /*@{*/
    cv::Rect   rect;  //!< Coordinates of the rectangle
    cv::Scalar color; //!< The rectangle color or brightness (grayscale image)
    int        thick; //!< The thickness of lines that make up the rectangle. Negative values, like #FILLED, mean that the function has to draw a filled rectangle
    int        lt;    //!< The type of the line. See #LineTypes
    int        shift; //!< The number of fractional bits in the point coordinates
    /*@{*/
};

/**
 * @brief This structure represents a circle to draw.
 *
 * Parameters match cv::circle().
 */
struct Circle
{
    /**
     * @brief Circle constructor
     *
     * @param  center_ The center of the circle
     * @param  radius_ The radius of the circle
     * @param  color_  The color of the  circle
     * @param  thick_  The thickness of the circle outline, if positive. Negative values, like #FILLED, mean that a filled circle is to be drawn
     * @param  lt_     The Type of the circle boundary. See #LineTypes
     * @param  shift_  The Number of fractional bits in the coordinates of the center and in the radius value
     */
    Circle(const cv::Point& center_,
           int radius_,
           const cv::Scalar& color_,
           int thick_ = 1,
           int lt_ = cv::LINE_8,
           int shift_ = 0) :
        center(center_), radius(radius_), color(color_), thick(thick_), lt(lt_), shift(shift_)
    {
    }

    /*@{*/
    cv::Point  center; //!< The center of the circle
    int        radius; //!< The radius of the circle
    cv::Scalar color;  //!< The color of the  circle
    int        thick;  //!< The thickness of the circle outline, if positive. Negative values, like #FILLED, mean that a filled circle is to be drawn
    int        lt;     //!< The Type of the circle boundary. See #LineTypes
    int        shift;  //!< The Number of fractional bits in the coordinates of the center and in the radius value
    /*@{*/
};

/**
 * @brief This structure represents a line to draw.
 *
 * Parameters match cv::line().
 */
struct Line
{
    /**
     * @brief Line constructor
     *
     * @param  pt1_    The first point of the line segment
     * @param  pt2_    The second point of the line segment
     * @param  color_  The line color
     * @param  thick_  The thickness of line
     * @param  lt_     The Type of the line. See #LineTypes
     * @param  shift_  The number of fractional bits in the point coordinates
    */
    Line(const cv::Point& pt1_,
         const cv::Point& pt2_,
         const cv::Scalar& color_,
         int thick_ = 1,
         int lt_ = cv::LINE_8,
         int shift_ = 0) :
        pt1(pt1_), pt2(pt2_), color(color_), thick(thick_), lt(lt_), shift(shift_)
    {
    }

    /*@{*/
    cv::Point  pt1;    //!< The first point of the line segment
    cv::Point  pt2;    //!< The second point of the line segment
    cv::Scalar color;  //!< The line color
    int        thick;  //!< The thickness of line
    int        lt;     //!< The Type of the line. See #LineTypes
    int        shift;  //!< The number of fractional bits in the point coordinates
    /*@{*/
};

/**
 * @brief This structure represents a mosaicing operation.
 *
 * Mosaicing is a very basic method to obfuscate regions in the image.
 */
struct Mosaic
{
    /**
     * @brief Mosaic constructor
     *
     * @param mos_    Coordinates of the mosaic
     * @param cellSz_ Cell size (same for X, Y). Note: mos size must be multiple of cell size
     * @param decim_  Decimation (0 stands for no decimation)
    */
    Mosaic(const cv::Rect& mos_,
           int cellSz_,
           int decim_) :
        mos(mos_), cellSz(cellSz_), decim(decim_)
    {
    }

    /*@{*/
    cv::Rect   mos;    //!< Coordinates of the mosaic
    int        cellSz; //!< Cell size (same for X, Y). Note: mosaic size must be a multiple of cell size
    int        decim;  //!< Decimation (0 stands for no decimation)
    /*@{*/
};

/**
 * @brief This structure represents an image to draw.
 *
 * Image is blended on a frame using the specified mask.
 */
struct Image
{
    /**
     * @brief Mosaic constructor
     *
     * @param  org_   The bottom-left corner of the image
     * @param  img_   Image to draw
     * @param  alpha_ Alpha channel for image to draw (same size and number of channels)
    */
    Image(const cv::Point& org_,
          const cv::Mat& img_,
          const cv::Mat& alpha_) :
        org(org_), img(img_), alpha(alpha_)
    {
    }

    /*@{*/
    cv::Point org;   //!< The bottom-left corner of the image
    cv::Mat   img;   //!< Image to draw
    cv::Mat   alpha; //!< Alpha channel for image to draw (same size and number of channels)
    /*@{*/
};

/**
 * @brief This structure represents a polygon to draw.
 */
struct Poly
{
    /**
     * @brief Mosaic constructor
     *
     * @param points_ Points to connect
     * @param color_  The line color
     * @param thick_  The thickness of line
     * @param lt_     The Type of the line. See #LineTypes
     * @param shift_  The number of fractional bits in the point coordinate
    */
    Poly(const std::vector<cv::Point>& points_,
         const cv::Scalar& color_,
         int thick_ = 1,
         int lt_ = cv::LINE_8,
         int shift_ = 0) :
        points(points_), color(color_), thick(thick_), lt(lt_), shift(shift_)
    {
    }

    /*@{*/
    std::vector<cv::Point> points;  //!< Points to connect
    cv::Scalar             color;   //!< The line color
    int                    thick;   //!< The thickness of line
    int                    lt;      //!< The Type of the line. See #LineTypes
    int                    shift;   //!< The number of fractional bits in the point coordinate
    /*@{*/
};

using Prim  = util::variant
    < Text
    , FText
    , Rect
    , Circle
    , Line
    , Mosaic
    , Image
    , Poly
    >;

using Prims     = std::vector<Prim>;
//! @} gapi_draw_prims

using GMat2     = std::tuple<cv::GMat,cv::GMat>;
using GMatDesc2 = std::tuple<cv::GMatDesc,cv::GMatDesc>;


//! @addtogroup gapi_draw_api
//! @{
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
//! @} gapi_draw_api

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

namespace detail
{
    template<> struct CompileArgTag<cv::gapi::wip::draw::freetype_font>
    {
        static const char* tag() { return "gapi.freetype_font"; }
    };
} // namespace detail

} // namespace cv

#endif // OPENCV_GAPI_RENDER_HPP
