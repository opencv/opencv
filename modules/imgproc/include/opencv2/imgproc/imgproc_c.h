/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_IMGPROC_IMGPROC_C_H
#define OPENCV_IMGPROC_IMGPROC_C_H

#include "opencv2/imgproc/types_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @addtogroup imgproc_c
@{
*/

/****************************************************************************************\
*                                     Drawing                                            *
\****************************************************************************************/

/****************************************************************************************\
*       Drawing functions work with images/matrices of arbitrary type.                   *
*       For color images the channel order is BGR[A]                                     *
*       Antialiasing is supported only for 8-bit image now.                              *
*       All the functions include parameter color that means rgb value (that may be      *
*       constructed with CV_RGB macro) for color images and brightness                   *
*       for grayscale images.                                                            *
*       If a drawn figure is partially or completely outside of the image, it is clipped.*
\****************************************************************************************/

#define CV_FILLED -1

#define CV_AA 16

/** @brief Draws 4-connected, 8-connected or antialiased line segment connecting two points
@see cv::line
*/
CVAPI(void)  cvLine( CvArr* img, CvPoint pt1, CvPoint pt2,
                     CvScalar color, int thickness CV_DEFAULT(1),
                     int line_type CV_DEFAULT(8), int shift CV_DEFAULT(0) );

/** @brief Draws a rectangle given two opposite corners of the rectangle (pt1 & pt2)

   if thickness<0 (e.g. thickness == CV_FILLED), the filled box is drawn
@see cv::rectangle
*/
CVAPI(void)  cvRectangle( CvArr* img, CvPoint pt1, CvPoint pt2,
                          CvScalar color, int thickness CV_DEFAULT(1),
                          int line_type CV_DEFAULT(8),
                          int shift CV_DEFAULT(0));


/** @brief Draws a circle with specified center and radius.

   Thickness works in the same way as with cvRectangle
@see cv::circle
*/
CVAPI(void)  cvCircle( CvArr* img, CvPoint center, int radius,
                       CvScalar color, int thickness CV_DEFAULT(1),
                       int line_type CV_DEFAULT(8), int shift CV_DEFAULT(0));

/** @brief Draws ellipse outline, filled ellipse, elliptic arc or filled elliptic sector

   depending on _thickness_, _start_angle_ and _end_angle_ parameters. The resultant figure
   is rotated by _angle_. All the angles are in degrees
@see cv::ellipse
*/
CVAPI(void)  cvEllipse( CvArr* img, CvPoint center, CvSize axes,
                        double angle, double start_angle, double end_angle,
                        CvScalar color, int thickness CV_DEFAULT(1),
                        int line_type CV_DEFAULT(8), int shift CV_DEFAULT(0));


/** @brief Fills an area bounded by one or more arbitrary polygons
@see cv::fillPoly
*/
CVAPI(void)  cvFillPoly( CvArr* img, CvPoint** pts, const int* npts,
                         int contours, CvScalar color,
                         int line_type CV_DEFAULT(8), int shift CV_DEFAULT(0) );

/** @brief Draws one or more polygonal curves
@see cv::polylines
*/
CVAPI(void)  cvPolyLine( CvArr* img, CvPoint** pts, const int* npts, int contours,
                         int is_closed, CvScalar color, int thickness CV_DEFAULT(1),
                         int line_type CV_DEFAULT(8), int shift CV_DEFAULT(0) );


/** @brief Initializes line iterator.

Initially, line_iterator->ptr will point to pt1 (or pt2, see left_to_right description) location in
the image. Returns the number of pixels on the line between the ending points.
@see cv::LineIterator
*/
CVAPI(int)  cvInitLineIterator( const CvArr* image, CvPoint pt1, CvPoint pt2,
                                CvLineIterator* line_iterator,
                                int connectivity CV_DEFAULT(8),
                                int left_to_right CV_DEFAULT(0));

#define CV_NEXT_LINE_POINT( line_iterator )                     \
{                                                               \
    int _line_iterator_mask = (line_iterator).err < 0 ? -1 : 0; \
    (line_iterator).err += (line_iterator).minus_delta +        \
        ((line_iterator).plus_delta & _line_iterator_mask);     \
    (line_iterator).ptr += (line_iterator).minus_step +         \
        ((line_iterator).plus_step & _line_iterator_mask);      \
}


#define CV_FONT_HERSHEY_SIMPLEX         0
#define CV_FONT_HERSHEY_PLAIN           1
#define CV_FONT_HERSHEY_DUPLEX          2
#define CV_FONT_HERSHEY_COMPLEX         3
#define CV_FONT_HERSHEY_TRIPLEX         4
#define CV_FONT_HERSHEY_COMPLEX_SMALL   5
#define CV_FONT_HERSHEY_SCRIPT_SIMPLEX  6
#define CV_FONT_HERSHEY_SCRIPT_COMPLEX  7

#define CV_FONT_ITALIC                 16

#define CV_FONT_VECTOR0    CV_FONT_HERSHEY_SIMPLEX


/** Font structure */
typedef struct CvFont
{
  const char* nameFont;   //Qt:nameFont
  CvScalar color;       //Qt:ColorFont -> cvScalar(blue_component, green_component, red_component[, alpha_component])
    int         font_face;    //Qt: bool italic         /** =CV_FONT_* */
    const int*  ascii;      //!< font data and metrics
    const int*  greek;
    const int*  cyrillic;
    float       hscale, vscale;
    float       shear;      //!< slope coefficient: 0 - normal, >0 - italic
    int         thickness;    //!< Qt: weight               /** letters thickness */
    float       dx;       //!< horizontal interval between letters
    int         line_type;    //!< Qt: PointSize
}
CvFont;

/** @brief Initializes font structure (OpenCV 1.x API).

The function initializes the font structure that can be passed to text rendering functions.

@param font Pointer to the font structure initialized by the function
@param font_face Font name identifier. See cv::HersheyFonts and corresponding old CV_* identifiers.
@param hscale Horizontal scale. If equal to 1.0f , the characters have the original width
depending on the font type. If equal to 0.5f , the characters are of half the original width.
@param vscale Vertical scale. If equal to 1.0f , the characters have the original height depending
on the font type. If equal to 0.5f , the characters are of half the original height.
@param shear Approximate tangent of the character slope relative to the vertical line. A zero
value means a non-italic font, 1.0f means about a 45 degree slope, etc.
@param thickness Thickness of the text strokes
@param line_type Type of the strokes, see line description

@sa cvPutText
 */
CVAPI(void)  cvInitFont( CvFont* font, int font_face,
                         double hscale, double vscale,
                         double shear CV_DEFAULT(0),
                         int thickness CV_DEFAULT(1),
                         int line_type CV_DEFAULT(8));

CV_INLINE CvFont cvFont( double scale, int thickness CV_DEFAULT(1) )
{
    CvFont font;
    cvInitFont( &font, CV_FONT_HERSHEY_PLAIN, scale, scale, 0, thickness, CV_AA );
    return font;
}

/** @brief Renders text stroke with specified font and color at specified location.
   CvFont should be initialized with cvInitFont
@see cvInitFont, cvGetTextSize, cvFont, cv::putText
*/
CVAPI(void)  cvPutText( CvArr* img, const char* text, CvPoint org,
                        const CvFont* font, CvScalar color );


/** @brief Unpacks color value

if arrtype is CV_8UC?, _color_ is treated as packed color value, otherwise the first channels
(depending on arrtype) of destination scalar are set to the same value = _color_
*/
CVAPI(CvScalar)  cvColorToScalar( double packed_color, int arrtype );

/** @brief Returns the polygon points which make up the given ellipse.

The ellipse is define by the box of size 'axes' rotated 'angle' around the 'center'. A partial
sweep of the ellipse arc can be done by specifying arc_start and arc_end to be something other than
0 and 360, respectively. The input array 'pts' must be large enough to hold the result. The total
number of points stored into 'pts' is returned by this function.
@see cv::ellipse2Poly
*/
CVAPI(int) cvEllipse2Poly( CvPoint center, CvSize axes,
                 int angle, int arc_start, int arc_end, CvPoint * pts, int delta );

/** @brief Draws contour outlines or filled interiors on the image
@see cv::drawContours
*/
CVAPI(void)  cvDrawContours( CvArr *img, CvSeq* contour,
                             CvScalar external_color, CvScalar hole_color,
                             int max_level, int thickness CV_DEFAULT(1),
                             int line_type CV_DEFAULT(8),
                             CvPoint offset CV_DEFAULT(cvPoint(0,0)));

/** @} */

#ifdef __cplusplus
}
#endif

#endif
