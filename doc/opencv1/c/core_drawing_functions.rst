Drawing Functions
=================

.. highlight:: c


Drawing functions work with matrices/images of arbitrary depth.
The boundaries of the shapes can be rendered with antialiasing (implemented only for 8-bit images for now).
All the functions include the parameter color that uses a rgb value (that may be constructed
with 
``CV_RGB``
macro or the  :func:`cvScalar`  function 
) for color
images and brightness for grayscale images. For color images the order channel
is normally 
*Blue, Green, Red*
, this is what 
:func:`imshow`
, 
:func:`imread`
and 
:func:`imwrite`
expect
, so if you form a color using 
:func:`cvScalar`
, it should look like:


.. math::

    \texttt{cvScalar} (blue \_ component, green \_ component, red \_ component[, alpha \_ component]) 


If you are using your own image rendering and I/O functions, you can use any channel ordering, the drawing functions process each channel independently and do not depend on the channel order or even on the color space used. The whole image can be converted from BGR to RGB or to a different color space using 
:func:`cvtColor`
.

If a drawn figure is partially or completely outside the image, the drawing functions clip it. Also, many drawing functions can handle pixel coordinates specified with sub-pixel accuracy, that is, the coordinates can be passed as fixed-point numbers, encoded as integers. The number of fractional bits is specified by the 
``shift``
parameter and the real point coordinates are calculated as 
:math:`\texttt{Point}(x,y)\rightarrow\texttt{Point2f}(x*2^{-shift},y*2^{-shift})`
. This feature is especially effective wehn rendering antialiased shapes.

Also, note that the functions do not support alpha-transparency - when the target image is 4-channnel, then the 
``color[3]``
is simply copied to the repainted pixels. Thus, if you want to paint semi-transparent shapes, you can paint them in a separate buffer and then blend it with the main image.


.. index:: Circle

.. _Circle:

Circle
------






.. cfunction:: void cvCircle(  CvArr* img, CvPoint center, int radius, CvScalar color, int thickness=1, int lineType=8, int shift=0 )

    Draws a circle.





    
    :param img: Image where the circle is drawn 
    
    
    :param center: Center of the circle 
    
    
    :param radius: Radius of the circle 
    
    
    :param color: Circle color 
    
    
    :param thickness: Thickness of the circle outline if positive, otherwise this indicates that a filled circle is to be drawn 
    
    
    :param lineType: Type of the circle boundary, see  :ref:`Line`  description 
    
    
    :param shift: Number of fractional bits in the center coordinates and radius value 
    
    
    
The function draws a simple or filled circle with a
given center and radius.


.. index:: ClipLine

.. _ClipLine:

ClipLine
--------






.. cfunction:: int cvClipLine(  CvSize imgSize, CvPoint* pt1, CvPoint* pt2 )

    Clips the line against the image rectangle.





    
    :param imgSize: Size of the image 
    
    
    :param pt1: First ending point of the line segment.  It is modified by the function.  
    
    
    :param pt2: Second ending point of the line segment.  It is modified by the function.  
    
    
    
The function calculates a part of the line segment which is entirely within the image.
It returns 0 if the line segment is completely outside the image and 1 otherwise. 

.. index:: DrawContours

.. _DrawContours:

DrawContours
------------






.. cfunction:: void cvDrawContours(  CvArr *img, CvSeq* contour, CvScalar external_color, CvScalar hole_color, int max_level, int thickness=1, int lineType=8 )

    Draws contour outlines or interiors in an image.





    
    :param img: Image where the contours are to be drawn. As with any other drawing function, the contours are clipped with the ROI. 
    
    
    :param contour: Pointer to the first contour 
    
    
    :param external_color: Color of the external contours 
    
    
    :param hole_color: Color of internal contours (holes) 
    
    
    :param max_level: Maximal level for drawn contours. If 0, only ``contour``  is drawn. If 1, the contour and all contours following
        it on the same level are drawn. If 2, all contours following and all
        contours one level below the contours are drawn, and so forth. If the value
        is negative, the function does not draw the contours following after ``contour``  but draws the child contours of  ``contour``  up
        to the  :math:`|\texttt{max\_level}|-1`  level. 
    
    
    :param thickness: Thickness of lines the contours are drawn with.
        If it is negative (For example, =CV _ FILLED), the contour interiors are
        drawn. 
    
    
    :param lineType: Type of the contour segments, see  :ref:`Line`  description 
    
    
    
The function draws contour outlines in the image if 
:math:`\texttt{thickness} \ge 0`
or fills the area bounded by the contours if 
:math:`\texttt{thickness}<0`
.

Example: Connected component detection via contour functions




::


    
    #include "cv.h"
    #include "highgui.h"
    
    int main( int argc, char** argv )
    {
        IplImage* src;
        // the first command line parameter must be file name of binary 
        // (black-n-white) image
        if( argc == 2 && (src=cvLoadImage(argv[1], 0))!= 0)
        {
            IplImage* dst = cvCreateImage( cvGetSize(src), 8, 3 );
            CvMemStorage* storage = cvCreateMemStorage(0);
            CvSeq* contour = 0;
    
            cvThreshold( src, src, 1, 255, CV_THRESH_BINARY );
            cvNamedWindow( "Source", 1 );
            cvShowImage( "Source", src );
    
            cvFindContours( src, storage, &contour, sizeof(CvContour), 
               CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
            cvZero( dst );
    
            for( ; contour != 0; contour = contour->h_next )
            {
                CvScalar color = CV_RGB( rand()&255, rand()&255, rand()&255 );
                /* replace CV_FILLED with 1 to see the outlines */
                cvDrawContours( dst, contour, color, color, -1, CV_FILLED, 8 );
            }
    
            cvNamedWindow( "Components", 1 );
            cvShowImage( "Components", dst );
            cvWaitKey(0);
        }
    }
    

..


.. index:: Ellipse

.. _Ellipse:

Ellipse
-------






.. cfunction:: void cvEllipse(  CvArr* img, CvPoint center, CvSize axes, double angle, double start_angle, double end_angle, CvScalar color, int thickness=1, int lineType=8, int shift=0 )

    Draws a simple or thick elliptic arc or an fills ellipse sector.





    
    :param img: The image 
    
    
    :param center: Center of the ellipse 
    
    
    :param axes: Length of the ellipse axes 
    
    
    :param angle: Rotation angle 
    
    
    :param start_angle: Starting angle of the elliptic arc 
    
    
    :param end_angle: Ending angle of the elliptic arc. 
    
    
    :param color: Ellipse color 
    
    
    :param thickness: Thickness of the ellipse arc outline if positive, otherwise this indicates that a filled ellipse sector is to be drawn 
    
    
    :param lineType: Type of the ellipse boundary, see  :ref:`Line`  description 
    
    
    :param shift: Number of fractional bits in the center coordinates and axes' values 
    
    
    
The function draws a simple or thick elliptic
arc or fills an ellipse sector. The arc is clipped by the ROI rectangle.
A piecewise-linear approximation is used for antialiased arcs and
thick arcs. All the angles are given in degrees. The picture below
explains the meaning of the parameters.

Parameters of Elliptic Arc



.. image:: ../pics/ellipse.png




.. index:: EllipseBox

.. _EllipseBox:

EllipseBox
----------






.. cfunction:: void cvEllipseBox(  CvArr* img,  CvBox2D box,  CvScalar color,                     int thickness=1,  int lineType=8,  int shift=0 )

    Draws a simple or thick elliptic arc or fills an ellipse sector.





    
    :param img: Image 
    
    
    :param box: The enclosing box of the ellipse drawn 
    
    
    :param thickness: Thickness of the ellipse boundary 
    
    
    :param lineType: Type of the ellipse boundary, see  :ref:`Line`  description 
    
    
    :param shift: Number of fractional bits in the box vertex coordinates 
    
    
    
The function draws a simple or thick ellipse outline, or fills an ellipse. The functions provides a convenient way to draw an ellipse approximating some shape; that is what 
:ref:`CamShift`
and 
:ref:`FitEllipse`
do. The ellipse drawn is clipped by ROI rectangle. A piecewise-linear approximation is used for antialiased arcs and thick arcs.


.. index:: FillConvexPoly

.. _FillConvexPoly:

FillConvexPoly
--------------






.. cfunction:: void cvFillConvexPoly(  CvArr* img, CvPoint* pts, int npts, CvScalar color, int lineType=8, int shift=0 )

    Fills a convex polygon.





    
    :param img: Image 
    
    
    :param pts: Array of pointers to a single polygon 
    
    
    :param npts: Polygon vertex counter 
    
    
    :param color: Polygon color 
    
    
    :param lineType: Type of the polygon boundaries, see  :ref:`Line`  description 
    
    
    :param shift: Number of fractional bits in the vertex coordinates 
    
    
    
The function fills a convex polygon's interior.
This function is much faster than the function 
``cvFillPoly``
and can fill not only convex polygons but any monotonic polygon,
i.e., a polygon whose contour intersects every horizontal line (scan
line) twice at the most.



.. index:: FillPoly

.. _FillPoly:

FillPoly
--------






.. cfunction:: void cvFillPoly(  CvArr* img, CvPoint** pts, int* npts, int contours, CvScalar color, int lineType=8, int shift=0 )

    Fills a polygon's interior.





    
    :param img: Image 
    
    
    :param pts: Array of pointers to polygons 
    
    
    :param npts: Array of polygon vertex counters 
    
    
    :param contours: Number of contours that bind the filled region 
    
    
    :param color: Polygon color 
    
    
    :param lineType: Type of the polygon boundaries, see  :ref:`Line`  description 
    
    
    :param shift: Number of fractional bits in the vertex coordinates 
    
    
    
The function fills an area bounded by several
polygonal contours. The function fills complex areas, for example,
areas with holes, contour self-intersection, and so forth.


.. index:: GetTextSize

.. _GetTextSize:

GetTextSize
-----------






.. cfunction:: void cvGetTextSize(  const char* textString, const CvFont* font, CvSize* textSize, int* baseline )

    Retrieves the width and height of a text string.





    
    :param font: Pointer to the font structure 
    
    
    :param textString: Input string 
    
    
    :param textSize: Resultant size of the text string. Height of the text does not include the height of character parts that are below the baseline. 
    
    
    :param baseline: y-coordinate of the baseline relative to the bottom-most text point 
    
    
    
The function calculates the dimensions of a rectangle to enclose a text string when a specified font is used.


.. index:: InitFont

.. _InitFont:

InitFont
--------






.. cfunction:: void cvInitFont(  CvFont* font, int fontFace, double hscale, double vscale, double shear=0, int thickness=1, int lineType=8 )

    Initializes font structure.





    
    :param font: Pointer to the font structure initialized by the function 
    
    
    :param fontFace: Font name identifier. Only a subset of Hershey fonts  http://sources.isc.org/utils/misc/hershey-font.txt  are supported now:
          
        
               
            * **CV_FONT_HERSHEY_SIMPLEX** normal size sans-serif font 
            
              
            * **CV_FONT_HERSHEY_PLAIN** small size sans-serif font 
            
              
            * **CV_FONT_HERSHEY_DUPLEX** normal size sans-serif font (more complex than    ``CV_FONT_HERSHEY_SIMPLEX`` ) 
            
              
            * **CV_FONT_HERSHEY_COMPLEX** normal size serif font 
            
              
            * **CV_FONT_HERSHEY_TRIPLEX** normal size serif font (more complex than  ``CV_FONT_HERSHEY_COMPLEX`` ) 
            
              
            * **CV_FONT_HERSHEY_COMPLEX_SMALL** smaller version of  ``CV_FONT_HERSHEY_COMPLEX`` 
            
              
            * **CV_FONT_HERSHEY_SCRIPT_SIMPLEX** hand-writing style font 
            
              
            * **CV_FONT_HERSHEY_SCRIPT_COMPLEX** more complex variant of  ``CV_FONT_HERSHEY_SCRIPT_SIMPLEX`` 
            
              
            
         The parameter can be composited from one of the values above and an optional  ``CV_FONT_ITALIC``  flag, which indicates italic or oblique font. 
    
    
    :param hscale: Horizontal scale.  If equal to  ``1.0f`` , the characters have the original width depending on the font type. If equal to  ``0.5f`` , the characters are of half the original width. 
    
    
    :param vscale: Vertical scale. If equal to  ``1.0f`` , the characters have the original height depending on the font type. If equal to  ``0.5f`` , the characters are of half the original height. 
    
    
    :param shear: Approximate tangent of the character slope relative to the vertical line.  A zero value means a non-italic font,  ``1.0f``  means about a 45 degree slope, etc. 
    
    
    :param thickness: Thickness of the text strokes 
    
    
    :param lineType: Type of the strokes, see  :ref:`Line`  description 
    
    
    
The function initializes the font structure that can be passed to text rendering functions.



.. index:: InitLineIterator

.. _InitLineIterator:

InitLineIterator
----------------






.. cfunction:: int cvInitLineIterator(  const CvArr* image, CvPoint pt1, CvPoint pt2, CvLineIterator* line_iterator, int connectivity=8, int left_to_right=0 )

    Initializes the line iterator.





    
    :param image: Image to sample the line from 
    
    
    :param pt1: First ending point of the line segment 
    
    
    :param pt2: Second ending point of the line segment 
    
    
    :param line_iterator: Pointer to the line iterator state structure 
    
    
    :param connectivity: The scanned line connectivity, 4 or 8. 
    
    
    :param left_to_right: 
        If ( :math:`\texttt{left\_to\_right} = 0`  ) then the line is scanned in the specified order, from  ``pt1``  to  ``pt2`` .
        If ( :math:`\texttt{left\_to\_right} \ne 0` ) the line is scanned from left-most point to right-most. 
    
    
    
The function initializes the line
iterator and returns the number of pixels between the two end points.
Both points must be inside the image.
After the iterator has been
initialized, all the points on the raster line that connects the
two ending points may be retrieved by successive calls of
``CV_NEXT_LINE_POINT``
point.
The points on the line are
calculated one by one using a 4-connected or 8-connected Bresenham
algorithm.

Example: Using line iterator to calculate the sum of pixel values along the color line.




::


    
    
    CvScalar sum_line_pixels( IplImage* image, CvPoint pt1, CvPoint pt2 )
    {
        CvLineIterator iterator;
        int blue_sum = 0, green_sum = 0, red_sum = 0;
        int count = cvInitLineIterator( image, pt1, pt2, &iterator, 8, 0 );
    
        for( int i = 0; i < count; i++ ){
            blue_sum += iterator.ptr[0];
            green_sum += iterator.ptr[1];
            red_sum += iterator.ptr[2];
            CV_NEXT_LINE_POINT(iterator);
    
            /* print the pixel coordinates: demonstrates how to calculate the 
                                                            coordinates */
            {
            int offset, x, y;
            /* assume that ROI is not set, otherwise need to take it 
                                                    into account. */
            offset = iterator.ptr - (uchar*)(image->imageData);
            y = offset/image->widthStep;
            x = (offset - y*image->widthStep)/(3*sizeof(uchar) 
                                            /* size of pixel */);
            printf("(
            }
        }
        return cvScalar( blue_sum, green_sum, red_sum );
    }
    
    

..


.. index:: Line

.. _Line:

Line
----






.. cfunction:: void cvLine(  CvArr* img, CvPoint pt1, CvPoint pt2, CvScalar color, int thickness=1, int lineType=8, int shift=0 )

    Draws a line segment connecting two points.





    
    :param img: The image 
    
    
    :param pt1: First point of the line segment 
    
    
    :param pt2: Second point of the line segment 
    
    
    :param color: Line color 
    
    
    :param thickness: Line thickness 
    
    
    :param lineType: Type of the line:
           
        
                
            * **8** (or omitted) 8-connected line. 
            
               
            * **4** 4-connected line. 
            
               
            * **CV_AA** antialiased line. 
            
               
            
    
    
    :param shift: Number of fractional bits in the point coordinates 
    
    
    
The function draws the line segment between
``pt1``
and 
``pt2``
points in the image. The line is
clipped by the image or ROI rectangle. For non-antialiased lines
with integer coordinates the 8-connected or 4-connected Bresenham
algorithm is used. Thick lines are drawn with rounding endings.
Antialiased lines are drawn using Gaussian filtering. To specify
the line color, the user may use the macro
``CV_RGB( r, g, b )``
.


.. index:: PolyLine

.. _PolyLine:

PolyLine
--------






.. cfunction:: void cvPolyLine(  CvArr* img, CvPoint** pts, int* npts, int contours, int is_closed, CvScalar color, int thickness=1, int lineType=8, int shift=0 )

    Draws simple or thick polygons.





    
    :param pts: Array of pointers to polygons 
    
    
    :param npts: Array of polygon vertex counters 
    
    
    :param contours: Number of contours that bind the filled region 
    
    
    :param img: Image 
    
    
    :param is_closed: Indicates whether the polylines must be drawn
        closed. If closed, the function draws the line from the last vertex
        of every contour to the first vertex. 
    
    
    :param color: Polyline color 
    
    
    :param thickness: Thickness of the polyline edges 
    
    
    :param lineType: Type of the line segments, see  :ref:`Line`  description 
    
    
    :param shift: Number of fractional bits in the vertex coordinates 
    
    
    
The function draws single or multiple polygonal curves.


.. index:: PutText

.. _PutText:

PutText
-------






.. cfunction:: void cvPutText(  CvArr* img, const char* text, CvPoint org, const CvFont* font, CvScalar color )

    Draws a text string.





    
    :param img: Input image 
    
    
    :param text: String to print 
    
    
    :param org: Coordinates of the bottom-left corner of the first letter 
    
    
    :param font: Pointer to the font structure 
    
    
    :param color: Text color 
    
    
    
The function renders the text in the image with
the specified font and color. The printed text is clipped by the ROI
rectangle. Symbols that do not belong to the specified font are
replaced with the symbol for a rectangle.


.. index:: Rectangle

.. _Rectangle:

Rectangle
---------






.. cfunction:: void cvRectangle(  CvArr* img, CvPoint pt1, CvPoint pt2, CvScalar color, int thickness=1, int lineType=8, int shift=0 )

    Draws a simple, thick, or filled rectangle.





    
    :param img: Image 
    
    
    :param pt1: One of the rectangle's vertices 
    
    
    :param pt2: Opposite rectangle vertex 
    
    
    :param color: Line color (RGB) or brightness (grayscale image) 
    
    
    :param thickness: Thickness of lines that make up the rectangle. Negative values, e.g., CV _ FILLED, cause the function to draw a filled rectangle. 
    
    
    :param lineType: Type of the line, see  :ref:`Line`  description 
    
    
    :param shift: Number of fractional bits in the point coordinates 
    
    
    
The function draws a rectangle with two opposite corners 
``pt1``
and 
``pt2``
.


.. index:: CV_RGB

.. _CV_RGB:

CV_RGB
------






.. cfunction:: \#define CV_RGB( r, g, b )  cvScalar( (b), (g), (r) )

    Constructs a color value.





    
    :param red: Red component 
    
    
    :param grn: Green component 
    
    
    :param blu: Blue component 
    
    
    
