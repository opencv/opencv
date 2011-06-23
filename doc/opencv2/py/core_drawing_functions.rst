Drawing Functions
=================

.. highlight:: python


Drawing functions work with matrices/images of arbitrary depth.
The boundaries of the shapes can be rendered with antialiasing (implemented only for 8-bit images for now).
All the functions include the parameter color that uses a rgb value (that may be constructed
with 
``CV_RGB``
) for color
images and brightness for grayscale images. For color images the order channel
is normally 
*Blue, Green, Red*
, this is what 
:cpp:func:`imshow`
, 
:cpp:func:`imread`
and 
:cpp:func:`imwrite`
expect
If you are using your own image rendering and I/O functions, you can use any channel ordering, the drawing functions process each channel independently and do not depend on the channel order or even on the color space used. The whole image can be converted from BGR to RGB or to a different color space using 
:cpp:func:`cvtColor`
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




.. function:: Circle(img,center,radius,color,thickness=1,lineType=8,shift=0)-> None

    Draws a circle.





    
    :param img: Image where the circle is drawn 
    
    :type img: :class:`CvArr`
    
    
    :param center: Center of the circle 
    
    :type center: :class:`CvPoint`
    
    
    :param radius: Radius of the circle 
    
    :type radius: int
    
    
    :param color: Circle color 
    
    :type color: :class:`CvScalar`
    
    
    :param thickness: Thickness of the circle outline if positive, otherwise this indicates that a filled circle is to be drawn 
    
    :type thickness: int
    
    
    :param lineType: Type of the circle boundary, see  :ref:`Line`  description 
    
    :type lineType: int
    
    
    :param shift: Number of fractional bits in the center coordinates and radius value 
    
    :type shift: int
    
    
    
The function draws a simple or filled circle with a
given center and radius.


.. index:: ClipLine

.. _ClipLine:

ClipLine
--------




.. function:: ClipLine(imgSize, pt1, pt2) -> (clipped_pt1, clipped_pt2)

    Clips the line against the image rectangle.





    
    :param imgSize: Size of the image 
    
    :type imgSize: :class:`CvSize`
    
    
    :param pt1: First ending point of the line segment.  
    
    :type pt1: :class:`CvPoint`
    
    
    :param pt2: Second ending point of the line segment.  
    
    :type pt2: :class:`CvPoint`
    
    
    
The function calculates a part of the line segment which is entirely within the image.
If the line segment is outside the image, it returns None. If the line segment is inside the image it returns a new pair of points. 

.. index:: DrawContours

.. _DrawContours:

DrawContours
------------




.. function:: DrawContours(img,contour,external_color,hole_color,max_level,thickness=1,lineType=8,offset=(0,0))-> None

    Draws contour outlines or interiors in an image.





    
    :param img: Image where the contours are to be drawn. As with any other drawing function, the contours are clipped with the ROI. 
    
    :type img: :class:`CvArr`
    
    
    :param contour: Pointer to the first contour 
    
    :type contour: :class:`CvSeq`
    
    
    :param external_color: Color of the external contours 
    
    :type external_color: :class:`CvScalar`
    
    
    :param hole_color: Color of internal contours (holes) 
    
    :type hole_color: :class:`CvScalar`
    
    
    :param max_level: Maximal level for drawn contours. If 0, only ``contour``  is drawn. If 1, the contour and all contours following
        it on the same level are drawn. If 2, all contours following and all
        contours one level below the contours are drawn, and so forth. If the value
        is negative, the function does not draw the contours following after ``contour``  but draws the child contours of  ``contour``  up
        to the  :math:`|\texttt{max\_level}|-1`  level. 
    
    :type max_level: int
    
    
    :param thickness: Thickness of lines the contours are drawn with.
        If it is negative (For example, =CV _ FILLED), the contour interiors are
        drawn. 
    
    :type thickness: int
    
    
    :param lineType: Type of the contour segments, see  :ref:`Line`  description 
    
    :type lineType: int
    
    
    
The function draws contour outlines in the image if 
:math:`\texttt{thickness} \ge 0`
or fills the area bounded by the contours if 
:math:`\texttt{thickness}<0`
.


.. index:: Ellipse

.. _Ellipse:

Ellipse
-------




.. function:: Ellipse(img,center,axes,angle,start_angle,end_angle,color,thickness=1,lineType=8,shift=0)-> None

    Draws a simple or thick elliptic arc or an fills ellipse sector.





    
    :param img: The image 
    
    :type img: :class:`CvArr`
    
    
    :param center: Center of the ellipse 
    
    :type center: :class:`CvPoint`
    
    
    :param axes: Length of the ellipse axes 
    
    :type axes: :class:`CvSize`
    
    
    :param angle: Rotation angle 
    
    :type angle: float
    
    
    :param start_angle: Starting angle of the elliptic arc 
    
    :type start_angle: float
    
    
    :param end_angle: Ending angle of the elliptic arc. 
    
    :type end_angle: float
    
    
    :param color: Ellipse color 
    
    :type color: :class:`CvScalar`
    
    
    :param thickness: Thickness of the ellipse arc outline if positive, otherwise this indicates that a filled ellipse sector is to be drawn 
    
    :type thickness: int
    
    
    :param lineType: Type of the ellipse boundary, see  :ref:`Line`  description 
    
    :type lineType: int
    
    
    :param shift: Number of fractional bits in the center coordinates and axes' values 
    
    :type shift: int
    
    
    
The function draws a simple or thick elliptic
arc or fills an ellipse sector. The arc is clipped by the ROI rectangle.
A piecewise-linear approximation is used for antialiased arcs and
thick arcs. All the angles are given in degrees. The picture below
explains the meaning of the parameters.

Parameters of Elliptic Arc








.. index:: EllipseBox

.. _EllipseBox:

EllipseBox
----------




.. function:: EllipseBox(img,box,color,thickness=1,lineType=8,shift=0)-> None

    Draws a simple or thick elliptic arc or fills an ellipse sector.





    
    :param img: Image 
    
    :type img: :class:`CvArr`
    
    
    :param box: The enclosing box of the ellipse drawn 
    
    :type box: :class:`CvBox2D`
    
    
    :param thickness: Thickness of the ellipse boundary 
    
    :type thickness: int
    
    
    :param lineType: Type of the ellipse boundary, see  :ref:`Line`  description 
    
    :type lineType: int
    
    
    :param shift: Number of fractional bits in the box vertex coordinates 
    
    :type shift: int
    
    
    
The function draws a simple or thick ellipse outline, or fills an ellipse. The functions provides a convenient way to draw an ellipse approximating some shape; that is what 
:ref:`CamShift`
and 
:ref:`FitEllipse`
do. The ellipse drawn is clipped by ROI rectangle. A piecewise-linear approximation is used for antialiased arcs and thick arcs.


.. index:: FillConvexPoly

.. _FillConvexPoly:

FillConvexPoly
--------------




.. function:: FillConvexPoly(img,pn,color,lineType=8,shift=0)-> None

    Fills a convex polygon.





    
    :param img: Image 
    
    :type img: :class:`CvArr`
    
    
    :param pn: List of coordinate pairs 
    
    :type pn: :class:`CvPoints`
    
    
    :param color: Polygon color 
    
    :type color: :class:`CvScalar`
    
    
    :param lineType: Type of the polygon boundaries, see  :ref:`Line`  description 
    
    :type lineType: int
    
    
    :param shift: Number of fractional bits in the vertex coordinates 
    
    :type shift: int
    
    
    
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




.. function:: FillPoly(img,polys,color,lineType=8,shift=0)-> None

    Fills a polygon's interior.





    
    :param img: Image 
    
    :type img: :class:`CvArr`
    
    
    :param polys: List of lists of (x,y) pairs.  Each list of points is a polygon. 
    
    :type polys: list of lists of (x,y) pairs
    
    
    :param color: Polygon color 
    
    :type color: :class:`CvScalar`
    
    
    :param lineType: Type of the polygon boundaries, see  :ref:`Line`  description 
    
    :type lineType: int
    
    
    :param shift: Number of fractional bits in the vertex coordinates 
    
    :type shift: int
    
    
    
The function fills an area bounded by several
polygonal contours. The function fills complex areas, for example,
areas with holes, contour self-intersection, and so forth.


.. index:: GetTextSize

.. _GetTextSize:

GetTextSize
-----------




.. function:: GetTextSize(textString,font)-> (textSize,baseline)

    Retrieves the width and height of a text string.





    
    :param font: Pointer to the font structure 
    
    :type font: :class:`CvFont`
    
    
    :param textString: Input string 
    
    :type textString: str
    
    
    :param textSize: Resultant size of the text string. Height of the text does not include the height of character parts that are below the baseline. 
    
    :type textSize: :class:`CvSize`
    
    
    :param baseline: y-coordinate of the baseline relative to the bottom-most text point 
    
    :type baseline: int
    
    
    
The function calculates the dimensions of a rectangle to enclose a text string when a specified font is used.


.. index:: InitFont

.. _InitFont:

InitFont
--------




.. function:: InitFont(fontFace,hscale,vscale,shear=0,thickness=1,lineType=8)-> font

    Initializes font structure.





    
    :param font: Pointer to the font structure initialized by the function 
    
    :type font: :class:`CvFont`
    
    
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
    
    :type fontFace: int
    
    
    :param hscale: Horizontal scale.  If equal to  ``1.0f`` , the characters have the original width depending on the font type. If equal to  ``0.5f`` , the characters are of half the original width. 
    
    :type hscale: float
    
    
    :param vscale: Vertical scale. If equal to  ``1.0f`` , the characters have the original height depending on the font type. If equal to  ``0.5f`` , the characters are of half the original height. 
    
    :type vscale: float
    
    
    :param shear: Approximate tangent of the character slope relative to the vertical line.  A zero value means a non-italic font,  ``1.0f``  means about a 45 degree slope, etc. 
    
    :type shear: float
    
    
    :param thickness: Thickness of the text strokes 
    
    :type thickness: int
    
    
    :param lineType: Type of the strokes, see  :ref:`Line`  description 
    
    :type lineType: int
    
    
    
The function initializes the font structure that can be passed to text rendering functions.



.. index:: InitLineIterator

.. _InitLineIterator:

InitLineIterator
----------------




.. function:: InitLineIterator(image, pt1, pt2, connectivity=8, left_to_right=0) -> line_iterator

    Initializes the line iterator.





    
    :param image: Image to sample the line from 
    
    :type image: :class:`CvArr`
    
    
    :param pt1: First ending point of the line segment 
    
    :type pt1: :class:`CvPoint`
    
    
    :param pt2: Second ending point of the line segment 
    
    :type pt2: :class:`CvPoint`
    
    
    :param connectivity: The scanned line connectivity, 4 or 8. 
    
    :type connectivity: int
    
    
    :param left_to_right: 
        If ( :math:`\texttt{left\_to\_right} = 0`  ) then the line is scanned in the specified order, from  ``pt1``  to  ``pt2`` .
        If ( :math:`\texttt{left\_to\_right} \ne 0` ) the line is scanned from left-most point to right-most. 
    
    :type left_to_right: int
    
    
    :param line_iterator: Iterator over the pixels of the line 
    
    :type line_iterator: :class:`iter`
    
    
    
The function returns an iterator over the pixels connecting the two points.
The points on the line are
calculated one by one using a 4-connected or 8-connected Bresenham
algorithm.

Example: Using line iterator to calculate the sum of pixel values along a color line




.. doctest::


    
    >>> import cv
    >>> img = cv.LoadImageM("building.jpg", cv.CV_LOAD_IMAGE_COLOR)
    >>> li = cv.InitLineIterator(img, (100, 100), (125, 150))
    >>> red_sum = 0
    >>> green_sum = 0
    >>> blue_sum = 0
    >>> for (r, g, b) in li:
    ...     red_sum += r
    ...     green_sum += g
    ...     blue_sum += b
    >>> print red_sum, green_sum, blue_sum
    10935.0 9496.0 7946.0
    

..

or more concisely using 
`zip <http://docs.python.org/library/functions.html#zip>`_
:




.. doctest::


    
    >>> import cv
    >>> img = cv.LoadImageM("building.jpg", cv.CV_LOAD_IMAGE_COLOR)
    >>> li = cv.InitLineIterator(img, (100, 100), (125, 150))
    >>> print [sum(c) for c in zip(*li)]
    [10935.0, 9496.0, 7946.0]
    

..


.. index:: Line

.. _Line:

Line
----




.. function:: Line(img,pt1,pt2,color,thickness=1,lineType=8,shift=0)-> None

    Draws a line segment connecting two points.





    
    :param img: The image 
    
    :type img: :class:`CvArr`
    
    
    :param pt1: First point of the line segment 
    
    :type pt1: :class:`CvPoint`
    
    
    :param pt2: Second point of the line segment 
    
    :type pt2: :class:`CvPoint`
    
    
    :param color: Line color 
    
    :type color: :class:`CvScalar`
    
    
    :param thickness: Line thickness 
    
    :type thickness: int
    
    
    :param lineType: Type of the line:
           
        
                
            * **8** (or omitted) 8-connected line. 
            
               
            * **4** 4-connected line. 
            
               
            * **CV_AA** antialiased line. 
            
               
            
    
    :type lineType: int
    
    
    :param shift: Number of fractional bits in the point coordinates 
    
    :type shift: int
    
    
    
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




.. function:: PolyLine(img,polys,is_closed,color,thickness=1,lineType=8,shift=0)-> None

    Draws simple or thick polygons.





    
    :param polys: List of lists of (x,y) pairs.  Each list of points is a polygon. 
    
    :type polys: list of lists of (x,y) pairs
    
    
    :param img: Image 
    
    :type img: :class:`CvArr`
    
    
    :param is_closed: Indicates whether the polylines must be drawn
        closed. If closed, the function draws the line from the last vertex
        of every contour to the first vertex. 
    
    :type is_closed: int
    
    
    :param color: Polyline color 
    
    :type color: :class:`CvScalar`
    
    
    :param thickness: Thickness of the polyline edges 
    
    :type thickness: int
    
    
    :param lineType: Type of the line segments, see  :ref:`Line`  description 
    
    :type lineType: int
    
    
    :param shift: Number of fractional bits in the vertex coordinates 
    
    :type shift: int
    
    
    
The function draws single or multiple polygonal curves.


.. index:: PutText

.. _PutText:

PutText
-------




.. function:: PutText(img,text,org,font,color)-> None

    Draws a text string.





    
    :param img: Input image 
    
    :type img: :class:`CvArr`
    
    
    :param text: String to print 
    
    :type text: str
    
    
    :param org: Coordinates of the bottom-left corner of the first letter 
    
    :type org: :class:`CvPoint`
    
    
    :param font: Pointer to the font structure 
    
    :type font: :class:`CvFont`
    
    
    :param color: Text color 
    
    :type color: :class:`CvScalar`
    
    
    
The function renders the text in the image with
the specified font and color. The printed text is clipped by the ROI
rectangle. Symbols that do not belong to the specified font are
replaced with the symbol for a rectangle.


.. index:: Rectangle

.. _Rectangle:

Rectangle
---------




.. function:: Rectangle(img,pt1,pt2,color,thickness=1,lineType=8,shift=0)-> None

    Draws a simple, thick, or filled rectangle.





    
    :param img: Image 
    
    :type img: :class:`CvArr`
    
    
    :param pt1: One of the rectangle's vertices 
    
    :type pt1: :class:`CvPoint`
    
    
    :param pt2: Opposite rectangle vertex 
    
    :type pt2: :class:`CvPoint`
    
    
    :param color: Line color (RGB) or brightness (grayscale image) 
    
    :type color: :class:`CvScalar`
    
    
    :param thickness: Thickness of lines that make up the rectangle. Negative values, e.g., CV _ FILLED, cause the function to draw a filled rectangle. 
    
    :type thickness: int
    
    
    :param lineType: Type of the line, see  :ref:`Line`  description 
    
    :type lineType: int
    
    
    :param shift: Number of fractional bits in the point coordinates 
    
    :type shift: int
    
    
    
The function draws a rectangle with two opposite corners 
``pt1``
and 
``pt2``
.


.. index:: CV_RGB

.. _CV_RGB:

CV_RGB
------




.. function:: CV_RGB(red,grn,blu)->CvScalar

    Constructs a color value.





    
    :param red: Red component 
    
    :type red: float
    
    
    :param grn: Green component 
    
    :type grn: float
    
    
    :param blu: Blue component 
    
    :type blu: float
    
    
    
