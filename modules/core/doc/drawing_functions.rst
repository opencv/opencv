Drawing Functions
=================

.. highlight:: cpp

Drawing functions work with matrices/images of arbitrary depth.
The boundaries of the shapes can be rendered with antialiasing (implemented only for 8-bit images for now).
All the functions include the parameter ``color`` that uses an RGB value (that may be constructed
with ``CV_RGB`` or the  :ocv:class:`Scalar_`  constructor
) for color
images and brightness for grayscale images. For color images, the channel ordering
is normally *Blue, Green, Red*.
This is what :ocv:func:`imshow`, :ocv:func:`imread`, and :ocv:func:`imwrite` expect.
So, if you form a color using the
``Scalar`` constructor, it should look like:

.. math::

    \texttt{Scalar} (blue \_ component, green \_ component, red \_ component[, alpha \_ component])

If you are using your own image rendering and I/O functions, you can use any channel ordering. The drawing functions process each channel independently and do not depend on the channel order or even on the used color space. The whole image can be converted from BGR to RGB or to a different color space using
:ocv:func:`cvtColor` .

If a drawn figure is partially or completely outside the image, the drawing functions clip it. Also, many drawing functions can handle pixel coordinates specified with sub-pixel accuracy. This means that the coordinates can be passed as fixed-point numbers encoded as integers. The number of fractional bits is specified by the ``shift`` parameter and the real point coordinates are calculated as
:math:`\texttt{Point}(x,y)\rightarrow\texttt{Point2f}(x*2^{-shift},y*2^{-shift})` . This feature is especially effective when rendering antialiased shapes.

.. note:: The functions do not support alpha-transparency when the target image is 4-channel. In this case, the ``color[3]`` is simply copied to the repainted pixels. Thus, if you want to paint semi-transparent shapes, you can paint them in a separate buffer and then blend it with the main image.

circle
----------
Draws a circle.

.. ocv:function:: void circle(Mat& img, Point center, int radius,            const Scalar& color, int thickness=1,            int lineType=8, int shift=0)

.. ocv:pyfunction:: cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]]) -> None

.. ocv:cfunction:: void cvCircle( CvArr* img, CvPoint center, int radius, CvScalar color, int thickness=1, int lineType=8, int shift=0 )
.. ocv:pyoldfunction:: cv.Circle(img, center, radius, color, thickness=1, lineType=8, shift=0)-> None

    :param img: Image where the circle is drawn.

    :param center: Center of the circle.

    :param radius: Radius of the circle.

    :param color: Circle color.

    :param thickness: Thickness of the circle outline, if positive. Negative thickness means that a filled circle is to be drawn.

    :param lineType: Type of the circle boundary. See the  :ocv:func:`line`  description.

    :param shift: Number of fractional bits in the coordinates of the center and in the radius value.

The function ``circle`` draws a simple or filled circle with a given center and radius.

clipLine
------------
Clips the line against the image rectangle.

.. ocv:function:: bool clipLine(Size imgSize, Point& pt1, Point& pt2)

.. ocv:function:: bool clipLine(Rect imgRect, Point& pt1, Point& pt2)

.. ocv:pyfunction:: cv2.clipLine(imgRect, pt1, pt2) -> retval, pt1, pt2

.. ocv:cfunction:: int cvClipLine( CvSize imgSize, CvPoint* pt1, CvPoint* pt2 )
.. ocv:pyoldfunction:: cv.ClipLine(imgSize, pt1, pt2) -> (clippedPt1, clippedPt2)

    :param imgSize: Image size. The image rectangle is  ``Rect(0, 0, imgSize.width, imgSize.height)`` .     

    :param imgRect: Image rectangle.

    :param pt1: First line point.

    :param pt2: Second line point.

The functions ``clipLine`` calculate a part of the line segment that is entirely within the specified rectangle.
They return ``false`` if the line segment is completely outside the rectangle. Otherwise, they return ``true`` .

ellipse
-----------
Draws a simple or thick elliptic arc or fills an ellipse sector.

.. ocv:function:: void ellipse(Mat& img, Point center, Size axes,             double angle, double startAngle, double endAngle,             const Scalar& color, int thickness=1,             int lineType=8, int shift=0)

.. ocv:function:: void ellipse(Mat& img, const RotatedRect& box, const Scalar& color,             int thickness=1, int lineType=8)

.. ocv:pyfunction:: cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color[, thickness[, lineType[, shift]]]) -> None
.. ocv:pyfunction:: cv2.ellipse(img, box, color[, thickness[, lineType]]) -> None

.. ocv:cfunction:: void cvEllipse( CvArr* img, CvPoint center, CvSize axes, double angle, double startAngle, double endAngle, CvScalar color, int thickness=1, int lineType=8, int shift=0 )
.. ocv:pyoldfunction:: cv.Ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness=1, lineType=8, shift=0)-> None

.. ocv:cfunction:: void cvEllipseBox( CvArr* img, CvBox2D box, CvScalar color, int thickness=1, int lineType=8, int shift=0 )
.. ocv:pyoldfunction:: cv.EllipseBox(img, box, color, thickness=1, lineType=8, shift=0)-> None

    :param img: Image.

    :param center: Center of the ellipse.

    :param axes: Length of the ellipse axes.

    :param angle: Ellipse rotation angle in degrees.

    :param startAngle: Starting angle of the elliptic arc in degrees.

    :param endAngle: Ending angle of the elliptic arc in degrees.

    :param box: Alternative ellipse representation via  :ocv:class:`RotatedRect` or ``CvBox2D``. This means that the function draws an ellipse inscribed in the rotated rectangle.

    :param color: Ellipse color.

    :param thickness: Thickness of the ellipse arc outline, if positive. Otherwise, this indicates that a filled ellipse sector is to be drawn.

    :param lineType: Type of the ellipse boundary. See the  :ocv:func:`line`  description.

    :param shift: Number of fractional bits in the coordinates of the center and values of axes.

The functions ``ellipse`` with less parameters draw an ellipse outline, a filled ellipse, an elliptic arc, or a filled ellipse sector.
A piecewise-linear curve is used to approximate the elliptic arc boundary. If you need more control of the ellipse rendering, you can retrieve the curve using
:ocv:func:`ellipse2Poly` and then render it with
:ocv:func:`polylines` or fill it with
:ocv:func:`fillPoly` . If you use the first variant of the function and want to draw the whole ellipse, not an arc, pass ``startAngle=0`` and ``endAngle=360`` . The figure below explains the meaning of the parameters.

**Figure 1. Parameters of Elliptic Arc**

.. image:: pics/ellipse.png

ellipse2Poly
----------------
Approximates an elliptic arc with a polyline.

.. ocv:function:: void ellipse2Poly( Point center, Size axes, int angle,                   int startAngle, int endAngle, int delta,                   vector<Point>& pts )

.. ocv:pyfunction:: cv2.ellipse2Poly(center, axes, angle, arcStart, arcEnd, delta) -> pts

    :param center: Center of the arc.

    :param axes: Half-sizes of the arc. See the  :ocv:func:`ellipse`  for details.   
    
    :param angle: Rotation angle of the ellipse in degrees. See the  :ocv:func:`ellipse`  for details.   
    
    :param startAngle: Starting angle of the elliptic arc in degrees.

    :param endAngle: Ending angle of the elliptic arc in degrees.

    :param delta: Angle between the subsequent polyline vertices. It defines the approximation accuracy.

    :param pts: Output vector of polyline vertices.

The function ``ellipse2Poly`` computes the vertices of a polyline that approximates the specified elliptic arc. It is used by
:ocv:func:`ellipse` .



fillConvexPoly
------------------
Fills a convex polygon.

.. ocv:function:: void fillConvexPoly(Mat& img, const Point* pts, int npts,                    const Scalar& color, int lineType=8,                    int shift=0)

.. ocv:pyfunction:: cv2.fillConvexPoly(img, points, color[, lineType[, shift]]) -> None

.. ocv:cfunction:: void cvFillConvexPoly( CvArr* img, CvPoint* pts, int npts, CvScalar color, int lineType=8, int shift=0 )
.. ocv:pyoldfunction:: cv.FillConvexPoly(img, pn, color, lineType=8, shift=0)-> None

    :param img: Image.

    :param pts: Polygon vertices.

    :param npts: Number of polygon vertices.

    :param color: Polygon color.

    :param lineType: Type of the polygon boundaries. See the  :ocv:func:`line`  description.

    :param shift: Number of fractional bits in the vertex coordinates.

The function ``fillConvexPoly`` draws a filled convex polygon.
This function is much faster than the function ``fillPoly`` . It can fill not only convex polygons but any monotonic polygon without self-intersections,
that is, a polygon whose contour intersects every horizontal line (scan line) twice at the most (though, its top-most and/or the bottom edge could be horizontal).



fillPoly
------------
Fills the area bounded by one or more polygons.

.. ocv:function:: void fillPoly(Mat& img, const Point** pts,               const int* npts, int ncontours,              const Scalar& color, int lineType=8,              int shift=0, Point offset=Point() )

.. ocv:pyfunction:: cv2.fillPoly(img, pts, color[, lineType[, shift[, offset]]]) -> None

.. ocv:cfunction:: void cvFillPoly( CvArr* img, CvPoint** pts, int* npts, int ncontours, CvScalar color, int lineType=8, int shift=0 )
.. ocv:pyoldfunction:: cv.FillPoly(img, polys, color, lineType=8, shift=0)-> None

    :param img: Image.

    :param pts: Array of polygons where each polygon is represented as an array of points.

    :param npts: Array of polygon vertex counters.

    :param ncontours: Number of contours that bind the filled region.

    :param color: Polygon color.

    :param lineType: Type of the polygon boundaries. See the  :ocv:func:`line`  description.

    :param shift: Number of fractional bits in the vertex coordinates.

    :param offset: Optional offset of all points of the contours.

The function ``fillPoly`` fills an area bounded by several polygonal contours. The function can fill complex areas, for example,
areas with holes, contours with self-intersections (some of their parts), and so forth.



getTextSize
---------------
Calculates the width and height of a text string.

.. ocv:function:: Size getTextSize(const string& text, int fontFace,                 double fontScale, int thickness,                 int* baseLine)

.. ocv:pyfunction:: cv2.getTextSize(text, fontFace, fontScale, thickness) -> retval, baseLine

.. ocv:cfunction:: void cvGetTextSize( const char* textString, const CvFont* font, CvSize* textSize, int* baseline )
.. ocv:pyoldfunction:: cv.GetTextSize(textString, font)-> (textSize, baseline)

    :param text: Input text string.

    :param fontFace: Font to use. See the  :ocv:func:`putText` for details.    

    :param fontScale: Font scale. See the  :ocv:func:`putText`  for details.   

    :param thickness: Thickness of lines used to render the text. See  :ocv:func:`putText`  for details.   

    :param baseLine: Output parameter - y-coordinate of the baseline relative to the bottom-most text point.

The function ``getTextSize`` calculates and returns the size of a box that contains the specified text.
That is, the following code renders some text, the tight box surrounding it, and the baseline: ::

    // Use "y" to show that the baseLine is about
    string text = "Funny text inside the box";
    int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
    double fontScale = 2;
    int thickness = 3;

    Mat img(600, 800, CV_8UC3, Scalar::all(0));

    int baseline=0;
    Size textSize = getTextSize(text, fontFace,
                                fontScale, thickness, &baseline);
    baseline += thickness;

    // center the text
    Point textOrg((img.cols - textSize.width)/2,
                  (img.rows + textSize.height)/2);

    // draw the box
    rectangle(img, textOrg + Point(0, baseline),
              textOrg + Point(textSize.width, -textSize.height),
              Scalar(0,0,255));
    // ... and the baseline first
    line(img, textOrg + Point(0, thickness),
         textOrg + Point(textSize.width, thickness),
         Scalar(0, 0, 255));

    // then put the text itself
    putText(img, text, textOrg, fontFace, fontScale,
            Scalar::all(255), thickness, 8);


InitFont
--------
Initializes font structure (OpenCV 1.x API).

.. ocv:cfunction:: void cvInitFont(  CvFont* font, int fontFace, double hscale, double vscale, double shear=0, int thickness=1, int lineType=8 )

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


    :param lineType: Type of the strokes, see  :ocv:func:`line`  description 


The function initializes the font structure that can be passed to text rendering functions.

.. seealso:: :ocv:cfunc:`PutText`

.. _Line: 

line
--------
Draws a line segment connecting two points.

.. ocv:function:: void line(Mat& img, Point pt1, Point pt2, const Scalar& color,          int thickness=1, int lineType=8, int shift=0)

.. ocv:pyfunction:: cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> None

.. ocv:cfunction:: void cvLine( CvArr* img, CvPoint pt1, CvPoint pt2, CvScalar color, int thickness=1, int lineType=8, int shift=0 )
.. ocv:pyoldfunction:: cv.Line(img, pt1, pt2, color, thickness=1, lineType=8, shift=0)-> None

    :param img: Image.

    :param pt1: First point of the line segment.

    :param pt2: Second point of the line segment.

    :param color: Line color.

    :param thickness: Line thickness.

    :param lineType: Type of the line:

            * **8** (or omitted) - 8-connected line.

            * **4** - 4-connected line.

            * **CV_AA** - antialiased line.

    :param shift: Number of fractional bits in the point coordinates.

The function ``line`` draws the line segment between ``pt1`` and ``pt2`` points in the image. The line is clipped by the image boundaries. For non-antialiased lines with integer coordinates, the 8-connected or 4-connected Bresenham algorithm is used. Thick lines are drawn with rounding endings.
Antialiased lines are drawn using Gaussian filtering. To specify the line color, you may use the macro ``CV_RGB(r, g, b)`` .


LineIterator
------------
.. ocv:class:: LineIterator

Class for iterating pixels on a raster line. ::

    class LineIterator
    {
    public:
        // creates iterators for the line connecting pt1 and pt2
        // the line will be clipped on the image boundaries
        // the line is 8-connected or 4-connected
        // If leftToRight=true, then the iteration is always done
        // from the left-most point to the right most,
        // not to depend on the ordering of pt1 and pt2 parameters
        LineIterator(const Mat& img, Point pt1, Point pt2,
                     int connectivity=8, bool leftToRight=false);
        // returns pointer to the current line pixel
        uchar* operator *();
        // move the iterator to the next pixel
        LineIterator& operator ++();
        LineIterator operator ++(int);

        // internal state of the iterator
        uchar* ptr;
        int err, count;
        int minusDelta, plusDelta;
        int minusStep, plusStep;
    };

The class ``LineIterator`` is used to get each pixel of a raster line. It can be treated as versatile implementation of the Bresenham algorithm where you can stop at each pixel and do some extra processing, for example, grab pixel values along the line or draw a line with an effect (for example, with XOR operation).

The number of pixels along the line is stored in ``LineIterator::count`` . ::

    // grabs pixels along the line (pt1, pt2)
    // from 8-bit 3-channel image to the buffer
    LineIterator it(img, pt1, pt2, 8);
    vector<Vec3b> buf(it.count);

    for(int i = 0; i < it.count; i++, ++it)
        buf[i] = *(const Vec3b)*it;



rectangle
-------------
Draws a simple, thick, or filled up-right rectangle.

.. ocv:function:: void rectangle(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness=1,               int lineType=8, int shift=0)

.. ocv:function:: void rectangle(Mat& img, Rect r, const Scalar& color, int thickness=1,               int lineType=8, int shift=0)

.. ocv:pyfunction:: cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> None

.. ocv:cfunction:: void cvRectangle( CvArr* img, CvPoint pt1, CvPoint pt2, CvScalar color, int thickness=1, int lineType=8, int shift=0 )
.. ocv:pyoldfunction:: cv.Rectangle(img, pt1, pt2, color, thickness=1, lineType=8, shift=0)-> None

    :param img: Image.

    :param pt1: Vertex of the rectangle.

    :param pt2: Vertex of the rectangle opposite to  ``pt1`` .
    
    :param r: Alternative specification of the drawn rectangle.

    :param color: Rectangle color or brightness (grayscale image).

    :param thickness: Thickness of lines that make up the rectangle. Negative values, like  ``CV_FILLED`` , mean that the function has to draw a filled rectangle.

    :param lineType: Type of the line. See the  :ocv:func:`line`  description.

    :param shift: Number of fractional bits in the point coordinates.

The function ``rectangle`` draws a rectangle outline or a filled rectangle whose two opposite corners are ``pt1`` and ``pt2``, or ``r.tl()`` and ``r.br()-Point(1,1)``.



polylines
-------------
Draws several polygonal curves.

.. ocv:function:: void polylines(Mat& img, const Point** pts, const int* npts, int ncontours, bool isClosed, const Scalar& color, int thickness=1, int lineType=8, int shift=0 )

.. ocv:pyfunction:: cv2.polylines(img, pts, isClosed, color[, thickness[, lineType[, shift]]]) -> None

.. ocv:cfunction:: void cvPolyLine( CvArr* img, CvPoint** pts, int* npts, int contours, int isClosed, CvScalar color, int thickness=1, int lineType=8, int shift=0 )

.. ocv:pyoldfunction:: cv.PolyLine(img, polys, isClosed, color, thickness=1, lineType=8, shift=0)-> None

    :param img: Image.

    :param pts: Array of polygonal curves.

    :param npts: Array of polygon vertex counters.

    :param ncontours: Number of curves.

    :param isClosed: Flag indicating whether the drawn polylines are closed or not. If they are closed, the function draws a line from the last vertex of each curve to its first vertex.

    :param color: Polyline color.

    :param thickness: Thickness of the polyline edges.

    :param lineType: Type of the line segments. See the  :ocv:func:`line`  description.

    :param shift: Number of fractional bits in the vertex coordinates.

The function ``polylines`` draws one or more polygonal curves.



putText
-----------
Draws a text string.

.. ocv:function:: void putText( Mat& img, const string& text, Point org,              int fontFace, double fontScale, Scalar color,              int thickness=1, int lineType=8,              bool bottomLeftOrigin=false )

.. ocv:pyfunction:: cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, linetype[, bottomLeftOrigin]]]) -> None

.. ocv:cfunction:: void cvPutText( CvArr* img, const char* text, CvPoint org, const CvFont* font, CvScalar color )
.. ocv:pyoldfunction:: cv.PutText(img, text, org, font, color)-> None

    :param img: Image.

    :param text: Text string to be drawn.

    :param org: Bottom-left corner of the text string in the image.

    :param font: ``CvFont`` structure initialized using :ocv:cfunc:`InitFont`.

    :param fontFace: Font type. One of  ``FONT_HERSHEY_SIMPLEX``,  ``FONT_HERSHEY_PLAIN``, ``FONT_HERSHEY_DUPLEX``,  ``FONT_HERSHEY_COMPLEX``,  ``FONT_HERSHEY_TRIPLEX``, ``FONT_HERSHEY_COMPLEX_SMALL``,  ``FONT_HERSHEY_SCRIPT_SIMPLEX``, or  ``FONT_HERSHEY_SCRIPT_COMPLEX``,
           where each of the font ID's can be combined with  ``FONT_HERSHEY_ITALIC``  to get the slanted letters.

    :param fontScale: Font scale factor that is multiplied by the font-specific base size.

    :param color: Text color.

    :param thickness: Thickness of the lines used to draw a text.

    :param lineType: Line type. See the  ``line``  for details.

    :param bottomLeftOrigin: When true, the image data origin is at the bottom-left corner. Otherwise, it is at the top-left corner.

The function ``putText`` renders the specified text string in the image.
Symbols that cannot be rendered using the specified font are
replaced by question marks. See
:ocv:func:`getTextSize` for a text rendering code example.

