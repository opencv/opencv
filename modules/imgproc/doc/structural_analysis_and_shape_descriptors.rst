Structural Analysis and Shape Descriptors
=========================================

.. highlight:: cpp

.. index:: moments

moments
-----------
.. cpp:function:: Moments moments( InputArray array, bool binaryImage=false )

    Calculates all of the moments up to the third order of a polygon or rasterized shape where the class ``Moments`` is defined as: ::

    class Moments
    {
    public:
        Moments();
        Moments(double m00, double m10, double m01, double m20, double m11,
                double m02, double m30, double m21, double m12, double m03 );
        Moments( const CvMoments& moments );
        operator CvMoments() const;

        // spatial moments
        double  m00, m10, m01, m20, m11, m02, m30, m21, m12, m03;
        // central moments
        double  mu20, mu11, mu02, mu30, mu21, mu12, mu03;
        // central normalized moments
        double  nu20, nu11, nu02, nu30, nu21, nu12, nu03;
    };

    :param array: A raster image (single-channel, 8-bit or floating-point 2D array) or an array ( :math:`1 \times N`  or  :math:`N \times 1` ) of 2D points (``Point``  or  ``Point2f`` ).

    :param binaryImage: If it is true, all non-zero image pixels are treated as 1's. The parameter is used for images only.

The function computes moments, up to the 3rd order, of a vector shape or a rasterized shape.
In case of a raster image, the spatial moments
:math:`\texttt{Moments::m}_{ji}` are computed as:

.. math::

    \texttt{m} _{ji}= \sum _{x,y}  \left ( \texttt{array} (x,y)  \cdot x^j  \cdot y^i \right );

the central moments
:math:`\texttt{Moments::mu}_{ji}` are computed as:

.. math::

    \texttt{mu} _{ji}= \sum _{x,y}  \left ( \texttt{array} (x,y)  \cdot (x -  \bar{x} )^j  \cdot (y -  \bar{y} )^i \right )

where
:math:`(\bar{x}, \bar{y})` is the mass center:

.. math::

    \bar{x} = \frac{\texttt{m}_{10}}{\texttt{m}_{00}} , \; \bar{y} = \frac{\texttt{m}_{01}}{\texttt{m}_{00}};

and the normalized central moments
:math:`\texttt{Moments::nu}_{ij}` are computed as:

.. math::

    \texttt{nu} _{ji}= \frac{\texttt{mu}_{ji}}{\texttt{m}_{00}^{(i+j)/2+1}} .

**Note**:
:math:`\texttt{mu}_{00}=\texttt{m}_{00}`,
:math:`\texttt{nu}_{00}=1` 
:math:`\texttt{nu}_{10}=\texttt{mu}_{10}=\texttt{mu}_{01}=\texttt{mu}_{10}=0` , hence the values are not stored.

The moments of a contour are defined in the same way but computed using Green's formula
(see
http://en.wikipedia.org/wiki/Green_theorem
). So, due to a limited raster resolution, the moments computed for a contour are slightly different from the moments computed for the same rasterized contour.

See Also:
:cpp:func:`contourArea`,
:cpp:func:`arcLength`

.. index:: HuMoments

HuMoments
-------------
.. cpp:function:: void HuMoments( const Moments& moments, double h[7] )

    Calculates the seven Hu invariants.

    :param moments: Input moments computed with  :cpp:func:`moments` .
    :param h: Output Hu invariants.

The function calculates the seven Hu invariants (see
http://en.wikipedia.org/wiki/Image_moment
) defined as:

.. math::

    \begin{array}{l} h[0]= \eta _{20}+ \eta _{02} \\ h[1]=( \eta _{20}- \eta _{02})^{2}+4 \eta _{11}^{2} \\ h[2]=( \eta _{30}-3 \eta _{12})^{2}+ (3 \eta _{21}- \eta _{03})^{2} \\ h[3]=( \eta _{30}+ \eta _{12})^{2}+ ( \eta _{21}+ \eta _{03})^{2} \\ h[4]=( \eta _{30}-3 \eta _{12})( \eta _{30}+ \eta _{12})[( \eta _{30}+ \eta _{12})^{2}-3( \eta _{21}+ \eta _{03})^{2}]+(3 \eta _{21}- \eta _{03})( \eta _{21}+ \eta _{03})[3( \eta _{30}+ \eta _{12})^{2}-( \eta _{21}+ \eta _{03})^{2}] \\ h[5]=( \eta _{20}- \eta _{02})[( \eta _{30}+ \eta _{12})^{2}- ( \eta _{21}+ \eta _{03})^{2}]+4 \eta _{11}( \eta _{30}+ \eta _{12})( \eta _{21}+ \eta _{03}) \\ h[6]=(3 \eta _{21}- \eta _{03})( \eta _{21}+ \eta _{03})[3( \eta _{30}+ \eta _{12})^{2}-( \eta _{21}+ \eta _{03})^{2}]-( \eta _{30}-3 \eta _{12})( \eta _{21}+ \eta _{03})[3( \eta _{30}+ \eta _{12})^{2}-( \eta _{21}+ \eta _{03})^{2}] \\ \end{array}

where
:math:`\eta_{ji}` stands for
:math:`\texttt{Moments::nu}_{ji}` .

These values are proved to be invariants to the image scale, rotation, and reflection except the seventh one, whose sign is changed by reflection. This invariance is proved with the assumption of infinite image resolution. In case of raster images, the computed Hu invariants for the original and transformed images are a bit different.

See Also:
:cpp:func:`matchShapes`

.. index:: findContours

findContours
----------------
.. cpp:function:: void findContours( InputArray image, OutputArrayOfArrays contours,                   OutputArray hierarchy, int mode, int method, Point offset=Point())

.. cpp:function:: void findContours( InputArray image, OutputArrayOfArrays contours, int mode, int method, Point offset=Point())

    Finds contours in a binary image.

    :param image: Source, an 8-bit single-channel image. Non-zero pixels are treated as 1's. Zero pixels remain 0's, so the image is treated as  ``binary`` . You can use  :cpp:func:`compare` ,  :cpp:func:`inRange` ,  :cpp:func:`threshold` ,  :cpp:func:`adaptiveThreshold` ,  :cpp:func:`Canny` , and others to create a binary image out of a grayscale or color one. The function modifies the  ``image``  while extracting the contours.

    :param contours: Detected contours. Each contour is stored as a vector of points.

    :param hiararchy: Optional output vector containing information about the image topology. It has as many elements as the number of contours. For each contour  ``contours[i]`` , the elements  ``hierarchy[i][0]`` ,  ``hiearchy[i][1]`` ,  ``hiearchy[i][2]`` , and  ``hiearchy[i][3]``  are set to 0-based indices in  ``contours``  of the next and previous contours at the same hierarchical level: the first child contour and the parent contour, respectively. If for a contour  ``i``  there are no next, previous, parent, or nested contours, the corresponding elements of  ``hierarchy[i]``  will be negative.

    :param mode: Contour retrieval mode.

            * **CV_RETR_EXTERNAL** retrieves only the extreme outer contours. It sets  ``hierarchy[i][2]=hierarchy[i][3]=-1``  for all the contours.

            * **CV_RETR_LIST** retrieves all of the contours without establishing any hierarchical relationships.

            * **CV_RETR_CCOMP** retrieves all of the contours and organizes them into a two-level hierarchy. At the top level, there are external boundaries of the components. At the second level, there are boundaries of the holes. If there is another contour inside a hole of a connected component, it is still put at the top level.

            * **CV_RETR_TREE** retrieves all of the contours and reconstructs a full hierarchy of nested contours. This full hierarchy is built and shown in the OpenCV  ``contours.c``  demo.

    :param method: Contour approximation method.

            * **CV_CHAIN_APPROX_NONE** stores absolutely all the contour points. That is, any 2 subsequent points ``(x1,y1)`` and ``(x2,y2)`` of the contour will be either horizontal, vertical or diagonal neighbors, that is, ``max(abs(x1-x2),abs(y2-y1))==1``.

            * **CV_CHAIN_APPROX_SIMPLE** compresses horizontal, vertical, and diagonal segments and leaves only their end points. For example, an up-right rectangular contour is encoded with 4 points.

            * **CV_CHAIN_APPROX_TC89_L1,CV_CHAIN_APPROX_TC89_KCOS** applies one of the flavors of the Teh-Chin chain approximation algorithm. See  TehChin89 for details.

    :param offset: Optional offset by which every contour point is shifted. This is useful if the contours are extracted from the image ROI and then they should be analyzed in the whole image context.

The function retrieves contours from the binary image using the algorithm
Suzuki85
. The contours are a useful tool for shape analysis and object detection and recognition. See ``squares.c`` in the OpenCV sample directory.

**Note**:
Source ``image`` is modified by this function.

.. index:: drawContours

drawContours
----------------
.. cpp:function:: void drawContours( InputOutputArray image, InputArrayOfArrays contours,                   int contourIdx, const Scalar& color, int thickness=1, int lineType=8, InputArray hierarchy=None(), int maxLevel=INT_MAX, Point offset=Point() )

    Draws contours outlines or filled contours.

    :param image: Destination image.

    :param contours: All the input contours. Each contour is stored as a point vector.

    :param contourIdx: Parameter indicating a contour to draw. If it is negative, all the contours are drawn.

    :param color: Color of the contours.
	
    :param thickness: Thickness of lines the contours are drawn with. If it is negative (for example,  ``thickness=CV_FILLED`` ), the contour interiors are
        drawn.

    :param lineType: Line connectivity. See  :cpp:func:`line`  for details.

    :param hierarchy: Optional information about hierarchy. It is only needed if you want to draw only some of the  contours (see  ``maxLevel`` ).

    :param maxLevel: Maximal level for drawn contours. If it is 0, only
        the specified contour is drawn. If it is 1, the function draws the contour(s) and all the nested contours. If it is 2, the function draws the contours, all the nested contours, all the nested-to-nested contours, and so on. This parameter is only taken into account when there is  ``hierarchy``  available.

    :param offset: Optional contour shift parameter. Shift all the drawn contours by the specified  :math:`\texttt{offset}=(dx,dy)` .

The function draws contour outlines in the image if
:math:`\texttt{thickness} \ge 0` or fills the area bounded by the contours if
:math:`\texttt{thickness}<0` . Here is the example on how to retrieve connected components from the binary image and label them: ::

    #include "cv.h"
    #include "highgui.h"

    using namespace cv;

    int main( int argc, char** argv )
    {
        Mat src;
        // the first command-line parameter must be a filename of the binary
        // (black-n-white) image
        if( argc != 2 || !(src=imread(argv[1], 0)).data)
            return -1;

        Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);

        src = src > 1;
        namedWindow( "Source", 1 );
        imshow( "Source", src );

        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;

        findContours( src, contours, hierarchy,
            CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );

        // iterate through all the top-level contours,
        // draw each connected component with its own random color
        int idx = 0;
        for( ; idx >= 0; idx = hierarchy[idx][0] )
        {
            Scalar color( rand()&255, rand()&255, rand()&255 );
            drawContours( dst, contours, idx, color, CV_FILLED, 8, hierarchy );
        }

        namedWindow( "Components", 1 );
        imshow( "Components", dst );
        waitKey(0);
    }

.. index:: approxPolyDP

approxPolyDP
----------------
.. cpp:function:: void approxPolyDP( InputArray curve, OutputArray approxCurve, double epsilon, bool closed )

    Approximates a polygonal curve(s) with the specified precision.

    :param curve: Input vector of 2d point, stored in ``std::vector`` or ``Mat``.

    :param approxCurve: Result of the approximation. The type should match the type of the input curve.

    :param epsilon: Parameter specifying the approximation accuracy. This is the maximum distance between the original curve and its approximation.

    :param closed: If true, the approximated curve is closed (its first and last vertices are connected). Otherwise, it is not closed.

The functions ``approxPolyDP`` approximate a curve or a polygon with another curve/polygon with less vertices, so that the distance between them is less or equal to the specified precision. It uses the Douglas-Peucker algorithm
http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm

See http://code.ros.org/svn/opencv/trunk/opencv/samples/cpp/contours.cpp on how to use the function.

.. index:: arcLength

arcLength
-------------
.. cpp:function:: double arcLength( InputArray curve, bool closed )

    Calculates a contour perimeter or a curve length.

    :param curve: Input vector of 2D points, stored in ``std::vector`` or ``Mat``.

    :param closed: Flag indicating whether the curve is closed or not.

The function computes a curve length or a closed contour perimeter.

.. index:: boundingRect

boundingRect
----------------
.. cpp:function:: Rect boundingRect( InputArray points )

    Calculates the up-right bounding rectangle of a point set.

    :param points: Input 2D point set, stored in ``std::vector`` or ``Mat``.

The function calculates and returns the minimal up-right bounding rectangle for the specified point set.

.. index:: estimateRigidTransform

estimateRigidTransform
--------------------------
.. cpp:function:: Mat estimateRigidTransform( InputArray srcpt, InputArray dstpt, bool fullAffine )

    Computes an optimal affine transformation between two 2D point sets.

    :param srcpt: The first input 2D point set, stored in ``std::vector`` or ``Mat``.

    :param dst: The second input 2D point set of the same size and the same type as  ``A`` .

    :param fullAffine: If true, the function finds an optimal affine transformation with no additional resrictions (6 degrees of freedom). Otherwise, the class of transformations to choose from is limited to combinations of translation, rotation, and uniform scaling (5 degrees of freedom).

The function finds an optimal affine transform
:math:`[A|b]` (a
:math:`2 \times 3` floating-point matrix) that approximates best the transformation from
:math:`\texttt{srcpt}_i` to
:math:`\texttt{dstpt}_i` : 

.. math::

    [A^*|b^*] = arg  \min _{[A|b]}  \sum _i  \| \texttt{dstpt} _i - A { \texttt{srcpt} _i}^T - b  \| ^2

where
:math:`[A|b]` can be either arbitrary (when ``fullAffine=true`` ) or have form

.. math::

    \begin{bmatrix} a_{11} & a_{12} & b_1  \\ -a_{12} & a_{11} & b_2  \end{bmatrix}

when ``fullAffine=false`` .

See Also:
:cpp:func:`getAffineTransform`,
:cpp:func:`getPerspectiveTransform`,
:cpp:func:`findHomography`

.. index:: estimateAffine3D

estimateAffine3D
--------------------
.. cpp:function:: int estimateAffine3D(InputArray srcpt, InputArray dstpt, OutputArray out,                     OutputArray outliers, double ransacThreshold = 3.0, double confidence = 0.99)

    Computes an optimal affine transformation between two 3D point sets.

    :param srcpt: The first input 3D point set.

    :param dstpt: The second input 3D point set.

    :param out: Output 3D affine transformation matrix  :math:`3 \times 4` .

    :param outliers: Output vector indicating which points are outliers.

    :param ransacThreshold: Maximum reprojection error in the RANSAC algorithm to consider a point as an inlier.

    :param confidence: The confidence level, between 0 and 1, that the estimated transformation will have. Anything between 0.95 and 0.99 is usually good enough. Too close to 1 values can slow down the estimation too much, lower than 0.8-0.9 confidence values can result in an incorrectly estimated transformation.

The function estimates an optimal 3D affine transformation between two 3D point sets using the RANSAC algorithm.

.. index:: contourArea

contourArea
---------------
.. cpp:function:: double contourArea( InputArray contour )

    Calculates a contour area.

    :param contour: Input vector of 2d points (contour vertices), stored in ``std::vector`` or ``Mat``.

The function computes a contour area. Similarly to
:cpp:func:`moments` , the area is computed using the Green formula. Thus, the returned area and the number of non-zero pixels, if you draw the contour using
:cpp:func:`drawContours` or
:cpp:func:`fillPoly` , can be different.
Here is a short example: ::

    vector<Point> contour;
    contour.push_back(Point2f(0, 0));
    contour.push_back(Point2f(10, 0));
    contour.push_back(Point2f(10, 10));
    contour.push_back(Point2f(5, 4));

    double area0 = contourArea(contour);
    vector<Point> approx;
    approxPolyDP(contour, approx, 5, true);
    double area1 = contourArea(approx);

    cout << "area0 =" << area0 << endl <<
            "area1 =" << area1 << endl <<
            "approx poly vertices" << approx.size() << endl;

.. index:: convexHull

convexHull
--------------
.. cpp:function:: void convexHull( InputArray points, OutputArray hull, bool clockwise=false, bool returnPoints=true )

    Finds the convex hull of a point set.

    :param points: Input 2D point set, stored in ``std::vector`` or ``Mat``.

    :param hull: Output convex hull. It is either an integer vector of indices or vector of points. In the first case the ``hull`` elements are 0-based indices of the convex hull points in the original array (since the set of convex hull points is a subset of the original point set). In the second case ``hull`` elements will be the convex hull points themselves.

    :param clockwise: Orientation flag. If true, the output convex hull will be oriented clockwise. Otherwise, it will be oriented counter-clockwise. The usual screen coordinate system is assumed where the origin is at the top-left corner, x axis is oriented to the right, and y axis is oriented downwards.
    
    :param returnPoints: Operation flag. In the case of matrix, when the flag is true, the function will return convex hull points, otherwise it will return indices of the convex hull points. When the output array is ``std::vector``, the flag is ignored, and the output depends on the type of the vector - ``std::vector<int>`` implies ``returnPoints=true``, ``std::vector<Point>`` implies ``returnPoints=false``.

The functions find the convex hull of a 2D point set using the Sklansky's algorithm
Sklansky82
that has
*O(N logN)* complexity in the current implementation. See the OpenCV sample ``convexhull.cpp`` that demonstrates the usage of different function variants.

.. index:: fitEllipse

fitEllipse
--------------
.. cpp:function:: RotatedRect fitEllipse( InputArray points )

    Fits an ellipse around a set of 2D points.

    :param points: Input vector of 2D points, stored in ``std::vector<>`` or ``Mat``.

The function calculates the ellipse that fits (in least-squares sense) a set of 2D points best of all. It returns the rotated rectangle in which the ellipse is inscribed.

.. index:: fitLine

fitLine
-----------
.. cpp:function:: void fitLine( InputArray points, OutputArray line, int distType, double param, double reps, double aeps )

    Fits a line to a 2D or 3D point set.

    :param points: Input vector of 2D or 3D points, stored in ``std::vector<>`` or ``Mat``.

    :param line: Output line parameters. In case of 2D fitting it should be a vector of 4 elements (like ``Vec4f``) - ``(vx, vy, x0, y0)``,  where  ``(vx, vy)``  is a normalized vector collinear to the line and  ``(x0, y0)``  is a point on the line. In case of 3D fitting, it should be a vector of 6 elements (like  ``Vec6f``) - ``(vx, vy, vz, x0, y0, z0)``, where ``(vx, vy, vz)`` is a normalized vector collinear to the line and ``(x0, y0, z0)`` is a point on the line.

    :param distType: Distance used by the M-estimator (see the discussion).

    :param param: Numerical parameter ( ``C`` ) for some types of distances. If it is 0, an optimal value is chosen.

    :param reps, aeps: Sufficient accuracy for the radius (distance between the coordinate origin and the line) and angle, respectively. 0.01 would be a good default value for both.

The function ``fitLine`` fits a line to a 2D or 3D point set by minimizing
:math:`\sum_i \rho(r_i)` where
:math:`r_i` is a distance between the
:math:`i^{th}` point, the line and
:math:`\rho(r)` is a distance function, one of:

* distType=CV\_DIST\_L2

    .. math::

        \rho (r) = r^2/2  \quad \text{(the simplest and the fastest least-squares method)}

* distType=CV\_DIST\_L1

    .. math::

        \rho (r) = r

* distType=CV\_DIST\_L12

    .. math::

        \rho (r) = 2  \cdot ( \sqrt{1 + \frac{r^2}{2}} - 1)

* distType=CV\_DIST\_FAIR

    .. math::

        \rho \left (r \right ) = C^2  \cdot \left (  \frac{r}{C} -  \log{\left(1 + \frac{r}{C}\right)} \right )  \quad \text{where} \quad C=1.3998

* distType=CV\_DIST\_WELSCH

    .. math::

        \rho \left (r \right ) =  \frac{C^2}{2} \cdot \left ( 1 -  \exp{\left(-\left(\frac{r}{C}\right)^2\right)} \right )  \quad \text{where} \quad C=2.9846

* distType=CV\_DIST\_HUBER

    .. math::

        \rho (r) =  \fork{r^2/2}{if $r < C$}{C \cdot (r-C/2)}{otherwise} \quad \text{where} \quad C=1.345

The algorithm is based on the M-estimator (
http://en.wikipedia.org/wiki/M-estimator
) technique that iteratively fits the line using the weighted least-squares algorithm. After each iteration the weights
:math:`w_i` are adjusted to be inversely proportional to
:math:`\rho(r_i)` .

.. index:: isContourConvex

isContourConvex
-------------------
.. cpp:function:: bool isContourConvex( InputArray contour )

    Tests a contour convexity.

    :param contour: The input vector of 2D points, stored in ``std::vector<>`` or ``Mat``.

The function tests whether the input contour is convex or not. The contour must be simple, that is, without self-intersections. Otherwise, the function output is undefined.

.. index:: minAreaRect

minAreaRect
---------------
.. cpp:function:: RotatedRect minAreaRect( InputArray points )

    Finds a rotated rectangle of the minimum area enclosing the input 2D point set.

    :param points: The input vector of 2D points, stored in ``std::vector<>`` or ``Mat``.

The function calculates and returns the minimum-area bounding rectangle (possibly rotated) for a specified point set. See the OpenCV sample ``minarea.cpp`` .

.. index:: minEnclosingCircle

minEnclosingCircle
----------------------
.. cpp:function:: void minEnclosingCircle( InputArray points, Point2f& center, float& radius )

    Finds a circle of the minimum area enclosing a 2D point set.

    :param points: The input vector of 2D points, stored in ``std::vector<>`` or ``Mat``.

    :param center: Output center of the circle.

    :param radius: Output radius of the circle.

The function finds the minimal enclosing circle of a 2D point set using an iterative algorithm. See the OpenCV sample ``minarea.cpp`` .

.. index:: matchShapes

matchShapes
---------------
.. cpp:function:: double matchShapes( InputArray object1, InputArray object2, int method, double parameter=0 )

    Compares two shapes.

    :param object1: The first contour or grayscale image.

    :param object2: The second contour or grayscale image.

    :param method: Comparison method: ``CV_CONTOUR_MATCH_I1`` , \ ``CV_CONTOURS_MATCH_I2`` \
        or ``CV_CONTOURS_MATCH_I3``  (see the details below).

    :param parameter: Method-specific parameter (not supported now).

The function compares two shapes. All three implemented methods use the Hu invariants (see
:cpp:func:`HuMoments` ) as follows (
:math:`A` denotes ``object1``,:math:`B` denotes ``object2`` ):

* method=CV\_CONTOUR\_MATCH\_I1

    .. math::

        I_1(A,B) =  \sum _{i=1...7}  \left |  \frac{1}{m^A_i} -  \frac{1}{m^B_i} \right |

* method=CV\_CONTOUR\_MATCH\_I2

    .. math::

        I_2(A,B) =  \sum _{i=1...7}  \left | m^A_i - m^B_i  \right |

* method=CV\_CONTOUR\_MATCH\_I3

    .. math::

        I_3(A,B) =  \sum _{i=1...7}  \frac{ \left| m^A_i - m^B_i \right| }{ \left| m^A_i \right| }

where

.. math::

    \begin{array}{l} m^A_i =  \mathrm{sign} (h^A_i)  \cdot \log{h^A_i} \\ m^B_i =  \mathrm{sign} (h^B_i)  \cdot \log{h^B_i} \end{array}

and
:math:`h^A_i, h^B_i` are the Hu moments of
:math:`A` and
:math:`B` , respectively.

.. index:: pointPolygonTest

pointPolygonTest
--------------------
.. cpp:function:: double pointPolygonTest( InputArray contour, Point2f pt, bool measureDist )

    Performs a point-in-contour test.

    :param contour: Input contour.

    :param pt: Point tested against the contour.

    :param measureDist: If true, the function estimates the signed distance from the point to the nearest contour edge. Otherwise, the function only checks if the point is inside a contour or not.

The function determines whether the
point is inside a contour, outside, or lies on an edge (or coincides
with a vertex). It returns positive (inside), negative (outside), or zero (on an edge) value,
correspondingly. When ``measureDist=false`` , the return value
is +1, -1, and 0, respectively. Otherwise, the return value
is a signed distance between the point and the nearest contour
edge.

Here is a sample output of the function where each image pixel is tested against the contour.

.. image:: pics/pointpolygon.png

