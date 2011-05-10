Structural Analysis and Shape Descriptors
=========================================

.. highlight:: python



.. index:: ApproxChains

.. _ApproxChains:

ApproxChains
------------




.. function:: ApproxChains(src_seq,storage,method=CV_CHAIN_APPROX_SIMPLE,parameter=0,minimal_perimeter=0,recursive=0)-> chains

    Approximates Freeman chain(s) with a polygonal curve.





    
    :param src_seq: Pointer to the chain that can refer to other chains 
    
    :type src_seq: :class:`CvSeq`
    
    
    :param storage: Storage location for the resulting polylines 
    
    :type storage: :class:`CvMemStorage`
    
    
    :param method: Approximation method (see the description of the function  :ref:`FindContours` ) 
    
    :type method: int
    
    
    :param parameter: Method parameter (not used now) 
    
    :type parameter: float
    
    
    :param minimal_perimeter: Approximates only those contours whose perimeters are not less than  ``minimal_perimeter`` . Other chains are removed from the resulting structure 
    
    :type minimal_perimeter: int
    
    
    :param recursive: If not 0, the function approximates all chains that access can be obtained to from  ``src_seq``  by using the  ``h_next``  or  ``v_next links`` . If 0, the single chain is approximated 
    
    :type recursive: int
    
    
    
This is a stand-alone approximation routine. The function 
``cvApproxChains``
works exactly in the same way as 
:ref:`FindContours`
with the corresponding approximation flag. The function returns pointer to the first resultant contour. Other approximated contours, if any, can be accessed via the 
``v_next``
or 
``h_next``
fields of the returned structure.


.. index:: ApproxPoly

.. _ApproxPoly:

ApproxPoly
----------




.. function:: 
ApproxPoly(src_seq, storage, method, parameter=0, parameter2=0) -> sequence


    Approximates polygonal curve(s) with the specified precision.





    
    :param src_seq: Sequence of an array of points 
    
    :type src_seq: :class:`CvArr` or :class:`CvSeq`
    
    
    :param storage: Container for the approximated contours. If it is NULL, the input sequences' storage is used 
    
    :type storage: :class:`CvMemStorage`
    
    
    :param method: Approximation method; only  ``CV_POLY_APPROX_DP``  is supported, that corresponds to the Douglas-Peucker algorithm 
    
    :type method: int
    
    
    :param parameter: Method-specific parameter; in the case of  ``CV_POLY_APPROX_DP``  it is a desired approximation accuracy 
    
    :type parameter: float
    
    
    :param parameter2: If case if  ``src_seq``  is a sequence, the parameter determines whether the single sequence should be approximated or all sequences on the same level or below  ``src_seq``  (see  :ref:`FindContours`  for description of hierarchical contour structures). If  ``src_seq``  is an array CvMat* of points, the parameter specifies whether the curve is closed ( ``parameter2`` !=0) or not ( ``parameter2``  =0) 
    
    :type parameter2: int
    
    
    
The function approximates one or more curves and
returns the approximation result[s]. In the case of multiple curves,
the resultant tree will have the same structure as the input one (1:1
correspondence).


.. index:: ArcLength

.. _ArcLength:

ArcLength
---------




.. function:: ArcLength(curve,slice=CV_WHOLE_SEQ,isClosed=-1)-> double

    Calculates the contour perimeter or the curve length.





    
    :param curve: Sequence or array of the curve points 
    
    :type curve: :class:`CvArr` or :class:`CvSeq`
    
    
    :param slice: Starting and ending points of the curve, by default, the whole curve length is calculated 
    
    :type slice: :class:`CvSlice`
    
    
    :param isClosed: Indicates whether the curve is closed or not. There are 3 cases: 
        
               
        
        *   :math:`\texttt{isClosed}=0`  the curve is assumed to be unclosed.
               
        
        *   :math:`\texttt{isClosed}>0`  the curve is assumed to be closed.
               
        
        *   :math:`\texttt{isClosed}<0`  if curve is sequence, the flag  ``CV_SEQ_FLAG_CLOSED``  of  ``((CvSeq*)curve)->flags``  is checked to determine if the curve is closed or not, otherwise (curve is represented by array (CvMat*) of points) it is assumed to be unclosed. 
            
    
    :type isClosed: int
    
    
    
The function calculates the length or curve as the sum of lengths of segments between subsequent points


.. index:: BoundingRect

.. _BoundingRect:

BoundingRect
------------




.. function:: BoundingRect(points,update=0)-> CvRect

    Calculates the up-right bounding rectangle of a point set.





    
    :param points: 2D point set, either a sequence or vector ( ``CvMat`` ) of points 
    
    :type points: :class:`CvArr` or :class:`CvSeq`
    
    
    :param update: The update flag. See below. 
    
    :type update: int
    
    
    
The function returns the up-right bounding rectangle for a 2d point set.
Here is the list of possible combination of the flag values and type of 
``points``
:


.. table::

    ======  =========================  =======================================================================================================
    update  points                     action  \                                                                                              
    ======  =========================  =======================================================================================================
    0       ``CvContour*``             the bounding rectangle is not calculated, but it is taken from  ``rect`` field of the contour header. \
    1       ``CvContour*``             the bounding rectangle is calculated and written to  ``rect`` field of the contour header. \           
    0       ``CvSeq*`` or  ``CvMat*``  the bounding rectangle is calculated and returned. \                                                   
    1       ``CvSeq*`` or  ``CvMat*``  runtime error is raised. \                                                                             
    ======  =========================  =======================================================================================================


.. index:: BoxPoints

.. _BoxPoints:

BoxPoints
---------




.. function:: BoxPoints(box)-> points

    Finds the box vertices.





    
    :param box: Box 
    
    :type box: :class:`CvBox2D`
    
    
    :param points: Array of vertices 
    
    :type points: :class:`CvPoint2D32f_4`
    
    
    
The function calculates the vertices of the input 2d box.


.. index:: CalcPGH

.. _CalcPGH:

CalcPGH
-------




.. function:: CalcPGH(contour,hist)-> None

    Calculates a pair-wise geometrical histogram for a contour.





    
    :param contour: Input contour. Currently, only integer point coordinates are allowed 
    
    
    :param hist: Calculated histogram; must be two-dimensional 
    
    
    
The function calculates a
2D pair-wise geometrical histogram (PGH), described in
:ref:`Iivarinen97`
for the contour. The algorithm considers every pair of contour
edges. The angle between the edges and the minimum/maximum distances
are determined for every pair. To do this each of the edges in turn
is taken as the base, while the function loops through all the other
edges. When the base edge and any other edge are considered, the minimum
and maximum distances from the points on the non-base edge and line of
the base edge are selected. The angle between the edges defines the row
of the histogram in which all the bins that correspond to the distance
between the calculated minimum and maximum distances are incremented
(that is, the histogram is transposed relatively to the 
:ref:`Iivarninen97`
definition). The histogram can be used for contour matching.


.. index:: CalcEMD2

.. _CalcEMD2:

CalcEMD2
--------




.. function:: CalcEMD2(signature1, signature2, distance_type, distance_func = None, cost_matrix=None, flow=None, lower_bound=None, userdata = None) -> float

    Computes the "minimal work" distance between two weighted point configurations.





    
    :param signature1: First signature, a  :math:`\texttt{size1}\times \texttt{dims}+1`  floating-point matrix. Each row stores the point weight followed by the point coordinates. The matrix is allowed to have a single column (weights only) if the user-defined cost matrix is used 
    
    :type signature1: :class:`CvArr`
    
    
    :param signature2: Second signature of the same format as  ``signature1`` , though the number of rows may be different. The total weights may be different, in this case an extra "dummy" point is added to either  ``signature1``  or  ``signature2`` 
    
    :type signature2: :class:`CvArr`
    
    
    :param distance_type: Metrics used;  ``CV_DIST_L1, CV_DIST_L2`` , and  ``CV_DIST_C``  stand for one of the standard metrics;  ``CV_DIST_USER``  means that a user-defined function  ``distance_func``  or pre-calculated  ``cost_matrix``  is used 
    
    :type distance_type: int
    
    
    :param distance_func: The user-supplied distance function. It takes coordinates of two points  ``pt0``  and  ``pt1`` , and returns the distance between the points, with sigature ``
                func(pt0, pt1, userdata) -> float`` 
    
    :type distance_func: :class:`PyCallableObject`
    
    
    :param cost_matrix: The user-defined  :math:`\texttt{size1}\times \texttt{size2}`  cost matrix. At least one of  ``cost_matrix``  and  ``distance_func``  must be NULL. Also, if a cost matrix is used, lower boundary (see below) can not be calculated, because it needs a metric function 
    
    :type cost_matrix: :class:`CvArr`
    
    
    :param flow: The resultant  :math:`\texttt{size1} \times \texttt{size2}`  flow matrix:  :math:`\texttt{flow}_{i,j}`  is a flow from  :math:`i`  th point of  ``signature1``  to  :math:`j`  th point of  ``signature2`` 
    
    :type flow: :class:`CvArr`
    
    
    :param lower_bound: Optional input/output parameter: lower boundary of distance between the two signatures that is a distance between mass centers. The lower boundary may not be calculated if the user-defined cost matrix is used, the total weights of point configurations are not equal, or if the signatures consist of weights only (i.e. the signature matrices have a single column). The user  **must**  initialize  ``*lower_bound`` . If the calculated distance between mass centers is greater or equal to  ``*lower_bound``  (it means that the signatures are far enough) the function does not calculate EMD. In any case  ``*lower_bound``  is set to the calculated distance between mass centers on return. Thus, if user wants to calculate both distance between mass centers and EMD,  ``*lower_bound``  should be set to 0 
    
    :type lower_bound: float
    
    
    :param userdata: Pointer to optional data that is passed into the user-defined distance function 
    
    :type userdata: object
    
    
    
The function computes the earth mover distance and/or
a lower boundary of the distance between the two weighted point
configurations. One of the applications described in 
:ref:`RubnerSept98`
is
multi-dimensional histogram comparison for image retrieval. EMD is a a
transportation problem that is solved using some modification of a simplex
algorithm, thus the complexity is exponential in the worst case, though, on average
it is much faster. In the case of a real metric the lower boundary
can be calculated even faster (using linear-time algorithm) and it can
be used to determine roughly whether the two signatures are far enough
so that they cannot relate to the same object.


.. index:: CheckContourConvexity

.. _CheckContourConvexity:

CheckContourConvexity
---------------------




.. function:: CheckContourConvexity(contour)-> int

    Tests contour convexity.





    
    :param contour: Tested contour (sequence or array of points) 
    
    :type contour: :class:`CvArr` or :class:`CvSeq`
    
    
    
The function tests whether the input contour is convex or not. The contour must be simple, without self-intersections.


.. index:: CvConvexityDefect

.. _CvConvexityDefect:

CvConvexityDefect
-----------------



.. class:: CvConvexityDefect



A single contour convexity defect, represented by a tuple 
``(start, end, depthpoint, depth)``
.



    
    
    .. attribute:: start
    
    
    
        (x, y) point of the contour where the defect begins 
    
    
    
    .. attribute:: end
    
    
    
        (x, y) point of the contour where the defect ends 
    
    
    
    .. attribute:: depthpoint
    
    
    
        (x, y) point farthest from the convex hull point within the defect 
    
    
    
    .. attribute:: depth
    
    
    
        distance between the farthest point and the convex hull 
    
    
    


.. image:: ../pics/defects.png




.. index:: ContourArea

.. _ContourArea:

ContourArea
-----------




.. function:: ContourArea(contour,slice=CV_WHOLE_SEQ)-> double

    Calculates the area of a whole contour or a contour section.





    
    :param contour: Contour (sequence or array of vertices) 
    
    :type contour: :class:`CvArr` or :class:`CvSeq`
    
    
    :param slice: Starting and ending points of the contour section of interest, by default, the area of the whole contour is calculated 
    
    :type slice: :class:`CvSlice`
    
    
    
The function calculates the area of a whole contour
or a contour section. In the latter case the total area bounded by the
contour arc and the chord connecting the 2 selected points is calculated
as shown on the picture below:



.. image:: ../pics/contoursecarea.png



Orientation of the contour affects the area sign, thus the function may return a 
*negative*
result. Use the 
``fabs()``
function from C runtime to get the absolute value of the area.


.. index:: ContourFromContourTree

.. _ContourFromContourTree:

ContourFromContourTree
----------------------




.. function:: ContourFromContourTree(tree,storage,criteria)-> contour

    Restores a contour from the tree.





    
    :param tree: Contour tree 
    
    
    :param storage: Container for the reconstructed contour 
    
    
    :param criteria: Criteria, where to stop reconstruction 
    
    
    
The function restores the contour from its binary tree representation. The parameter 
``criteria``
determines the accuracy and/or the number of tree levels used for reconstruction, so it is possible to build an approximated contour. The function returns the reconstructed contour.


.. index:: ConvexHull2

.. _ConvexHull2:

ConvexHull2
-----------




.. function:: ConvexHull2(points,storage,orientation=CV_CLOCKWISE,return_points=0)-> convex_hull

    Finds the convex hull of a point set.





    
    :param points: Sequence or array of 2D points with 32-bit integer or floating-point coordinates 
    
    :type points: :class:`CvArr` or :class:`CvSeq`
    
    
    :param storage: The destination array (CvMat*) or memory storage (CvMemStorage*) that will store the convex hull. If it is an array, it should be 1d and have the same number of elements as the input array/sequence. On output the header is modified as to truncate the array down to the hull size.  If  ``storage``  is NULL then the convex hull will be stored in the same storage as the input sequence 
    
    :type storage: :class:`CvMemStorage`
    
    
    :param orientation: Desired orientation of convex hull:  ``CV_CLOCKWISE``  or  ``CV_COUNTER_CLOCKWISE`` 
    
    :type orientation: int
    
    
    :param return_points: If non-zero, the points themselves will be stored in the hull instead of indices if  ``storage``  is an array, or pointers if  ``storage``  is memory storage 
    
    :type return_points: int
    
    
    
The function finds the convex hull of a 2D point set using Sklansky's algorithm. If 
``storage``
is memory storage, the function creates a sequence containing the hull points or pointers to them, depending on 
``return_points``
value and returns the sequence on output.  If 
``storage``
is a CvMat, the function returns NULL.


.. index:: ConvexityDefects

.. _ConvexityDefects:

ConvexityDefects
----------------




.. function:: ConvexityDefects(contour,convexhull,storage)-> convexity_defects

    Finds the convexity defects of a contour.





    
    :param contour: Input contour 
    
    :type contour: :class:`CvArr` or :class:`CvSeq`
    
    
    :param convexhull: Convex hull obtained using  :ref:`ConvexHull2`  that should contain pointers or indices to the contour points, not the hull points themselves (the  ``return_points``  parameter in  :ref:`ConvexHull2`  should be 0) 
    
    :type convexhull: :class:`CvSeq`
    
    
    :param storage: Container for the output sequence of convexity defects. If it is NULL, the contour or hull (in that order) storage is used 
    
    :type storage: :class:`CvMemStorage`
    
    
    
The function finds all convexity defects of the input contour and returns a sequence of the CvConvexityDefect structures.


.. index:: CreateContourTree

.. _CreateContourTree:

CreateContourTree
-----------------




.. function:: CreateContourTree(contour,storage,threshold)-> contour_tree

    Creates a hierarchical representation of a contour.





    
    :param contour: Input contour 
    
    
    :param storage: Container for output tree 
    
    
    :param threshold: Approximation accuracy 
    
    
    
The function creates a binary tree representation for the input 
``contour``
and returns the pointer to its root. If the parameter 
``threshold``
is less than or equal to 0, the function creates a full binary tree representation. If the threshold is greater than 0, the function creates a representation with the precision 
``threshold``
: if the vertices with the interceptive area of its base line are less than 
``threshold``
, the tree should not be built any further. The function returns the created tree.


.. index:: FindContours

.. _FindContours:

FindContours
------------




.. function:: FindContours(image, storage, mode=CV_RETR_LIST, method=CV_CHAIN_APPROX_SIMPLE, offset=(0,0)) -> cvseq

    Finds the contours in a binary image.





    
    :param image: The source, an 8-bit single channel image. Non-zero pixels are treated as 1's, zero pixels remain 0's - the image is treated as  ``binary`` . To get such a binary image from grayscale, one may use  :ref:`Threshold` ,  :ref:`AdaptiveThreshold`  or  :ref:`Canny` . The function modifies the source image's content 
    
    :type image: :class:`CvArr`
    
    
    :param storage: Container of the retrieved contours 
    
    :type storage: :class:`CvMemStorage`
    
    
    :param mode: Retrieval mode 
        
                
            * **CV_RETR_EXTERNAL** retrives only the extreme outer contours 
            
               
            * **CV_RETR_LIST** retrieves all of the contours and puts them in the list 
            
               
            * **CV_RETR_CCOMP** retrieves all of the contours and organizes them into a two-level hierarchy: on the top level are the external boundaries of the components, on the second level are the boundaries of the holes 
            
               
            * **CV_RETR_TREE** retrieves all of the contours and reconstructs the full hierarchy of nested contours 
            
            
    
    :type mode: int
    
    
    :param method: Approximation method (for all the modes, except  ``CV_LINK_RUNS`` , which uses built-in approximation) 
        
                
            * **CV_CHAIN_CODE** outputs contours in the Freeman chain code. All other methods output polygons (sequences of vertices) 
            
               
            * **CV_CHAIN_APPROX_NONE** translates all of the points from the chain code into points 
            
               
            * **CV_CHAIN_APPROX_SIMPLE** compresses horizontal, vertical, and diagonal segments and leaves only their end points 
            
               
            * **CV_CHAIN_APPROX_TC89_L1,CV_CHAIN_APPROX_TC89_KCOS** applies one of the flavors of the Teh-Chin chain approximation algorithm. 
            
               
            * **CV_LINK_RUNS** uses a completely different contour retrieval algorithm by linking horizontal segments of 1's. Only the  ``CV_RETR_LIST``  retrieval mode can be used with this method. 
            
            
    
    :type method: int
    
    
    :param offset: Offset, by which every contour point is shifted. This is useful if the contours are extracted from the image ROI and then they should be analyzed in the whole image context 
    
    :type offset: :class:`CvPoint`
    
    
    
The function retrieves contours from the binary image using the algorithm
Suzuki85
. The contours are a useful tool for shape analysis and
object detection and recognition.

The function retrieves contours from the
binary image and returns the number of retrieved contours. The
pointer 
``first_contour``
is filled by the function. It will
contain a pointer to the first outermost contour or 
``NULL``
if no
contours are detected (if the image is completely black). Other
contours may be reached from 
``first_contour``
using the
``h_next``
and 
``v_next``
links. The sample in the
:ref:`DrawContours`
discussion shows how to use contours for
connected component detection. Contours can be also used for shape
analysis and object recognition - see
``squares.py``
in the OpenCV sample directory.

**Note:**
the source 
``image``
is modified by this function.


.. index:: FitEllipse2

.. _FitEllipse2:

FitEllipse2
-----------




.. function:: FitEllipse2(points)-> Box2D

    Fits an ellipse around a set of 2D points.





    
    :param points: Sequence or array of points 
    
    :type points: :class:`CvArr`
    
    
    
The function calculates the ellipse that fits best
(in least-squares sense) around a set of 2D points. The meaning of the
returned structure fields is similar to those in 
:ref:`Ellipse`
except
that 
``size``
stores the full lengths of the ellipse axises,
not half-lengths.


.. index:: FitLine

.. _FitLine:

FitLine
-------




.. function:: FitLine(points, dist_type, param, reps, aeps) -> line

    Fits a line to a 2D or 3D point set.





    
    :param points: Sequence or array of 2D or 3D points with 32-bit integer or floating-point coordinates 
    
    :type points: :class:`CvArr`
    
    
    :param dist_type: The distance used for fitting (see the discussion) 
    
    :type dist_type: int
    
    
    :param param: Numerical parameter ( ``C`` ) for some types of distances, if 0 then some optimal value is chosen 
    
    :type param: float
    
    
    :param reps: Sufficient accuracy for the radius (distance between the coordinate origin and the line).  0.01 is a good default value. 
    
    :type reps: float
    
    
    :param aeps: Sufficient accuracy for the angle.  0.01 is a good default value. 
    
    :type aeps: float
    
    
    :param line: The output line parameters. In the case of a 2d fitting,
        it is    a tuple   of 4 floats  ``(vx, vy, x0, y0)``  where  ``(vx, vy)``  is a normalized vector collinear to the
        line and  ``(x0, y0)``  is some point on the line. in the case of a
        3D fitting it is    a tuple   of 6 floats  ``(vx, vy, vz, x0, y0, z0)`` 
        where  ``(vx, vy, vz)``  is a normalized vector collinear to the line
        and  ``(x0, y0, z0)``  is some point on the line 
    
    :type line: object
    
    
    
The function fits a line to a 2D or 3D point set by minimizing 
:math:`\sum_i \rho(r_i)`
where 
:math:`r_i`
is the distance between the 
:math:`i`
th point and the line and 
:math:`\rho(r)`
is a distance function, one of:



    

* dist\_type=CV\_DIST\_L2
    
    
    .. math::
    
        \rho (r) = r^2/2  \quad \text{(the simplest and the fastest least-squares method)} 
    
    
    

* dist\_type=CV\_DIST\_L1
    
    
    .. math::
    
        \rho (r) = r  
    
    
    

* dist\_type=CV\_DIST\_L12
    
    
    .. math::
    
        \rho (r) = 2  \cdot ( \sqrt{1 + \frac{r^2}{2}} - 1)  
    
    
    

* dist\_type=CV\_DIST\_FAIR
    
    
    .. math::
    
        \rho \left (r \right ) = C^2  \cdot \left (  \frac{r}{C} -  \log{\left(1 + \frac{r}{C}\right)} \right )  \quad \text{where} \quad C=1.3998  
    
    
    

* dist\_type=CV\_DIST\_WELSCH
    
    
    .. math::
    
        \rho \left (r \right ) =  \frac{C^2}{2} \cdot \left ( 1 -  \exp{\left(-\left(\frac{r}{C}\right)^2\right)} \right )  \quad \text{where} \quad C=2.9846  
    
    
    

* dist\_type=CV\_DIST\_HUBER
    
    
    .. math::
    
        \rho (r) =  \fork{r^2/2}{if $r < C$}{C \cdot (r-C/2)}{otherwise} \quad \text{where} \quad C=1.345 
    
    
    
    

.. index:: GetCentralMoment

.. _GetCentralMoment:

GetCentralMoment
----------------




.. function:: GetCentralMoment(moments, x_order, y_order) -> double

    Retrieves the central moment from the moment state structure.





    
    :param moments: Pointer to the moment state structure 
    
    :type moments: :class:`CvMoments`
    
    
    :param x_order: x order of the retrieved moment,  :math:`\texttt{x\_order} >= 0` 
    
    :type x_order: int
    
    
    :param y_order: y order of the retrieved moment,  :math:`\texttt{y\_order} >= 0`  and  :math:`\texttt{x\_order} + \texttt{y\_order} <= 3` 
    
    :type y_order: int
    
    
    
The function retrieves the central moment, which in the case of image moments is defined as:



.. math::

    \mu _{x \_ order,  \, y \_ order} =  \sum _{x,y} (I(x,y)  \cdot (x-x_c)^{x \_ order}  \cdot (y-y_c)^{y \_ order}) 


where 
:math:`x_c,y_c`
are the coordinates of the gravity center:



.. math::

    x_c= \frac{M_{10}}{M_{00}} , y_c= \frac{M_{01}}{M_{00}} 



.. index:: GetHuMoments

.. _GetHuMoments:

GetHuMoments
------------




.. function:: GetHuMoments(moments) -> hu

    Calculates the seven Hu invariants.





    
    :param moments: The input moments, computed with  :ref:`Moments` 
    
    :type moments: :class:`CvMoments`
    
    
    :param hu: The output Hu invariants 
    
    :type hu: object
    
    
    
The function calculates the seven Hu invariants, see 
http://en.wikipedia.org/wiki/Image_moment
, that are defined as:



.. math::

    \begin{array}{l} hu_1= \eta _{20}+ \eta _{02} \\ hu_2=( \eta _{20}- \eta _{02})^{2}+4 \eta _{11}^{2} \\ hu_3=( \eta _{30}-3 \eta _{12})^{2}+ (3 \eta _{21}- \eta _{03})^{2} \\ hu_4=( \eta _{30}+ \eta _{12})^{2}+ ( \eta _{21}+ \eta _{03})^{2} \\ hu_5=( \eta _{30}-3 \eta _{12})( \eta _{30}+ \eta _{12})[( \eta _{30}+ \eta _{12})^{2}-3( \eta _{21}+ \eta _{03})^{2}]+(3 \eta _{21}- \eta _{03})( \eta _{21}+ \eta _{03})[3( \eta _{30}+ \eta _{12})^{2}-( \eta _{21}+ \eta _{03})^{2}] \\ hu_6=( \eta _{20}- \eta _{02})[( \eta _{30}+ \eta _{12})^{2}- ( \eta _{21}+ \eta _{03})^{2}]+4 \eta _{11}( \eta _{30}+ \eta _{12})( \eta _{21}+ \eta _{03}) \\ hu_7=(3 \eta _{21}- \eta _{03})( \eta _{21}+ \eta _{03})[3( \eta _{30}+ \eta _{12})^{2}-( \eta _{21}+ \eta _{03})^{2}]-( \eta _{30}-3 \eta _{12})( \eta _{21}+ \eta _{03})[3( \eta _{30}+ \eta _{12})^{2}-( \eta _{21}+ \eta _{03})^{2}] \\ \end{array} 


where 
:math:`\eta_{ji}`
denote the normalized central moments.

These values are proved to be invariant to the image scale, rotation, and reflection except the seventh one, whose sign is changed by reflection. Of course, this invariance was proved with the assumption of infinite image resolution. In case of a raster images the computed Hu invariants for the original and transformed images will be a bit different.




.. doctest::


    
    >>> import cv
    >>> original = cv.LoadImageM("building.jpg", cv.CV_LOAD_IMAGE_GRAYSCALE)
    >>> print cv.GetHuMoments(cv.Moments(original))
    (0.0010620951868446141, 1.7962726159653835e-07, 1.4932744974469421e-11, 4.4832441315737963e-12, -1.0819359198251739e-23, -9.5726503811945833e-16, -3.5050592804744648e-23)
    >>> flipped = cv.CloneMat(original)
    >>> cv.Flip(original, flipped)
    >>> print cv.GetHuMoments(cv.Moments(flipped))
    (0.0010620951868446141, 1.796272615965384e-07, 1.4932744974469935e-11, 4.4832441315740249e-12, -1.0819359198259393e-23, -9.572650381193327e-16, 3.5050592804745877e-23)
    

..


.. index:: GetNormalizedCentralMoment

.. _GetNormalizedCentralMoment:

GetNormalizedCentralMoment
--------------------------




.. function:: GetNormalizedCentralMoment(moments, x_order, y_order) -> double

    Retrieves the normalized central moment from the moment state structure.





    
    :param moments: Pointer to the moment state structure 
    
    :type moments: :class:`CvMoments`
    
    
    :param x_order: x order of the retrieved moment,  :math:`\texttt{x\_order} >= 0` 
    
    :type x_order: int
    
    
    :param y_order: y order of the retrieved moment,  :math:`\texttt{y\_order} >= 0`  and  :math:`\texttt{x\_order} + \texttt{y\_order} <= 3` 
    
    :type y_order: int
    
    
    
The function retrieves the normalized central moment:



.. math::

    \eta _{x \_ order,  \, y \_ order} =  \frac{\mu_{x\_order, \, y\_order}}{M_{00}^{(y\_order+x\_order)/2+1}} 



.. index:: GetSpatialMoment

.. _GetSpatialMoment:

GetSpatialMoment
----------------




.. function:: GetSpatialMoment(moments, x_order, y_order) -> double

    Retrieves the spatial moment from the moment state structure.





    
    :param moments: The moment state, calculated by  :ref:`Moments` 
    
    :type moments: :class:`CvMoments`
    
    
    :param x_order: x order of the retrieved moment,  :math:`\texttt{x\_order} >= 0` 
    
    :type x_order: int
    
    
    :param y_order: y order of the retrieved moment,  :math:`\texttt{y\_order} >= 0`  and  :math:`\texttt{x\_order} + \texttt{y\_order} <= 3` 
    
    :type y_order: int
    
    
    
The function retrieves the spatial moment, which in the case of image moments is defined as:



.. math::

    M_{x \_ order,  \, y \_ order} =  \sum _{x,y} (I(x,y)  \cdot x^{x \_ order}  \cdot y^{y \_ order}) 


where 
:math:`I(x,y)`
is the intensity of the pixel 
:math:`(x, y)`
.


.. index:: MatchContourTrees

.. _MatchContourTrees:

MatchContourTrees
-----------------




.. function:: MatchContourTrees(tree1,tree2,method,threshold)-> double

    Compares two contours using their tree representations.





    
    :param tree1: First contour tree 
    
    
    :param tree2: Second contour tree 
    
    
    :param method: Similarity measure, only  ``CV_CONTOUR_TREES_MATCH_I1``  is supported 
    
    
    :param threshold: Similarity threshold 
    
    
    
The function calculates the value of the matching measure for two contour trees. The similarity measure is calculated level by level from the binary tree roots. If at a certain level the difference between contours becomes less than 
``threshold``
, the reconstruction process is interrupted and the current difference is returned.


.. index:: MatchShapes

.. _MatchShapes:

MatchShapes
-----------




.. function:: MatchShapes(object1,object2,method,parameter=0)-> None

    Compares two shapes.





    
    :param object1: First contour or grayscale image 
    
    :type object1: :class:`CvSeq`
    
    
    :param object2: Second contour or grayscale image 
    
    :type object2: :class:`CvSeq`
    
    
    :param method: Comparison method;
          ``CV_CONTOUR_MATCH_I1`` , 
          ``CV_CONTOURS_MATCH_I2``  
        or 
          ``CV_CONTOURS_MATCH_I3`` 
    
    :type method: int
    
    
    :param parameter: Method-specific parameter (is not used now) 
    
    :type parameter: float
    
    
    
The function compares two shapes. The 3 implemented methods all use Hu moments (see 
:ref:`GetHuMoments`
) (
:math:`A`
is 
``object1``
, 
:math:`B`
is 
``object2``
):



    

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

    \begin{array}{l} m^A_i = sign(h^A_i)  \cdot \log{h^A_i} m^B_i = sign(h^B_i)  \cdot \log{h^B_i} \end{array} 


and 
:math:`h^A_i, h^B_i`
are the Hu moments of 
:math:`A`
and 
:math:`B`
respectively.



.. index:: MinAreaRect2

.. _MinAreaRect2:

MinAreaRect2
------------




.. function:: MinAreaRect2(points,storage=NULL)-> CvBox2D

    Finds the circumscribed rectangle of minimal area for a given 2D point set.





    
    :param points: Sequence or array of points 
    
    :type points: :class:`CvArr` or :class:`CvSeq`
    
    
    :param storage: Optional temporary memory storage 
    
    :type storage: :class:`CvMemStorage`
    
    
    
The function finds a circumscribed rectangle of the minimal area for a 2D point set by building a convex hull for the set and applying the rotating calipers technique to the hull.

Picture. Minimal-area bounding rectangle for contour



.. image:: ../pics/minareabox.png




.. index:: MinEnclosingCircle

.. _MinEnclosingCircle:

MinEnclosingCircle
------------------




.. function:: MinEnclosingCircle(points)-> (int,center,radius)

    Finds the circumscribed circle of minimal area for a given 2D point set.





    
    :param points: Sequence or array of 2D points 
    
    :type points: :class:`CvArr` or :class:`CvSeq`
    
    
    :param center: Output parameter; the center of the enclosing circle 
    
    :type center: :class:`CvPoint2D32f`
    
    
    :param radius: Output parameter; the radius of the enclosing circle 
    
    :type radius: float
    
    
    
The function finds the minimal circumscribed
circle for a 2D point set using an iterative algorithm. It returns nonzero
if the resultant circle contains all the input points and zero otherwise
(i.e. the algorithm failed).


.. index:: Moments

.. _Moments:

Moments
-------




.. function:: Moments(arr, binary = 0) -> moments

    Calculates all of the moments up to the third order of a polygon or rasterized shape.





    
    :param arr: Image (1-channel or 3-channel with COI set) or polygon (CvSeq of points or a vector of points) 
    
    :type arr: :class:`CvArr` or :class:`CvSeq`
    
    
    :param moments: Pointer to returned moment's state structure 
    
    :type moments: :class:`CvMoments`
    
    
    :param binary: (For images only) If the flag is non-zero, all of the zero pixel values are treated as zeroes, and all of the others are treated as 1's 
    
    :type binary: int
    
    
    
The function calculates spatial and central moments up to the third order and writes them to 
``moments``
. The moments may then be used then to calculate the gravity center of the shape, its area, main axises and various shape characeteristics including 7 Hu invariants.


.. index:: PointPolygonTest

.. _PointPolygonTest:

PointPolygonTest
----------------




.. function:: PointPolygonTest(contour,pt,measure_dist)-> double

    Point in contour test.





    
    :param contour: Input contour 
    
    :type contour: :class:`CvArr` or :class:`CvSeq`
    
    
    :param pt: The point tested against the contour 
    
    :type pt: :class:`CvPoint2D32f`
    
    
    :param measure_dist: If it is non-zero, the function estimates the distance from the point to the nearest contour edge 
    
    :type measure_dist: int
    
    
    
The function determines whether the
point is inside a contour, outside, or lies on an edge (or coinsides
with a vertex). It returns positive, negative or zero value,
correspondingly. When 
:math:`\texttt{measure\_dist} =0`
, the return value
is +1, -1 and 0, respectively. When 
:math:`\texttt{measure\_dist} \ne 0`
,
it is a signed distance between the point and the nearest contour
edge.

Here is the sample output of the function, where each image pixel is tested against the contour.



.. image:: ../pics/pointpolygon.png



