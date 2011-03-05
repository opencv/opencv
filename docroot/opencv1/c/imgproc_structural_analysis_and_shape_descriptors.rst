Structural Analysis and Shape Descriptors
=========================================

.. highlight:: c



.. index:: ApproxChains

.. _ApproxChains:

ApproxChains
------------

`id=0.432936866636 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/ApproxChains>`__




.. cfunction:: CvSeq* cvApproxChains(  CvSeq* src_seq, CvMemStorage* storage, int method=CV_CHAIN_APPROX_SIMPLE, double parameter=0, int minimal_perimeter=0, int recursive=0 )

    Approximates Freeman chain(s) with a polygonal curve.





    
    :param src_seq: Pointer to the chain that can refer to other chains 
    
    
    :param storage: Storage location for the resulting polylines 
    
    
    :param method: Approximation method (see the description of the function  :ref:`FindContours` ) 
    
    
    :param parameter: Method parameter (not used now) 
    
    
    :param minimal_perimeter: Approximates only those contours whose perimeters are not less than  ``minimal_perimeter`` . Other chains are removed from the resulting structure 
    
    
    :param recursive: If not 0, the function approximates all chains that access can be obtained to from  ``src_seq``  by using the  ``h_next``  or  ``v_next links`` . If 0, the single chain is approximated 
    
    
    
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

`id=0.861831385172 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/ApproxPoly>`__




.. cfunction:: CvSeq* cvApproxPoly(  const void* src_seq, int header_size, CvMemStorage* storage, int method, double parameter, int parameter2=0 )

    Approximates polygonal curve(s) with the specified precision.





    
    :param src_seq: Sequence of an array of points 
    
    
    :param header_size: Header size of the approximated curve[s] 
    
    
    :param storage: Container for the approximated contours. If it is NULL, the input sequences' storage is used 
    
    
    :param method: Approximation method; only  ``CV_POLY_APPROX_DP``  is supported, that corresponds to the Douglas-Peucker algorithm 
    
    
    :param parameter: Method-specific parameter; in the case of  ``CV_POLY_APPROX_DP``  it is a desired approximation accuracy 
    
    
    :param parameter2: If case if  ``src_seq``  is a sequence, the parameter determines whether the single sequence should be approximated or all sequences on the same level or below  ``src_seq``  (see  :ref:`FindContours`  for description of hierarchical contour structures). If  ``src_seq``  is an array CvMat* of points, the parameter specifies whether the curve is closed ( ``parameter2`` !=0) or not ( ``parameter2``  =0) 
    
    
    
The function approximates one or more curves and
returns the approximation result[s]. In the case of multiple curves,
the resultant tree will have the same structure as the input one (1:1
correspondence).


.. index:: ArcLength

.. _ArcLength:

ArcLength
---------

`id=0.382186875357 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/ArcLength>`__




.. cfunction:: double cvArcLength(  const void* curve, CvSlice slice=CV_WHOLE_SEQ, int isClosed=-1 )

    Calculates the contour perimeter or the curve length.





    
    :param curve: Sequence or array of the curve points 
    
    
    :param slice: Starting and ending points of the curve, by default, the whole curve length is calculated 
    
    
    :param isClosed: Indicates whether the curve is closed or not. There are 3 cases: 
        
               
        
        *   :math:`\texttt{isClosed}=0`  the curve is assumed to be unclosed.
               
        
        *   :math:`\texttt{isClosed}>0`  the curve is assumed to be closed.
               
        
        *   :math:`\texttt{isClosed}<0`  if curve is sequence, the flag  ``CV_SEQ_FLAG_CLOSED``  of  ``((CvSeq*)curve)->flags``  is checked to determine if the curve is closed or not, otherwise (curve is represented by array (CvMat*) of points) it is assumed to be unclosed. 
            
    
    
    
The function calculates the length or curve as the sum of lengths of segments between subsequent points


.. index:: BoundingRect

.. _BoundingRect:

BoundingRect
------------

`id=0.99193394782 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/BoundingRect>`__




.. cfunction:: CvRect cvBoundingRect( CvArr* points, int update=0 )

    Calculates the up-right bounding rectangle of a point set.





    
    :param points: 2D point set, either a sequence or vector ( ``CvMat`` ) of points 
    
    
    :param update: The update flag. See below. 
    
    
    
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

`id=0.15348377114 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/BoxPoints>`__




.. cfunction:: void cvBoxPoints(  CvBox2D box, CvPoint2D32f pt[4] )

    Finds the box vertices.





    
    :param box: Box 
    
    
    :param points: Array of vertices 
    
    
    
The function calculates the vertices of the input 2d box.

Here is the function code:




::


    
    void cvBoxPoints( CvBox2D box, CvPoint2D32f pt[4] )
    {
        float a = (float)cos(box.angle)*0.5f;
        float b = (float)sin(box.angle)*0.5f;
    
        pt[0].x = box.center.x - a*box.size.height - b*box.size.width;
        pt[0].y = box.center.y + b*box.size.height - a*box.size.width;
        pt[1].x = box.center.x + a*box.size.height - b*box.size.width;
        pt[1].y = box.center.y - b*box.size.height - a*box.size.width;
        pt[2].x = 2*box.center.x - pt[0].x;
        pt[2].y = 2*box.center.y - pt[0].y;
        pt[3].x = 2*box.center.x - pt[1].x;
        pt[3].y = 2*box.center.y - pt[1].y;
    }
    

..


.. index:: CalcPGH

.. _CalcPGH:

CalcPGH
-------

`id=0.713512953819 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/CalcPGH>`__




.. cfunction:: void cvCalcPGH( const CvSeq* contour, CvHistogram* hist )

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

`id=0.642501185958 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/CalcEMD2>`__




.. cfunction:: float cvCalcEMD2(  const CvArr* signature1, const CvArr* signature2, int distance_type, CvDistanceFunction distance_func=NULL, const CvArr* cost_matrix=NULL, CvArr* flow=NULL, float* lower_bound=NULL, void* userdata=NULL )

    Computes the "minimal work" distance between two weighted point configurations.





    
    :param signature1: First signature, a  :math:`\texttt{size1}\times \texttt{dims}+1`  floating-point matrix. Each row stores the point weight followed by the point coordinates. The matrix is allowed to have a single column (weights only) if the user-defined cost matrix is used 
    
    
    :param signature2: Second signature of the same format as  ``signature1`` , though the number of rows may be different. The total weights may be different, in this case an extra "dummy" point is added to either  ``signature1``  or  ``signature2`` 
    
    
    :param distance_type: Metrics used;  ``CV_DIST_L1, CV_DIST_L2`` , and  ``CV_DIST_C``  stand for one of the standard metrics;  ``CV_DIST_USER``  means that a user-defined function  ``distance_func``  or pre-calculated  ``cost_matrix``  is used 
    
    
    :param distance_func: The user-supplied distance function. It takes coordinates of two points and returns the distance between the points ``
                typedef float (*CvDistanceFunction)(const float* f1, const float* f2, void* userdata);`` 
    
    
    :param cost_matrix: The user-defined  :math:`\texttt{size1}\times \texttt{size2}`  cost matrix. At least one of  ``cost_matrix``  and  ``distance_func``  must be NULL. Also, if a cost matrix is used, lower boundary (see below) can not be calculated, because it needs a metric function 
    
    
    :param flow: The resultant  :math:`\texttt{size1} \times \texttt{size2}`  flow matrix:  :math:`\texttt{flow}_{i,j}`  is a flow from  :math:`i`  th point of  ``signature1``  to  :math:`j`  th point of  ``signature2`` 
    
    
    :param lower_bound: Optional input/output parameter: lower boundary of distance between the two signatures that is a distance between mass centers. The lower boundary may not be calculated if the user-defined cost matrix is used, the total weights of point configurations are not equal, or if the signatures consist of weights only (i.e. the signature matrices have a single column). The user  **must**  initialize  ``*lower_bound`` . If the calculated distance between mass centers is greater or equal to  ``*lower_bound``  (it means that the signatures are far enough) the function does not calculate EMD. In any case  ``*lower_bound``  is set to the calculated distance between mass centers on return. Thus, if user wants to calculate both distance between mass centers and EMD,  ``*lower_bound``  should be set to 0 
    
    
    :param userdata: Pointer to optional data that is passed into the user-defined distance function 
    
    
    
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

`id=0.596409711678 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/CheckContourConvexity>`__




.. cfunction:: int cvCheckContourConvexity( const CvArr* contour )

    Tests contour convexity.





    
    :param contour: Tested contour (sequence or array of points) 
    
    
    
The function tests whether the input contour is convex or not. The contour must be simple, without self-intersections.


.. index:: CvConvexityDefect

.. _CvConvexityDefect:

CvConvexityDefect
-----------------

`id=0.0456666449216 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/CvConvexityDefect>`__

.. ctype:: CvConvexityDefect



Structure describing a single contour convexity defect.




::


    
    typedef struct CvConvexityDefect
    {
        CvPoint* start; /* point of the contour where the defect begins */
        CvPoint* end; /* point of the contour where the defect ends */
        CvPoint* depth_point; /* the farthest from the convex hull point within the defect */
        float depth; /* distance between the farthest point and the convex hull */
    } CvConvexityDefect;
    

..



.. image:: ../pics/defects.png




.. index:: ContourArea

.. _ContourArea:

ContourArea
-----------

`id=0.579530349862 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/ContourArea>`__




.. cfunction:: double cvContourArea(  const CvArr* contour,  CvSlice slice=CV_WHOLE_SEQ )

    Calculates the area of a whole contour or a contour section.





    
    :param contour: Contour (sequence or array of vertices) 
    
    
    :param slice: Starting and ending points of the contour section of interest, by default, the area of the whole contour is calculated 
    
    
    
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

`id=0.283577660364 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/ContourFromContourTree>`__




.. cfunction:: CvSeq* cvContourFromContourTree(  const CvContourTree* tree, CvMemStorage* storage, CvTermCriteria criteria )

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

`id=0.07365440701 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/ConvexHull2>`__




.. cfunction:: CvSeq* cvConvexHull2(  const CvArr* input, void* storage=NULL, int orientation=CV_CLOCKWISE, int return_points=0 )

    Finds the convex hull of a point set.





    
    :param points: Sequence or array of 2D points with 32-bit integer or floating-point coordinates 
    
    
    :param storage: The destination array (CvMat*) or memory storage (CvMemStorage*) that will store the convex hull. If it is an array, it should be 1d and have the same number of elements as the input array/sequence. On output the header is modified as to truncate the array down to the hull size.  If  ``storage``  is NULL then the convex hull will be stored in the same storage as the input sequence 
    
    
    :param orientation: Desired orientation of convex hull:  ``CV_CLOCKWISE``  or  ``CV_COUNTER_CLOCKWISE`` 
    
    
    :param return_points: If non-zero, the points themselves will be stored in the hull instead of indices if  ``storage``  is an array, or pointers if  ``storage``  is memory storage 
    
    
    
The function finds the convex hull of a 2D point set using Sklansky's algorithm. If 
``storage``
is memory storage, the function creates a sequence containing the hull points or pointers to them, depending on 
``return_points``
value and returns the sequence on output.  If 
``storage``
is a CvMat, the function returns NULL.

Example. Building convex hull for a sequence or array of points




::


    
    #include "cv.h"
    #include "highgui.h"
    #include <stdlib.h>
    
    #define ARRAY  0 /* switch between array/sequence method by replacing 0<=>1 */
    
    void main( int argc, char** argv )
    {
        IplImage* img = cvCreateImage( cvSize( 500, 500 ), 8, 3 );
        cvNamedWindow( "hull", 1 );
    
    #if !ARRAY
            CvMemStorage* storage = cvCreateMemStorage();
    #endif
    
        for(;;)
        {
            int i, count = rand()
            CvPoint pt0;
    #if !ARRAY
            CvSeq* ptseq = cvCreateSeq( CV_SEQ_KIND_GENERIC|CV_32SC2,
                                        sizeof(CvContour),
                                        sizeof(CvPoint),
                                        storage );
            CvSeq* hull;
    
            for( i = 0; i < count; i++ )
            {
                pt0.x = rand() 
                pt0.y = rand() 
                cvSeqPush( ptseq, &pt0 );
            }
            hull = cvConvexHull2( ptseq, 0, CV_CLOCKWISE, 0 );
            hullcount = hull->total;
    #else
            CvPoint* points = (CvPoint*)malloc( count * sizeof(points[0]));
            int* hull = (int*)malloc( count * sizeof(hull[0]));
            CvMat point_mat = cvMat( 1, count, CV_32SC2, points );
            CvMat hull_mat = cvMat( 1, count, CV_32SC1, hull );
    
            for( i = 0; i < count; i++ )
            {
                pt0.x = rand() 
                pt0.y = rand() 
                points[i] = pt0;
            }
            cvConvexHull2( &point_mat, &hull_mat, CV_CLOCKWISE, 0 );
            hullcount = hull_mat.cols;
    #endif
            cvZero( img );
            for( i = 0; i < count; i++ )
            {
    #if !ARRAY
                pt0 = *CV_GET_SEQ_ELEM( CvPoint, ptseq, i );
    #else
                pt0 = points[i];
    #endif
                cvCircle( img, pt0, 2, CV_RGB( 255, 0, 0 ), CV_FILLED );
            }
    
    #if !ARRAY
            pt0 = **CV_GET_SEQ_ELEM( CvPoint*, hull, hullcount - 1 );
    #else
            pt0 = points[hull[hullcount-1]];
    #endif
    
            for( i = 0; i < hullcount; i++ )
            {
    #if !ARRAY
                CvPoint pt = **CV_GET_SEQ_ELEM( CvPoint*, hull, i );
    #else
                CvPoint pt = points[hull[i]];
    #endif
                cvLine( img, pt0, pt, CV_RGB( 0, 255, 0 ));
                pt0 = pt;
            }
    
            cvShowImage( "hull", img );
    
            int key = cvWaitKey(0);
            if( key == 27 ) // 'ESC'
                break;
    
    #if !ARRAY
            cvClearMemStorage( storage );
    #else
            free( points );
            free( hull );
    #endif
        }
    }
    

..


.. index:: ConvexityDefects

.. _ConvexityDefects:

ConvexityDefects
----------------

`id=0.246826049247 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/ConvexityDefects>`__




.. cfunction:: CvSeq* cvConvexityDefects(  const CvArr* contour, const CvArr* convexhull, CvMemStorage* storage=NULL )

    Finds the convexity defects of a contour.





    
    :param contour: Input contour 
    
    
    :param convexhull: Convex hull obtained using  :ref:`ConvexHull2`  that should contain pointers or indices to the contour points, not the hull points themselves (the  ``return_points``  parameter in  :ref:`ConvexHull2`  should be 0) 
    
    
    :param storage: Container for the output sequence of convexity defects. If it is NULL, the contour or hull (in that order) storage is used 
    
    
    
The function finds all convexity defects of the input contour and returns a sequence of the CvConvexityDefect structures.


.. index:: CreateContourTree

.. _CreateContourTree:

CreateContourTree
-----------------

`id=0.116090901246 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/CreateContourTree>`__




.. cfunction:: CvContourTree* cvCreateContourTree(  const CvSeq* contour, CvMemStorage* storage, double threshold )

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


.. index:: EndFindContours

.. _EndFindContours:

EndFindContours
---------------

`id=0.772927708524 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/EndFindContours>`__




.. cfunction:: CvSeq* cvEndFindContours(  CvContourScanner* scanner )

    Finishes the scanning process.





    
    :param scanner: Pointer to the contour scanner 
    
    
    
The function finishes the scanning process and returns a pointer to the first contour on the highest level.


.. index:: FindContours

.. _FindContours:

FindContours
------------

`id=0.804514745402 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/FindContours>`__




.. cfunction:: int cvFindContours( CvArr* image, CvMemStorage* storage, CvSeq** first_contour,                    int header_size=sizeof(CvContour), int mode=CV_RETR_LIST,                    int method=CV_CHAIN_APPROX_SIMPLE, CvPoint offset=cvPoint(0,0) )

    Finds the contours in a binary image.





    
    :param image: The source, an 8-bit single channel image. Non-zero pixels are treated as 1's, zero pixels remain 0's - the image is treated as  ``binary`` . To get such a binary image from grayscale, one may use  :ref:`Threshold` ,  :ref:`AdaptiveThreshold`  or  :ref:`Canny` . The function modifies the source image's content 
    
    
    :param storage: Container of the retrieved contours 
    
    
    :param first_contour: Output parameter, will contain the pointer to the first outer contour 
    
    
    :param header_size: Size of the sequence header,  :math:`\ge \texttt{sizeof(CvChain)}`  if  :math:`\texttt{method} =\texttt{CV\_CHAIN\_CODE}` ,
        and  :math:`\ge \texttt{sizeof(CvContour)}`  otherwise 
    
    
    :param mode: Retrieval mode 
        
                
            * **CV_RETR_EXTERNAL** retrives only the extreme outer contours 
            
               
            * **CV_RETR_LIST** retrieves all of the contours and puts them in the list 
            
               
            * **CV_RETR_CCOMP** retrieves all of the contours and organizes them into a two-level hierarchy: on the top level are the external boundaries of the components, on the second level are the boundaries of the holes 
            
               
            * **CV_RETR_TREE** retrieves all of the contours and reconstructs the full hierarchy of nested contours 
            
            
    
    
    :param method: Approximation method (for all the modes, except  ``CV_LINK_RUNS`` , which uses built-in approximation) 
        
                
            * **CV_CHAIN_CODE** outputs contours in the Freeman chain code. All other methods output polygons (sequences of vertices) 
            
               
            * **CV_CHAIN_APPROX_NONE** translates all of the points from the chain code into points 
            
               
            * **CV_CHAIN_APPROX_SIMPLE** compresses horizontal, vertical, and diagonal segments and leaves only their end points 
            
               
            * **CV_CHAIN_APPROX_TC89_L1,CV_CHAIN_APPROX_TC89_KCOS** applies one of the flavors of the Teh-Chin chain approximation algorithm. 
            
               
            * **CV_LINK_RUNS** uses a completely different contour retrieval algorithm by linking horizontal segments of 1's. Only the  ``CV_RETR_LIST``  retrieval mode can be used with this method. 
            
            
    
    
    :param offset: Offset, by which every contour point is shifted. This is useful if the contours are extracted from the image ROI and then they should be analyzed in the whole image context 
    
    
    
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
``squares.c``
in the OpenCV sample directory.

**Note:**
the source 
``image``
is modified by this function.


.. index:: FindNextContour

.. _FindNextContour:

FindNextContour
---------------

`id=0.251954589601 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/FindNextContour>`__




.. cfunction:: CvSeq* cvFindNextContour(  CvContourScanner scanner )

    Finds the next contour in the image.





    
    :param scanner: Contour scanner initialized by  :ref:`StartFindContours`   
    
    
    
The function locates and retrieves the next contour in the image and returns a pointer to it. The function returns NULL if there are no more contours.


.. index:: FitEllipse2

.. _FitEllipse2:

FitEllipse2
-----------

`id=0.639828157054 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/FitEllipse2>`__




.. cfunction:: CvBox2D cvFitEllipse2(  const CvArr* points )

    Fits an ellipse around a set of 2D points.





    
    :param points: Sequence or array of points 
    
    
    
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

`id=0.0204712084438 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/FitLine>`__




.. cfunction:: void  cvFitLine(  const CvArr* points, int dist_type, double param, double reps, double aeps, float* line )

    Fits a line to a 2D or 3D point set.





    
    :param points: Sequence or array of 2D or 3D points with 32-bit integer or floating-point coordinates 
    
    
    :param dist_type: The distance used for fitting (see the discussion) 
    
    
    :param param: Numerical parameter ( ``C`` ) for some types of distances, if 0 then some optimal value is chosen 
    
    
    :param reps: Sufficient accuracy for the radius (distance between the coordinate origin and the line).  0.01 is a good default value. 
    
    
    :param aeps: Sufficient accuracy for the angle.  0.01 is a good default value. 
    
    
    :param line: The output line parameters. In the case of a 2d fitting,
        it is  an array     of 4 floats  ``(vx, vy, x0, y0)``  where  ``(vx, vy)``  is a normalized vector collinear to the
        line and  ``(x0, y0)``  is some point on the line. in the case of a
        3D fitting it is  an array     of 6 floats  ``(vx, vy, vz, x0, y0, z0)`` 
        where  ``(vx, vy, vz)``  is a normalized vector collinear to the line
        and  ``(x0, y0, z0)``  is some point on the line 
    
    
    
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

`id=0.574094648001 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/GetCentralMoment>`__




.. cfunction:: double cvGetCentralMoment(  CvMoments* moments, int x_order, int y_order )

    Retrieves the central moment from the moment state structure.





    
    :param moments: Pointer to the moment state structure 
    
    
    :param x_order: x order of the retrieved moment,  :math:`\texttt{x\_order} >= 0` 
    
    
    :param y_order: y order of the retrieved moment,  :math:`\texttt{y\_order} >= 0`  and  :math:`\texttt{x\_order} + \texttt{y\_order} <= 3` 
    
    
    
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

`id=0.56722466619 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/GetHuMoments>`__




.. cfunction:: void cvGetHuMoments( const CvMoments* moments,CvHuMoments* hu )

    Calculates the seven Hu invariants.





    
    :param moments: The input moments, computed with  :ref:`Moments` 
    
    
    :param hu: The output Hu invariants 
    
    
    
The function calculates the seven Hu invariants, see 
http://en.wikipedia.org/wiki/Image_moment
, that are defined as:



.. math::

    \begin{array}{l} hu_1= \eta _{20}+ \eta _{02} \\ hu_2=( \eta _{20}- \eta _{02})^{2}+4 \eta _{11}^{2} \\ hu_3=( \eta _{30}-3 \eta _{12})^{2}+ (3 \eta _{21}- \eta _{03})^{2} \\ hu_4=( \eta _{30}+ \eta _{12})^{2}+ ( \eta _{21}+ \eta _{03})^{2} \\ hu_5=( \eta _{30}-3 \eta _{12})( \eta _{30}+ \eta _{12})[( \eta _{30}+ \eta _{12})^{2}-3( \eta _{21}+ \eta _{03})^{2}]+(3 \eta _{21}- \eta _{03})( \eta _{21}+ \eta _{03})[3( \eta _{30}+ \eta _{12})^{2}-( \eta _{21}+ \eta _{03})^{2}] \\ hu_6=( \eta _{20}- \eta _{02})[( \eta _{30}+ \eta _{12})^{2}- ( \eta _{21}+ \eta _{03})^{2}]+4 \eta _{11}( \eta _{30}+ \eta _{12})( \eta _{21}+ \eta _{03}) \\ hu_7=(3 \eta _{21}- \eta _{03})( \eta _{21}+ \eta _{03})[3( \eta _{30}+ \eta _{12})^{2}-( \eta _{21}+ \eta _{03})^{2}]-( \eta _{30}-3 \eta _{12})( \eta _{21}+ \eta _{03})[3( \eta _{30}+ \eta _{12})^{2}-( \eta _{21}+ \eta _{03})^{2}] \\ \end{array} 


where 
:math:`\eta_{ji}`
denote the normalized central moments.

These values are proved to be invariant to the image scale, rotation, and reflection except the seventh one, whose sign is changed by reflection. Of course, this invariance was proved with the assumption of infinite image resolution. In case of a raster images the computed Hu invariants for the original and transformed images will be a bit different.


.. index:: GetNormalizedCentralMoment

.. _GetNormalizedCentralMoment:

GetNormalizedCentralMoment
--------------------------

`id=0.460978782732 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/GetNormalizedCentralMoment>`__




.. cfunction:: double cvGetNormalizedCentralMoment(  CvMoments* moments, int x_order, int y_order )

    Retrieves the normalized central moment from the moment state structure.





    
    :param moments: Pointer to the moment state structure 
    
    
    :param x_order: x order of the retrieved moment,  :math:`\texttt{x\_order} >= 0` 
    
    
    :param y_order: y order of the retrieved moment,  :math:`\texttt{y\_order} >= 0`  and  :math:`\texttt{x\_order} + \texttt{y\_order} <= 3` 
    
    
    
The function retrieves the normalized central moment:



.. math::

    \eta _{x \_ order,  \, y \_ order} =  \frac{\mu_{x\_order, \, y\_order}}{M_{00}^{(y\_order+x\_order)/2+1}} 



.. index:: GetSpatialMoment

.. _GetSpatialMoment:

GetSpatialMoment
----------------

`id=0.768768789318 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/GetSpatialMoment>`__




.. cfunction:: double cvGetSpatialMoment(  CvMoments* moments,  int x_order,  int y_order )

    Retrieves the spatial moment from the moment state structure.





    
    :param moments: The moment state, calculated by  :ref:`Moments` 
    
    
    :param x_order: x order of the retrieved moment,  :math:`\texttt{x\_order} >= 0` 
    
    
    :param y_order: y order of the retrieved moment,  :math:`\texttt{y\_order} >= 0`  and  :math:`\texttt{x\_order} + \texttt{y\_order} <= 3` 
    
    
    
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

`id=0.555027093069 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/MatchContourTrees>`__




.. cfunction:: double cvMatchContourTrees(  const CvContourTree* tree1, const CvContourTree* tree2, int method, double threshold )

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

`id=0.492880753336 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/MatchShapes>`__




.. cfunction:: double cvMatchShapes(  const void* object1, const void* object2, int method, double parameter=0 )

    Compares two shapes.





    
    :param object1: First contour or grayscale image 
    
    
    :param object2: Second contour or grayscale image 
    
    
    :param method: Comparison method;
          ``CV_CONTOUR_MATCH_I1`` , 
          ``CV_CONTOURS_MATCH_I2``  
        or 
          ``CV_CONTOURS_MATCH_I3`` 
    
    
    :param parameter: Method-specific parameter (is not used now) 
    
    
    
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

`id=0.325416946848 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/MinAreaRect2>`__




.. cfunction:: CvBox2D  cvMinAreaRect2(  const CvArr* points, CvMemStorage* storage=NULL )

    Finds the circumscribed rectangle of minimal area for a given 2D point set.





    
    :param points: Sequence or array of points 
    
    
    :param storage: Optional temporary memory storage 
    
    
    
The function finds a circumscribed rectangle of the minimal area for a 2D point set by building a convex hull for the set and applying the rotating calipers technique to the hull.

Picture. Minimal-area bounding rectangle for contour



.. image:: ../pics/minareabox.png




.. index:: MinEnclosingCircle

.. _MinEnclosingCircle:

MinEnclosingCircle
------------------

`id=0.232805538989 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/MinEnclosingCircle>`__




.. cfunction:: int cvMinEnclosingCircle(  const CvArr* points, CvPoint2D32f* center, float* radius )

    Finds the circumscribed circle of minimal area for a given 2D point set.





    
    :param points: Sequence or array of 2D points 
    
    
    :param center: Output parameter; the center of the enclosing circle 
    
    
    :param radius: Output parameter; the radius of the enclosing circle 
    
    
    
The function finds the minimal circumscribed
circle for a 2D point set using an iterative algorithm. It returns nonzero
if the resultant circle contains all the input points and zero otherwise
(i.e. the algorithm failed).


.. index:: Moments

.. _Moments:

Moments
-------

`id=0.145895685877 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/Moments>`__




.. cfunction:: void cvMoments(  const CvArr* arr, CvMoments* moments, int binary=0 )

    Calculates all of the moments up to the third order of a polygon or rasterized shape.





    
    :param arr: Image (1-channel or 3-channel with COI set) or polygon (CvSeq of points or a vector of points) 
    
    
    :param moments: Pointer to returned moment's state structure 
    
    
    :param binary: (For images only) If the flag is non-zero, all of the zero pixel values are treated as zeroes, and all of the others are treated as 1's 
    
    
    
The function calculates spatial and central moments up to the third order and writes them to 
``moments``
. The moments may then be used then to calculate the gravity center of the shape, its area, main axises and various shape characeteristics including 7 Hu invariants.


.. index:: PointPolygonTest

.. _PointPolygonTest:

PointPolygonTest
----------------

`id=0.21757803031 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/PointPolygonTest>`__




.. cfunction:: double cvPointPolygonTest(  const CvArr* contour, CvPoint2D32f pt, int measure_dist )

    Point in contour test.





    
    :param contour: Input contour 
    
    
    :param pt: The point tested against the contour 
    
    
    :param measure_dist: If it is non-zero, the function estimates the distance from the point to the nearest contour edge 
    
    
    
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




.. index:: PointSeqFromMat

.. _PointSeqFromMat:

PointSeqFromMat
---------------

`id=0.728001629164 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/PointSeqFromMat>`__




.. cfunction:: CvSeq* cvPointSeqFromMat(  int seq_kind, const CvArr* mat, CvContour* contour_header, CvSeqBlock* block )

    Initializes a point sequence header from a point vector.





    
    :param seq_kind: Type of the point sequence: point set (0), a curve ( ``CV_SEQ_KIND_CURVE`` ), closed curve ( ``CV_SEQ_KIND_CURVE+CV_SEQ_FLAG_CLOSED`` ) etc. 
    
    
    :param mat: Input matrix. It should be a continuous, 1-dimensional vector of points, that is, it should have type  ``CV_32SC2``  or  ``CV_32FC2`` 
    
    
    :param contour_header: Contour header, initialized by the function 
    
    
    :param block: Sequence block header, initialized by the function 
    
    
    
The function initializes a sequence
header to create a "virtual" sequence in which elements reside in
the specified matrix. No data is copied. The initialized sequence
header may be passed to any function that takes a point sequence
on input. No extra elements can be added to the sequence,
but some may be removed. The function is a specialized variant of
:ref:`MakeSeqHeaderForArray`
and uses
the latter internally. It returns a pointer to the initialized contour
header. Note that the bounding rectangle (field 
``rect``
of
``CvContour``
strucuture) is not initialized by the function. If
you need one, use 
:ref:`BoundingRect`
.

Here is a simple usage example.




::


    
    CvContour header;
    CvSeqBlock block;
    CvMat* vector = cvCreateMat( 1, 3, CV_32SC2 );
    
    CV_MAT_ELEM( *vector, CvPoint, 0, 0 ) = cvPoint(100,100);
    CV_MAT_ELEM( *vector, CvPoint, 0, 1 ) = cvPoint(100,200);
    CV_MAT_ELEM( *vector, CvPoint, 0, 2 ) = cvPoint(200,100);
    
    IplImage* img = cvCreateImage( cvSize(300,300), 8, 3 );
    cvZero(img);
    
    cvDrawContours( img,
        cvPointSeqFromMat(CV_SEQ_KIND_CURVE+CV_SEQ_FLAG_CLOSED,
                          vector,
                          &header,
                          &block),
                    CV_RGB(255,0,0),
                    CV_RGB(255,0,0),
                    0, 3, 8, cvPoint(0,0));
    

..


.. index:: ReadChainPoint

.. _ReadChainPoint:

ReadChainPoint
--------------

`id=0.760176226481 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/ReadChainPoint>`__




.. cfunction:: CvPoint cvReadChainPoint( CvChainPtReader* reader )

    Gets the next chain point.





    
    :param reader: Chain reader state 
    
    
    
The function returns the current chain point and updates the reader position.


.. index:: StartFindContours

.. _StartFindContours:

StartFindContours
-----------------

`id=0.411171934048 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/StartFindContours>`__




.. cfunction:: CvContourScanner cvStartFindContours( CvArr* image, CvMemStorage* storage,                                      int header_size=sizeof(CvContour),                                      int mode=CV_RETR_LIST,                                      int method=CV_CHAIN_APPROX_SIMPLE,                                      CvPoint offset=cvPoint(0,0) )

    Initializes the contour scanning process.





    
    :param image: The 8-bit, single channel, binary source image 
    
    
    :param storage: Container of the retrieved contours 
    
    
    :param header_size: Size of the sequence header,  :math:`>=sizeof(CvChain)`  if  ``method``  =CV _ CHAIN _ CODE, and  :math:`>=sizeof(CvContour)`  otherwise 
    
    
    :param mode: Retrieval mode; see  :ref:`FindContours` 
    
    
    :param method: Approximation method. It has the same meaning in  :ref:`FindContours` , but  ``CV_LINK_RUNS``  can not be used here 
    
    
    :param offset: ROI offset; see  :ref:`FindContours` 
    
    
    
The function initializes and returns a pointer to the contour scanner. The scanner is used in 
:ref:`FindNextContour`
to retrieve the rest of the contours.


.. index:: StartReadChainPoints

.. _StartReadChainPoints:

StartReadChainPoints
--------------------

`id=0.532234897641 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/StartReadChainPoints>`__




.. cfunction:: void cvStartReadChainPoints( CvChain* chain, CvChainPtReader* reader )

    Initializes the chain reader.



The function initializes a special reader.


.. index:: SubstituteContour

.. _SubstituteContour:

SubstituteContour
-----------------

`id=0.692706172642 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/SubstituteContour>`__




.. cfunction:: void cvSubstituteContour(  CvContourScanner scanner,  CvSeq* new_contour )

    Replaces a retrieved contour.





    
    :param scanner: Contour scanner initialized by  :ref:`StartFindContours`   
    
    
    :param new_contour: Substituting contour 
    
    
    
The function replaces the retrieved
contour, that was returned from the preceding call of
:ref:`FindNextContour`
and stored inside the contour scanner
state, with the user-specified contour. The contour is inserted
into the resulting structure, list, two-level hierarchy, or tree,
depending on the retrieval mode. If the parameter 
``new_contour``
is 
``NULL``
, the retrieved contour is not included in the
resulting structure, nor are any of its children that might be added
to this structure later.

