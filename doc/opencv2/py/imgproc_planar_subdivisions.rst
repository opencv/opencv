Planar Subdivisions
===================

.. highlight:: python



.. index:: CvSubdiv2D

.. _CvSubdiv2D:

CvSubdiv2D
----------



.. class:: CvSubdiv2D



Planar subdivision.



    
    
    .. attribute:: edges
    
    
    
        A  :ref:`CvSet`  of  :ref:`CvSubdiv2DEdge` 
    
    
    
Planar subdivision is the subdivision of a plane into a set of
non-overlapped regions (facets) that cover the whole plane. The above
structure describes a subdivision built on a 2d point set, where the points
are linked together and form a planar graph, which, together with a few
edges connecting the exterior subdivision points (namely, convex hull points)
with infinity, subdivides a plane into facets by its edges.

For every subdivision there exists a dual subdivision in which facets and
points (subdivision vertices) swap their roles, that is, a facet is
treated as a vertex (called a virtual point below) of the dual subdivision and
the original subdivision vertices become facets. On the picture below
original subdivision is marked with solid lines and dual subdivision
with dotted lines.







OpenCV subdivides a plane into triangles using Delaunay's
algorithm. Subdivision is built iteratively starting from a dummy
triangle that includes all the subdivision points for sure. In this
case the dual subdivision is a Voronoi diagram of the input 2d point set. The
subdivisions can be used for the 3d piece-wise transformation of a plane,
morphing, fast location of points on the plane, building special graphs
(such as NNG,RNG) and so forth.


.. index:: CvSubdiv2DPoint

.. _CvSubdiv2DPoint:

CvSubdiv2DPoint
---------------



.. class:: CvSubdiv2DPoint



Point of original or dual subdivision.



    
    
    .. attribute:: first
    
    
    
        A connected  :ref:`CvSubdiv2DEdge` 
    
    
    
    .. attribute:: pt
    
    
    
        Position, as a  :ref:`CvPoint2D32f` 
    
    
    

.. index:: CalcSubdivVoronoi2D

.. _CalcSubdivVoronoi2D:

CalcSubdivVoronoi2D
-------------------




.. function:: CalcSubdivVoronoi2D(subdiv)-> None

    Calculates the coordinates of Voronoi diagram cells.





    
    :param subdiv: Delaunay subdivision, in which all the points are already added 
    
    :type subdiv: :class:`CvSubdiv2D`
    
    
    
The function calculates the coordinates
of virtual points. All virtual points corresponding to some vertex of the
original subdivision form (when connected together) a boundary of the Voronoi
cell at that point.


.. index:: ClearSubdivVoronoi2D

.. _ClearSubdivVoronoi2D:

ClearSubdivVoronoi2D
--------------------




.. function:: ClearSubdivVoronoi2D(subdiv)-> None

    Removes all virtual points.





    
    :param subdiv: Delaunay subdivision 
    
    :type subdiv: :class:`CvSubdiv2D`
    
    
    
The function removes all of the virtual points. It
is called internally in 
:ref:`CalcSubdivVoronoi2D`
if the subdivision
was modified after previous call to the function.



.. index:: CreateSubdivDelaunay2D

.. _CreateSubdivDelaunay2D:

CreateSubdivDelaunay2D
----------------------




.. function:: CreateSubdivDelaunay2D(rect,storage)-> delaunay_triangulation

    Creates an empty Delaunay triangulation.





    
    :param rect: Rectangle that includes all of the 2d points that are to be added to the subdivision 
    
    :type rect: :class:`CvRect`
    
    
    :param storage: Container for subdivision 
    
    :type storage: :class:`CvMemStorage`
    
    
    
The function creates an empty Delaunay
subdivision, where 2d points can be added using the function
:ref:`SubdivDelaunay2DInsert`
. All of the points to be added must be within
the specified rectangle, otherwise a runtime error will be raised.

Note that the triangulation is a single large triangle that covers the given rectangle.  Hence the three vertices of this triangle are outside the rectangle 
``rect``
.


.. index:: FindNearestPoint2D

.. _FindNearestPoint2D:

FindNearestPoint2D
------------------




.. function:: FindNearestPoint2D(subdiv,pt)-> point

    Finds the closest subdivision vertex to the given point.





    
    :param subdiv: Delaunay or another subdivision 
    
    :type subdiv: :class:`CvSubdiv2D`
    
    
    :param pt: Input point 
    
    :type pt: :class:`CvPoint2D32f`
    
    
    
The function is another function that
locates the input point within the subdivision. It finds the subdivision vertex that
is the closest to the input point. It is not necessarily one of vertices
of the facet containing the input point, though the facet (located using
:ref:`Subdiv2DLocate`
) is used as a starting
point. The function returns a pointer to the found subdivision vertex.


.. index:: Subdiv2DEdgeDst

.. _Subdiv2DEdgeDst:

Subdiv2DEdgeDst
---------------




.. function:: Subdiv2DEdgeDst(edge)-> point

    Returns the edge destination.





    
    :param edge: Subdivision edge (not a quad-edge) 
    
    :type edge: :class:`CvSubdiv2DEdge`
    
    
    
The function returns the edge destination. The
returned pointer may be NULL if the edge is from dual subdivision and
the virtual point coordinates are not calculated yet. The virtual points
can be calculated using the function 
:ref:`CalcSubdivVoronoi2D`
.


.. index:: Subdiv2DGetEdge

.. _Subdiv2DGetEdge:

Subdiv2DGetEdge
---------------




.. function:: Subdiv2DGetEdge(edge,type)-> CvSubdiv2DEdge

    Returns one of the edges related to the given edge.





    
    :param edge: Subdivision edge (not a quad-edge) 
    
    :type edge: :class:`CvSubdiv2DEdge`
    
    
    :param type: Specifies which of the related edges to return, one of the following: 
    
    :type type: :class:`CvNextEdgeType`
    
    
    
        
        * **CV_NEXT_AROUND_ORG** next around the edge origin ( ``eOnext``  on the picture below if  ``e``  is the input edge) 
        
        
        * **CV_NEXT_AROUND_DST** next around the edge vertex ( ``eDnext`` ) 
        
        
        * **CV_PREV_AROUND_ORG** previous around the edge origin (reversed  ``eRnext`` ) 
        
        
        * **CV_PREV_AROUND_DST** previous around the edge destination (reversed  ``eLnext`` ) 
        
        
        * **CV_NEXT_AROUND_LEFT** next around the left facet ( ``eLnext`` ) 
        
        
        * **CV_NEXT_AROUND_RIGHT** next around the right facet ( ``eRnext`` ) 
        
        
        * **CV_PREV_AROUND_LEFT** previous around the left facet (reversed  ``eOnext`` ) 
        
        
        * **CV_PREV_AROUND_RIGHT** previous around the right facet (reversed  ``eDnext`` ) 
        
        
        
    
    






The function returns one of the edges related to the input edge.


.. index:: Subdiv2DNextEdge

.. _Subdiv2DNextEdge:

Subdiv2DNextEdge
----------------




.. function:: Subdiv2DNextEdge(edge)-> CvSubdiv2DEdge

    Returns next edge around the edge origin





    
    :param edge: Subdivision edge (not a quad-edge) 
    
    :type edge: :class:`CvSubdiv2DEdge`
    
    
    






The function returns the next edge around the edge origin: 
``eOnext``
on the picture above if 
``e``
is the input edge)


.. index:: Subdiv2DLocate

.. _Subdiv2DLocate:

Subdiv2DLocate
--------------




.. function:: Subdiv2DLocate(subdiv, pt) -> (loc, where)

    Returns the location of a point within a Delaunay triangulation.





    
    :param subdiv: Delaunay or another subdivision 
    
    :type subdiv: :class:`CvSubdiv2D`
    
    
    :param pt: The point to locate 
    
    :type pt: :class:`CvPoint2D32f`
    
    
    :param loc: The location of the point within the triangulation 
    
    :type loc: int
    
    
    :param where: The edge or vertex.  See below. 
    
    :type where: :class:`CvSubdiv2DEdge`, :class:`CvSubdiv2DPoint`
    
    
    
The function locates the input point within the subdivision. There are 5 cases:



    

*
    The point falls into some facet.                          
    ``loc``
    is 
    ``CV_PTLOC_INSIDE``
    and 
    ``where``
    is one of edges of the facet.
     
    

*
    The point falls onto the edge.                            
    ``loc``
    is 
    ``CV_PTLOC_ON_EDGE``
    and 
    ``where``
    is the edge.
     
    

*
    The point coincides with one of the subdivision vertices. 
    ``loc``
    is 
    ``CV_PTLOC_VERTEX``
    and 
    ``where``
    is the vertex.
     
    

*
    The point is outside the subdivsion reference rectangle.  
    ``loc``
    is 
    ``CV_PTLOC_OUTSIDE_RECT``
    and 
    ``where``
    is None.
     
    

*
    One of input arguments is invalid. The function raises an exception.
    
    

.. index:: Subdiv2DRotateEdge

.. _Subdiv2DRotateEdge:

Subdiv2DRotateEdge
------------------




.. function:: Subdiv2DRotateEdge(edge,rotate)-> CvSubdiv2DEdge

    Returns another edge of the same quad-edge.





    
    :param edge: Subdivision edge (not a quad-edge) 
    
    :type edge: :class:`CvSubdiv2DEdge`
    
    
    :param rotate: Specifies which of the edges of the same quad-edge as the input one to return, one of the following: 
        
                
            * **0** the input edge ( ``e``  on the picture below if  ``e``  is the input edge) 
            
               
            * **1** the rotated edge ( ``eRot`` ) 
            
               
            * **2** the reversed edge (reversed  ``e``  (in green)) 
            
               
            * **3** the reversed rotated edge (reversed  ``eRot``  (in green)) 
            
            
    
    :type rotate: int
    
    
    






The function returns one of the edges of the same quad-edge as the input edge.


.. index:: SubdivDelaunay2DInsert

.. _SubdivDelaunay2DInsert:

SubdivDelaunay2DInsert
----------------------




.. function:: SubdivDelaunay2DInsert(subdiv,pt)-> point

    Inserts a single point into a Delaunay triangulation.





    
    :param subdiv: Delaunay subdivision created by the function  :ref:`CreateSubdivDelaunay2D` 
    
    :type subdiv: :class:`CvSubdiv2D`
    
    
    :param pt: Inserted point 
    
    :type pt: :class:`CvPoint2D32f`
    
    
    
The function inserts a single point into a subdivision and modifies the subdivision topology appropriately. If a point with the same coordinates exists already, no new point is added. The function returns a pointer to the allocated point. No virtual point coordinates are calculated at this stage.

