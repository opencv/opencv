Planar Subdivisions
===================

.. highlight:: c



.. index:: CvSubdiv2D

.. _CvSubdiv2D:

CvSubdiv2D
----------



.. ctype:: CvSubdiv2D



Planar subdivision.




::


    
    #define CV_SUBDIV2D_FIELDS()    \
        CV_GRAPH_FIELDS()           \
        int  quad_edges;            \
        int  is_geometry_valid;     \
        CvSubdiv2DEdge recent_edge; \
        CvPoint2D32f  topleft;      \
        CvPoint2D32f  bottomright;
    
    typedef struct CvSubdiv2D
    {
        CV_SUBDIV2D_FIELDS()
    }
    CvSubdiv2D;
    

..

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



.. image:: ../pics/subdiv.png



OpenCV subdivides a plane into triangles using Delaunay's
algorithm. Subdivision is built iteratively starting from a dummy
triangle that includes all the subdivision points for sure. In this
case the dual subdivision is a Voronoi diagram of the input 2d point set. The
subdivisions can be used for the 3d piece-wise transformation of a plane,
morphing, fast location of points on the plane, building special graphs
(such as NNG,RNG) and so forth.


.. index:: CvQuadEdge2D

.. _CvQuadEdge2D:

CvQuadEdge2D
------------



.. ctype:: CvQuadEdge2D



Quad-edge of planar subdivision.




::


    
    /* one of edges within quad-edge, lower 2 bits is index (0..3)
       and upper bits are quad-edge pointer */
    typedef long CvSubdiv2DEdge;
    
    /* quad-edge structure fields */
    #define CV_QUADEDGE2D_FIELDS()     \
        int flags;                     \
        struct CvSubdiv2DPoint* pt[4]; \
        CvSubdiv2DEdge  next[4];
    
    typedef struct CvQuadEdge2D
    {
        CV_QUADEDGE2D_FIELDS()
    }
    CvQuadEdge2D;
    
    

..

Quad-edge is a basic element of subdivision containing four edges (e, eRot, reversed e and reversed eRot):



.. image:: ../pics/quadedge.png




.. index:: CvSubdiv2DPoint

.. _CvSubdiv2DPoint:

CvSubdiv2DPoint
---------------



.. ctype:: CvSubdiv2DPoint



Point of original or dual subdivision.




::


    
    #define CV_SUBDIV2D_POINT_FIELDS()\
        int            flags;      \
        CvSubdiv2DEdge first;      \
        CvPoint2D32f   pt;         \
        int id;
    
    #define CV_SUBDIV2D_VIRTUAL_POINT_FLAG (1 << 30)
    
    typedef struct CvSubdiv2DPoint
    {
        CV_SUBDIV2D_POINT_FIELDS()
    }
    CvSubdiv2DPoint;
    

..



    

* id
    This integer can be used to index auxillary data associated with each vertex of the planar subdivision
    
    

.. index:: CalcSubdivVoronoi2D

.. _CalcSubdivVoronoi2D:

CalcSubdivVoronoi2D
-------------------






.. cfunction:: void cvCalcSubdivVoronoi2D(  CvSubdiv2D* subdiv )

    Calculates the coordinates of Voronoi diagram cells.





    
    :param subdiv: Delaunay subdivision, in which all the points are already added 
    
    
    
The function calculates the coordinates
of virtual points. All virtual points corresponding to some vertex of the
original subdivision form (when connected together) a boundary of the Voronoi
cell at that point.


.. index:: ClearSubdivVoronoi2D

.. _ClearSubdivVoronoi2D:

ClearSubdivVoronoi2D
--------------------






.. cfunction:: void cvClearSubdivVoronoi2D( CvSubdiv2D* subdiv )

    Removes all virtual points.





    
    :param subdiv: Delaunay subdivision 
    
    
    
The function removes all of the virtual points. It
is called internally in 
:ref:`CalcSubdivVoronoi2D`
if the subdivision
was modified after previous call to the function.



.. index:: CreateSubdivDelaunay2D

.. _CreateSubdivDelaunay2D:

CreateSubdivDelaunay2D
----------------------






.. cfunction:: CvSubdiv2D* cvCreateSubdivDelaunay2D(  CvRect rect, CvMemStorage* storage )

    Creates an empty Delaunay triangulation.





    
    :param rect: Rectangle that includes all of the 2d points that are to be added to the subdivision 
    
    
    :param storage: Container for subdivision 
    
    
    
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






.. cfunction:: CvSubdiv2DPoint* cvFindNearestPoint2D(  CvSubdiv2D* subdiv, CvPoint2D32f pt )

    Finds the closest subdivision vertex to the given point.





    
    :param subdiv: Delaunay or another subdivision 
    
    
    :param pt: Input point 
    
    
    
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






.. cfunction:: CvSubdiv2DPoint* cvSubdiv2DEdgeDst(  CvSubdiv2DEdge edge )

    Returns the edge destination.





    
    :param edge: Subdivision edge (not a quad-edge) 
    
    
    
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






.. cfunction:: CvSubdiv2DEdge  cvSubdiv2DGetEdge( CvSubdiv2DEdge edge, CvNextEdgeType type )

    Returns one of the edges related to the given edge.





    
    :param edge: Subdivision edge (not a quad-edge) 
    
    
    :param type: Specifies which of the related edges to return, one of the following: 
    
    
    
        
        * **CV_NEXT_AROUND_ORG** next around the edge origin ( ``eOnext``  on the picture below if  ``e``  is the input edge) 
        
        
        * **CV_NEXT_AROUND_DST** next around the edge vertex ( ``eDnext`` ) 
        
        
        * **CV_PREV_AROUND_ORG** previous around the edge origin (reversed  ``eRnext`` ) 
        
        
        * **CV_PREV_AROUND_DST** previous around the edge destination (reversed  ``eLnext`` ) 
        
        
        * **CV_NEXT_AROUND_LEFT** next around the left facet ( ``eLnext`` ) 
        
        
        * **CV_NEXT_AROUND_RIGHT** next around the right facet ( ``eRnext`` ) 
        
        
        * **CV_PREV_AROUND_LEFT** previous around the left facet (reversed  ``eOnext`` ) 
        
        
        * **CV_PREV_AROUND_RIGHT** previous around the right facet (reversed  ``eDnext`` ) 
        
        
        
    
    


.. image:: ../pics/quadedge.png



The function returns one of the edges related to the input edge.


.. index:: Subdiv2DNextEdge

.. _Subdiv2DNextEdge:

Subdiv2DNextEdge
----------------






.. cfunction:: CvSubdiv2DEdge  cvSubdiv2DNextEdge( CvSubdiv2DEdge edge )

    Returns next edge around the edge origin





    
    :param edge: Subdivision edge (not a quad-edge) 
    
    
    


.. image:: ../pics/quadedge.png



The function returns the next edge around the edge origin: 
``eOnext``
on the picture above if 
``e``
is the input edge)


.. index:: Subdiv2DLocate

.. _Subdiv2DLocate:

Subdiv2DLocate
--------------






.. cfunction:: CvSubdiv2DPointLocation  cvSubdiv2DLocate(  CvSubdiv2D* subdiv, CvPoint2D32f pt, CvSubdiv2DEdge* edge, CvSubdiv2DPoint** vertex=NULL )

    Returns the location of a point within a Delaunay triangulation.





    
    :param subdiv: Delaunay or another subdivision 
    
    
    :param pt: The point to locate 
    
    
    :param edge: The output edge the point falls onto or right to 
    
    
    :param vertex: Optional output vertex double pointer the input point coinsides with 
    
    
    
The function locates the input point within the subdivision. There are 5 cases:



    

*
    The point falls into some facet. The function returns 
    ``CV_PTLOC_INSIDE``
    and 
    ``*edge``
    will contain one of edges of the facet.
     
    

*
    The point falls onto the edge. The function returns 
    ``CV_PTLOC_ON_EDGE``
    and 
    ``*edge``
    will contain this edge.
     
    

*
    The point coincides with one of the subdivision vertices. The function returns 
    ``CV_PTLOC_VERTEX``
    and 
    ``*vertex``
    will contain a pointer to the vertex.
     
    

*
    The point is outside the subdivsion reference rectangle. The function returns 
    ``CV_PTLOC_OUTSIDE_RECT``
    and no pointers are filled.
     
    

*
    One of input arguments is invalid. A runtime error is raised or, if silent or "parent" error processing mode is selected, 
    ``CV_PTLOC_ERROR``
    is returnd.
    
    

.. index:: Subdiv2DRotateEdge

.. _Subdiv2DRotateEdge:

Subdiv2DRotateEdge
------------------






.. cfunction:: CvSubdiv2DEdge  cvSubdiv2DRotateEdge(  CvSubdiv2DEdge edge, int rotate )

    Returns another edge of the same quad-edge.





    
    :param edge: Subdivision edge (not a quad-edge) 
    
    
    :param rotate: Specifies which of the edges of the same quad-edge as the input one to return, one of the following: 
        
                
            * **0** the input edge ( ``e``  on the picture below if  ``e``  is the input edge) 
            
               
            * **1** the rotated edge ( ``eRot`` ) 
            
               
            * **2** the reversed edge (reversed  ``e``  (in green)) 
            
               
            * **3** the reversed rotated edge (reversed  ``eRot``  (in green)) 
            
            
    
    
    


.. image:: ../pics/quadedge.png



The function returns one of the edges of the same quad-edge as the input edge.


.. index:: SubdivDelaunay2DInsert

.. _SubdivDelaunay2DInsert:

SubdivDelaunay2DInsert
----------------------






.. cfunction:: CvSubdiv2DPoint*  cvSubdivDelaunay2DInsert(  CvSubdiv2D* subdiv, CvPoint2D32f pt)

    Inserts a single point into a Delaunay triangulation.





    
    :param subdiv: Delaunay subdivision created by the function  :ref:`CreateSubdivDelaunay2D` 
    
    
    :param pt: Inserted point 
    
    
    
The function inserts a single point into a subdivision and modifies the subdivision topology appropriately. If a point with the same coordinates exists already, no new point is added. The function returns a pointer to the allocated point. No virtual point coordinates are calculated at this stage.

