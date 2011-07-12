#!/usr/bin/python
"""
the script demostrates iterative construction of
delaunay triangulation and voronoi tesselation

Original Author (C version): ?
Converted to Python by: Roman Stanchak
"""
import cv2.cv as cv
import random

def draw_subdiv_point( img, fp, color ):
    cv.Circle( img, (cv.Round(fp[0]), cv.Round(fp[1])), 3, color, cv.CV_FILLED, 8, 0 );

def draw_subdiv_edge( img, edge, color ):
    org_pt = cv.Subdiv2DEdgeOrg(edge);
    dst_pt = cv.Subdiv2DEdgeDst(edge);

    if org_pt and dst_pt :
    
        org = org_pt.pt;
        dst = dst_pt.pt;

        iorg = ( cv.Round( org[0] ), cv.Round( org[1] ));
        idst = ( cv.Round( dst[0] ), cv.Round( dst[1] ));

        cv.Line( img, iorg, idst, color, 1, cv.CV_AA, 0 );


def draw_subdiv( img, subdiv, delaunay_color, voronoi_color ):
    
    for edge in subdiv.edges:
        edge_rot = cv.Subdiv2DRotateEdge( edge, 1 )

        draw_subdiv_edge( img, edge_rot, voronoi_color );
        draw_subdiv_edge( img, edge, delaunay_color );


def locate_point( subdiv, fp, img, active_color ):

    (res, e0) = cv.Subdiv2DLocate( subdiv, fp );

    if res in [ cv.CV_PTLOC_INSIDE, cv.CV_PTLOC_ON_EDGE ]:
        e = e0
        while True:
            draw_subdiv_edge( img, e, active_color );
            e = cv.Subdiv2DGetEdge(e, cv.CV_NEXT_AROUND_LEFT);
            if e == e0:
                break

    draw_subdiv_point( img, fp, active_color );


def draw_subdiv_facet( img, edge ):

    t = edge;
    count = 0;

    # count number of edges in facet
    while count == 0 or t != edge:
        count+=1
        t = cv.Subdiv2DGetEdge( t, cv.CV_NEXT_AROUND_LEFT );

    buf = []

    # gather points
    t = edge;
    for i in range(count):
        assert t>4
        pt = cv.Subdiv2DEdgeOrg( t );
        if not pt: 
            break;
        buf.append( ( cv.Round(pt.pt[0]), cv.Round(pt.pt[1]) ) );
        t = cv.Subdiv2DGetEdge( t, cv.CV_NEXT_AROUND_LEFT );

    if( len(buf)==count ):
        pt = cv.Subdiv2DEdgeDst( cv.Subdiv2DRotateEdge( edge, 1 ));
        cv.FillConvexPoly( img, buf, cv.RGB(random.randrange(256),random.randrange(256),random.randrange(256)), cv.CV_AA, 0 );
        cv.PolyLine( img, [buf], 1, cv.RGB(0,0,0), 1, cv.CV_AA, 0);
        draw_subdiv_point( img, pt.pt, cv.RGB(0,0,0));

def paint_voronoi( subdiv, img ):

    cv.CalcSubdivVoronoi2D( subdiv );

    for edge in subdiv.edges:

        # left
        draw_subdiv_facet( img, cv.Subdiv2DRotateEdge( edge, 1 ));

        # right
        draw_subdiv_facet( img, cv.Subdiv2DRotateEdge( edge, 3 ));

if __name__ == '__main__':
    win = "source";
    rect = ( 0, 0, 600, 600 );

    active_facet_color = cv.RGB( 255, 0, 0 );
    delaunay_color  = cv.RGB( 0,0,0);
    voronoi_color = cv.RGB(0, 180, 0);
    bkgnd_color = cv.RGB(255,255,255);

    img = cv.CreateImage( (rect[2],rect[3]), 8, 3 );
    cv.Set( img, bkgnd_color );

    cv.NamedWindow( win, 1 );

    storage = cv.CreateMemStorage(0);
    subdiv = cv.CreateSubdivDelaunay2D( rect, storage );

    print "Delaunay triangulation will be build now interactively."
    print "To stop the process, press any key\n";

    for i in range(200):
        fp = ( random.random()*(rect[2]-10)+5, random.random()*(rect[3]-10)+5 )

        locate_point( subdiv, fp, img, active_facet_color );
        cv.ShowImage( win, img );

        if( cv.WaitKey( 100 ) >= 0 ):
            break;

        cv.SubdivDelaunay2DInsert( subdiv, fp );
        cv.CalcSubdivVoronoi2D( subdiv );
        cv.Set( img, bkgnd_color );
        draw_subdiv( img, subdiv, delaunay_color, voronoi_color );
        cv.ShowImage( win, img );

        if( cv.WaitKey( 100 ) >= 0 ):
            break;
    

    cv.Set( img, bkgnd_color );
    paint_voronoi( subdiv, img );
    cv.ShowImage( win, img );

    cv.WaitKey(0);

    cv.DestroyWindow( win );
