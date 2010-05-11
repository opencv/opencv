#!/usr/bin/python
"""
the script demostrates iterative construction of
delaunay triangulation and voronoi tesselation

Original Author (C version): ?
Converted to Python by: Roman Stanchak
"""
from opencv.cv import *
from opencv.highgui import *
from random import random,randint

def draw_subdiv_point( img, fp, color ):
    cvCircle( img, cvPoint(cvRound(fp.x), cvRound(fp.y)), 3, color, CV_FILLED, 8, 0 );



def draw_subdiv_edge( img, edge, color ):
    org_pt = cvSubdiv2DEdgeOrg(edge);
    dst_pt = cvSubdiv2DEdgeDst(edge);

    if org_pt and dst_pt :
    
        org = org_pt.pt;
        dst = dst_pt.pt;

        iorg = cvPoint( cvRound( org.x ), cvRound( org.y ));
        idst = cvPoint( cvRound( dst.x ), cvRound( dst.y ));

        cvLine( img, iorg, idst, color, 1, CV_AA, 0 );


def draw_subdiv( img, subdiv, delaunay_color, voronoi_color ):
    
    total = subdiv.edges.total;
    elem_size = subdiv.edges.elem_size;

    for edge in subdiv.edges:
        edge_rot = cvSubdiv2DRotateEdge( edge, 1 )

        if( CV_IS_SET_ELEM( edge )):
            draw_subdiv_edge( img, edge_rot, voronoi_color );
            draw_subdiv_edge( img, edge, delaunay_color );


def locate_point( subdiv, fp, img, active_color ):

    [res, e0, p] = cvSubdiv2DLocate( subdiv, fp );

    if e0:
        e = e0
        while True:
            draw_subdiv_edge( img, e, active_color );
            e = cvSubdiv2DGetEdge(e,CV_NEXT_AROUND_LEFT);
            if e == e0:
                break

    draw_subdiv_point( img, fp, active_color );


def draw_subdiv_facet( img, edge ):

    t = edge;
    count = 0;

    # count number of edges in facet
    while count == 0 or t != edge:
        count+=1
        t = cvSubdiv2DGetEdge( t, CV_NEXT_AROUND_LEFT );

    buf = []

    # gather points
    t = edge;
    for i in range(count):
        assert t>4
        pt = cvSubdiv2DEdgeOrg( t );
        if not pt: 
            break;
        buf.append( cvPoint( cvRound(pt.pt.x), cvRound(pt.pt.y) ) );
        t = cvSubdiv2DGetEdge( t, CV_NEXT_AROUND_LEFT );

    if( len(buf)==count ):
        pt = cvSubdiv2DEdgeDst( cvSubdiv2DRotateEdge( edge, 1 ));
        cvFillConvexPoly( img, buf, CV_RGB(randint(0,255),randint(0,255),randint(0,255)), CV_AA, 0 );
        cvPolyLine( img, [buf], 1, CV_RGB(0,0,0), 1, CV_AA, 0);
        draw_subdiv_point( img, pt.pt, CV_RGB(0,0,0));

def paint_voronoi( subdiv, img ):
    total = subdiv.edges.total;
    elem_size = subdiv.edges.elem_size;

    cvCalcSubdivVoronoi2D( subdiv );

    for edge in subdiv.edges:

        if( CV_IS_SET_ELEM( edge )):
            # left
            draw_subdiv_facet( img, cvSubdiv2DRotateEdge( edge, 1 ));

            # right
            draw_subdiv_facet( img, cvSubdiv2DRotateEdge( edge, 3 ));

if __name__ == '__main__':
    win = "source";
    rect = cvRect( 0, 0, 600, 600 );

    active_facet_color = CV_RGB( 255, 0, 0 );
    delaunay_color  = CV_RGB( 0,0,0);
    voronoi_color = CV_RGB(0, 180, 0);
    bkgnd_color = CV_RGB(255,255,255);

    img = cvCreateImage( cvSize(rect.width,rect.height), 8, 3 );
    cvSet( img, bkgnd_color );

    cvNamedWindow( win, 1 );

    storage = cvCreateMemStorage(0);
    subdiv = cvCreateSubdivDelaunay2D( rect, storage );

    print "Delaunay triangulation will be build now interactively."
    print "To stop the process, press any key\n";

    for i in range(200):
        fp = cvPoint2D32f( random()*(rect.width-10)+5, random()*(rect.height-10)+5 )

        locate_point( subdiv, fp, img, active_facet_color );
        cvShowImage( win, img );

        if( cvWaitKey( 100 ) >= 0 ):
            break;

        cvSubdivDelaunay2DInsert( subdiv, fp );
        cvCalcSubdivVoronoi2D( subdiv );
        cvSet( img, bkgnd_color );
        draw_subdiv( img, subdiv, delaunay_color, voronoi_color );
        cvShowImage( win, img );

        if( cvWaitKey( 100 ) >= 0 ):
            break;
    

    cvSet( img, bkgnd_color );
    paint_voronoi( subdiv, img );
    cvShowImage( win, img );

    cvWaitKey(0);

    cvDestroyWindow( win );
