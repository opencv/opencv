#!/usr/bin/python
#
# The full "Square Detector" program.
# It loads several images subsequentally and tries to find squares in
# each image
#

from opencv.cv import *
from opencv.highgui import *
from math import sqrt

thresh = 50;
img = None;
img0 = None;
storage = None;
wndname = "Square Detection Demo";

def angle( pt1, pt2, pt0 ):
    dx1 = pt1.x - pt0.x;
    dy1 = pt1.y - pt0.y;
    dx2 = pt2.x - pt0.x;
    dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);

def findSquares4( img, storage ):
    N = 11;
    sz = cvSize( img.width & -2, img.height & -2 );
    timg = cvCloneImage( img ); # make a copy of input image
    gray = cvCreateImage( sz, 8, 1 );
    pyr = cvCreateImage( cvSize(sz.width/2, sz.height/2), 8, 3 );
    # create empty sequence that will contain points -
    # 4 points per square (the square's vertices)
    squares = cvCreateSeq( 0, sizeof_CvSeq, sizeof_CvPoint, storage );
    squares = CvSeq_CvPoint.cast( squares )

    # select the maximum ROI in the image
    # with the width and height divisible by 2
    subimage = cvGetSubRect( timg, cvRect( 0, 0, sz.width, sz.height ))

    # down-scale and upscale the image to filter out the noise
    cvPyrDown( subimage, pyr, 7 );
    cvPyrUp( pyr, subimage, 7 );
    tgray = cvCreateImage( sz, 8, 1 );
    # find squares in every color plane of the image
    for c in range(3):
        # extract the c-th color plane
        channels = [None, None, None]
        channels[c] = tgray
        cvSplit( subimage, channels[0], channels[1], channels[2], None ) 
        for l in range(N):
            # hack: use Canny instead of zero threshold level.
            # Canny helps to catch squares with gradient shading
            if( l == 0 ):
                # apply Canny. Take the upper threshold from slider
                # and set the lower to 0 (which forces edges merging)
                cvCanny( tgray, gray, 0, thresh, 5 );
                # dilate canny output to remove potential
                # holes between edge segments
                cvDilate( gray, gray, None, 1 );
            else:
                # apply threshold if l!=0:
                #     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                cvThreshold( tgray, gray, (l+1)*255/N, 255, CV_THRESH_BINARY );

            # find contours and store them all as a list
            count, contours = cvFindContours( gray, storage, sizeof_CvContour,
                CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0) );

            if not contours:
                continue
            
            # test each contour
            for contour in contours.hrange():
                # approximate contour with accuracy proportional
                # to the contour perimeter
                result = cvApproxPoly( contour, sizeof_CvContour, storage,
                    CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0 );
                # square contours should have 4 vertices after approximation
                # relatively large area (to filter out noisy contours)
                # and be convex.
                # Note: absolute value of an area is used because
                # area may be positive or negative - in accordance with the
                # contour orientation
                if( result.total == 4 and 
                    abs(cvContourArea(result)) > 1000 and 
                    cvCheckContourConvexity(result) ):
                    s = 0;
                    for i in range(5):
                        # find minimum angle between joint
                        # edges (maximum of cosine)
                        if( i >= 2 ):
                            t = abs(angle( result[i], result[i-2], result[i-1]))
                            if s<t:
                                s=t
                    # if cosines of all angles are small
                    # (all angles are ~90 degree) then write quandrange
                    # vertices to resultant sequence
                    if( s < 0.3 ):
                        for i in range(4):
                            squares.append( result[i] )

    return squares;

# the function draws all the squares in the image
def drawSquares( img, squares ):
    cpy = cvCloneImage( img );
    # read 4 sequence elements at a time (all vertices of a square)
    i=0
    while i<squares.total:
        pt = []
        # read 4 vertices
        pt.append( squares[i] )
        pt.append( squares[i+1] )
        pt.append( squares[i+2] )
        pt.append( squares[i+3] )

        # draw the square as a closed polyline
        cvPolyLine( cpy, [pt], 1, CV_RGB(0,255,0), 3, CV_AA, 0 );
        i+=4

    # show the resultant image
    cvShowImage( wndname, cpy );

def on_trackbar( a ):
    if( img ):
        drawSquares( img, findSquares4( img, storage ) );

names =  ["../c/pic1.png", "../c/pic2.png", "../c/pic3.png",
          "../c/pic4.png", "../c/pic5.png", "../c/pic6.png" ];

if __name__ == "__main__":
    # create memory storage that will contain all the dynamic data
    storage = cvCreateMemStorage(0);
    for name in names:
        img0 = cvLoadImage( name, 1 );
        if not img0:
            print "Couldn't load %s" % name
            continue;
        img = cvCloneImage( img0 );
        # create window and a trackbar (slider) with parent "image" and set callback
        # (the slider regulates upper threshold, passed to Canny edge detector)
        cvNamedWindow( wndname, 1 );
        cvCreateTrackbar( "canny thresh", wndname, thresh, 1000, on_trackbar );
        # force the image processing
        on_trackbar(0);
        # wait for key.
        # Also the function cvWaitKey takes care of event processing
        c = cvWaitKey(0);
        # clear memory storage - reset free space position
        cvClearMemStorage( storage );
        if( c == '\x1b' ):
            break;
    cvDestroyWindow( wndname );
