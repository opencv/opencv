#!/usr/bin/python
"""
This program is demonstration for ellipse fitting. Program finds 
contours and approximate it by ellipses.

Trackbar specify threshold parametr.

White lines is contours. Red lines is fitting ellipses.

Original C implementation by:  Denis Burenkov.
Python implementation by: Roman Stanchak
"""

import sys
from opencv import cv
from opencv import highgui

image02 = None
image03 = None
image04 = None

def process_image( slider_pos ): 
    """
    Define trackbar callback functon. This function find contours,
    draw it and approximate it by ellipses.
    """
    stor = cv.cvCreateMemStorage(0);
    
    # Threshold the source image. This needful for cv.cvFindContours().
    cv.cvThreshold( image03, image02, slider_pos, 255, cv.CV_THRESH_BINARY );
    
    # Find all contours.
    nb_contours, cont = cv.cvFindContours (image02,
            stor,
            cv.sizeof_CvContour,
            cv.CV_RETR_LIST,
            cv.CV_CHAIN_APPROX_NONE,
            cv.cvPoint (0,0))
    
    # Clear images. IPL use.
    cv.cvZero(image02);
    cv.cvZero(image04);
    
    # This cycle draw all contours and approximate it by ellipses.
    for c in cont.hrange():
        count = c.total; # This is number point in contour

        # Number point must be more than or equal to 6 (for cv.cvFitEllipse_32f).        
        if( count < 6 ):
            continue;
        
        # Alloc memory for contour point set.    
        PointArray = cv.cvCreateMat(1, count, cv.CV_32SC2)
        PointArray2D32f= cv.cvCreateMat( 1, count, cv.CV_32FC2)
        
        # Get contour point set.
        cv.cvCvtSeqToArray(c, PointArray, cv.cvSlice(0, cv.CV_WHOLE_SEQ_END_INDEX));
        
        # Convert CvPoint set to CvBox2D32f set.
        cv.cvConvert( PointArray, PointArray2D32f )
        
        box = cv.CvBox2D()

        # Fits ellipse to current contour.
        box = cv.cvFitEllipse2(PointArray2D32f);
        
        # Draw current contour.
        cv.cvDrawContours(image04, c, cv.CV_RGB(255,255,255), cv.CV_RGB(255,255,255),0,1,8,cv.cvPoint(0,0));
        
        # Convert ellipse data from float to integer representation.
        center = cv.CvPoint()
        size = cv.CvSize()
        center.x = cv.cvRound(box.center.x);
        center.y = cv.cvRound(box.center.y);
        size.width = cv.cvRound(box.size.width*0.5);
        size.height = cv.cvRound(box.size.height*0.5);
        box.angle = -box.angle;
        
        # Draw ellipse.
        cv.cvEllipse(image04, center, size,
                  box.angle, 0, 360,
                  cv.CV_RGB(0,0,255), 1, cv.CV_AA, 0);
    
    # Show image. HighGUI use.
    highgui.cvShowImage( "Result", image04 );


if __name__ == '__main__':
    argc = len(sys.argv)
    filename = "../c/stuff.jpg"
    if(argc == 2):
        filename = sys.argv[1]
    
    slider_pos = 70

    # load image and force it to be grayscale
    image03 = highgui.cvLoadImage(filename, 0)
    if not image03:
        print "Could not load image " + filename
        sys.exit(-1)

    # Create the destination images
    image02 = cv.cvCloneImage( image03 );
    image04 = cv.cvCloneImage( image03 );

    # Create windows.
    highgui.cvNamedWindow("Source", 1);
    highgui.cvNamedWindow("Result", 1);

    # Show the image.
    highgui.cvShowImage("Source", image03);

    # Create toolbars. HighGUI use.
    highgui.cvCreateTrackbar( "Threshold", "Result", slider_pos, 255, process_image );


    process_image( 1 );

    #Wait for a key stroke; the same function arranges events processing                
    print "Press any key to exit"
    highgui.cvWaitKey(0);

    highgui.cvDestroyWindow("Source");
    highgui.cvDestroyWindow("Result");

