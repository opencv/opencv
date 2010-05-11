#!/usr/bin/python
from opencv.cv import *
from opencv.highgui import *
import sys

if __name__ == "__main__":
    laplace = None
    colorlaplace = None
    planes = [ None, None, None ]
    capture = None
    
    if len(sys.argv)==1:
        capture = cvCreateCameraCapture( 0 )
    elif len(sys.argv)==2 and sys.argv[1].isdigit():
        capture = cvCreateCameraCapture( int(sys.argv[1]) )
    elif len(sys.argv)==2:
        capture = cvCreateFileCapture( sys.argv[1] ) 

    if not capture:
        print "Could not initialize capturing..."
        sys.exit(-1)
        
    cvNamedWindow( "Laplacian", 1 )

    while True:
        frame = cvQueryFrame( capture )
        if not frame:
            cvWaitKey(0)
            break

        if not laplace:
            for i in range( len(planes) ):
                planes[i] = cvCreateImage( cvSize(frame.width,frame.height), 8, 1 )
            laplace = cvCreateImage( cvSize(frame.width,frame.height), IPL_DEPTH_16S, 1 )
            colorlaplace = cvCreateImage( cvSize(frame.width,frame.height), 8, 3 )

        cvSplit( frame, planes[0], planes[1], planes[2], None )
        for plane in planes:
            cvLaplace( plane, laplace, 3 )
            cvConvertScaleAbs( laplace, plane, 1, 0 )

        cvMerge( planes[0], planes[1], planes[2], None, colorlaplace )

        cvShowImage("Laplacian", colorlaplace )

        if cvWaitKey(10) != -1:
            break

    cvDestroyWindow("Laplacian")
