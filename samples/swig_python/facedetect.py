#!/usr/bin/python
"""
This program is demonstration for face and object detection using haar-like features.
The program finds faces in a camera image or video stream and displays a red box around them.

Original C implementation by:  ?
Python implementation by: Roman Stanchak
"""
import sys
from opencv.cv import *
from opencv.highgui import *


# Global Variables
cascade = None
storage = cvCreateMemStorage(0)
cascade_name = "../../data/haarcascades/haarcascade_frontalface_alt.xml"
input_name = "../c/lena.jpg"

# Parameters for haar detection
# From the API:
# The default parameters (scale_factor=1.1, min_neighbors=3, flags=0) are tuned 
# for accurate yet slow object detection. For a faster operation on real video 
# images the settings are: 
# scale_factor=1.2, min_neighbors=2, flags=CV_HAAR_DO_CANNY_PRUNING, 
# min_size=<minimum possible face size
min_size = cvSize(20,20)
image_scale = 1.3
haar_scale = 1.2
min_neighbors = 2
haar_flags = 0


def detect_and_draw( img ):
    # allocate temporary images
    gray = cvCreateImage( cvSize(img.width,img.height), 8, 1 )
    small_img = cvCreateImage((cvRound(img.width/image_scale),
			       cvRound (img.height/image_scale)), 8, 1 )

    # convert color input image to grayscale
    cvCvtColor( img, gray, CV_BGR2GRAY )

    # scale input image for faster processing
    cvResize( gray, small_img, CV_INTER_LINEAR )

    cvEqualizeHist( small_img, small_img )
    
    cvClearMemStorage( storage )

    if( cascade ):
        t = cvGetTickCount()
        faces = cvHaarDetectObjects( small_img, cascade, storage,
                                     haar_scale, min_neighbors, haar_flags, min_size )
        t = cvGetTickCount() - t
        print "detection time = %gms" % (t/(cvGetTickFrequency()*1000.))
        if faces:
            for face_rect in faces:
                # the input to cvHaarDetectObjects was resized, so scale the 
                # bounding box of each face and convert it to two CvPoints
                pt1 = cvPoint( int(face_rect.x*image_scale), int(face_rect.y*image_scale))
                pt2 = cvPoint( int((face_rect.x+face_rect.width)*image_scale),
                               int((face_rect.y+face_rect.height)*image_scale) )
                cvRectangle( img, pt1, pt2, CV_RGB(255,0,0), 3, 8, 0 )

    cvShowImage( "result", img )


if __name__ == '__main__':

    if len(sys.argv) > 1:

        if sys.argv[1].startswith("--cascade="):
            cascade_name = sys.argv[1][ len("--cascade="): ]
            if len(sys.argv) > 2:
                input_name = sys.argv[2]

        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print "Usage: facedetect --cascade=\"<cascade_path>\" [filename|camera_index]\n" 
            sys.exit(-1)

        else:
            input_name = sys.argv[1]
    
    # the OpenCV API says this function is obsolete, but we can't
    # cast the output of cvLoad to a HaarClassifierCascade, so use this anyways
    # the size parameter is ignored
    cascade = cvLoadHaarClassifierCascade( cascade_name, cvSize(1,1) )
    
    if not cascade:
        print "ERROR: Could not load classifier cascade"
        sys.exit(-1)
    

    if input_name.isdigit():
        capture = cvCreateCameraCapture( int(input_name) )
    else:
        capture = cvCreateFileCapture( input_name ) 

    cvNamedWindow( "result", 1 )

    if capture:
        frame_copy = None
        while True:
            frame = cvQueryFrame( capture )
            if not frame:
                cvWaitKey(0)
                break
            if not frame_copy:
                frame_copy = cvCreateImage( cvSize(frame.width,frame.height),
                                            IPL_DEPTH_8U, frame.nChannels )
            if frame.origin == IPL_ORIGIN_TL:
                cvCopy( frame, frame_copy )
            else:
                cvFlip( frame, frame_copy, 0 )
            
            detect_and_draw( frame_copy )

            if( cvWaitKey( 10 ) >= 0 ):
                break

    else:
        image = cvLoadImage( input_name, 1 )

        if image:
            detect_and_draw( image )
            cvWaitKey(0)

    cvDestroyWindow("result")
