#! /usr/bin/env octave
## This program is demonstration for face and object detection using haar-like features.
## The program finds faces in a camera image or video stream and displays a red box around them.

## Original C implementation by:  ?
## Python implementation by: Roman Stanchak
## Octave implementation by: Xavier Delacour
addpath("/home/x/opencv2/interfaces/swig/octave");
source("/home/x/opencv2/interfaces/swig/octave/PKG_ADD_template");
debug_on_error(true);
debug_on_warning(true);
crash_dumps_octave_core (0)
cv;
highgui;


## Global Variables
global g;
g.cascade = [];
g.storage = cvCreateMemStorage(0);
g.cascade_name = "../../data/haarcascades/haarcascade_frontalface_alt.xml";
g.input_name = "../c/lena.jpg";

## Parameters for haar detection
## From the API:
## The default parameters (scale_factor=1.1, min_neighbors=3, flags=0) are tuned 
## for accurate yet slow object detection. For a faster operation on real video 
## images the settings are: 
## scale_factor=1.2, min_neighbors=2, flags=CV_HAAR_DO_CANNY_PRUNING, 
## min_size=<minimum possible face size
g.min_size = cvSize(20,20);
g.image_scale = 1.3;
g.haar_scale = 1.2;
g.min_neighbors = 2;
g.haar_flags = 0;


function detect_and_draw( img )
  global g;
  global cv;

  gray = cvCreateImage( cvSize(img.width,img.height), 8, 1 );
  small_img = cvCreateImage( cvSize( cvRound (img.width/g.image_scale),
				    cvRound (img.height/g.image_scale)), 8, 1 );
  cvCvtColor( img, gray, cv.CV_BGR2GRAY );
  cvResize( gray, small_img, cv.CV_INTER_LINEAR );

  cvEqualizeHist( small_img, small_img );
  
  cvClearMemStorage( g.storage );

  if( swig_this(g.cascade) )
    tic
    faces = cvHaarDetectObjects( small_img, g.cascade, g.storage,
                                g.haar_scale, g.min_neighbors, g.haar_flags, g.min_size );
    toc
    if (swig_this(faces))
      for r = CvSeq_map(faces),
	r = r{1};
        pt1 = cvPoint( int32(r.x*g.image_scale), int32(r.y*g.image_scale));
        pt2 = cvPoint( int32((r.x+r.width)*g.image_scale), int32((r.y+r.height)*g.image_scale) );
        cvRectangle( img, pt1, pt2, CV_RGB(255,0,0), 3, 8, 0 );
      endfor
    endif
  endif

  cvShowImage( "result", img );
endfunction


if (size(argv, 2) > 0 && (strcmp(argv(){1}, "--help") || strcmp(argv(){1}, "-h")))
  printf("Usage: facedetect --cascade \"<cascade_path>\" [filename|camera_index]\n");
  exit(-1);
endif

if (size(argv, 2) >= 2)
  if (strcmp(argv(){1},"--cascade"))
    g.cascade_name = argv(){2};
    if (size(argv, 2) >= 3)
      g.input_name = argv(){3};
    endif
  endif
elseif (size(argv, 2) == 1)
  g.input_name = argv(){1};
endif

## the OpenCV API says this function is obsolete, but we can't
## cast the output of cvLoad to a HaarClassifierCascade, so use this anyways
## the size parameter is ignored
g.cascade = cvLoadHaarClassifierCascade( g.cascade_name, cvSize(1,1) );

if (!swig_this(g.cascade))
  printf("ERROR: Could not load classifier cascade\n");
  exit(-1);
endif

g.input_name

if (all(isdigit(g.input_name)))
  capture = cvCreateCameraCapture( sscanf(g.input_name, "%i") );
else
  capture = cvCreateFileCapture( g.input_name );
endif

capture

cvNamedWindow( "result", 1 );

if( swig_this(capture) )
  frame_copy = [];
  while (true)
    frame = cvQueryFrame( capture );
    if( ! swig_this(frame) )
      cvWaitKey(0);
    endif
    if( !swig_this(frame_copy) )
      frame_copy = cvCreateImage( cvSize(frame.width,frame.height),
                                 IPL_DEPTH_8U, frame.nChannels );
    endif
    if( frame.origin == IPL_ORIGIN_TL )
      cvCopy( frame, frame_copy );
    else
      cvFlip( frame, frame_copy, 0 );
    endif
    
    detect_and_draw( frame_copy );

    if( cvWaitKey( 10 ) == 27 )
      break;
    endif
  endwhile

else
  image = cvLoadImage( g.input_name, 1 );
  
  if( swig_this(image) )

    detect_and_draw( image );
    cvWaitKey(0);
  endif
endif

cvDestroyWindow("result");
