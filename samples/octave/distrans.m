#! /usr/bin/env octave
cv;
highgui;

global g;

g.wndname = "Distance transform";
g.tbarname = "Threshold";

## The output images
g.dist = 0;
g.dist8u1 = 0;
g.dist8u2 = 0;
g.dist8u = 0;
g.dist32s = 0;

g.gray = 0;
g.edge = 0;

## define a trackbar callback
function on_trackbar( edge_thresh )
  global g;
  global cv;

  cvThreshold( g.gray, g.edge, double(edge_thresh), double(edge_thresh), cv.CV_THRESH_BINARY );
  ## Distance transform                  
  cvDistTransform( g.edge, g.dist, cv.CV_DIST_L2, cv.CV_DIST_MASK_5, [], [] );

  cvConvertScale( g.dist, g.dist, 5000.0, 0 );
  cvPow( g.dist, g.dist, 0.5 );

  cvConvertScale( g.dist, g.dist32s, 1.0, 0.5 );
  cvAndS( g.dist32s, cvScalarAll(255), g.dist32s, [] );
  cvConvertScale( g.dist32s, g.dist8u1, 1, 0 );
  cvConvertScale( g.dist32s, g.dist32s, -1, 0 );
  cvAddS( g.dist32s, cvScalarAll(255), g.dist32s, [] );
  cvConvertScale( g.dist32s, g.dist8u2, 1, 0 );
  cvMerge( g.dist8u1, g.dist8u2, g.dist8u2, [], g.dist8u );
  cvShowImage( g.wndname, g.dist8u );
endfunction


edge_thresh = 100;

filename = "../c/stuff.jpg";
if (size(argv, 1) > 1)
  filename = argv(){1};
endif

g.gray = cvLoadImage( filename, 0 );
if (!swig_this(g.gray))
  printf("Failed to load %s\n",filename);
  exit(-1);
endif

## Create the output image
g.dist = cvCreateImage( cvSize(g.gray.width,g.gray.height), IPL_DEPTH_32F, 1 );
g.dist8u1 = cvCloneImage( g.gray );
g.dist8u2 = cvCloneImage( g.gray );
g.dist8u = cvCreateImage( cvSize(g.gray.width,g.gray.height), IPL_DEPTH_8U, 3 );
g.dist32s = cvCreateImage( cvSize(g.gray.width,g.gray.height), IPL_DEPTH_32S, 1 );

## Convert to grayscale
g.edge = cvCloneImage( g.gray );

## Create a window
cvNamedWindow( g.wndname, 1 );

## create a toolbar 
cvCreateTrackbar( g.tbarname, g.wndname, edge_thresh, 255, @on_trackbar );

## Show the image
on_trackbar(edge_thresh);

## Wait for a key stroke; the same function arranges events processing
cvWaitKey(0);
