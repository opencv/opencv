#! /usr/bin/env octave
addpath("/home/x/opencv2/interfaces/swig/octave");
source("/home/x/opencv2/interfaces/swig/octave/PKG_ADD_template");
debug_on_error(true);
debug_on_warning(true);
crash_dumps_octave_core (0)
cv;
highgui;

laplace = [];
colorlaplace = [];
planes = { [], [], [] };
capture = [];

if (size(argv, 2)==0)
  capture = cvCreateCameraCapture( -1 );
elseif (size(argv, 2)==1 && all(isdigit(argv(){1})))
  capture = cvCreateCameraCapture( int32(argv(){1}) );
elseif (size(argv, 2)==1)
  capture = cvCreateFileCapture( argv(){1} );
endif

if (!swig_this(capture))
  printf("Could not initialize capturing...\n");
  exit(-1)
endif

cvNamedWindow( "Laplacian", 1 );

while (true),
  frame = cvQueryFrame( capture );
  if (!swig_this(frame))
    break
  endif

  if (!swig_this(laplace))
    for i=1:size(planes,2),
      planes{i} = cvCreateImage( \
				cvSize(frame.width,frame.height), \
				8, 1 );
    endfor
    laplace = cvCreateImage( cvSize(frame.width,frame.height), IPL_DEPTH_16S, 1 );
    colorlaplace = cvCreateImage( \
				 cvSize(frame.width,frame.height), \
				 8, 3 );
  endif

  cvSplit( frame, planes{1}, planes{2}, planes{3}, [] );
  for plane = planes,
    plane = plane{1};
    cvLaplace( plane, laplace, 3 );
    cvConvertScaleAbs( laplace, plane, 1, 0 );
  endfor

  cvMerge( planes{1}, planes{2}, planes{3}, [], colorlaplace );
#  colorlaplace.origin = frame.origin;

  cvShowImage("Laplacian", colorlaplace );

  if (cvWaitKey(10) == 27)
    break;
  endif
endwhile

cvDestroyWindow("Laplacian");
