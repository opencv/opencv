#! /usr/bin/env octave
cv;
highgui;

global g;
g.src=[];
g.dst=[];
g.src2=[];

function on_mouse( event, x, y, flags, param )
  global g;
  global cv;
  global highgui;

  if(!swig_this(g.src) )
    return;
  endif

  if (event==highgui.CV_EVENT_LBUTTONDOWN)
    cvLogPolar( g.src, g.dst, cvPoint2D32f(x,y), 40, cv.CV_INTER_LINEAR+cv.CV_WARP_FILL_OUTLIERS );
    cvLogPolar( g.dst, g.src2, cvPoint2D32f(x,y), 40, cv.CV_INTER_LINEAR+cv.CV_WARP_FILL_OUTLIERS+cv.CV_WARP_INVERSE_MAP );
    cvShowImage( "log-polar", g.dst );
    cvShowImage( "inverse log-polar", g.src2 );
  endif
endfunction

filename = "../c/fruits.jpg"
if (size(argv, 1)>1)
  filename=argv(){1};
endif

g.src = cvLoadImage(filename,1);
if (!swig_this(g.src))
  printf("Could not open %s",filename);
  exit(-1)
endif

cvNamedWindow( "original",1 );
cvNamedWindow( "log-polar", 1 );
cvNamedWindow( "inverse log-polar", 1 );


g.dst = cvCreateImage( cvSize(256,256), 8, 3 );
g.src2 = cvCreateImage( cvGetSize(g.src), 8, 3 );

cvSetMouseCallback( "original", @on_mouse );
on_mouse( CV_EVENT_LBUTTONDOWN, g.src.width/2, g.src.height/2, [], []);

cvShowImage( "original", g.src );
cvWaitKey();
