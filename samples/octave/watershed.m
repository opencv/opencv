#! /usr/bin/env octave

cv;
highgui;

global g;
g.marker_mask = [];
g.markers = [];
g.img0 = []
g.img = []
g.img_gray = [] 
g.wshed = []
g.prev_pt = cvPoint(-1,-1);

function on_mouse( event, x, y, flags, param )
  global g;
  global cv;
  global highgui;

  if( !swig_this( g.img) )
    return;
  endif
  if( event == highgui.CV_EVENT_LBUTTONUP || ! bitand(flags,highgui.CV_EVENT_FLAG_LBUTTON) )
    g.prev_pt = cvPoint(-1,-1);
  elseif( event == highgui.CV_EVENT_LBUTTONDOWN )
    g.prev_pt = cvPoint(x,y);
  elseif( event == highgui.CV_EVENT_MOUSEMOVE && bitand(flags,highgui.CV_EVENT_FLAG_LBUTTON) )
    pt = cvPoint(x,y);
    if( g.prev_pt.x < 0 )
      g.prev_pt = pt;
    endif
    cvLine( g.marker_mask, g.prev_pt, pt, cvScalarAll(255), 5, 8, 0 );
    cvLine( g.img, g.prev_pt, pt, cvScalarAll(255), 5, 8, 0 );
    g.prev_pt = pt;
    cvShowImage( "image", g.img );
  endif
endfunction

filename = "../c/fruits.jpg";
if (size(argv, 1)>=1)
  filename = argv(){1};
endif

rng = cvRNG(-1);
g.img0 = cvLoadImage(filename,1);
if (!swig_this(g.img0))
  print "Error opening image '%s'" % filename
  exit(-1)
endif

printf("Hot keys:\n");
printf("\tESC - quit the program\n");
printf("\tr - restore the original image\n");
printf("\tw - run watershed algorithm\n");
printf("\t  (before that, roughly outline several g.markers on the image)\n");

cvNamedWindow( "image", 1 );
cvNamedWindow( "watershed transform", 1 );

g.img = cvCloneImage( g.img0 );
g.img_gray = cvCloneImage( g.img0 );
g.wshed = cvCloneImage( g.img0 );
g.marker_mask = cvCreateImage( cvGetSize(g.img), 8, 1 );
g.markers = cvCreateImage( cvGetSize(g.img), IPL_DEPTH_32S, 1 );

cvCvtColor( g.img, g.marker_mask, CV_BGR2GRAY );
cvCvtColor( g.marker_mask, g.img_gray, CV_GRAY2BGR );

cvZero( g.marker_mask );
cvZero( g.wshed );

cvShowImage( "image", g.img );
cvShowImage( "watershed transform", g.wshed );

cvSetMouseCallback( "image", @on_mouse, [] );
while (true)
  c = cvWaitKey(0);
  if (c=='\x1b')
    break;
  endif
  if (c == 'r')
    cvZero( g.marker_mask );
    cvCopy( g.img0, g.img );
    cvShowImage( "image", g.img );
  endif
  if (c == 'w')
    storage = cvCreateMemStorage(0);
    comp_count = 0;
    ##cvSaveImage( "g.wshed_mask.png", g.marker_mask );
    ##g.marker_mask = cvLoadImage( "g.wshed_mask.png", 0 );
    [nb_cont, contours] = cvFindContours( g.marker_mask, storage, \
				       sizeof_CvContour, \
				       CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
    cvZero( g.markers );
    swig_this(contours)
    while (swig_this(contours))
      cvDrawContours( g.markers, contours, cvScalarAll(comp_count+1), \
                     cvScalarAll(comp_count+1), -1, -1, 8, cvPoint(0,0) );
      contours=contours.h_next;
      comp_count+=1;
    endwhile
    comp_count
    color_tab = cvCreateMat( comp_count, 1, CV_8UC3 );
    for i=0:comp_count-1,
      color_tab(i) = cvScalar( mod(cvRandInt(rng),180) + 50,  \
                              mod(cvRandInt(rng),180) + 50,  \
                              mod(cvRandInt(rng),180) + 50 );
    endfor
    t = int32(cvGetTickCount());
    cvWatershed( g.img0, g.markers );
    t = int32(cvGetTickCount()) - t;
    ##print "exec time = %f" % t/(cvGetTickFrequency()*1000.)

    cvSet( g.wshed, cvScalarAll(255) );

    ## paint the watershed image
    for j=0:g.markers.height-1,
      for i=0:g.markers.width-1,
	{j,i}
	idx = g.markers({j,i});
	if (idx==-1)
          continue
	endif
	idx = idx-1;
	g.wshed({j,i}) = color_tab({idx,0});
      endfor
    endfor

    cvAddWeighted( g.wshed, 0.5, g.img_gray, 0.5, 0, g.wshed );
    cvShowImage( "watershed transform", g.wshed );
    cvWaitKey();
  endif
endwhile
