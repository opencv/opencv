#! /usr/bin/env octave
## This program is demonstration for ellipse fitting. Program finds 
## contours and approximate it by ellipses.

## Trackbar specify threshold parametr.

## White lines is contours. Red lines is fitting ellipses.

## Original C implementation by:  Denis Burenkov.
## Python implementation by: Roman Stanchak
## Octave implementation by: Xavier Delacour

cv;
highgui;

global g;
g.image02 = [];
g.image03 = [];
g.image04 = [];

function process_image( slider_pos )
  global g;
  global cv;
  global highgui;

  ##  Define trackbar callback functon. This function find contours,
  ##    draw it and approximate it by ellipses.
  stor = cv.cvCreateMemStorage(0);
  
  ## Threshold the source image. This needful for cv.cvFindContours().
  cv.cvThreshold( g.image03, g.image02, slider_pos, 255, cv.CV_THRESH_BINARY );
  
  ## Find all contours.
  [nb_contours, cont] = cv.cvFindContours (g.image02,stor,cv.sizeof_CvContour,cv.CV_RETR_LIST,cv.CV_CHAIN_APPROX_NONE,cv.cvPoint (0,0));
  
  ## Clear images. IPL use.
  cv.cvZero(g.image02);
  cv.cvZero(g.image04);
  
  ## This cycle draw all contours and approximate it by ellipses.
  for c = cv.CvSeq_hrange(cont),
    c = c{1};
    count = c.total; # This is number point in contour

    ## Number point must be more than or equal to 6 (for cv.cvFitEllipse_32f).        
    if( count < 6 )
      continue;
    endif
    
    ## Alloc memory for contour point set.    
    PointArray = cv.cvCreateMat(1, count, cv.CV_32SC2);
    PointArray2D32f= cv.cvCreateMat( 1, count, cv.CV_32FC2);
    
    ## Get contour point set.
    cv.cvCvtSeqToArray(c, PointArray, cv.cvSlice(0, cv.CV_WHOLE_SEQ_END_INDEX));
    
    ## Convert CvPoint set to CvBox2D32f set.
    cv.cvConvert( PointArray, PointArray2D32f );
    
    box = cv.CvBox2D();

    ## Fits ellipse to current contour.
    box = cv.cvFitEllipse2(PointArray2D32f);
    
    ## Draw current contour.
    cv.cvDrawContours(g.image04, c, cv.CV_RGB(255,255,255), cv.CV_RGB(255,255,255),0,1,8,cv.cvPoint(0,0));
    
    ## Convert ellipse data from float to integer representation.
    center = cv.CvPoint();
    size = cv.CvSize();
    center.x = cv.cvRound(box.center.x);
    center.y = cv.cvRound(box.center.y);
    size.width = cv.cvRound(box.size.width*0.5);
    size.height = cv.cvRound(box.size.height*0.5);
    box.angle = -box.angle;
    
    ## Draw ellipse.
    cv.cvEllipse(g.image04, center, size,box.angle, 0, 360,cv.CV_RGB(0,0,255), 1, cv.CV_AA, 0);
  endfor    

  ## Show image. HighGUI use.
  highgui.cvShowImage( "Result", g.image04 );
endfunction

argc = size(argv, 1);
filename = "../c/stuff.jpg";
if(argc == 2)
  filename = argv(){1};
endif

slider_pos = 70;

## load image and force it to be grayscale
g.image03 = highgui.cvLoadImage(filename, 0);
if (!swig_this( g.image03))
  printf("Could not load image %s\n", filename);
  exit(-1);
endif

## Create the destination images
g.image02 = cv.cvCloneImage( g.image03 );
g.image04 = cv.cvCloneImage( g.image03 );

## Create windows.
highgui.cvNamedWindow("Source", 1);
highgui.cvNamedWindow("Result", 1);

## Show the image.
highgui.cvShowImage("Source", g.image03);

## Create toolbars. HighGUI use.
highgui.cvCreateTrackbar( "Threshold", "Result", slider_pos, 255, @process_image );


process_image( 1 );

## Wait for a key stroke; the same function arranges events processing                
printf("Press any key to exit\n");
highgui.cvWaitKey(0);

highgui.cvDestroyWindow("Source");
highgui.cvDestroyWindow("Result");

