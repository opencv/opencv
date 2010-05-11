#! /usr/bin/env octave
##
## The full "Square Detector" program.
## It loads several images subsequentally and tries to find squares in
## each image
##

cv;
highgui;

global g;

g.thresh = 50;
g.img = [];
g.img0 = [];
g.storage = [];
g.wndname = "Square Detection Demo";

function ret = compute_angle( pt1, pt2, pt0 )
  dx1 = pt1.x - pt0.x;
  dy1 = pt1.y - pt0.y;
  dx2 = pt2.x - pt0.x;
  dy2 = pt2.y - pt0.y;
  ret = (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
endfunction

function squares = findSquares4( img, storage )
  global g;
  global cv;

  N = 11;
  sz = cvSize( img.width, img.height );
  timg = cvCloneImage( img ); # make a copy of input image
  gray = cvCreateImage( sz, 8, 1 );
  pyr = cvCreateImage( cvSize(int32(sz.width/2), int32(sz.height/2)), 8, 3 );
  ## create empty sequence that will contain points -
  ## 4 points per square (the square's vertices)
  squares = cvCreateSeq( 0, cv.sizeof_CvSeq, cv.sizeof_CvPoint, storage );
  squares = cv.CvSeq_CvPoint.cast( squares );

  ## select the maximum ROI in the image
  ## with the width and height divisible by 2
  subimage = cvGetSubRect( timg, cvRect( 0, 0, sz.width, sz.height ));

  ## down-scale and upscale the image to filter out the noise
  cvPyrDown( subimage, pyr, 7 );
  cvPyrUp( pyr, subimage, 7 );
  tgray = cvCreateImage( sz, 8, 1 );
  ## find squares in every color plane of the image
  for c=1:3,
    ## extract the c-th color plane
    channels = {[], [], []};
    channels{c} = tgray;
    cvSplit( subimage, channels{1}, channels{2}, channels{3}, [] ) ;
    for l=1:N,
      ## hack: use Canny instead of zero threshold level.
      ## Canny helps to catch squares with gradient shading
      if( l == 1 )
	## apply Canny. Take the upper threshold from slider
	## and set the lower to 0 (which forces edges merging)
        cvCanny( tgray, gray, 0, g.thresh, 5 );
	## dilate canny output to remove potential
	## holes between edge segments
        cvDilate( gray, gray, [], 1 );
      else
	## apply threshold if l!=0
	##     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
        cvThreshold( tgray, gray, l*255/N, 255, cv.CV_THRESH_BINARY );
      endif

      ## find contours and store them all as a list
      [count, contours] = cvFindContours( gray, storage, cv.sizeof_CvContour, cv.CV_RETR_LIST, cv.CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0) );

      if (!swig_this(contours))
        continue;
      endif
     
      ## test each contour
      for contour = CvSeq_hrange(contours),
	## approximate contour with accuracy proportional
	## to the contour perimeter
        result = cvApproxPoly( contour, cv.sizeof_CvContour, storage, cv.CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0 );
	## square contours should have 4 vertices after approximation
	## relatively large area (to filter out noisy contours)
	## and be convex.
	## Note: absolute value of an area is used because
	## area may be positive or negative - in accordance with the
	## contour orientation
        if( result.total == 4 &&
           abs(cvContourArea(result)) > 1000 &&
           cvCheckContourConvexity(result) )
          s = 0;
          for i=1:5,
	    ## find minimum angle between joint
	    ## edges (maximum of cosine)
            if( i > 2 )
              t = abs(compute_angle( result{i}, result{i-2}, result{i-1}));
              if (s<t)
                s=t;
              endif
            endif
          endfor
	  ## if cosines of all angles are small
	  ## (all angles are ~90 degree) then write quandrange
	  ## vertices to resultant sequence
          if( s < 0.3 )
            for i=1:4,
              squares.append( result{i} )
            endfor
          endif
        endif
      endfor
    endfor
  endfor
endfunction

## the function draws all the squares in the image
function drawSquares( img, squares )
  global g;
  global cv;

  cpy = cvCloneImage( img );
  ## read 4 sequence elements at a time (all vertices of a square)
  i=0;
  while (i<squares.total)
    pt = { squares{i}, squares{i+1}, squares{i+2}, squares{i+3} };

    ## draw the square as a closed polyline
    cvPolyLine( cpy, {pt}, 1, CV_RGB(0,255,0), 3, cv.CV_AA, 0 );
    i+=4;
  endwhile

  ## show the resultant image
  cvShowImage( g.wndname, cpy );
endfunction

function on_trackbar( a )
  global g;

  if( swig_this(g.img) )
    drawSquares( g.img, findSquares4( g.img, g.storage ) );
  endif
endfunction

g.names =  {"../c/pic1.png", "../c/pic2.png", "../c/pic3.png", \
            "../c/pic4.png", "../c/pic5.png", "../c/pic6.png" };

## create memory storage that will contain all the dynamic data
g.storage = cvCreateMemStorage(0);
for name = g.names,
  g.img0 = cvLoadImage( name, 1 );
  if (!swig_this(g.img0))
    printf("Couldn't load %s\n",name);
    continue;
  endif
  g.img = cvCloneImage( g.img0 );
  ## create window and a trackbar (slider) with parent "image" and set callback
  ## (the slider regulates upper threshold, passed to Canny edge detector)
  cvNamedWindow( g.wndname, 1 );
  cvCreateTrackbar( "canny thresh", g.wndname, g.thresh, 1000, @on_trackbar );
  ## force the image processing
  on_trackbar(0);
  ## wait for key.
  ## Also the function cvWaitKey takes care of event processing
  c = cvWaitKey(0);
  ## clear memory storage - reset free space position
  cvClearMemStorage( g.storage );
  if( c == '\x1b' )
    break;
  endif
endfor
cvDestroyWindow( g.wndname );

