#! /usr/bin/env octave

cv
highgui

CLOCKS_PER_SEC = 1.0
MHI_DURATION = 1;
MAX_TIME_DELTA = 0.5;
MIN_TIME_DELTA = 0.05;
N = 4;
buf = range(10) 
last = 0;
mhi = []; # MHI
orient = []; # orientation
mask = []; # valid orientation mask
segmask = []; # motion segmentation map
storage = []; # temporary storage

function update_mhi( img, dst, diff_threshold )
  global last
  global mhi
  global storage
  global mask
  global orient
  global segmask
  timestamp = time.clock()/CLOCKS_PER_SEC; # get current time in seconds
  size = cvSize(img.width,img.height); # get current frame size
  idx1 = last;
  if (! mhi || mhi.width != size.width || mhi.height != size.height)
    for i=0:N-1,
      buf[i] = cvCreateImage( size, IPL_DEPTH_8U, 1 );
      cvZero( buf[i] );
      mhi = cvCreateImage( size, IPL_DEPTH_32F, 1 );
      cvZero( mhi ); # clear MHI at the beginning
      orient = cvCreateImage( size, IPL_DEPTH_32F, 1 );
      segmask = cvCreateImage( size, IPL_DEPTH_32F, 1 );
      mask = cvCreateImage( size, IPL_DEPTH_8U, 1 );

      cvCvtColor( img, buf[last], CV_BGR2GRAY ); # convert frame to grayscale
      idx2 = (last + 1) % N; # index of (last - (N-1))th frame
      last = idx2;
      silh = buf[idx2];
      cvAbsDiff( buf[idx1], buf[idx2], silh ); # get difference between frames
      cvThreshold( silh, silh, diff_threshold, 1, CV_THRESH_BINARY ); # and threshold it
      cvUpdateMotionHistory( silh, mhi, timestamp, MHI_DURATION ); # update MHI
      cvCvtScale( mhi, mask, 255./MHI_DURATION,
		 (MHI_DURATION - timestamp)*255./MHI_DURATION );
      cvZero( dst );
      cvMerge( mask, [], [], [], dst );
      cvCalcMotionGradient( mhi, mask, orient, MAX_TIME_DELTA, MIN_TIME_DELTA, 3 );
      if( not storage )
	storage = cvCreateMemStorage(0);
      else
	cvClearMemStorage(storage);
	seq = cvSegmentMotion( mhi, segmask, storage, timestamp, MAX_TIME_DELTA );
	for i=-1:seq.total-1,
          if( i < 0 )  # case of the whole image
            comp_rect = cvRect( 0, 0, size.width, size.height );
            color = CV_RGB(255,255,255);
            magnitude = 100.;
          else  # i-th motion component
            comp_rect = seq[i].rect 
            if( comp_rect.width + comp_rect.height < 100 ) # reject very small components
              continue;
	    endif
	  endif
          color = CV_RGB(255,0,0);
          magnitude = 30.;
          silh_roi = cvGetSubRect(silh, comp_rect);
          mhi_roi = cvGetSubRect( mhi, comp_rect );
          orient_roi = cvGetSubRect( orient, comp_rect );
          mask_roi = cvGetSubRect( mask, comp_rect );
          angle = cvCalcGlobalOrientation( orient_roi, mask_roi, mhi_roi, timestamp, MHI_DURATION);
          angle = 360.0 - angle;  # adjust for images with top-left origin
          count = cvNorm( silh_roi, [], CV_L1, [] ); # calculate number of points within silhouette ROI
          if( count < comp_rect.width * comp_rect.height * 0.05 )
            continue;
	  endif
          center = cvPoint( (comp_rect.x + comp_rect.width/2),
                           (comp_rect.y + comp_rect.height/2) );
          cvCircle( dst, center, cvRound(magnitude*1.2), color, 3, CV_AA, 0 );
          cvLine( dst, center, cvPoint( cvRound( center.x + magnitude*cos(angle*CV_PI/180)),
				       cvRound( center.y - magnitude*sin(angle*CV_PI/180))), \
		 color, 3, CV_AA, 0 );
	endfor
      endif
    endfor
  endif
endfunction

motion = 0;
capture = 0;

if (size(argv, 1)==1)
  capture = cvCreateCameraCapture( 0 )
elseif (size(argv, 1)==2 && all(isdigit(argv(1, :))))
  capture = cvCreateCameraCapture( int32(argv(1, :)) )
elseif (size(argv, 1)==2)
  capture = cvCreateFileCapture( argv(1, :) ); 
endif

if (!capture)
  print "Could not initialize capturing..."
  exit(-1)
endif

cvNamedWindow( "Motion", 1 );
while (true)
  image = cvQueryFrame( capture );
  if( image )
    if( ! motion )
      motion = cvCreateImage( cvSize(image.width,image.height), 8, 3 );
      cvZero( motion );
      motion.origin = image.origin;
    endif
    update_mhi( image, motion, 30 );
    cvShowImage( "Motion", motion );
    if( cvWaitKey(10) != -1 )
      break;
    endif
  else
    break
  endif
endwhile

cvDestroyWindow( "Motion" );
