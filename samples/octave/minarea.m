#! /usr/bin/env octave

cv;
highgui;

function ret = randint(a, b)
  ret = int32(rand() * (b - a) + a);
endfunction

function minarea_array(img, count)
  global cv;
  global highgui;
  pointMat = cvCreateMat( count, 1, cv.CV_32SC2 );
  for i=0:count-1,
    pointMat(i) = cvPoint( randint(img.width/4, img.width*3/4), randint(img.height/4, img.height*3/4) );
  endfor

  box = cvMinAreaRect2( pointMat );
  box_vtx = cvBoxPoints( box );
  [success, center, radius] = cvMinEnclosingCircle( pointMat );
  cv.cvZero( img );
  for i=0:count-1,
    cvCircle( img, cvGet1D(pointMat,i), 2, CV_RGB( 255, 0, 0 ), \
	     cv.CV_FILLED, cv.CV_AA, 0 );
  endfor

  box_vtx = {cvPointFrom32f(box_vtx{1}), \
             cvPointFrom32f(box_vtx{2}), \
             cvPointFrom32f(box_vtx{3}), \
             cvPointFrom32f(box_vtx{4})};
  cvCircle( img, cvPointFrom32f(center), cvRound(radius), CV_RGB(255, 255, 0), 1, cv.CV_AA, 0 );
  cvPolyLine( img, {box_vtx}, 1, CV_RGB(0,255,255), 1, cv.CV_AA ) ;
endfunction


function minarea_seq(img, count, storage)
  global cv;
  global highgui;
  ptseq = cvCreateSeq( bitor(cv.CV_SEQ_KIND_GENERIC, cv.CV_32SC2), cv.sizeof_CvContour, cv.sizeof_CvPoint, storage );
  ptseq = cv.CvSeq_CvPoint.cast( ptseq );
  for i=0:count-1,
    pt0 = cvPoint( randint(img.width/4, img.width*3/4), randint(img.height/4, img.height*3/4) );
    cvSeqPush( ptseq, pt0 );
  endfor
  box = cvMinAreaRect2( ptseq );
  box_vtx = cvBoxPoints( box );
  [success, center, radius] = cvMinEnclosingCircle( ptseq );
  cv.cvZero( img );
  for pt = CvSeq_map(ptseq),
    pt = pt{1};
    cvCircle( img, pt, 2, CV_RGB( 255, 0, 0 ), cv.CV_FILLED, cv.CV_AA, 0 );
  endfor

  box_vtx = {cvPointFrom32f(box_vtx{1}), \
             cvPointFrom32f(box_vtx{2}), \
             cvPointFrom32f(box_vtx{3}), \
             cvPointFrom32f(box_vtx{4})};
  cvCircle( img, cvPointFrom32f(center), cvRound(radius), CV_RGB(255, 255, 0), 1, cv.CV_AA, 0 );
  cvPolyLine( img, {box_vtx}, 1, CV_RGB(0,255,255), 1, cv.CV_AA );
  cvClearMemStorage( storage );
endfunction

img = cvCreateImage( cvSize( 500, 500 ), 8, 3 );
storage = cvCreateMemStorage(0);

cvNamedWindow( "rect & circle", 1 );

use_seq=false;

while (true),
  count = randint(1,100);
  if (use_seq)
    minarea_seq(img, count, storage);
  else
    minarea_array(img, count);
  endif

  cvShowImage("rect & circle", img);
  key = cvWaitKey();
  if( key == '\x1b' );
    break;
  endif

  use_seq = !use_seq;
endwhile
