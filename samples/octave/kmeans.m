#! /usr/bin/env octave
cv;
highgui;
MAX_CLUSTERS=5;

function ret = randint(v1, v2)
  ret = int32(rand() * (v2 - v1) + v1);
end

color_tab = { \
             CV_RGB(255,0,0), \
             CV_RGB(0,255,0), \
             CV_RGB(100,100,255), \
             CV_RGB(255,0,255), \
             CV_RGB(255,255,0)};
img = cvCreateImage( cvSize( 500, 500 ), 8, 3 );
rng = cvRNG(-1);

cvNamedWindow( "clusters", 1 );

while (true),
  cluster_count = randint(2, MAX_CLUSTERS);
  sample_count = randint(1, 1000);
  points = cvCreateMat( sample_count, 1, CV_32FC2 );
  clusters = cvCreateMat( sample_count, 1, CV_32SC1 );
  
  ## generate random sample from multigaussian distribution
  for k=0:cluster_count-1,
    center = CvPoint();
    center.x = mod(cvRandInt(rng), img.width);
    center.y = mod(cvRandInt(rng), img.height);
    first = k*sample_count/cluster_count;
    last = sample_count;
    if (k != cluster_count)
      last = (k+1)*sample_count/cluster_count;
    endif

    point_chunk = cvGetRows(points, first, last);
    
    cvRandArr( rng, point_chunk, CV_RAND_NORMAL, \
              cvScalar(center.x,center.y,0,0), \
              cvScalar(img.width*0.1,img.height*0.1,0,0));
  endfor
  

  ## shuffle samples 
  cvRandShuffle( points, rng );

  cvKMeans2( points, cluster_count, clusters, \
            cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0 ));

  cvZero( img );

  for i=0:sample_count-1,
    cluster_idx = clusters(i);
    pt = points(i);
    cvCircle( img, pt, 2, color_tab{cluster_idx + 1}, CV_FILLED, CV_AA, 0 );

cvCircle( img, pt, 2, color_tab{cluster_idx + 1}, CV_FILLED, CV_AA, 0 );
  endfor
  

  cvShowImage( "clusters", img );

  key = cvWaitKey(0);
  if( key == 27 || key == 'q' || key == 'Q' )
    break;
  endif
endwhile


cvDestroyWindow( "clusters" );
