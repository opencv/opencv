#! /usr/bin/env octave
##   Tracking of rotating point.
##   Rotation speed is constant.
##   Both state and measurements vectors are 1D (a point angle),
##   Measurement is the real point angle + gaussian noise.
##   The real and the estimated points are connected with yellow line segment,
##   the real and the measured points are connected with red line segment.
##   (if Kalman filter works correctly,
##    the yellow segment should be shorter than the red one).
##   Pressing any key (except ESC) will reset the tracking with a different speed.
##   Pressing ESC will stop the program.

cv;
highgui;

global img;

function ret=calc_point(angle)
  global img;
  ret=cvPoint( cvRound(img.width/2 + img.width/3*cos(angle)), \
              cvRound(img.height/2 - img.width/3*sin(angle)));
endfunction

function draw_cross( center, color, d )
  global img;
  global CV_AA;
  cvLine( img, cvPoint( center.x - d, center.y - d ),
         cvPoint( center.x + d, center.y + d ), color, 1, CV_AA, 0); 
  cvLine( img, cvPoint( center.x + d, center.y - d ),                
         cvPoint( center.x - d, center.y + d ), \
	 color, 1, CV_AA, 0 );
endfunction

A = [ 1, 1; 0, 1 ];

img = cvCreateImage( cvSize(500,500), 8, 3 );
kalman = cvCreateKalman( 2, 1, 0 );
state = cvCreateMat( 2, 1, CV_32FC1 );  # (phi, delta_phi)
process_noise = cvCreateMat( 2, 1, CV_32FC1 );
measurement = cvCreateMat( 1, 1, CV_32FC1 );
rng = cvRNG(-1);
code = -1;

cvZero( measurement );
cvNamedWindow( "Kalman", 1 );

while (true),
  cvRandArr( rng, state, CV_RAND_NORMAL, cvRealScalar(0), cvRealScalar(0.1) );

  kalman.transition_matrix = mat2cv(A, CV_32FC1);
  cvSetIdentity( kalman.measurement_matrix, cvRealScalar(1) );
  cvSetIdentity( kalman.process_noise_cov, cvRealScalar(1e-5) );
  cvSetIdentity( kalman.measurement_noise_cov, cvRealScalar(1e-1) );
  cvSetIdentity( kalman.error_cov_post, cvRealScalar(1));
  cvRandArr( rng, kalman.state_post, CV_RAND_NORMAL, cvRealScalar(0), cvRealScalar(0.1) );

  while (true),

    state_angle = state(0);
    state_pt = calc_point(state_angle);

    prediction = cvKalmanPredict( kalman );
    predict_angle = prediction(0);
    predict_pt = calc_point(predict_angle);

    cvRandArr( rng, measurement, CV_RAND_NORMAL, cvRealScalar(0), \
              cvRealScalar(sqrt(kalman.measurement_noise_cov(0))) );

    ## generate measurement 
    cvMatMulAdd( kalman.measurement_matrix, state, measurement, measurement );

    measurement_angle = measurement(0);
    measurement_pt = calc_point(measurement_angle);
    
    ## plot points 
    cvZero( img );
    draw_cross( state_pt, CV_RGB(255,255,255), 3 );
    draw_cross( measurement_pt, CV_RGB(255,0,0), 3 );
    draw_cross( predict_pt, CV_RGB(0,255,0), 3 );
    cvLine( img, state_pt, measurement_pt, CV_RGB(255,0,0), 3, CV_AA, 0 );
    cvLine( img, state_pt, predict_pt, CV_RGB(255,255,0), 3, CV_AA, 0 );
    
    cvKalmanCorrect( kalman, measurement );

    cvRandArr( rng, process_noise, CV_RAND_NORMAL, cvRealScalar(0), \
              cvRealScalar(sqrt(kalman.process_noise_cov(0)(0))));
    cvMatMulAdd( kalman.transition_matrix, state, process_noise, state );

    cvShowImage( "Kalman", img );
    code = cvWaitKey( 100 );
    
    if( code > 0 )
      break;
    endif
  endwhile
  
  if( code == '\x1b' || code == 'q' || code == 'Q' )
    break;
  endif
endwhile

cvDestroyWindow("Kalman");
