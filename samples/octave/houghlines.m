#! /usr/bin/env octave
## This is a standalone program. Pass an image name as a first parameter of the program.

cv;
highgui;

## toggle between CV_HOUGH_STANDARD and CV_HOUGH_PROBILISTIC
USE_STANDARD=0;

filename = "../../docs/ref/pics/building.jpg"
if (size(argv, 1)>=1)
  filename = argv(){1};
endif

src=cvLoadImage(filename, 0);
if (!swig_this(src))
  printf("Error opening image %s\n",filename);
  exit(-1);
endif

dst = cvCreateImage( cvGetSize(src), 8, 1 );
color_dst = cvCreateImage( cvGetSize(src), 8, 3 );
storage = cvCreateMemStorage(0);
lines = 0;
cvCanny( src, dst, 50, 200, 3 );
cvCvtColor( dst, color_dst, CV_GRAY2BGR );

if (USE_STANDARD)
  lines = cvHoughLines2( dst, storage, CV_HOUGH_STANDARD, 1, CV_PI/180, 100, 0, 0 );

  for i=0:min(lines.total, 100)-1,
    line = lines{i};
    rho = line{0};
    theta = line{1};
    pt1 = CvPoint();
    pt2 = CvPoint();
    a = cos(theta);
    b = sin(theta);
    x0 = a*rho;
    y0 = b*rho;
    pt1.x = cvRound(x0 + 1000*(-b));
    pt1.y = cvRound(y0 + 1000*(a));
    pt2.x = cvRound(x0 - 1000*(-b));
    pt2.y = cvRound(y0 - 1000*(a));
    cvLine( color_dst, pt1, pt2, CV_RGB(255,0,0), 3, 8 );
  endfor

else
  lines = cvHoughLines2( dst, storage, CV_HOUGH_PROBABILISTIC, 1, CV_PI/180, 50, 50, 10 );
  for line = CvSeq_map(lines),
    line = line{1};
    cvLine( color_dst, line{0}, line{1}, CV_RGB(255,0,0), 3, 8 );
  endfor
endif

cvNamedWindow( "Source", 1 );
cvShowImage( "Source", src );

cvNamedWindow( "Hough", 1 );
cvShowImage( "Hough", color_dst );

cvWaitKey(0);
