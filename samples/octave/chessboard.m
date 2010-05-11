#! /usr/bin/env octave
cv;
highgui;

arg_list=argv();

cvNamedWindow("win");
if (!size(arg_list,1))
  error("must specify filename");
  exit
endif
filename = arg_list{1};
im = cvLoadImage(filename, CV_LOAD_IMAGE_GRAYSCALE);
im3 = cvLoadImage(filename, CV_LOAD_IMAGE_COLOR);
chessboard_dim = cvSize( 5, 6 );

[found_all, corners] = cvFindChessboardCorners( im, chessboard_dim );

cvDrawChessboardCorners( im3, chessboard_dim, corners, found_all );

cvShowImage("win", im3);
cvWaitKey();

