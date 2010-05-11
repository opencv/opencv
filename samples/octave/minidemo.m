#! /usr/bin/env octave


cvNamedWindow("win", CV_WINDOW_AUTOSIZE);
cap = cvCreateFileCapture("/home/x/work/sneaker/dvgrab-001.avi");
img = cvQueryFrame(cap);

printf("Got frame of dimensions (%i x %i)",img.width,img.height);

cvShowImage("win", img);
cvMoveWindow("win", 200, 200);
cvWaitKey(0);

octimg = cv2im(img);

