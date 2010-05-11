#! /usr/bin/env octave

printf("OpenCV Octave version of contours\n");

## import the necessary things for OpenCV
cv;


global _SIZE;
global _red;
global _green;
global _blue;
global contours;

## some default constants
_SIZE = 500;
_DEFAULT_LEVEL = 3;

## definition of some colors
_red = cvScalar (0, 0, 255, 0);
_green = cvScalar (0, 255, 0, 0);
_white = cvRealScalar (255);
_black = cvRealScalar (0);

## the callback on the trackbar, to set the level of contours we want
## to display
function on_trackbar (position)
  global cv;
  global _SIZE;
  global _red;
  global _green;
  global _blue;
  global contours;

  ## create the image for putting in it the founded contours
  contours_image = cvCreateImage (cvSize (_SIZE, _SIZE), 8, 3);

  ## compute the real level of display, given the current position
  levels = position - 3;

  ## initialisation
  _contours = contours;
  
  if (levels <= 0)
    ## zero or negative value
    ## => get to the nearest face to make it look more funny
    _contours = contours.h_next.h_next.h_next;
  endif
  
  ## first, clear the image where we will draw contours
  cvSetZero (contours_image);
  
  ## draw contours in red and green
  cvDrawContours (contours_image, _contours, _red, _green, levels, 3, cv.CV_AA, cvPoint (0, 0));

  ## finally, show the image
  cvShowImage ("contours", contours_image);
endfunction

## create the image where we want to display results
image = cvCreateImage (cvSize (_SIZE, _SIZE), 8, 1);

## start with an empty image
cvSetZero (image);

## draw the original picture
for i=0:6-1,
  dx = mod(i,2) * 250 - 30;
  dy = (i / 2) * 150;
  
  cvEllipse (image,
                cvPoint (dx + 150, dy + 100),
                cvSize (100, 70),
                0, 0, 360, _white, -1, 8, 0);
  cvEllipse (image,
                cvPoint (dx + 115, dy + 70),
                cvSize (30, 20),
                0, 0, 360, _black, -1, 8, 0);
  cvEllipse (image,
                cvPoint (dx + 185, dy + 70),
                cvSize (30, 20),
                0, 0, 360, _black, -1, 8, 0);
  cvEllipse (image,
                cvPoint (dx + 115, dy + 70),
                cvSize (15, 15),
                0, 0, 360, _white, -1, 8, 0);
  cvEllipse (image,
                cvPoint (dx + 185, dy + 70),
                cvSize (15, 15),
                0, 0, 360, _white, -1, 8, 0);
  cvEllipse (image,
                cvPoint (dx + 115, dy + 70),
                cvSize (5, 5),
                0, 0, 360, _black, -1, 8, 0);
  cvEllipse (image,
                cvPoint (dx + 185, dy + 70),
                cvSize (5, 5),
                0, 0, 360, _black, -1, 8, 0);
  cvEllipse (image,
                cvPoint (dx + 150, dy + 100),
                cvSize (10, 5),
                0, 0, 360, _black, -1, 8, 0);
  cvEllipse (image,
                cvPoint (dx + 150, dy + 150),
                cvSize (40, 10),
                0, 0, 360, _black, -1, 8, 0);
  cvEllipse (image,
                cvPoint (dx + 27, dy + 100),
                cvSize (20, 35),
                0, 0, 360, _white, -1, 8, 0);
  cvEllipse (image,
                cvPoint (dx + 273, dy + 100),
                cvSize (20, 35),
                0, 0, 360, _white, -1, 8, 0);
endfor

## create window and display the original picture in it
cvNamedWindow ("image", 1);
cvShowImage ("image", image);

## create the storage area
storage = cvCreateMemStorage (0);

## find the contours
[nb_contours, contours] = cvFindContours (image, storage, sizeof_CvContour, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cvPoint (0,0));

## comment this out if you do not want approximation
contours = cvApproxPoly (contours, sizeof_CvContour, storage, CV_POLY_APPROX_DP, 3, 1);

## create the window for the contours
cvNamedWindow ("contours", 1);

## create the trackbar, to enable the change of the displayed level
cvCreateTrackbar ("levels+3", "contours", 3, 7, @on_trackbar);

## call one time the callback, so we will have the 1st display done
on_trackbar (_DEFAULT_LEVEL);

## wait a key pressed to end
cvWaitKey (0);
