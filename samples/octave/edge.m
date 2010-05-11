#! /usr/bin/env octave

printf("OpenCV Octave version of edge\n");

global g;

## import the necessary things for OpenCV
cv;
highgui;

## some definitions
g.win_name = "Edge";
g.trackbar_name = "Threshold";

## the callback on the trackbar
function on_trackbar (position)
  global g;
  global cv;
  global highgui;

  cv.cvSmooth (g.gray, g.edge, cv.CV_BLUR, 3, 3, 0);
  cv.cvNot (g.gray, g.edge);

  ## run the edge dector on gray scale
  cv.cvCanny (g.gray, g.edge, position, position * 3, 3);

  ## reset
  cv.cvSetZero (g.col_edge);

  ## copy edge points
  cv.cvCopy (g.image, g.col_edge, g.edge);
  
  ## show the image
  highgui.cvShowImage (g.win_name, g.col_edge);
endfunction

filename = "../c/fruits.jpg";

if (size(argv, 1)>1)
  filename = argv(){1};
endif

## load the image gived on the command line
g.image = highgui.cvLoadImage (filename);

if (!swig_this(g.image))
  printf("Error loading image '%s'",filename);
  exit(-1);
endif

## create the output image
g.col_edge = cv.cvCreateImage (cv.cvSize (g.image.width, g.image.height), 8, 3);

## convert to grayscale
g.gray = cv.cvCreateImage (cv.cvSize (g.image.width, g.image.height), 8, 1);
g.edge = cv.cvCreateImage (cv.cvSize (g.image.width, g.image.height), 8, 1);
cv.cvCvtColor (g.image, g.gray, cv.CV_BGR2GRAY);

## create the window
highgui.cvNamedWindow (g.win_name, highgui.CV_WINDOW_AUTOSIZE);

## create the trackbar
highgui.cvCreateTrackbar (g.trackbar_name, g.win_name, 1, 100, @on_trackbar);

## show the image
on_trackbar (0);

## wait a key pressed to end
highgui.cvWaitKey (0);
