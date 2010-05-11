#! /usr/bin/env octave
cv;
highgui;

global src;
global image
global element
global element_shape
global global_pos;
global dest;
src = 0;
image = 0;
dest = 0;
element = 0;
element_shape = CV_SHAPE_RECT;
global_pos = 0;

function Opening(pos)
  global src;
  global image
  global element
  global element_shape
  global global_pos;
  global dest;
  element = cvCreateStructuringElementEx( pos*2+1, pos*2+1, pos, pos, element_shape, [] );
  cvErode(src,image,element,1);
  cvDilate(image,dest,element,1);
  cvShowImage("Opening&Closing window",dest);
endfunction
function Closing(pos)
  global src;
  global image
  global element
  global element_shape
  global global_pos;
  global dest;
  element = cvCreateStructuringElementEx( pos*2+1, pos*2+1, pos, pos, element_shape, [] );
  cvDilate(src,image,element,1);
  cvErode(image,dest,element,1);
  cvShowImage("Opening&Closing window",dest);
endfunction
function Erosion(pos)
  global src;
  global image
  global element
  global element_shape
  global global_pos;
  global dest;
  element = cvCreateStructuringElementEx( pos*2+1, pos*2+1, pos, pos, element_shape, [] );
  cvErode(src,dest,element,1);
  cvShowImage("Erosion&Dilation window",dest);
endfunction
function Dilation(pos)
  global src;
  global image
  global element
  global element_shape
  global global_pos;
  global dest;
  element = cvCreateStructuringElementEx( pos*2+1, pos*2+1, pos, pos, element_shape, [] );
  cvDilate(src,dest,element,1);
  cvShowImage("Erosion&Dilation window",dest);
endfunction

filename = "../c/baboon.jpg";
if (size(argv, 1)==1)
  filename = argv(){1};
endif
src = cvLoadImage(filename,1);
if (! swig_this(src))
  exit(-1);
endif

image = cvCloneImage(src);
dest = cvCloneImage(src);
cvNamedWindow("Opening&Closing window",1);
cvNamedWindow("Erosion&Dilation window",1);
cvShowImage("Opening&Closing window",src);
cvShowImage("Erosion&Dilation window",src);
cvCreateTrackbar("Open","Opening&Closing window",global_pos,10,@Opening);
cvCreateTrackbar("Close","Opening&Closing window",global_pos,10,@Closing);
cvCreateTrackbar("Dilate","Erosion&Dilation window",global_pos,10,@Dilation);
cvCreateTrackbar("Erode","Erosion&Dilation window",global_pos,10,@Erosion);
cvWaitKey(0);
cvDestroyWindow("Opening&Closing window");
cvDestroyWindow("Erosion&Dilation window");
