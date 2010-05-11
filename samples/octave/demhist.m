#! /usr/bin/env octave
cv;
highgui;

file_name = "../c/baboon.jpg";

global Gbrightness;
global Gcontrast;
global hist_size;
global ranges;
global src_image;
global dst_image;
global hist_image;
global hist;
global lut;

_brightness = 100;
_contrast = 100;
Gbrightness = 100;
Gcontrast = 100;

hist_size = 64;
range_0={0,256};
ranges = { range_0 };
src_image=[];
dst_image=[];
hist_image=[];
hist=[];
lut=cvCreateMat(256,1,CV_8U);

## brightness/contrast callback function
function update_brightness( val )
  global Gbrightness    # global tag is required, or we get UnboundLocalError
  Gbrightness = val;
  update_brightcont( );
endfunction

function update_contrast( val )
  global Gcontrast;     # global tag is required, or we get UnboundLocalError
  Gcontrast = val;
  update_brightcont( );
endfunction

function update_brightcont()
  global Gbrightness;
  global Gcontrast;
  global hist_size;
  global ranges;
  global src_image;
  global dst_image;
  global hist_image;
  global hist;
  global lut;
  global cvCalcHist; # use cv namespace for these instead
  global cvZero;
  global cvScale;

  brightness = Gbrightness - 100;
  contrast = Gcontrast - 100;
  max_value = 0;

  ## The algorithm is by Werner D. Streidt
  ## (http://visca.com/ffactory/archives/5-99/msg00021.html)
  if( contrast > 0 )
    delta = 127.*contrast/100;
    a = 255./(255. - delta*2);
    b = a*(brightness - delta);
  else
    delta = -128.*contrast/100;
    a = (256.-delta*2)/255.;
    b = a*brightness + delta;
  endif

  for i=0:256-1,
    v = cvRound(a*i + b);
    if( v < 0 )
      v = 0;
    endif
    if( v > 255 )
      v = 255;
    endif
    lut(i) = v;
  endfor
  
  cvLUT( src_image, dst_image, lut );
  cvShowImage( "image", dst_image );

  cvCalcHist( dst_image, hist, 0, [] );
  cvZero( dst_image );
  [min_value, max_value] = cvGetMinMaxHistValue( hist );
  cvScale( hist.bins, hist.bins, double(hist_image.height)/max_value, 0 );
  ##cvNormalizeHist( hist, 1000 );

  cvSet( hist_image, cvScalarAll(255));
  bin_w = cvRound(double(hist_image.width)/hist_size);

  for i=0:hist_size-1,
    cvRectangle( hist_image, cvPoint(i*bin_w, hist_image.height), cvPoint((i+1)*bin_w, hist_image.height - cvRound(cvGetReal1D(hist.bins,i))), cvScalarAll(0), -1, 8, 0 );
  endfor
  
  cvShowImage( "histogram", hist_image );
endfunction


## Load the source image. HighGUI use.
if size(argv, 1)>1
  file_name = argv(){1}
endif

src_image = cvLoadImage( file_name, 0 );

if (!swig_this(src_image))
  printf("Image was not loaded.\n");
  exit(-1);
endif


dst_image = cvCloneImage(src_image);
hist_image = cvCreateImage(cvSize(320,200), 8, 1);
hist = cvCreateHist({hist_size}, CV_HIST_ARRAY, ranges, 1);

cvNamedWindow("image", 0);
cvNamedWindow("histogram", 0);

cvCreateTrackbar("brightness", "image", _brightness, 200, @update_brightness);
cvCreateTrackbar("contrast", "image", _contrast, 200, @update_contrast);

update_brightcont();
cvWaitKey(0);
