#! /usr/bin/env octave

## import the necessary things for OpenCV
addpath("/home/x/opencv2/interfaces/swig/octave");
source("/home/x/opencv2/interfaces/swig/octave/PKG_ADD_template");
debug_on_error(true);
debug_on_warning(true);
crash_dumps_octave_core (0)
cv;
highgui;

#############################################################################
## definition of some constants

## how many bins we want for the histogram, and their ranges
hdims = 16;
hranges = {0, 180};

## ranges for the limitation of the histogram
vmin = 10;
vmax = 256;
smin = 30;

## the range we want to monitor
hsv_min = cv.cvScalar (0, smin, vmin, 0);
hsv_max = cv.cvScalar (180, 256, vmax, 0);

#############################################################################
## some useful functions

function rgb = hsv2rgb (hue)
  global cv;
  ## convert the hue value to the corresponding rgb value

  sector_data = [0, 2, 1; 1, 2, 0; 1, 0, 2; 2, 0, 1; 2, 1, 0; 0, 1, 2]+1;
  hue *= 0.1 / 3;
  sector = cv.cvFloor (hue);
  p = cv.cvRound (255 * (hue - sector));
  if (bitand(sector,1))
    p = bitxor(p,255);
  endif

  rgb = zeros(1,3);
  rgb (sector_data (sector+1, 1)) = 255;
  rgb (sector_data (sector+1, 2)) = 0;
  rgb (sector_data (sector+1, 3)) = p;
  
  rgb = cv.cvScalar (rgb (3), rgb (2), rgb (1), 0);
endfunction

#############################################################################
## so, here is the main part of the program

## a small welcome
printf("OpenCV Octave wrapper test\n");
printf("OpenCV version: %s (%d, %d, %d)\n",
       cv.CV_VERSION,cv.CV_MAJOR_VERSION,
       cv.CV_MINOR_VERSION,cv.CV_SUBMINOR_VERSION);

## first, create the necessary windows
highgui.cvNamedWindow ('Camera', highgui.CV_WINDOW_AUTOSIZE);
highgui.cvNamedWindow ('Histogram', highgui.CV_WINDOW_AUTOSIZE);

## move the new window to a better place
#highgui.cvMoveWindow ('Camera', 10, 40);
#highgui.cvMoveWindow ('Histogram', 10, 270);

try
  ## try to get the device number from the command line
  device = int32 (argv(){1});
  have_device = true;
catch
  ## no device number on the command line, assume we want the 1st device
  device = -1;
end_try_catch

## no argument on the command line, try to use the camera
capture = highgui.cvCreateCameraCapture (device);

## set the wanted image size from the camera
highgui.cvSetCaptureProperty (capture, \
                              highgui.CV_CAP_PROP_FRAME_WIDTH, 320);
highgui.cvSetCaptureProperty (capture, \
                              highgui.CV_CAP_PROP_FRAME_HEIGHT, 240);

## create an image to put in the histogram
histimg = cv.cvCreateImage (cv.cvSize (320,240), 8, 3);

## init the image of the histogram to black
cv.cvSetZero (histimg);

## capture the 1st frame to get some propertie on it
frame = highgui.cvQueryFrame (capture);

## get some properties of the frame
frame_size = cv.cvGetSize (frame);

## compute which selection of the frame we want to monitor
selection = cv.cvRect (0, 0, frame.width, frame.height);

## create some images usefull later
hue = cv.cvCreateImage (frame_size, 8, 1);
mask = cv.cvCreateImage (frame_size, 8, 1);
hsv = cv.cvCreateImage (frame_size, 8, 3 );

## create the histogram
hist = cv.cvCreateHist ({hdims}, cv.CV_HIST_ARRAY, {hranges}, 1);

while (1)  ## do forever
  
  ## 1. capture the current image
  frame = highgui.cvQueryFrame (capture);
  if (swig_this(frame)==0);
    ## no image captured... end the processing
    break
  endif
  
  ## mirror the captured image
  cv.cvFlip (frame, [], 1);

  ## compute the hsv version of the image 
  cv.cvCvtColor (frame, hsv, cv.CV_BGR2HSV);

  ## compute which pixels are in the wanted range
  cv.cvInRangeS (hsv, hsv_min, hsv_max, mask);

  ## extract the hue from the hsv array
  cv.cvSplit (hsv, hue, [], [], []);

  ## select the rectangle of interest in the hue/mask arrays
  hue_roi = cv.cvGetSubRect (hue, selection);
  mask_roi = cv.cvGetSubRect (mask, selection);

  ## it's time to compute the histogram
  cv.cvCalcHist (hue_roi, hist, 0, mask_roi);

  ## extract the min and max value of the histogram
  [min_val, max_val, min_idx, max_idx] = cv.cvGetMinMaxHistValue (hist);

  ## compute the scale factor
  if (max_val > 0)
    scale = 255. / max_val;
  else
    scale = 0.;
  endif

  ## scale the histograms
  cv.cvConvertScale (hist.bins, hist.bins, scale, 0);

  ## clear the histogram image
  cv.cvSetZero (histimg);

  ## compute the width for each bin do display
  bin_w = histimg.width / hdims;
  
  for  (i=0:hdims-1)
    ## for all the bins

    ## get the value, and scale to the size of the hist image
    val = cv.cvRound (cv.cvGetReal1D (hist.bins, i)
		      * histimg.height / 255);

    ## compute the color
    color = hsv2rgb (i * 180. / hdims);

    ## draw the rectangle in the wanted color
    cv.cvRectangle (histimg,
		    cv.cvPoint (i * bin_w, histimg.height),
		    cv.cvPoint ((i + 1) * bin_w, histimg.height - val),
		    color, -1, 8, 0);

    ## we can now display the images
    highgui.cvShowImage ('Camera', frame);
    highgui.cvShowImage ('Histogram', histimg);
  endfor

  ## handle events
  k = highgui.cvWaitKey (5);

  if (k == 27)
    ## user has press the ESC key, so exit
    break;
  endif
endwhile
