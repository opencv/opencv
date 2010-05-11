#! /usr/bin/env octave

## import the necessary things for OpenCV
addpath("/home/x/opencv2/interfaces/swig/octave");
source("/home/x/opencv2/interfaces/swig/octave/PKG_ADD_template");
debug_on_error(true);
debug_on_warning(true);
crash_dumps_octave_core (0)
cv;
highgui;

## the codec existing in cvcapp.cpp,
## need to have a better way to specify them in the future
## WARNING: I have see only MPEG1VIDEO working on my computer
H263 = 0x33363255;
H263I = 0x33363249;
MSMPEG4V3 = 0x33564944;
MPEG4 = 0x58564944;
MSMPEG4V2 = 0x3234504D;
MJPEG = 0x47504A4D;
MPEG1VIDEO = 0x314D4950;
AC3 = 0x2000;
MP2 = 0x50;
FLV1 = 0x31564C46;

#############################################################################
## so, here is the main part of the program

## a small welcome
printf("OpenCV Octave capture video\n");

## first, create the necessary window
highgui.cvNamedWindow ('Camera', highgui.CV_WINDOW_AUTOSIZE);

## move the new window to a better place
#highgui.cvMoveWindow ('Camera', 10, 10);

try
  ## try to get the device number from the command line
  device = int32 (argv(){1});
catch
  ## no device number on the command line, assume we want the 1st device
  device = -1;
end_try_catch

## no argument on the command line, try to use the camera
capture = highgui.cvCreateCameraCapture (device);

## check that capture device is OK
if (!swig_this(capture))
  printf("Error opening capture device\n");
  exit (1);
endif

## capture the 1st frame to get some propertie on it
frame = highgui.cvQueryFrame (capture);

## get size of the frame
frame_size = cv.cvGetSize (frame);

## get the frame rate of the capture device
fps = highgui.cvGetCaptureProperty (capture, highgui.CV_CAP_PROP_FPS);
if (fps == 0)
  ## no fps getted, so set it to 30 by default
  fps = 30;
endif

## create the writer
writer = highgui.cvCreateVideoWriter ("captured.mpg", MPEG1VIDEO,
                                      fps, frame_size, true);

## check the writer is OK
if (!swig_this(writer))
  printf("Error opening writer\n");
  exit(1);
endif

while (1)
  ## do forever

  ## 1. capture the current image
  frame = highgui.cvQueryFrame (capture);
  if (swig_this(frame) == 0)
    ## no image captured... end the processing
    break
  endif

  ## write the frame to the output file
  highgui.cvWriteFrame (writer, frame);

  ## display the frames to have a visual output
  highgui.cvShowImage ('Camera', frame);

  ## handle events
  k = highgui.cvWaitKey (5);

  if (k & 0x100 == 27)
    ## user has press the ESC key, so exit
    break
  endif
endwhile


## end working with the writer
## not working at this time... Need to implement some typemaps...
## but exiting without calling it is OK in this simple application
##highgui.cvReleaseVideoWriter (writer)
