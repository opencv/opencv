#! /usr/bin/env octave

printf("OpenCV Octave version of lkdemo\n");

## import the necessary things for OpenCV
cv;
highgui;

#############################################################################
## some "constants"

win_size = 10;
MAX_COUNT = 500;

#############################################################################
## some "global" variables

global g;

g.image = [];
g.pt = [];
g.add_remove_pt = false;
g.flags = 0;
g.night_mode = false;
g.need_to_init = true;

g

#############################################################################
## the mouse callback

## the callback on the trackbar
function on_mouse (event, x, y, flags, param)
  global g;
  global cv;
  global highgui;

  if (swig_this(g.image) == 0)
    ## not initialized, so skip
    return;
  endif

  if (g.image.origin != 0)
    ## different origin
    y = g.image.height - y;
  endif

  if (event == highgui.CV_EVENT_LBUTTONDOWN)
    ## user has click, so memorize it
    pt = cv.cvPoint (x, y);
    g.add_remove_pt = true;
  endif
endfunction

#############################################################################
## so, here is the main part of the program


filename = "/home/x/work/sneaker/dvgrab-001.avi";
if (size(argv, 1)>1)
  filename=argv(){1};
endif

capture = highgui.cvCreateFileCapture (filename);

## check that capture device is OK
if (!swig_this(capture))
  printf("Error opening capture device\n");
  exit(1)
endif

## display a small howto use it
printf("Hot keys: \n");
printf("\tESC - quit the program\n");
printf("\tr - auto-initialize tracking\n");
printf("\tc - delete all the points\n");
printf("\tn - switch the \"night\" mode on/off\n");
printf("To add/remove a feature point click it\n");

## first, create the necessary windows
highgui.cvNamedWindow ('LkDemo', 1);

## register the mouse callback
highgui.cvSetMouseCallback ('LkDemo', @on_mouse, []);

while (1)
  ## do forever

  ## 1. capture the current image
  frame = highgui.cvQueryFrame (capture);
  if (swig_this(frame) == 0)
    ## no image captured... end the processing
    break
  endif

  if (swig_this(g.image) == 0),
    ## create the images we need
    g.image = cv.cvCreateImage (cv.cvGetSize (frame), 8, 3);
#    g.image.origin = frame.origin;
    g.grey = cv.cvCreateImage (cv.cvGetSize (frame), 8, 1);
    g.prev_grey = cv.cvCreateImage (cv.cvGetSize (frame), 8, 1);
    g.pyramid = cv.cvCreateImage (cv.cvGetSize (frame), 8, 1);
    g.prev_pyramid = cv.cvCreateImage (cv.cvGetSize (frame), 8, 1);
    g.points = {[], []};
  endif

  ## copy the frame, so we can draw on it
  cv.cvCopy (frame, g.image)

  ## create a grey version of the image
  cv.cvCvtColor (g.image, g.grey, cv.CV_BGR2GRAY)

  if (g.night_mode)
    ## night mode: only display the points
    cv.cvSetZero (g.image);
  endif

  if (g.need_to_init)
    ## we want to search all the good points

    ## create the wanted images
    eig = cv.cvCreateImage (cv.cvGetSize (g.grey), 32, 1);
    temp = cv.cvCreateImage (cv.cvGetSize (g.grey), 32, 1);

    ## the default parameters
    quality = 0.01;
    min_distance = 10;

    ## search the good points
    g.points {1} = cv.cvGoodFeaturesToTrack (g.grey, eig, temp,MAX_COUNT,quality, min_distance, [], 3, 0, 0.04);

    ## refine the corner locations
    cv.cvFindCornerSubPix (g.grey,g.points {1},cv.cvSize (win_size, win_size), cv.cvSize (-1, -1),cv.cvTermCriteria (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS,20, 0.03));
    
  elseif (size (g.points {1}, 2) > 0)
    ## we have points, so display them

    ## calculate the optical flow
    [tmp, status] = cv.cvCalcOpticalFlowPyrLK (g.prev_grey, g.grey, g.prev_pyramid, g.pyramid,g.points {1}, size (g.points {1},2),cv.cvSize (win_size, win_size), 3,size (g.points {1}, 2),[],cv.cvTermCriteria (bitor(cv.CV_TERMCRIT_ITER,cv.CV_TERMCRIT_EPS),20, 0.03),g.flags);
    g.points {2} = tmp;

    ## initializations
    point_counter = -1;
    new_points = {};

    for the_point = g.points {2},
      the_point = the_point{1};
      ## go trough all the points

      ## increment the counter
      point_counter += 1;

      if (g.add_remove_pt)
	## we have a point to add, so see if it is close to
	## another one. If yes, don't use it
        dx = pt.x - the_point.x;
        dy = pt.y - the_point.y;
        if (dx * dx + dy * dy <= 25)
	  ## too close
          g.add_remove_pt = 0;
          continue;
	endif
      endif

      if (!status {point_counter+1})
	## we will disable this point
        continue;
      endif

      ## this point is a correct point
      new_points{end+1} = the_point;

      ## draw the current point
      cv.cvCircle (g.image, {the_point.x, the_point.y},3, cv.cvScalar (0, 255, 0, 0),-1, 8, 0);
    endfor

    ## set back the points we keep;
    points {1} = new_points;
  endif

  if (g.add_remove_pt)
    ## we want to add a point
    points {1} = append (points {1}, cv.cvPointTo32f (pt));

    ## refine the corner locations
    g.points {1} = cv.cvFindCornerSubPix  \
    (g.grey, {points {1}}, cv.cvSize (win_size, win_size), cv.cvSize \
     (-1, -1), cv.cvTermCriteria (bitor(cv.CV_TERMCRIT_ITER, cv.CV_TERMCRIT_EPS),20, 0.03));

    ## we are no more in "add_remove_pt" mode
    g.add_remove_pt = false
  endif

  ## swapping
  tmp = g.prev_grey; g.prev_grey = g.grey; g.grey = tmp;
  tmp = g.prev_pyramid; g.prev_pyramid = g.pyramid; g.pyramid = tmp;
  tmp = g.points{1}; g.points{1} = g.points{2}; g.points{2} = tmp;
  g.need_to_init = false;
  
  ## we can now display the image
  highgui.cvShowImage ('LkDemo', g.image)

  ## handle events
  c = highgui.cvWaitKey (10);

  if (c == 27)
    ## user has press the ESC key, so exit
    break
  endif

  ## processing depending on the character
  if (c == int32('r'))
    g.need_to_init = true;
  elseif (c == int32('c'))
    g.points = {[], []};
  elseif (c == int32('n'))
    g.night_mode = !g.night_mode;
  endif
endwhile
