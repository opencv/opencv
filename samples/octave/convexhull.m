#! /usr/bin/env octave

printf("OpenCV Octave version of convexhull\n");

## import the necessary things for OpenCV
cv;
highgui;

## how many points we want at max
_MAX_POINTS = 100;

## create the image where we want to display results
image = cv.cvCreateImage (cv.cvSize (500, 500), 8, 3);

## create the window to put the image in
highgui.cvNamedWindow ('hull', highgui.CV_WINDOW_AUTOSIZE);

while (true)
  ## do forever

  ## get a random number of points
  count = int32(rand()*_MAX_POINTS)+1

  ## initialisations
  points = {};
  
  for i=1:count,
    ## generate a random point
    points{i} = cv.cvPoint  \
    (int32(rand() * (image.width / 2) + image.width / 4), \
     int32(rand() * (image.height / 2) + image.height / 4)); \
  endfor

  ## compute the convex hull
  hull = cv.cvConvexHull2 (points, cv.CV_CLOCKWISE, 0);

  ## start with an empty image
  cv.cvSetZero (image);

  for i=1:count,
    ## draw all the points
    cv.cvCircle (image, points {i}, 2, \
                 cv.cvScalar (0, 0, 255, 0), \
                 cv.CV_FILLED, cv.CV_AA, 0);
  endfor

  ## start the line from the last point
  pt0 = points {hull [-1]};

  for point_index = 1:hull.rows,
    ## connect the previous point to the current one

    ## get the current one
    pt1 = points {point_index};

    ## draw
    cv.cvLine (image, pt0, pt1, \
               cv.cvScalar (0, 255, 0, 0), \
               1, cv.CV_AA, 0);

    ## now, current one will be the previous one for the next iteration
    pt0 = pt1;
  endfor

  ## display the final image
  highgui.cvShowImage ('hull', image);

  ## handle events, and wait a key pressed
  k = highgui.cvWaitKey (0);
  if (k == '\x1b')
    ## user has press the ESC key, so exit
    break
  endif
endwhile
