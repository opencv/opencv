#! /usr/bin/env octave

printf("OpenCV Octave version of drawing\n");

## import the necessary things for OpenCV
cv;
highgui;

function ret=random_color ()
  ret = CV_RGB(int32(rand()*255), int32(rand()*255), int32(rand()*255));
endfunction


## some "constants"
width = 1000;
height = 700;
window_name = "Drawing Demo";
number = 100;
delay = 5;
line_type = cv.CV_AA;  # change it to 8 to see non-antialiased graphics

## create the source image
image = cv.cvCreateImage (cv.cvSize (width, height), 8, 3);

## create window and display the original picture in it
highgui.cvNamedWindow (window_name, 1);
cv.cvSetZero (image);
highgui.cvShowImage (window_name, image);

## draw some lines
for i=0:number-1,
  pt1 = cv.cvPoint (int32(rand() * 2 * width - width),
                    int32(rand() * 2 * height - height));
  pt2 = cv.cvPoint (int32(rand() * 2 * width - width),
                    int32(rand() * 2 * height - height));
  cv.cvLine (image, pt1, pt2,
             random_color (),
             int32(rand() * 10),
             line_type, 0);
  
  highgui.cvShowImage (window_name, image);
  highgui.cvWaitKey (delay);
endfor

## draw some rectangles
for i=0:number-1,
  pt1 = cv.cvPoint (int32(rand() * 2 * width - width),
                    int32(rand() * 2 * height - height));
  pt2 = cv.cvPoint (int32(rand() * 2 * width - width),
                    int32(rand() * 2 * height - height));
  cv.cvRectangle (image, pt1, pt2,
                  random_color (),
                  int32(rand() * 10 - 1),
                  line_type, 0);
  
  highgui.cvShowImage (window_name, image);
  highgui.cvWaitKey (delay);
endfor

## draw some ellipes
for i=0:number-1,
  pt1 = cv.cvPoint (int32(rand() * 2 * width - width),
                    int32(rand() * 2 * height - height));
  sz = cv.cvSize (int32(rand() * 200),
                  int32(rand() * 200));
  angle = rand() * 1000 * 0.180;
  cv.cvEllipse (image, pt1, sz, angle, angle - 100, angle + 200,
                random_color (),
                int32(rand() * 10 - 1),
                line_type, 0);
  
  highgui.cvShowImage (window_name, image);
  highgui.cvWaitKey (delay);
endfor

## init the list of polylines
nb_polylines = 2;
polylines_size = 3;
pt = cell(1, nb_polylines);
for a=1:nb_polylines,
  pt{a} = cell(1,polylines_size);
endfor

## draw some polylines
for i=0:number-1,
  for a=1:nb_polylines,
    for b=1:polylines_size,
      pt {a}{b} = cv.cvPoint (int32(rand() * 2 * width - width), \
                              int32(rand() * 2 * height - height));
    endfor
  endfor
  cv.cvPolyLine (image, pt, 1, random_color(), int32(rand() * 8 + 1), line_type, 0);

  highgui.cvShowImage (window_name, image);
  highgui.cvWaitKey (delay);
endfor

## draw some filled polylines
for i=0:number-1,
  for a=1:nb_polylines,
    for b=1:polylines_size,
      pt {a}{b} = cv.cvPoint (int32(rand() * 2 * width - width),
                              int32(rand() * 2 * height - height));
    endfor
  endfor
  cv.cvFillPoly (image, pt, random_color (), line_type, 0);

  highgui.cvShowImage (window_name, image);
  highgui.cvWaitKey (delay);
endfor

## draw some circles
for i=0:number-1,
  pt1 = cv.cvPoint (int32(rand() * 2 * width - width),
                    int32(rand() * 2 * height - height));
  cv.cvCircle (image, pt1, int32(rand() * 300), random_color (), \
	       int32(rand() * 10 - 1), line_type, 0);
  
  highgui.cvShowImage (window_name, image);
  highgui.cvWaitKey (delay);
endfor

## draw some text
for i=0:number-1,
  pt1 = cv.cvPoint (int32(rand() * 2 * width - width), \
                    int32(rand() * 2 * height - height));
  font = cv.cvInitFont (int32(rand() * 8), \
                        rand() * 100 * 0.05 + 0.01, \
                        rand() * 100 * 0.05 + 0.01, \
                        rand() * 5 * 0.1, \
                        int32(rand() * 10), \
                        line_type);

  cv.cvPutText (image, "Testing text rendering!", \
                pt1, font, \
                random_color ());
  
  highgui.cvShowImage (window_name, image);
  highgui.cvWaitKey (delay);
endfor

## prepare a text, and get it's properties
font = cv.cvInitFont (cv.CV_FONT_HERSHEY_COMPLEX, \
                      3, 3, 0.0, 5, line_type);
[text_size, ymin] = cv.cvGetTextSize ("OpenCV forever!", font);
pt1.x = int32((width - text_size.width) / 2);
pt1.y = int32((height + text_size.height) / 2);
image2 = cv.cvCloneImage(image);

## now, draw some OpenCV pub ;-)
for i=0:255-1,
  cv.cvSubS (image2, cv.cvScalarAll (i), image, []);
  cv.cvPutText (image, "OpenCV forever!",
                pt1, font, cv.cvScalar (255, i, i));
  highgui.cvShowImage (window_name, image);
  highgui.cvWaitKey (delay);
endfor

## wait some key to end
highgui.cvWaitKey (0);

