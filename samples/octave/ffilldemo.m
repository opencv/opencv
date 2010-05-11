#! /usr/bin/env octave
cv;
highgui;

global g;
g.color_img0 = [];
g.mask = [];
g.color_img = [];
g.gray_img0 = [];
g.gray_img = [];
g.ffill_case = 1;
g.lo_diff = 20
g.up_diff = 20;
g.connectivity = 4;
g.is_color = 1;
g.is_mask = 0;
g.new_mask_val = 255;

function ret = randint(v1, v2)
  ret = int32(rand() * (v2 - v1) + v1);
end

function update_lo( pos )
  g.lo_diff = pos;
endfunction
function update_up( pos )
  g.up_diff = pos;
endfunction

function on_mouse( event, x, y, flags, param )
  global g;
  global cv;
  global highgui;

  if( !swig_this(g.color_img) )
    return;
  endif

  if (event == highgui.CV_EVENT_LBUTTONDOWN)
    comp = cv.CvConnectedComp();
    my_mask = [];
    seed = cvPoint(x,y);
    if (g.ffill_case==0)
      lo = 0;
      up = 0;
      flags = g.connectivity + bitshift(g.new_mask_val,8);
    else
      lo = g.lo_diff;
      up = g.up_diff;
      flags = g.connectivity + bitshift(g.new_mask_val,8) + \
	  cv.CV_FLOODFILL_FIXED_RANGE;
    endif
    color = CV_RGB( randint(0,255), randint(0,255), randint(0,255) );    

    if( g.is_mask )
      my_mask = g.mask;
      cvThreshold( g.mask, g.mask, 1, 128, cv.CV_THRESH_BINARY );
    endif
    
    if( g.is_color )
      cv.cvFloodFill( g.color_img, seed, color, cv.CV_RGB( lo, lo, lo ),
                  CV_RGB( up, up, up ), comp, flags, my_mask );
      cvShowImage( "image", g.color_img );
      
    else
      
      brightness = cvRealScalar((r*2 + g*7 + b + 5)/10);
      cvFloodFill( g.gray_img, seed, brightness, cvRealScalar(lo),
                  cvRealScalar(up), comp, flags, my_mask );
      cvShowImage( "image", g.gray_img );
    endif
    

    printf("%i pixels were repainted\n", comp.area);

    if( g.is_mask )
      cvShowImage( "mask", g.mask );
    endif
  endif
endfunction




filename = "../c/fruits.jpg";
if (size(argv, 1)>0)
  filename=argv(){1};
endif

g.color_img0 = cvLoadImage(filename,1);
if (!swig_this(g.color_img0))
  printf("Could not open %s\n",filename);
  exit(-1);
endif

printf("Hot keys:\n");
printf("\tESC - quit the program\n");
printf("\tc - switch color/grayscale mode\n");
printf("\tm - switch mask mode\n");
printf("\tr - restore the original image\n");
printf("\ts - use null-range floodfill\n");
printf("\tf - use gradient floodfill with fixed(absolute) range\n");
printf("\tg - use gradient floodfill with floating(relative) range\n");
printf("\t4 - use 4-g.connectivity mode\n");
printf("\t8 - use 8-g.connectivity mode\n");

g.color_img = cvCloneImage( g.color_img0 );
g.gray_img0 = cvCreateImage( cvSize(g.color_img.width, g.color_img.height), 8, 1 );
cvCvtColor( g.color_img, g.gray_img0, CV_BGR2GRAY );
g.gray_img = cvCloneImage( g.gray_img0 );
g.mask = cvCreateImage( cvSize(g.color_img.width + 2, g.color_img.height + 2), 8, 1 );

cvNamedWindow( "image", 1 );
cvCreateTrackbar( "g.lo_diff", "image", g.lo_diff, 255, @update_lo);
cvCreateTrackbar( "g.up_diff", "image", g.up_diff, 255, @update_up);

cvSetMouseCallback( "image", @on_mouse );

while (true)
  if( g.is_color )
    cvShowImage( "image", g.color_img );
  else
    cvShowImage( "image", g.gray_img );
  endif

  c = cvWaitKey(0);
  if (c==27)
    printf("Exiting ...\n");
    exit(0)
  elseif (c=='c')
    if( g.is_color )
      
      print("Grayscale mode is set");
      cvCvtColor( g.color_img, g.gray_img, CV_BGR2GRAY );
      g.is_color = 0;
      
    else
      
      print("Color mode is set");
      cvCopy( g.color_img0, g.color_img, [] );
      cvZero( g.mask );
      g.is_color = 1;
    endif
    
  elseif (c=='m')
    if( g.is_mask )
      cvDestroyWindow( "mask" );
      g.is_mask = 0;
      
    else
      cvNamedWindow( "mask", 0 );
      cvZero( g.mask );
      cvShowImage( "mask", g.mask );
      g.is_mask = 1;
    endif
    
  elseif (c=='r')
    printf("Original image is restored");
    cvCopy( g.color_img0, g.color_img, [] );
    cvCopy( g.gray_img0, g.gray_img, [] );
    cvZero( g.mask );
  elseif (c=='s')
    printf("Simple floodfill mode is set");
    g.ffill_case = 0;
  elseif (c=='f')
    printf("Fixed Range floodfill mode is set");
    g.ffill_case = 1;
  elseif (c=='g')
    printf("Gradient (floating range) floodfill mode is set");
    g.ffill_case = 2;
  elseif (c=='4')
    printf("4-g.connectivity mode is set");
    g.connectivity = 4;
  elseif (c=='8')
    printf("8-g.connectivity mode is set");
    g.connectivity = 8;
  endif

endwhile
