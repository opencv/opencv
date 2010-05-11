#! /usr/bin/env octave
cv;
highgui;

global g;
inpaint_mask = [];
g.img0 = [];
g.img = [];
g.inpainted = [];
g.prev_pt = cvPoint(-1,-1);

function on_mouse( event, x, y, flags, param )
  global g;
  global cv;
  global highgui;
  if (!swig_this(g.img))
    return;
  endif

  if (event == highgui.CV_EVENT_LBUTTONUP || ! (bitand(flags,highgui.CV_EVENT_FLAG_LBUTTON)))
    g.prev_pt = cvPoint(-1,-1);
  elseif (event == highgui.CV_EVENT_LBUTTONDOWN)
    g.prev_pt = cvPoint(x,y);
  elseif (event == highgui.CV_EVENT_MOUSEMOVE && bitand(flags,highgui.CV_EVENT_FLAG_LBUTTON))
    pt = cvPoint(x,y);
    if (g.prev_pt.x < 0)
      g.prev_pt = pt;
    endif
    cvLine( g.inpaint_mask, g.prev_pt, pt, cvScalarAll(255), 5, 8, 0 );
    cvLine( g.img, g.prev_pt, pt, cvScalarAll(255), 5, 8, 0 );
    g.prev_pt = pt;
    cvShowImage( "image", g.img );
  endif
endfunction

filename = "../c/fruits.jpg";
if (size(argv, 1)>=1)
  filename = argv(){1};
endif

g.img0 = cvLoadImage(filename,-1);
if (!swig_this(g.img0))
  printf("Can't open image '%s'\n", filename);
  exit(1);
endif

printf("Hot keys:\n");
printf("\tESC - quit the program\n");
printf("\tr - restore the original image\n");
printf("\ti or ENTER - run inpainting algorithm\n");
printf("\t\t(before running it, paint something on the image)\n");

cvNamedWindow( "image", 1 );

g.img = cvCloneImage( g.img0 );
g.inpainted = cvCloneImage( g.img0 );
g.inpaint_mask = cvCreateImage( cvGetSize(g.img), 8, 1 );

cvZero( g.inpaint_mask );
cvZero( g.inpainted );
cvShowImage( "image", g.img );
cvShowImage( "watershed transform", g.inpainted );
cvSetMouseCallback( "image", @on_mouse, [] );

while (true)
  c = cvWaitKey(0);

  if( c == 27 || c=='q')
    break;
  endif

  if( c == 'r' )
    cvZero( g.inpaint_mask );
    cvCopy( g.img0, g.img );
    cvShowImage( "image", g.img );
  endif

  if( c == 'i' || c == '\012' )
    cvNamedWindow( "g.inpainted image", 1 );
    cvInpaint( g.img, g.inpaint_mask, g.inpainted, 3, CV_INPAINT_TELEA );
    cvShowImage( "g.inpainted image", g.inpainted );
  endif
endwhile

