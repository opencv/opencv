#!/usr/bin/python
import sys
import random
from opencv.cv import *
from opencv.highgui import *

color_img0=None;
mask=None;
color_img=None;
gray_img0 = None;
gray_img = None;
ffill_case = 1;
lo_diff = 20
up_diff = 20;
connectivity = 4;
is_color = 1;
is_mask = 0;
new_mask_val = 255;

def update_lo( pos ):
    lo_diff = pos
def update_up( pos ):
    up_diff = pos

def on_mouse( event, x, y, flags, param ):

    if( not color_img ):
        return;

    if event==CV_EVENT_LBUTTONDOWN:
            comp = CvConnectedComp()
            my_mask = None
            seed = cvPoint(x,y);
            if ffill_case==0:
                lo = up = 0
                flags = connectivity + (new_mask_val << 8)
            else:
                lo = lo_diff;
                up = up_diff;
                flags = connectivity + (new_mask_val << 8) + CV_FLOODFILL_FIXED_RANGE
            b = random.randint(0,255)
            g = random.randint(0,255)
            r = random.randint(0,255)

            if( is_mask ):
                my_mask = mask
                cvThreshold( mask, mask, 1, 128, CV_THRESH_BINARY );
               
            if( is_color ):
            
                color = CV_RGB( r, g, b );
                cvFloodFill( color_img, seed, color, CV_RGB( lo, lo, lo ),
                             CV_RGB( up, up, up ), comp, flags, my_mask );
                cvShowImage( "image", color_img );
            
            else:
            
                brightness = cvRealScalar((r*2 + g*7 + b + 5)/10);
                cvFloodFill( gray_img, seed, brightness, cvRealScalar(lo),
                             cvRealScalar(up), comp, flags, my_mask );
                cvShowImage( "image", gray_img );
            

            print "%g pixels were repainted" % comp.area;

            if( is_mask ):
                cvShowImage( "mask", mask );
        
    


if __name__ == "__main__":
    
    filename = "../c/fruits.jpg"
    if len(sys.argv)>1:
        filename=argv[1]
    
    color_img0 = cvLoadImage(filename,1)
    if not color_img0:
        print "Could not open %s" % filename
        sys.exit(-1)

    print "Hot keys:"
    print "\tESC - quit the program"
    print "\tc - switch color/grayscale mode"
    print "\tm - switch mask mode"
    print "\tr - restore the original image"
    print "\ts - use null-range floodfill"
    print "\tf - use gradient floodfill with fixed(absolute) range"
    print "\tg - use gradient floodfill with floating(relative) range"
    print "\t4 - use 4-connectivity mode"
    print "\t8 - use 8-connectivity mode"
        
    color_img = cvCloneImage( color_img0 );
    gray_img0 = cvCreateImage( cvSize(color_img.width, color_img.height), 8, 1 );
    cvCvtColor( color_img, gray_img0, CV_BGR2GRAY );
    gray_img = cvCloneImage( gray_img0 );
    mask = cvCreateImage( cvSize(color_img.width + 2, color_img.height + 2), 8, 1 );

    cvNamedWindow( "image", 1 );
    cvCreateTrackbar( "lo_diff", "image", lo_diff, 255, update_lo);
    cvCreateTrackbar( "up_diff", "image", up_diff, 255, update_up);

    cvSetMouseCallback( "image", on_mouse );

    while True: 
        if( is_color ):
            cvShowImage( "image", color_img );
        else:
            cvShowImage( "image", gray_img );

        c = cvWaitKey(0);
        if c=='\x1b':
            print("Exiting ...");
            sys.exit(0)
        elif c=='c':
            if( is_color ):
            
                print("Grayscale mode is set");
                cvCvtColor( color_img, gray_img, CV_BGR2GRAY );
                is_color = 0;
            
            else:
            
                print("Color mode is set");
                cvCopy( color_img0, color_img, None );
                cvZero( mask );
                is_color = 1;
            
        elif c=='m':
            if( is_mask ):
                cvDestroyWindow( "mask" );
                is_mask = 0;
            
            else:
                cvNamedWindow( "mask", 0 );
                cvZero( mask );
                cvShowImage( "mask", mask );
                is_mask = 1;
            
        elif c=='r':
            print("Original image is restored");
            cvCopy( color_img0, color_img, None );
            cvCopy( gray_img0, gray_img, None );
            cvZero( mask );
        elif c=='s':
            print("Simple floodfill mode is set");
            ffill_case = 0;
        elif c=='f':
            print("Fixed Range floodfill mode is set");
            ffill_case = 1;
        elif c=='g':
            print("Gradient (floating range) floodfill mode is set");
            ffill_case = 2;
        elif c=='4':
            print("4-connectivity mode is set");
            connectivity = 4;
        elif c=='8':
            print("8-connectivity mode is set");
            connectivity = 8;
