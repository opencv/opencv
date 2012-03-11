#!/usr/bin/python
import sys
import random
import urllib2
import cv2.cv as cv

im=None;
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

    if event == cv.CV_EVENT_LBUTTONDOWN:
            my_mask = None
            seed = (x,y);
            if ffill_case==0:
                lo = up = 0
                flags = connectivity + (new_mask_val << 8)
            else:
                lo = lo_diff;
                up = up_diff;
                flags = connectivity + (new_mask_val << 8) + cv.CV_FLOODFILL_FIXED_RANGE
            b = random.randint(0,255)
            g = random.randint(0,255)
            r = random.randint(0,255)

            if( is_mask ):
                my_mask = mask
                cv.Threshold( mask, mask, 1, 128, cv.CV_THRESH_BINARY );
               
            if( is_color ):
            
                color = cv.CV_RGB( r, g, b );
                comp = cv.FloodFill( color_img, seed, color, cv.CV_RGB( lo, lo, lo ),
                             cv.CV_RGB( up, up, up ), flags, my_mask );
                cv.ShowImage( "image", color_img );
            
            else:
            
                brightness = cv.RealScalar((r*2 + g*7 + b + 5)/10);
                comp = cv.FloodFill( gray_img, seed, brightness, cv.RealScalar(lo),
                             cv.RealScalar(up), flags, my_mask );
                cv.ShowImage( "image", gray_img );
            

            print "%g pixels were repainted" % comp[0]

            if( is_mask ):
                cv.ShowImage( "mask", mask );
        
    


if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        im = cv.LoadImage( sys.argv[1], cv.CV_LOAD_IMAGE_COLOR)
    else:
        url = 'http://code.opencv.org/svn/opencv/trunk/opencv/samples/c/fruits.jpg'
        filedata = urllib2.urlopen(url).read()
        imagefiledata = cv.CreateMatHeader(1, len(filedata), cv.CV_8UC1)
        cv.SetData(imagefiledata, filedata, len(filedata))
        im = cv.DecodeImage(imagefiledata, cv.CV_LOAD_IMAGE_COLOR)

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
        
    color_img = cv.CloneImage( im );
    gray_img0 = cv.CreateImage( (color_img.width, color_img.height), 8, 1 );
    cv.CvtColor( color_img, gray_img0, cv.CV_BGR2GRAY );
    gray_img = cv.CloneImage( gray_img0 );
    mask = cv.CreateImage( (color_img.width + 2, color_img.height + 2), 8, 1 );

    cv.NamedWindow( "image", 1 );
    cv.CreateTrackbar( "lo_diff", "image", lo_diff, 255, update_lo);
    cv.CreateTrackbar( "up_diff", "image", up_diff, 255, update_up);

    cv.SetMouseCallback( "image", on_mouse );

    while True: 
        if( is_color ):
            cv.ShowImage( "image", color_img );
        else:
            cv.ShowImage( "image", gray_img );

        c = cv.WaitKey(0) % 0x100
        if c == 27:
            print("Exiting ...");
            sys.exit(0)
        elif c == ord('c'):
            if( is_color ):
            
                print("Grayscale mode is set");
                cv.CvtColor( color_img, gray_img, cv.CV_BGR2GRAY );
                is_color = 0;
            
            else:
            
                print("Color mode is set");
                cv.Copy( im, color_img, None );
                cv.Zero( mask );
                is_color = 1;
            
        elif c == ord('m'):
            if( is_mask ):
                cv.DestroyWindow( "mask" );
                is_mask = 0;
            
            else:
                cv.NamedWindow( "mask", 0 );
                cv.Zero( mask );
                cv.ShowImage( "mask", mask );
                is_mask = 1;
            
        elif c == ord('r'):
            print("Original image is restored");
            cv.Copy( im, color_img, None );
            cv.Copy( gray_img0, gray_img, None );
            cv.Zero( mask );
        elif c == ord('s'):
            print("Simple floodfill mode is set");
            ffill_case = 0;
        elif c == ord('f'):
            print("Fixed Range floodfill mode is set");
            ffill_case = 1;
        elif c == ord('g'):
            print("Gradient (floating range) floodfill mode is set");
            ffill_case = 2;
        elif c == ord('4'):
            print("4-connectivity mode is set");
            connectivity = 4;
        elif c == ord('8'):
            print("8-connectivity mode is set");
            connectivity = 8;
