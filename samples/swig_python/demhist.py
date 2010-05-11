#!/usr/bin/python
from opencv.cv import *
from opencv.highgui import *
import sys

file_name = "../c/baboon.jpg";

_brightness = 100
_contrast = 100
Gbrightness = 100
Gcontrast = 100

hist_size = 64
range_0=[0,256]
ranges = [ range_0 ]
src_image=None
dst_image=None
hist_image=None
hist=None
lut=cvCreateMat(256,1,CV_8U)

# brightness/contrast callback function
def update_brightness( val ):
    global Gbrightness    # global tag is required, or we get UnboundLocalError
    Gbrightness = val
    update_brightcont( )

def update_contrast( val ):
    global Gcontrast     # global tag is required, or we get UnboundLocalError
    Gcontrast = val
    update_brightcont( )

def update_brightcont():
    # no global tag required for images ???

    brightness = Gbrightness - 100;
    contrast = Gcontrast - 100;
    max_value = 0;

     # The algorithm is by Werner D. Streidt
     # (http://visca.com/ffactory/archives/5-99/msg00021.html)
    if( contrast > 0 ):
        delta = 127.*contrast/100;
        a = 255./(255. - delta*2);
        b = a*(brightness - delta);
    else:
        delta = -128.*contrast/100;
        a = (256.-delta*2)/255.;
        b = a*brightness + delta;

    for i in range(256):
        v = cvRound(a*i + b);
        if( v < 0 ):
            v = 0;
        if( v > 255 ):
            v = 255;
        lut[i] = v;
    
    cvLUT( src_image, dst_image, lut );
    cvShowImage( "image", dst_image );

    cvCalcHist( dst_image, hist, 0, None );
    cvZero( dst_image );
    min_value, max_value = cvGetMinMaxHistValue( hist );
    cvScale( hist.bins, hist.bins, float(hist_image.height)/max_value, 0 );
    #cvNormalizeHist( hist, 1000 );

    cvSet( hist_image, cvScalarAll(255));
    bin_w = cvRound(float(hist_image.width)/hist_size);

    for i in range(hist_size):
        cvRectangle( hist_image, cvPoint(i*bin_w, hist_image.height),
                     cvPoint((i+1)*bin_w, hist_image.height - cvRound(cvGetReal1D(hist.bins,i))),
                     cvScalarAll(0), -1, 8, 0 );
   
    cvShowImage( "histogram", hist_image );


if __name__ == "__main__":
    # Load the source image. HighGUI use.
    if len(sys.argv)>1:
        file_name = sys.argv[1]

    src_image = cvLoadImage( file_name, 0 );

    if not src_image:
        print "Image was not loaded.";
        sys.exit(-1)


    dst_image = cvCloneImage(src_image);
    hist_image = cvCreateImage(cvSize(320,200), 8, 1);
    hist = cvCreateHist([hist_size], CV_HIST_ARRAY, ranges, 1);

    cvNamedWindow("image", 0);
    cvNamedWindow("histogram", 0);

    cvCreateTrackbar("brightness", "image", _brightness, 200, update_brightness);
    cvCreateTrackbar("contrast", "image", _contrast, 200, update_contrast);

    update_brightcont();
    cvWaitKey(0);
