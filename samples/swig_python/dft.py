#!/usr/bin/python
from opencv.cv import *
from opencv.highgui import *
import sys

# Rearrange the quadrants of Fourier image so that the origin is at
# the image center
# src & dst arrays of equal size & type
def cvShiftDFT(src_arr, dst_arr ):

    size = cvGetSize(src_arr)
    dst_size = cvGetSize(dst_arr)

    if(dst_size.width != size.width or 
            dst_size.height != size.height) :
        cvError( CV_StsUnmatchedSizes, "cvShiftDFT", "Source and Destination arrays must have equal sizes", __FILE__, __LINE__ )    

    if(src_arr is dst_arr):
        tmp = cvCreateMat(size.height/2, size.width/2, cvGetElemType(src_arr))
    
    cx = size.width/2
    cy = size.height/2 # image center

    q1 = cvGetSubRect( src_arr, cvRect(0,0,cx, cy) )
    q2 = cvGetSubRect( src_arr, cvRect(cx,0,cx,cy) )
    q3 = cvGetSubRect( src_arr, cvRect(cx,cy,cx,cy) )
    q4 = cvGetSubRect( src_arr, cvRect(0,cy,cx,cy) )
    d1 = cvGetSubRect( src_arr, cvRect(0,0,cx,cy) )
    d2 = cvGetSubRect( src_arr, cvRect(cx,0,cx,cy) )
    d3 = cvGetSubRect( src_arr, cvRect(cx,cy,cx,cy) )
    d4 = cvGetSubRect( src_arr, cvRect(0,cy,cx,cy) )

    if(src_arr is not dst_arr):
        if( not CV_ARE_TYPES_EQ( q1, d1 )):
            cvError( CV_StsUnmatchedFormats, "cvShiftDFT", "Source and Destination arrays must have the same format", __FILE__, __LINE__ )    
        
        cvCopy(q3, d1)
        cvCopy(q4, d2)
        cvCopy(q1, d3)
        cvCopy(q2, d4)
    
    else:
        cvCopy(q3, tmp)
        cvCopy(q1, q3)
        cvCopy(tmp, q1)
        cvCopy(q4, tmp)
        cvCopy(q2, q4)
        cvCopy(tmp, q2)

if __name__ == "__main__":
    
    im = cvLoadImage( sys.argv[1], CV_LOAD_IMAGE_GRAYSCALE)

    realInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 1)
    imaginaryInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 1)
    complexInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 2)

    cvScale(im, realInput, 1.0, 0.0)
    cvZero(imaginaryInput)
    cvMerge(realInput, imaginaryInput, None, None, complexInput)

    dft_M = cvGetOptimalDFTSize( im.height - 1 )
    dft_N = cvGetOptimalDFTSize( im.width - 1 )

    dft_A = cvCreateMat( dft_M, dft_N, CV_64FC2 )
    image_Re = cvCreateImage( cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1)
    image_Im = cvCreateImage( cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1)

    # copy A to dft_A and pad dft_A with zeros
    tmp = cvGetSubRect( dft_A, cvRect(0,0, im.width, im.height))
    cvCopy( complexInput, tmp, None )
    if(dft_A.width > im.width):
        tmp = cvGetSubRect( dft_A, cvRect(im.width,0, dft_N - im.width, im.height))
        cvZero( tmp )

    # no need to pad bottom part of dft_A with zeros because of
    # use nonzero_rows parameter in cvDFT() call below

    cvDFT( dft_A, dft_A, CV_DXT_FORWARD, complexInput.height )

    cvNamedWindow("win", 0)
    cvNamedWindow("magnitude", 0)
    cvShowImage("win", im)

    # Split Fourier in real and imaginary parts
    cvSplit( dft_A, image_Re, image_Im, None, None )

    # Compute the magnitude of the spectrum Mag = sqrt(Re^2 + Im^2)
    cvPow( image_Re, image_Re, 2.0)
    cvPow( image_Im, image_Im, 2.0)
    cvAdd( image_Re, image_Im, image_Re, None)
    cvPow( image_Re, image_Re, 0.5 )

    # Compute log(1 + Mag)
    cvAddS( image_Re, cvScalarAll(1.0), image_Re, None ) # 1 + Mag
    cvLog( image_Re, image_Re ) # log(1 + Mag)


    # Rearrange the quadrants of Fourier image so that the origin is at
    # the image center
    cvShiftDFT( image_Re, image_Re )

    min, max, pt1, pt2 = cvMinMaxLoc(image_Re)
    cvScale(image_Re, image_Re, 1.0/(max-min), 1.0*(-min)/(max-min))
    cvShowImage("magnitude", image_Re)

    cvWaitKey(0)
