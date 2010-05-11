#!/usr/bin/python
import cv
import sys
import urllib2

# Rearrange the quadrants of Fourier image so that the origin is at
# the image center
# src & dst arrays of equal size & type
def cvShiftDFT(src_arr, dst_arr ):

    size = cv.GetSize(src_arr)
    dst_size = cv.GetSize(dst_arr)

    if dst_size != size:
        cv.Error( cv.CV_StsUnmatchedSizes, "cv.ShiftDFT", "Source and Destination arrays must have equal sizes", __FILE__, __LINE__ )    

    if(src_arr is dst_arr):
        tmp = cv.CreateMat(size[1]/2, size[0]/2, cv.GetElemType(src_arr))
    
    cx = size[0] / 2
    cy = size[1] / 2 # image center

    q1 = cv.GetSubRect( src_arr, (0,0,cx, cy) )
    q2 = cv.GetSubRect( src_arr, (cx,0,cx,cy) )
    q3 = cv.GetSubRect( src_arr, (cx,cy,cx,cy) )
    q4 = cv.GetSubRect( src_arr, (0,cy,cx,cy) )
    d1 = cv.GetSubRect( src_arr, (0,0,cx,cy) )
    d2 = cv.GetSubRect( src_arr, (cx,0,cx,cy) )
    d3 = cv.GetSubRect( src_arr, (cx,cy,cx,cy) )
    d4 = cv.GetSubRect( src_arr, (0,cy,cx,cy) )

    if(src_arr is not dst_arr):
        if( not cv.CV_ARE_TYPES_EQ( q1, d1 )):
            cv.Error( cv.CV_StsUnmatchedFormats, "cv.ShiftDFT", "Source and Destination arrays must have the same format", __FILE__, __LINE__ )    
        
        cv.Copy(q3, d1)
        cv.Copy(q4, d2)
        cv.Copy(q1, d3)
        cv.Copy(q2, d4)
    
    else:
        cv.Copy(q3, tmp)
        cv.Copy(q1, q3)
        cv.Copy(tmp, q1)
        cv.Copy(q4, tmp)
        cv.Copy(q2, q4)
        cv.Copy(tmp, q2)

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        im = cv.LoadImage( sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        url = 'https://code.ros.org/svn/opencv/trunk/opencv/samples/c/baboon.jpg'
        filedata = urllib2.urlopen(url).read()
        imagefiledata = cv.CreateMatHeader(1, len(filedata), cv.CV_8UC1)
        cv.SetData(imagefiledata, filedata, len(filedata))
        im = cv.DecodeImageM(imagefiledata, cv.CV_LOAD_IMAGE_GRAYSCALE)

    realInput = cv.CreateImage( cv.GetSize(im), cv.IPL_DEPTH_64F, 1)
    imaginaryInput = cv.CreateImage( cv.GetSize(im), cv.IPL_DEPTH_64F, 1)
    complexInput = cv.CreateImage( cv.GetSize(im), cv.IPL_DEPTH_64F, 2)

    cv.Scale(im, realInput, 1.0, 0.0)
    cv.Zero(imaginaryInput)
    cv.Merge(realInput, imaginaryInput, None, None, complexInput)

    dft_M = cv.GetOptimalDFTSize( im.height - 1 )
    dft_N = cv.GetOptimalDFTSize( im.width - 1 )

    dft_A = cv.CreateMat( dft_M, dft_N, cv.CV_64FC2 )
    image_Re = cv.CreateImage( (dft_N, dft_M), cv.IPL_DEPTH_64F, 1)
    image_Im = cv.CreateImage( (dft_N, dft_M), cv.IPL_DEPTH_64F, 1)

    # copy A to dft_A and pad dft_A with zeros
    tmp = cv.GetSubRect( dft_A, (0,0, im.width, im.height))
    cv.Copy( complexInput, tmp, None )
    if(dft_A.width > im.width):
        tmp = cv.GetSubRect( dft_A, (im.width,0, dft_N - im.width, im.height))
        cv.Zero( tmp )

    # no need to pad bottom part of dft_A with zeros because of
    # use nonzero_rows parameter in cv.FT() call below

    cv.DFT( dft_A, dft_A, cv.CV_DXT_FORWARD, complexInput.height )

    cv.NamedWindow("win", 0)
    cv.NamedWindow("magnitude", 0)
    cv.ShowImage("win", im)

    # Split Fourier in real and imaginary parts
    cv.Split( dft_A, image_Re, image_Im, None, None )

    # Compute the magnitude of the spectrum Mag = sqrt(Re^2 + Im^2)
    cv.Pow( image_Re, image_Re, 2.0)
    cv.Pow( image_Im, image_Im, 2.0)
    cv.Add( image_Re, image_Im, image_Re, None)
    cv.Pow( image_Re, image_Re, 0.5 )

    # Compute log(1 + Mag)
    cv.AddS( image_Re, cv.ScalarAll(1.0), image_Re, None ) # 1 + Mag
    cv.Log( image_Re, image_Re ) # log(1 + Mag)


    # Rearrange the quadrants of Fourier image so that the origin is at
    # the image center
    cvShiftDFT( image_Re, image_Re )

    min, max, pt1, pt2 = cv.MinMaxLoc(image_Re)
    cv.Scale(image_Re, image_Re, 1.0/(max-min), 1.0*(-min)/(max-min))
    cv.ShowImage("magnitude", image_Re)

    cv.WaitKey(0)
