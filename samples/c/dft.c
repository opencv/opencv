#define CV_NO_BACKWARD_COMPATIBILITY

#include <cxcore.h>
#include <cv.h>
#include <highgui.h>

// Rearrange the quadrants of Fourier image so that the origin is at
// the image center
// src & dst arrays of equal size & type
void cvShiftDFT(CvArr * src_arr, CvArr * dst_arr )
{
    CvMat * tmp=0;
    CvMat q1stub, q2stub;
    CvMat q3stub, q4stub;
    CvMat d1stub, d2stub;
    CvMat d3stub, d4stub;
    CvMat * q1, * q2, * q3, * q4;
    CvMat * d1, * d2, * d3, * d4;

    CvSize size = cvGetSize(src_arr);
    CvSize dst_size = cvGetSize(dst_arr);
    int cx, cy;

    if(dst_size.width != size.width ||
       dst_size.height != size.height){
        cvError( CV_StsUnmatchedSizes, "cvShiftDFT", "Source and Destination arrays must have equal sizes", __FILE__, __LINE__ );
    }

    if(src_arr==dst_arr){
        tmp = cvCreateMat(size.height/2, size.width/2, cvGetElemType(src_arr));
    }

    cx = size.width/2;
    cy = size.height/2; // image center

    q1 = cvGetSubRect( src_arr, &q1stub, cvRect(0,0,cx, cy) );
    q2 = cvGetSubRect( src_arr, &q2stub, cvRect(cx,0,cx,cy) );
    q3 = cvGetSubRect( src_arr, &q3stub, cvRect(cx,cy,cx,cy) );
    q4 = cvGetSubRect( src_arr, &q4stub, cvRect(0,cy,cx,cy) );
    d1 = cvGetSubRect( src_arr, &d1stub, cvRect(0,0,cx,cy) );
    d2 = cvGetSubRect( src_arr, &d2stub, cvRect(cx,0,cx,cy) );
    d3 = cvGetSubRect( src_arr, &d3stub, cvRect(cx,cy,cx,cy) );
    d4 = cvGetSubRect( src_arr, &d4stub, cvRect(0,cy,cx,cy) );

    if(src_arr!=dst_arr){
        if( !CV_ARE_TYPES_EQ( q1, d1 )){
            cvError( CV_StsUnmatchedFormats, "cvShiftDFT", "Source and Destination arrays must have the same format", __FILE__, __LINE__ );
        }
        cvCopy(q3, d1, 0);
        cvCopy(q4, d2, 0);
        cvCopy(q1, d3, 0);
        cvCopy(q2, d4, 0);
    }
    else{
        cvCopy(q3, tmp, 0);
        cvCopy(q1, q3, 0);
        cvCopy(tmp, q1, 0);
        cvCopy(q4, tmp, 0);
        cvCopy(q2, q4, 0);
        cvCopy(tmp, q2, 0);
    }
}

int main(int argc, char ** argv)
{
    const char* filename = argc >=2 ? argv[1] : "lena.jpg";
    IplImage * im;

    IplImage * realInput;
    IplImage * imaginaryInput;
    IplImage * complexInput;
    int dft_M, dft_N;
    CvMat* dft_A, tmp;
    IplImage * image_Re;
    IplImage * image_Im;
    double m, M;

    im = cvLoadImage( filename, CV_LOAD_IMAGE_GRAYSCALE );
    if( !im )
        return -1;

    realInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 1);
    imaginaryInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 1);
    complexInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 2);

    cvScale(im, realInput, 1.0, 0.0);
    cvZero(imaginaryInput);
    cvMerge(realInput, imaginaryInput, NULL, NULL, complexInput);

    dft_M = cvGetOptimalDFTSize( im->height - 1 );
    dft_N = cvGetOptimalDFTSize( im->width - 1 );

    dft_A = cvCreateMat( dft_M, dft_N, CV_64FC2 );
    image_Re = cvCreateImage( cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1);
    image_Im = cvCreateImage( cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1);

    // copy A to dft_A and pad dft_A with zeros
    cvGetSubRect( dft_A, &tmp, cvRect(0,0, im->width, im->height));
    cvCopy( complexInput, &tmp, NULL );
    if( dft_A->cols > im->width )
    {
        cvGetSubRect( dft_A, &tmp, cvRect(im->width,0, dft_A->cols - im->width, im->height));
        cvZero( &tmp );
    }

    // no need to pad bottom part of dft_A with zeros because of
    // use nonzero_rows parameter in cvDFT() call below

    cvDFT( dft_A, dft_A, CV_DXT_FORWARD, complexInput->height );

    cvNamedWindow("win", 0);
    cvNamedWindow("magnitude", 0);
    cvShowImage("win", im);

    // Split Fourier in real and imaginary parts
    cvSplit( dft_A, image_Re, image_Im, 0, 0 );

    // Compute the magnitude of the spectrum Mag = sqrt(Re^2 + Im^2)
    cvPow( image_Re, image_Re, 2.0);
    cvPow( image_Im, image_Im, 2.0);
    cvAdd( image_Re, image_Im, image_Re, NULL);
    cvPow( image_Re, image_Re, 0.5 );

    // Compute log(1 + Mag)
    cvAddS( image_Re, cvScalarAll(1.0), image_Re, NULL ); // 1 + Mag
    cvLog( image_Re, image_Re ); // log(1 + Mag)


    // Rearrange the quadrants of Fourier image so that the origin is at
    // the image center
    cvShiftDFT( image_Re, image_Re );

    cvMinMaxLoc(image_Re, &m, &M, NULL, NULL, NULL);
    cvScale(image_Re, image_Re, 1.0/(M-m), 1.0*(-m)/(M-m));
    cvShowImage("magnitude", image_Re);

    cvWaitKey(-1);
    return 0;
}
