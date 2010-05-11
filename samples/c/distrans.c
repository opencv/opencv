#define CV_NO_BACKWARD_COMPATIBILITY

#ifdef _CH_
#pragma package <opencv>
#endif

#include "cv.h"
#include "highgui.h"
#include <stdio.h>

char wndname[] = "Distance transform";
char tbarname[] = "Threshold";
int mask_size = CV_DIST_MASK_5;
int build_voronoi = 0;
int edge_thresh = 100;
int dist_type = CV_DIST_L1;

// The output and temporary images
IplImage* dist = 0;
IplImage* dist8u1 = 0;
IplImage* dist8u2 = 0;
IplImage* dist8u = 0;
IplImage* dist32s = 0;

IplImage* gray = 0;
IplImage* edge = 0;
IplImage* labels = 0;

// threshold trackbar callback
void on_trackbar( int dummy )
{
    static const uchar colors[][3] =
    {
        {0,0,0},
        {255,0,0},
        {255,128,0},
        {255,255,0},
        {0,255,0},
        {0,128,255},
        {0,255,255},
        {0,0,255},
        {255,0,255}
    };

    int msize = mask_size;
    int _dist_type = build_voronoi ? CV_DIST_L2 : dist_type;

    cvThreshold( gray, edge, (float)edge_thresh, (float)edge_thresh, CV_THRESH_BINARY );

    if( build_voronoi )
        msize = CV_DIST_MASK_5;

    if( _dist_type == CV_DIST_L1 )
    {
        cvDistTransform( edge, edge, _dist_type, msize, NULL, NULL );
        cvConvert( edge, dist );
    }
    else
        cvDistTransform( edge, dist, _dist_type, msize, NULL, build_voronoi ? labels : NULL );

    if( !build_voronoi )
    {
        // begin "painting" the distance transform result
        cvConvertScale( dist, dist, 5000.0, 0 );
        cvPow( dist, dist, 0.5 );

        cvConvertScale( dist, dist32s, 1.0, 0.5 );
        cvAndS( dist32s, cvScalarAll(255), dist32s, 0 );
        cvConvertScale( dist32s, dist8u1, 1, 0 );
        cvConvertScale( dist32s, dist32s, -1, 0 );
        cvAddS( dist32s, cvScalarAll(255), dist32s, 0 );
        cvConvertScale( dist32s, dist8u2, 1, 0 );
        cvMerge( dist8u1, dist8u2, dist8u2, 0, dist8u );
        // end "painting" the distance transform result
    }
    else
    {
        int i, j;
        for( i = 0; i < labels->height; i++ )
        {
            int* ll = (int*)(labels->imageData + i*labels->widthStep);
            float* dd = (float*)(dist->imageData + i*dist->widthStep);
            uchar* d = (uchar*)(dist8u->imageData + i*dist8u->widthStep);
            for( j = 0; j < labels->width; j++ )
            {
                int idx = ll[j] == 0 || dd[j] == 0 ? 0 : (ll[j]-1)%8 + 1;
                int b = cvRound(colors[idx][0]);
                int g = cvRound(colors[idx][1]);
                int r = cvRound(colors[idx][2]);
                d[j*3] = (uchar)b;
                d[j*3+1] = (uchar)g;
                d[j*3+2] = (uchar)r;
            }
        }
    }

    cvShowImage( wndname, dist8u );
}

int main( int argc, char** argv )
{
    char* filename = argc == 2 ? argv[1] : (char*)"stuff.jpg";

    if( (gray = cvLoadImage( filename, 0 )) == 0 )
        return -1;

    printf( "Hot keys: \n"
        "\tESC - quit the program\n"
        "\tC - use C/Inf metric\n"
        "\tL1 - use L1 metric\n"
        "\tL2 - use L2 metric\n"
        "\t3 - use 3x3 mask\n"
        "\t5 - use 5x5 mask\n"
        "\t0 - use precise distance transform\n"
        "\tv - switch Voronoi diagram mode on/off\n"
        "\tSPACE - loop through all the modes\n" );

    dist = cvCreateImage( cvGetSize(gray), IPL_DEPTH_32F, 1 );
    dist8u1 = cvCloneImage( gray );
    dist8u2 = cvCloneImage( gray );
    dist8u = cvCreateImage( cvGetSize(gray), IPL_DEPTH_8U, 3 );
    dist32s = cvCreateImage( cvGetSize(gray), IPL_DEPTH_32S, 1 );
    edge = cvCloneImage( gray );
    labels = cvCreateImage( cvGetSize(gray), IPL_DEPTH_32S, 1 );

    cvNamedWindow( wndname, 1 );

    cvCreateTrackbar( tbarname, wndname, &edge_thresh, 255, on_trackbar );

    for(;;)
    {
        int c;

        // Call to update the view
        on_trackbar(0);

        c = cvWaitKey(0);

        if( (char)c == 27 )
            break;

        if( (char)c == 'c' || (char)c == 'C' )
            dist_type = CV_DIST_C;
        else if( (char)c == '1' )
            dist_type = CV_DIST_L1;
        else if( (char)c == '2' )
            dist_type = CV_DIST_L2;
        else if( (char)c == '3' )
            mask_size = CV_DIST_MASK_3;
        else if( (char)c == '5' )
            mask_size = CV_DIST_MASK_5;
        else if( (char)c == '0' )
            mask_size = CV_DIST_MASK_PRECISE;
        else if( (char)c == 'v' )
            build_voronoi ^= 1;
        else if( (char)c == ' ' )
        {
            if( build_voronoi )
            {
                build_voronoi = 0;
                mask_size = CV_DIST_MASK_3;
                dist_type = CV_DIST_C;
            }
            else if( dist_type == CV_DIST_C )
                dist_type = CV_DIST_L1;
            else if( dist_type == CV_DIST_L1 )
                dist_type = CV_DIST_L2;
            else if( mask_size == CV_DIST_MASK_3 )
                mask_size = CV_DIST_MASK_5;
            else if( mask_size == CV_DIST_MASK_5 )
                mask_size = CV_DIST_MASK_PRECISE;
            else if( mask_size == CV_DIST_MASK_PRECISE )
                build_voronoi = 1;
        }
    }

    cvReleaseImage( &gray );
    cvReleaseImage( &edge );
    cvReleaseImage( &dist );
    cvReleaseImage( &dist8u );
    cvReleaseImage( &dist8u1 );
    cvReleaseImage( &dist8u2 );
    cvReleaseImage( &dist32s );
    cvReleaseImage( &labels );

    cvDestroyWindow( wndname );

    return 0;
}
