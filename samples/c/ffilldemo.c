#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

IplImage* color_img0;
IplImage* mask;
IplImage* color_img;
IplImage* gray_img0 = NULL;
IplImage* gray_img = NULL;
int ffill_case = 1;
int lo_diff = 20, up_diff = 20;
int connectivity = 4;
int is_color = 1;
int is_mask = 0;
int new_mask_val = 255;

void on_mouse( int event, int x, int y, int flags, void* param )
{
    if( !color_img )
        return;

    switch( event )
    {
    case CV_EVENT_LBUTTONDOWN:
        {
            CvPoint seed = cvPoint(x,y);
            int lo = ffill_case == 0 ? 0 : lo_diff;
            int up = ffill_case == 0 ? 0 : up_diff;
            int flags = connectivity + (new_mask_val << 8) +
                        (ffill_case == 1 ? CV_FLOODFILL_FIXED_RANGE : 0);
            int b = rand() & 255, g = rand() & 255, r = rand() & 255;
            CvConnectedComp comp;

            if( is_mask )
                cvThreshold( mask, mask, 1, 128, CV_THRESH_BINARY );

            if( is_color )
            {
                CvScalar color = CV_RGB( r, g, b );
                cvFloodFill( color_img, seed, color, CV_RGB( lo, lo, lo ),
                             CV_RGB( up, up, up ), &comp, flags, is_mask ? mask : NULL );
                cvShowImage( "image", color_img );
            }
            else
            {
                CvScalar brightness = cvRealScalar((r*2 + g*7 + b + 5)/10);
                cvFloodFill( gray_img, seed, brightness, cvRealScalar(lo),
                             cvRealScalar(up), &comp, flags, is_mask ? mask : NULL );
                cvShowImage( "image", gray_img );
            }

            printf("%g pixels were repainted\n", comp.area );

            if( is_mask )
                cvShowImage( "mask", mask );
        }
        break;
    }
}

int main( int argc, char** argv )
{
    char* filename = argc >= 2 ? argv[1] : (char*)"fruits.jpg";

    if( (color_img0 = cvLoadImage(filename,1)) == 0 )
        return 0;

    printf( "Hot keys: \n"
            "\tESC - quit the program\n"
            "\tc - switch color/grayscale mode\n"
            "\tm - switch mask mode\n"
            "\tr - restore the original image\n"
            "\ts - use null-range floodfill\n"
            "\tf - use gradient floodfill with fixed(absolute) range\n"
            "\tg - use gradient floodfill with floating(relative) range\n"
            "\t4 - use 4-connectivity mode\n"
            "\t8 - use 8-connectivity mode\n" );

    color_img = cvCloneImage( color_img0 );
    gray_img0 = cvCreateImage( cvSize(color_img->width, color_img->height), 8, 1 );
    cvCvtColor( color_img, gray_img0, CV_BGR2GRAY );
    gray_img = cvCloneImage( gray_img0 );
    mask = cvCreateImage( cvSize(color_img->width + 2, color_img->height + 2), 8, 1 );

    cvNamedWindow( "image", 0 );
    cvCreateTrackbar( "lo_diff", "image", &lo_diff, 255, NULL );
    cvCreateTrackbar( "up_diff", "image", &up_diff, 255, NULL );

    cvSetMouseCallback( "image", on_mouse, 0 );

    for(;;)
    {
        int c;

        if( is_color )
            cvShowImage( "image", color_img );
        else
            cvShowImage( "image", gray_img );

        c = cvWaitKey(0);
        switch( (char) c )
        {
        case '\x1b':
            printf("Exiting ...\n");
            goto exit_main;
        case 'c':
            if( is_color )
            {
                printf("Grayscale mode is set\n");
                cvCvtColor( color_img, gray_img, CV_BGR2GRAY );
                is_color = 0;
            }
            else
            {
                printf("Color mode is set\n");
                cvCopy( color_img0, color_img, NULL );
                cvZero( mask );
                is_color = 1;
            }
            break;
        case 'm':
            if( is_mask )
            {
                cvDestroyWindow( "mask" );
                is_mask = 0;
            }
            else
            {
                cvNamedWindow( "mask", 0 );
                cvZero( mask );
                cvShowImage( "mask", mask );
                is_mask = 1;
            }
            break;
        case 'r':
            printf("Original image is restored\n");
            cvCopy( color_img0, color_img, NULL );
            cvCopy( gray_img0, gray_img, NULL );
            cvZero( mask );
            break;
        case 's':
            printf("Simple floodfill mode is set\n");
            ffill_case = 0;
            break;
        case 'f':
            printf("Fixed Range floodfill mode is set\n");
            ffill_case = 1;
            break;
        case 'g':
            printf("Gradient (floating range) floodfill mode is set\n");
            ffill_case = 2;
            break;
        case '4':
            printf("4-connectivity mode is set\n");
            connectivity = 4;
            break;
        case '8':
            printf("8-connectivity mode is set\n");
            connectivity = 8;
            break;
        }
    }

exit_main:

    cvDestroyWindow( "test" );
    cvReleaseImage( &gray_img );
    cvReleaseImage( &gray_img0 );
    cvReleaseImage( &color_img );
    cvReleaseImage( &color_img0 );
    cvReleaseImage( &mask );

    return 1;
}

#ifdef _EiC
main(1,"ffilldemo.c");
#endif
