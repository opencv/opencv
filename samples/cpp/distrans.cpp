#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdio.h>

using namespace cv;

int maskSize0 = CV_DIST_MASK_5;
int voronoiType = -1;
int edgeThresh = 100;
int distType0 = CV_DIST_L1;

// The output and temporary images
Mat gray;

// threshold trackbar callback
static void onTrackbar( int, void* )
{
    static const Scalar colors[] =
    {
        Scalar(0,0,0),
        Scalar(255,0,0),
        Scalar(255,128,0),
        Scalar(255,255,0),
        Scalar(0,255,0),
        Scalar(0,128,255),
        Scalar(0,255,255),
        Scalar(0,0,255),
        Scalar(255,0,255)
    };

    int maskSize = voronoiType >= 0 ? CV_DIST_MASK_5 : maskSize0;
    int distType = voronoiType >= 0 ? CV_DIST_L2 : distType0;

    Mat edge = gray >= edgeThresh, dist, labels, dist8u;

    if( voronoiType < 0 )
        distanceTransform( edge, dist, distType, maskSize );
    else
        distanceTransform( edge, dist, labels, distType, maskSize, voronoiType );

    if( voronoiType < 0 )
    {
        // begin "painting" the distance transform result
        dist *= 5000;
        pow(dist, 0.5, dist);

        Mat dist32s, dist8u1, dist8u2;

        dist.convertTo(dist32s, CV_32S, 1, 0.5);
        dist32s &= Scalar::all(255);

        dist32s.convertTo(dist8u1, CV_8U, 1, 0);
        dist32s *= -1;

        dist32s += Scalar::all(255);
        dist32s.convertTo(dist8u2, CV_8U);

        Mat planes[] = {dist8u1, dist8u2, dist8u2};
        merge(planes, 3, dist8u);
    }
    else
    {
        dist8u.create(labels.size(), CV_8UC3);
        for( int i = 0; i < labels.rows; i++ )
        {
            const int* ll = (const int*)labels.ptr(i);
            const float* dd = (const float*)dist.ptr(i);
            uchar* d = (uchar*)dist8u.ptr(i);
            for( int j = 0; j < labels.cols; j++ )
            {
                int idx = ll[j] == 0 || dd[j] == 0 ? 0 : (ll[j]-1)%8 + 1;
                float scale = 1.f/(1 + dd[j]*dd[j]*0.0004f);
                int b = cvRound(colors[idx][0]*scale);
                int g = cvRound(colors[idx][1]*scale);
                int r = cvRound(colors[idx][2]*scale);
                d[j*3] = (uchar)b;
                d[j*3+1] = (uchar)g;
                d[j*3+2] = (uchar)r;
            }
        }
    }

    imshow("Distance Map", dist8u );
}

static void help()
{
    printf("\nProgram to demonstrate the use of the distance transform function between edge images.\n"
            "Usage:\n"
            "./distrans [image_name -- default image is stuff.jpg]\n"
            "\nHot keys: \n"
            "\tESC - quit the program\n"
            "\tC - use C/Inf metric\n"
            "\tL1 - use L1 metric\n"
            "\tL2 - use L2 metric\n"
            "\t3 - use 3x3 mask\n"
            "\t5 - use 5x5 mask\n"
            "\t0 - use precise distance transform\n"
            "\tv - switch to Voronoi diagram mode\n"
            "\tp - switch to pixel-based Voronoi diagram mode\n"
            "\tSPACE - loop through all the modes\n\n");
}

const char* keys =
{
    "{1| |stuff.jpg|input image file}"
};

int main( int argc, const char** argv )
{
    help();
    CommandLineParser parser(argc, argv, keys);
    string filename = parser.get<string>("1");
    gray = imread(filename.c_str(), 0);
    if(gray.empty())
    {
        printf("Cannot read image file: %s\n", filename.c_str());
        help();
        return -1;
    }

    namedWindow("Distance Map", 1);
    createTrackbar("Brightness Threshold", "Distance Map", &edgeThresh, 255, onTrackbar, 0);

    for(;;)
    {
        // Call to update the view
        onTrackbar(0, 0);

        int c = cvWaitKey(0) & 255;

        if( c == 27 )
            break;

        if( c == 'c' || c == 'C' || c == '1' || c == '2' ||
            c == '3' || c == '5' || c == '0' )
            voronoiType = -1;

        if( c == 'c' || c == 'C' )
            distType0 = CV_DIST_C;
        else if( c == '1' )
            distType0 = CV_DIST_L1;
        else if( c == '2' )
            distType0 = CV_DIST_L2;
        else if( c == '3' )
            maskSize0 = CV_DIST_MASK_3;
        else if( c == '5' )
            maskSize0 = CV_DIST_MASK_5;
        else if( c == '0' )
            maskSize0 = CV_DIST_MASK_PRECISE;
        else if( c == 'v' )
            voronoiType = 0;
        else if( c == 'p' )
            voronoiType = 1;
        else if( c == ' ' )
        {
            if( voronoiType == 0 )
                voronoiType = 1;
            else if( voronoiType == 1 )
            {
                voronoiType = -1;
                maskSize0 = CV_DIST_MASK_3;
                distType0 = CV_DIST_C;
            }
            else if( distType0 == CV_DIST_C )
                distType0 = CV_DIST_L1;
            else if( distType0 == CV_DIST_L1 )
                distType0 = CV_DIST_L2;
            else if( maskSize0 == CV_DIST_MASK_3 )
                maskSize0 = CV_DIST_MASK_5;
            else if( maskSize0 == CV_DIST_MASK_5 )
                maskSize0 = CV_DIST_MASK_PRECISE;
            else if( maskSize0 == CV_DIST_MASK_PRECISE )
                voronoiType = 0;
        }
    }

    return 0;
}
