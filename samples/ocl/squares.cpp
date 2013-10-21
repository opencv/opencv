// The "Square Detector" program.
// It loads several images sequentially and tries to find squares in
// each image

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ocl/ocl.hpp"
#include <iostream>
#include <math.h>
#include <string.h>

using namespace cv;
using namespace std;

#define ACCURACY_CHECK 1

#if ACCURACY_CHECK
// check if two vectors of vector of points are near or not
// prior assumption is that they are in correct order
static bool checkPoints(
    vector< vector<Point> > set1,
    vector< vector<Point> > set2,
    int maxDiff = 5)
{
    if(set1.size() != set2.size())
    {
        return false;
    }

    for(vector< vector<Point> >::iterator it1 = set1.begin(), it2 = set2.begin();
            it1 < set1.end() && it2 < set2.end(); it1 ++, it2 ++)
    {
        vector<Point> pts1 = *it1;
        vector<Point> pts2 = *it2;


        if(pts1.size() != pts2.size())
        {
            return false;
        }
        for(size_t i = 0; i < pts1.size(); i ++)
        {
            Point pt1 = pts1[i], pt2 = pts2[i];
            if(std::abs(pt1.x - pt2.x) > maxDiff ||
                    std::abs(pt1.y - pt2.y) > maxDiff)
            {
                return false;
            }
        }
    }
    return true;
}
#endif

int thresh = 50, N = 11;
const char* wndname = "OpenCL Square Detection Demo";


// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}


// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
static void findSquares( const Mat& image, vector<vector<Point> >& squares )
{
    squares.clear();
    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    pyrUp(pyr, timg, image.size());
    vector<vector<Point> > contours;

    // find squares in every color plane of the image
    for( int c = 0; c < 3; c++ )
    {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                cv::threshold(gray0, gray, (l+1)*255/N, 255, THRESH_BINARY);
            }

            // find contours and store them all as a list
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            vector<Point> approx;

            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if( approx.size() == 4 &&
                        fabs(contourArea(Mat(approx))) > 1000 &&
                        isContourConvex(Mat(approx)) )
                {
                    double maxCosine = 0;

                    for( int j = 2; j < 5; j++ )
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if( maxCosine < 0.3 )
                        squares.push_back(approx);
                }
            }
        }
    }
}


// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
static void findSquares_ocl( const Mat& image, vector<vector<Point> >& squares )
{
    squares.clear();

    Mat gray;
    cv::ocl::oclMat pyr_ocl, timg_ocl, gray0_ocl, gray_ocl;

    // down-scale and upscale the image to filter out the noise
    ocl::pyrDown(ocl::oclMat(image), pyr_ocl);
    ocl::pyrUp(pyr_ocl, timg_ocl);

    vector<vector<Point> > contours;
    vector<cv::ocl::oclMat> gray0s;
    ocl::split(timg_ocl, gray0s); // split 3 channels into a vector of oclMat
    // find squares in every color plane of the image
    for( int c = 0; c < 3; c++ )
    {
        gray0_ocl = gray0s[c];
        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 )
            {
                // do canny on OpenCL device
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                cv::ocl::Canny(gray0_ocl, gray_ocl, 0, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                ocl::dilate(gray_ocl, gray_ocl, Mat(), Point(-1,-1));
                gray = Mat(gray_ocl);
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                cv::ocl::threshold(gray0_ocl, gray_ocl, (l+1)*255/N, 255, THRESH_BINARY);
                gray = gray_ocl;
            }

            // find contours and store them all as a list
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            vector<Point> approx;
            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if( approx.size() == 4 &&
                        fabs(contourArea(Mat(approx))) > 1000 &&
                        isContourConvex(Mat(approx)) )
                {
                    double maxCosine = 0;
                    for( int j = 2; j < 5; j++ )
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if( maxCosine < 0.3 )
                        squares.push_back(approx);
                }
            }
        }
    }
}


// the function draws all the squares in the image
static void drawSquares( Mat& image, const vector<vector<Point> >& squares )
{
    for( size_t i = 0; i < squares.size(); i++ )
    {
        const Point* p = &squares[i][0];
        int n = (int)squares[i].size();
        polylines(image, &p, &n, 1, true, Scalar(0,255,0), 3, LINE_AA);
    }
}


// draw both pure-C++ and ocl square results onto a single image
static Mat drawSquaresBoth( const Mat& image,
                            const vector<vector<Point> >& sqsCPP,
                            const vector<vector<Point> >& sqsOCL
)
{
    Mat imgToShow(Size(image.cols * 2, image.rows), image.type());
    Mat lImg = imgToShow(Rect(Point(0, 0), image.size()));
    Mat rImg = imgToShow(Rect(Point(image.cols, 0), image.size()));
    image.copyTo(lImg);
    image.copyTo(rImg);
    drawSquares(lImg, sqsCPP);
    drawSquares(rImg, sqsOCL);
    float fontScale = 0.8f;
    Scalar white = Scalar::all(255), black = Scalar::all(0);

    putText(lImg, "C++", Point(10, 20), FONT_HERSHEY_COMPLEX_SMALL, fontScale, black, 2);
    putText(rImg, "OCL", Point(10, 20), FONT_HERSHEY_COMPLEX_SMALL, fontScale, black, 2);
    putText(lImg, "C++", Point(10, 20), FONT_HERSHEY_COMPLEX_SMALL, fontScale, white, 1);
    putText(rImg, "OCL", Point(10, 20), FONT_HERSHEY_COMPLEX_SMALL, fontScale, white, 1);

    return imgToShow;
}


int main(int argc, char** argv)
{
    const char* keys =
        "{ i | input   |                    | specify input image }"
        "{ o | output  | squares_output.jpg | specify output save path}";
    CommandLineParser cmd(argc, argv, keys);
    string inputName = cmd.get<string>("i");
    string outfile = cmd.get<string>("o");
    if(inputName.empty())
    {
        cout << "Available options:" << endl;
        cmd.printMessage();
        return 0;
    }

    int iterations = 10;
    namedWindow( wndname, 1 );
    vector<vector<Point> > squares_cpu, squares_ocl;

    Mat image = imread(inputName, 1);
    if( image.empty() )
    {
        cout << "Couldn't load " << inputName << endl;
        return -1;
    }
    int j = iterations;
    int64 t_ocl = 0, t_cpp = 0;
    //warm-ups
    cout << "warming up ..." << endl;
    findSquares(image, squares_cpu);
    findSquares_ocl(image, squares_ocl);


#if ACCURACY_CHECK
    cout << "Checking ocl accuracy ... " << endl;
    cout << (checkPoints(squares_cpu, squares_ocl) ? "Pass" : "Failed") << endl;
#endif
    do
    {
        int64 t_start = cv::getTickCount();
        findSquares(image, squares_cpu);
        t_cpp += cv::getTickCount() - t_start;


        t_start  = cv::getTickCount();
        findSquares_ocl(image, squares_ocl);
        t_ocl += cv::getTickCount() - t_start;
        cout << "run loop: " << j << endl;
    }
    while(--j);
    cout << "cpp average time: " << 1000.0f * (double)t_cpp / getTickFrequency() / iterations << "ms" << endl;
    cout << "ocl average time: " << 1000.0f * (double)t_ocl / getTickFrequency() / iterations << "ms" << endl;

    Mat result = drawSquaresBoth(image, squares_cpu, squares_ocl);
    imshow(wndname, result);
    imwrite(outfile, result);
    waitKey(0);

    return 0;
}
