
#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int thresh = 50, N = 11;
const char* wndname = "Square Detection Demo";

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
static void findSquares( const UMat& image, vector<vector<Point> >& squares )
{
    squares.clear();
    UMat pyr, timg, gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    pyrUp(pyr, timg, image.size());
    vector<vector<Point> > contours;

    // find squares in every color plane of the image
    for( int c = 0; c < 3; c++ )
    {
        int ch[] = {c, 0};
        mixChannels(timg, gray0, ch, 1);

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
                dilate(gray, gray, UMat(), Point(-1,-1));
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                threshold(gray0, gray, (l+1)*255/N, 255, THRESH_BINARY);
            }

            // find contours and store them all as a list
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            vector<Point> approx;

            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter

                approxPolyDP(contours[i], approx, arcLength(contours[i], true)*0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if( approx.size() == 4 &&
                        fabs(contourArea(approx)) > 1000 &&
                        isContourConvex(approx) )
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
static void drawSquares( UMat& _image, const vector<vector<Point> >& squares )
{
    Mat image = _image.getMat(ACCESS_WRITE);
    for( size_t i = 0; i < squares.size(); i++ )
    {
        const Point* p = &squares[i][0];
        int n = (int)squares[i].size();
        polylines(image, &p, &n, 1, true, Scalar(0,255,0), 3, LINE_AA);
    }
}


// draw both pure-C++ and ocl square results onto a single image
static UMat drawSquaresBoth( const UMat& image,
                            const vector<vector<Point> >& sqs)
{
    UMat imgToShow(Size(image.cols, image.rows), image.type());
    image.copyTo(imgToShow);

    drawSquares(imgToShow, sqs);

    return imgToShow;
}


int main(int argc, char** argv)
{
    const char* keys =
        "{ i input    | ../data/pic1.png   | specify input image }"
        "{ o output   | squares_output.jpg | specify output save path}"
        "{ h help     |                    | print help message }"
        "{ m cpu_mode |                    | run without OpenCL }";

    CommandLineParser cmd(argc, argv, keys);

    if(cmd.has("help"))
    {
        cout << "Usage : " << argv[0] << " [options]" << endl;
        cout << "Available options:" << endl;
        cmd.printMessage();
        return EXIT_SUCCESS;
    }
    if (cmd.has("cpu_mode"))
    {
        ocl::setUseOpenCL(false);
        cout << "OpenCL was disabled" << endl;
    }

    string inputName = cmd.get<string>("i");
    string outfile = cmd.get<string>("o");

    int iterations = 10;
    namedWindow( wndname, WINDOW_AUTOSIZE );
    vector<vector<Point> > squares;

    UMat image;
    imread(inputName, 1).copyTo(image);
    if( image.empty() )
    {
        cout << "Couldn't load " << inputName << endl;
        cmd.printMessage();
        return EXIT_FAILURE;
    }

    int j = iterations;
    int64 t_cpp = 0;
    //warm-ups
    cout << "warming up ..." << endl;
    findSquares(image, squares);

    do
    {
        int64 t_start = getTickCount();
        findSquares(image, squares);
        t_cpp += cv::getTickCount() - t_start;

        t_start  = getTickCount();

        cout << "run loop: " << j << endl;
    }
    while(--j);
    cout << "average time: " << 1000.0f * (double)t_cpp / getTickFrequency() / iterations << "ms" << endl;

    UMat result = drawSquaresBoth(image, squares);
    imshow(wndname, result);
    imwrite(outfile, result);
    waitKey(0);

    return EXIT_SUCCESS;
}
