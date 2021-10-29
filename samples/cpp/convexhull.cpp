#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

static void help(char** argv)
{
    cout << "\nThis sample program demonstrates the use of the convexHull() function\n"
         << "Call:\n"
         << argv[0] << endl;
}

int main( int argc, char** argv )
{
    CommandLineParser parser(argc, argv, "{help h||}");
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }
    Mat img(500, 500, CV_8UC3);
    RNG& rng = theRNG();

    for(;;)
    {
        int i, count = (unsigned)rng%100 + 1;

        vector<Point> points;

        for( i = 0; i < count; i++ )
        {
            Point pt;
            pt.x = rng.uniform(img.cols/4, img.cols*3/4);
            pt.y = rng.uniform(img.rows/4, img.rows*3/4);

            points.push_back(pt);
        }

        vector<Point> hull;
        convexHull(points, hull, true);

        img = Scalar::all(0);
        for( i = 0; i < count; i++ )
            circle(img, points[i], 3, Scalar(0, 0, 255), FILLED, LINE_AA);

        polylines(img, hull, true, Scalar(0, 255, 0), 1, LINE_AA);
        imshow("hull", img);

        char key = (char)waitKey();
        if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
            break;
    }

    return 0;
}
