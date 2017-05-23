#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace cv;
using namespace std;

static void help()
{
    cout << "This program demonstrates finding the minimum enclosing box or circle of a set\n"
            "of points using functions: minAreaRect() minEnclosingCircle().\n"
            "Random points are generated and then enclosed.\n"
            "Call:\n"
            "./minarea\n"
            "Using OpenCV version %s\n" << CV_VERSION << "\n" << endl;
}

int main( int /*argc*/, char** /*argv*/ )
{
    help();

    Mat img(500, 500, CV_8UC3);
    RNG& rng = theRNG();

    for(;;)
    {
        int count = rng.uniform(3, 10);
        std::cout << "using " << count << " random points." << std::endl;
        vector<Point> points;
        for( int i = 0; i < count; i++ )
        {
            Point pt;
            pt.x = rng.uniform(img.cols/4, img.cols*3/4);
            pt.y = rng.uniform(img.rows/4, img.rows*3/4);

            points.push_back(pt);
        }

        RotatedRect box = minAreaRect(Mat(points));

        Point2f center, vtx[4];
        float radius = 0;
        minEnclosingCircle(Mat(points), center, radius);
        box.points(vtx);

        // compare radius and minimal distance
        double max_dist = 0;
        Point max_begin, max_end;
        for (int i = 0; i < count; ++i) {
            for (int j = i + 1; j < count; ++j) {
                double distance = norm(points[i] - points[j]);
                if (distance > max_dist) {
                    max_dist = distance;
                    max_begin = points[i];
                    max_end = points[j];
                }
            }
        }

        std::cout << "2*radius: " << 2*radius << ", max distance: " << max_dist << ", rel. diff.: " << 2*radius / max_dist << std::endl;

        img = Scalar::all(0);
        // min rectangle
        for(int i = 0; i < 4; i++ )
            line(img, vtx[i], vtx[(i+1)%4], Scalar(0, 255, 0), 1, CV_AA);

        // min circle
        circle(img, center, cvRound(radius), Scalar(0, 255, 255), 1, CV_AA);

        // points
        for(int i = 0; i < count; i++ )
            circle( img, points[i], 3, Scalar(0, 0, 255), CV_FILLED, CV_AA );

        // max dist
        line(img, max_begin, max_end, Scalar(255, 0, 255));

        imshow( "rect & circle", img );

        char key = (char)waitKey();
        if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
            break;
    }

    return 0;
}
