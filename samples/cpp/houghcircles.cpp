#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>

using namespace cv;
using namespace std;

static void help()
{
    cout << "\nThis program demonstrates circle finding with the Hough transform.\n"
            "Usage:\n"
            "./houghcircles <image_name>, Default is ../data/board.jpg\n" << endl;
}

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv,
        "{help h ||}{@image|../data/board.jpg|}"
    );
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    //![load]
    string filename = parser.get<string>("@image");
    Mat img = imread(filename, IMREAD_COLOR);
    if(img.empty())
    {
        help();
        cout << "can not open " << filename << endl;
        return -1;
    }
    //![load]

    //![convert_to_gray]
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    //![convert_to_gray]

    //![reduce_noise]
    medianBlur(gray, gray, 5);
    //![reduce_noise]

    //![houghcircles]
    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
                 gray.rows/16, // change this value to detect circles with different distances to each other
                 100, 30, 1, 30 // change the last two parameters
                                // (min_radius & max_radius) to detect larger circles
                 );
    //![houghcircles]

    //![draw]
    for( size_t i = 0; i < circles.size(); i++ )
    {
        Vec3i c = circles[i];
        circle( img, Point(c[0], c[1]), c[2], Scalar(0,0,255), 3, LINE_AA);
        circle( img, Point(c[0], c[1]), 2, Scalar(0,255,0), 3, LINE_AA);
    }
    //![draw]

    //![display]
    imshow("detected circles", img);
    waitKey();
    //![display]

    return 0;
}
