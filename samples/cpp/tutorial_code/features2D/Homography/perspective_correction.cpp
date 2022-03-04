#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

namespace
{
enum Pattern { CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };

Scalar randomColor( RNG& rng )
{
  int icolor = (unsigned int) rng;
  return Scalar( icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255 );
}

void perspectiveCorrection(const string &img1Path, const string &img2Path, const Size &patternSize, RNG &rng)
{
    Mat img1 = imread( samples::findFile(img1Path) );
    Mat img2 = imread( samples::findFile(img2Path) );

    //! [find-corners]
    vector<Point2f> corners1, corners2;
    bool found1 = findChessboardCorners(img1, patternSize, corners1);
    bool found2 = findChessboardCorners(img2, patternSize, corners2);
    //! [find-corners]

    if (!found1 || !found2)
    {
        cout << "Error, cannot find the chessboard corners in both images." << endl;
        return;
    }

    //! [estimate-homography]
    Mat H = findHomography(corners1, corners2);
    cout << "H:\n" << H << endl;
    //! [estimate-homography]

    //! [warp-chessboard]
    Mat img1_warp;
    warpPerspective(img1, img1_warp, H, img1.size());
    //! [warp-chessboard]

    Mat img_draw_warp;
    hconcat(img2, img1_warp, img_draw_warp);
    imshow("Desired chessboard view / Warped source chessboard view", img_draw_warp);

    //! [compute-transformed-corners]
    Mat img_draw_matches;
    hconcat(img1, img2, img_draw_matches);
    for (size_t i = 0; i < corners1.size(); i++)
    {
        Mat pt1 = (Mat_<double>(3,1) << corners1[i].x, corners1[i].y, 1);
        Mat pt2 = H * pt1;
        pt2 /= pt2.at<double>(2);

        Point end( (int) (img1.cols + pt2.at<double>(0)), (int) pt2.at<double>(1) );
        line(img_draw_matches, corners1[i], end, randomColor(rng), 2);
    }

    imshow("Draw matches", img_draw_matches);
    waitKey();
    //! [compute-transformed-corners]
}

const char* params
    = "{ help h         |       | print usage }"
      "{ image1         | left02.jpg | path to the source chessboard image }"
      "{ image2         | left01.jpg | path to the desired chessboard image }"
      "{ width bw       | 9     | chessboard width }"
      "{ height bh      | 6     | chessboard height }";
}

int main(int argc, char *argv[])
{
    cv::RNG rng( 0xFFFFFFFF );
    CommandLineParser parser(argc, argv, params);

    if (parser.has("help"))
    {
        parser.about("Code for homography tutorial.\n"
            "Example 2: perspective correction.\n");
        parser.printMessage();
        return 0;
    }

    Size patternSize(parser.get<int>("width"), parser.get<int>("height"));
    perspectiveCorrection(parser.get<String>("image1"),
                          parser.get<String>("image2"),
                          patternSize, rng);

    return 0;
}
