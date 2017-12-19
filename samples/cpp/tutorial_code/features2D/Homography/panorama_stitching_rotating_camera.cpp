#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace
{
void basicPanoramaStitching(const string &img1Path, const string &img2Path)
{
    Mat img1 = imread(img1Path);
    Mat img2 = imread(img2Path);

    //! [camera-pose-from-Blender-at-location-1]
    Mat c1Mo = (Mat_<double>(4,4) << 0.9659258723258972, 0.2588190734386444, 0.0, 1.5529145002365112,
                                     0.08852133899927139, -0.3303661346435547, -0.9396926164627075, -0.10281121730804443,
                                     -0.24321036040782928, 0.9076734185218811, -0.342020183801651, 6.130080699920654,
                                     0, 0, 0, 1);
    //! [camera-pose-from-Blender-at-location-1]

    //! [camera-pose-from-Blender-at-location-2]
    Mat c2Mo = (Mat_<double>(4,4) << 0.9659258723258972, -0.2588190734386444, 0.0, -1.5529145002365112,
                                     -0.08852133899927139, -0.3303661346435547, -0.9396926164627075, -0.10281121730804443,
                                     0.24321036040782928, 0.9076734185218811, -0.342020183801651, 6.130080699920654,
                                     0, 0, 0, 1);
    //! [camera-pose-from-Blender-at-location-2]

    //! [camera-intrinsics-from-Blender]
    Mat cameraMatrix = (Mat_<double>(3,3) << 700.0, 0.0, 320.0,
                                             0.0, 700.0, 240.0,
                                             0, 0, 1);
    //! [camera-intrinsics-from-Blender]

    //! [extract-rotation]
    Mat R1 = c1Mo(Range(0,3), Range(0,3));
    Mat R2 = c2Mo(Range(0,3), Range(0,3));
    //! [extract-rotation]

    //! [compute-rotation-displacement]
    //c1Mo * oMc2
    Mat R_2to1 = R1*R2.t();
    //! [compute-rotation-displacement]

    //! [compute-homography]
    Mat H = cameraMatrix * R_2to1 * cameraMatrix.inv();
    H /= H.at<double>(2,2);
    cout << "H:\n" << H << endl;
    //! [compute-homography]

    //! [stitch]
    Mat img_stitch;
    warpPerspective(img2, img_stitch, H, Size(img2.cols*2, img2.rows));
    Mat half = img_stitch(Rect(0, 0, img1.cols, img1.rows));
    img1.copyTo(half);
    //! [stitch]

    Mat img_compare;
    Mat img_space = Mat::zeros(Size(50, img1.rows), CV_8UC3);
    hconcat(img1, img_space, img_compare);
    hconcat(img_compare, img2, img_compare);
    imshow("Compare images", img_compare);

    imshow("Panorama stitching", img_stitch);
    waitKey();
}

const char* params
    = "{ help h         |                              | print usage }"
      "{ image1         | ../data/Blender_Suzanne1.jpg | path to the first Blender image }"
      "{ image2         | ../data/Blender_Suzanne2.jpg | path to the second Blender image }";
}

int main(int argc, char *argv[])
{
    CommandLineParser parser(argc, argv, params);

    if (parser.has("help"))
    {
        parser.about( "Code for homography tutorial.\n"
                      "Example 5: basic panorama stitching from a rotating camera.\n" );
        parser.printMessage();
        return 0;
    }

    basicPanoramaStitching(parser.get<String>("image1"), parser.get<String>("image2"));

    return 0;
}
