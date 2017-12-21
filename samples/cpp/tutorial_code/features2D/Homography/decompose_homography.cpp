#include <iostream>
#include <opencv2/opencv_modules.hpp>
#ifdef HAVE_OPENCV_ARUCO
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

using namespace std;
using namespace cv;

namespace
{
enum Pattern { CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };

void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f>& corners, Pattern patternType = CHESSBOARD)
{
    corners.resize(0);

    switch (patternType) {
    case CHESSBOARD:
    case CIRCLES_GRID:
        for( int i = 0; i < boardSize.height; i++ )
            for( int j = 0; j < boardSize.width; j++ )
                corners.push_back(Point3f(float(j*squareSize),
                                          float(i*squareSize), 0));
        break;

    case ASYMMETRIC_CIRCLES_GRID:
        for( int i = 0; i < boardSize.height; i++ )
            for( int j = 0; j < boardSize.width; j++ )
                corners.push_back(Point3f(float((2*j + i % 2)*squareSize),
                                          float(i*squareSize), 0));
        break;

    default:
        CV_Error(Error::StsBadArg, "Unknown pattern type\n");
    }
}

Mat computeHomography(const Mat &R_1to2, const Mat &tvec_1to2, const double d_inv, const Mat &normal)
{
    Mat homography = R_1to2 + d_inv * tvec_1to2*normal.t();
    return homography;
}

void computeC2MC1(const Mat &R1, const Mat &tvec1, const Mat &R2, const Mat &tvec2,
                  Mat &R_1to2, Mat &tvec_1to2)
{
    //c2Mc1 = c2Mo * oMc1 = c2Mo * c1Mo.inv()
    R_1to2 = R2 * R1.t();
    tvec_1to2 = R2 * (-R1.t()*tvec1) + tvec2;
}

void decomposeHomography(const string &img1Path, const string &img2Path, const Size &patternSize,
                         const float squareSize, const string &intrinsicsPath)
{
    Mat img1 = imread(img1Path);
    Mat img2 = imread(img2Path);

    vector<Point2f> corners1, corners2;
    bool found1 = findChessboardCorners(img1, patternSize, corners1);
    bool found2 = findChessboardCorners(img2, patternSize, corners2);

    if (!found1 || !found2)
    {
        cout << "Error, cannot find the chessboard corners in both images." << endl;
        return;
    }

    //! [compute-poses]
    vector<Point3f> objectPoints;
    calcChessboardCorners(patternSize, squareSize, objectPoints);

    FileStorage fs(intrinsicsPath, FileStorage::READ);
    Mat cameraMatrix, distCoeffs;
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;

    Mat rvec1, tvec1;
    solvePnP(objectPoints, corners1, cameraMatrix, distCoeffs, rvec1, tvec1);
    Mat rvec2, tvec2;
    solvePnP(objectPoints, corners2, cameraMatrix, distCoeffs, rvec2, tvec2);
    //! [compute-poses]

    //! [compute-camera-displacement]
    Mat R1, R2;
    Rodrigues(rvec1, R1);
    Rodrigues(rvec2, R2);

    Mat R_1to2, t_1to2;
    computeC2MC1(R1, tvec1, R2, tvec2, R_1to2, t_1to2);
    Mat rvec_1to2;
    Rodrigues(R_1to2, rvec_1to2);
    //! [compute-camera-displacement]

    //! [compute-plane-normal-at-camera-pose-1]
    Mat normal = (Mat_<double>(3,1) << 0, 0, 1);
    Mat normal1 = R1*normal;
    //! [compute-plane-normal-at-camera-pose-1]

    //! [compute-plane-distance-to-the-camera-frame-1]
    Mat origin(3, 1, CV_64F, Scalar(0));
    Mat origin1 = R1*origin + tvec1;
    double d_inv1 = 1.0 / normal1.dot(origin1);
    //! [compute-plane-distance-to-the-camera-frame-1]

    //! [compute-homography-from-camera-displacement]
    Mat homography_euclidean = computeHomography(R_1to2, t_1to2, d_inv1, normal1);
    Mat homography = cameraMatrix * homography_euclidean * cameraMatrix.inv();

    homography /= homography.at<double>(2,2);
    homography_euclidean /= homography_euclidean.at<double>(2,2);
    //! [compute-homography-from-camera-displacement]

    //! [decompose-homography-from-camera-displacement]
    vector<Mat> Rs_decomp, ts_decomp, normals_decomp;
    int solutions = decomposeHomographyMat(homography, cameraMatrix, Rs_decomp, ts_decomp, normals_decomp);
    cout << "Decompose homography matrix computed from the camera displacement:" << endl << endl;
    for (int i = 0; i < solutions; i++)
    {
      double factor_d1 = 1.0 / d_inv1;
      Mat rvec_decomp;
      Rodrigues(Rs_decomp[i], rvec_decomp);
      cout << "Solution " << i << ":" << endl;
      cout << "rvec from homography decomposition: " << rvec_decomp.t() << endl;
      cout << "rvec from camera displacement: " << rvec_1to2.t() << endl;
      cout << "tvec from homography decomposition: " << ts_decomp[i].t() << " and scaled by d: " << factor_d1 * ts_decomp[i].t() << endl;
      cout << "tvec from camera displacement: " << t_1to2.t() << endl;
      cout << "plane normal from homography decomposition: " << normals_decomp[i].t() << endl;
      cout << "plane normal at camera 1 pose: " << normal1.t() << endl << endl;
    }
    //! [decompose-homography-from-camera-displacement]

    //! [estimate homography]
    Mat H = findHomography(corners1, corners2);
    //! [estimate homography]

    //! [decompose-homography-estimated-by-findHomography]
    solutions = decomposeHomographyMat(H, cameraMatrix, Rs_decomp, ts_decomp, normals_decomp);
    cout << "Decompose homography matrix estimated by findHomography():" << endl << endl;
    for (int i = 0; i < solutions; i++)
    {
      double factor_d1 = 1.0 / d_inv1;
      Mat rvec_decomp;
      Rodrigues(Rs_decomp[i], rvec_decomp);
      cout << "Solution " << i << ":" << endl;
      cout << "rvec from homography decomposition: " << rvec_decomp.t() << endl;
      cout << "rvec from camera displacement: " << rvec_1to2.t() << endl;
      cout << "tvec from homography decomposition: " << ts_decomp[i].t() << " and scaled by d: " << factor_d1 * ts_decomp[i].t() << endl;
      cout << "tvec from camera displacement: " << t_1to2.t() << endl;
      cout << "plane normal from homography decomposition: " << normals_decomp[i].t() << endl;
      cout << "plane normal at camera 1 pose: " << normal1.t() << endl << endl;
    }
    //! [decompose-homography-estimated-by-findHomography]
}

const char* params
    = "{ help h         |       | print usage }"
      "{ image1         | ../data/left02.jpg | path to the source chessboard image }"
      "{ image2         | ../data/left01.jpg | path to the desired chessboard image }"
      "{ intrinsics     | ../data/left_intrinsics.yml | path to camera intrinsics }"
      "{ width bw       | 9     | chessboard width }"
      "{ height bh      | 6     | chessboard height }"
      "{ square_size    | 0.025 | chessboard square size }";
}

int main(int argc, char *argv[])
{
    CommandLineParser parser(argc, argv, params);

    if ( parser.has("help") )
    {
        parser.about( "Code for homography tutorial.\n"
                      "Example 4: decompose the homography matrix.\n" );
        parser.printMessage();
        return 0;
    }

    Size patternSize(parser.get<int>("width"), parser.get<int>("height"));
    float squareSize = (float) parser.get<double>("square_size");
    decomposeHomography(parser.get<String>("image1"),
                        parser.get<String>("image2"),
                        patternSize, squareSize,
                        parser.get<String>("intrinsics"));

    return 0;
}
#else
int main()
{
    std::cerr << "FATAL ERROR: This sample requires opencv_aruco module (from opencv_contrib)" << std::endl;
    return 0;
}
#endif
