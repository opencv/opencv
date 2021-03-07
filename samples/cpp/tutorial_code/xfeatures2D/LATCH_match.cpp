#include <iostream>

#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_XFEATURES2D

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>

// If you find this code useful, please add a reference to the following paper in your work:
// Gil Levi and Tal Hassner, "LATCH: Learned Arrangements of Three Patch Codes", arXiv preprint arXiv:1501.03719, 15 Jan. 2015

using namespace std;
using namespace cv;

const float inlier_threshold = 2.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio

int main(int argc, char* argv[])
{
    CommandLineParser parser(argc, argv,
                             "{@img1 | graf1.png | input image 1}"
                             "{@img2 | graf3.png | input image 2}"
                             "{@homography | H1to3p.xml | homography matrix}");
    Mat img1 = imread( samples::findFile( parser.get<String>("@img1") ), IMREAD_GRAYSCALE);
    Mat img2 = imread( samples::findFile( parser.get<String>("@img2") ), IMREAD_GRAYSCALE);

    Mat homography;
    FileStorage fs( samples::findFile( parser.get<String>("@homography") ), FileStorage::READ);
    fs.getFirstTopLevelNode() >> homography;

    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;

    Ptr<cv::ORB> orb_detector = cv::ORB::create(10000);

    Ptr<xfeatures2d::LATCH> latch = xfeatures2d::LATCH::create();


    orb_detector->detect(img1, kpts1);
    latch->compute(img1, kpts1, desc1);

    orb_detector->detect(img2, kpts2);
    latch->compute(img2, kpts2, desc2);

    BFMatcher matcher(NORM_HAMMING);
    vector< vector<DMatch> > nn_matches;
    matcher.knnMatch(desc1, desc2, nn_matches, 2);

    vector<KeyPoint> matched1, matched2, inliers1, inliers2;
    vector<DMatch> good_matches;
    for (size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;

        if (dist1 < nn_match_ratio * dist2) {
            matched1.push_back(kpts1[first.queryIdx]);
            matched2.push_back(kpts2[first.trainIdx]);
        }
    }

    for (unsigned i = 0; i < matched1.size(); i++) {
        Mat col = Mat::ones(3, 1, CV_64F);
        col.at<double>(0) = matched1[i].pt.x;
        col.at<double>(1) = matched1[i].pt.y;

        col = homography * col;
        col /= col.at<double>(2);
        double dist = sqrt(pow(col.at<double>(0) - matched2[i].pt.x, 2) +
            pow(col.at<double>(1) - matched2[i].pt.y, 2));

        if (dist < inlier_threshold) {
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(matched1[i]);
            inliers2.push_back(matched2[i]);
            good_matches.push_back(DMatch(new_i, new_i, 0));
        }
    }

    Mat res;
    drawMatches(img1, inliers1, img2, inliers2, good_matches, res);
    imwrite("latch_result.png", res);


    double inlier_ratio = inliers1.size() * 1.0 / matched1.size();
    cout << "LATCH Matching Results" << endl;
    cout << "*******************************" << endl;
    cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
    cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
    cout << "# Matches:                            \t" << matched1.size() << endl;
    cout << "# Inliers:                            \t" << inliers1.size() << endl;
    cout << "# Inliers Ratio:                      \t" << inlier_ratio << endl;
    cout << endl;

    imshow("result", res);
    waitKey();

    return 0;
}

#else

int main()
{
    std::cerr << "OpenCV was built without xfeatures2d module" << std::endl;
    return 0;
}

#endif
