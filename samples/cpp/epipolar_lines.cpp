// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <iostream>

using namespace cv;

int main(int args, char** argv) {
    std::string img_name1, img_name2;
    if (args < 3) {
       CV_Error(Error::StsBadArg,
                "Path to two images \nFor example: "
                "./epipolar_lines img1.jpg img2.jpg");
    } else {
       img_name1 = argv[1];
       img_name2 = argv[2];
    }

    Mat image1 = imread(img_name1);
    Mat image2 = imread(img_name2);
    Mat descriptors1, descriptors2;
    std::vector<KeyPoint> keypoints1, keypoints2;

    Ptr<SIFT> detector = SIFT::create();
    detector->detect(image1, keypoints1);
    detector->detect(image2, keypoints2);
    detector->compute(image1, keypoints1, descriptors1);
    detector->compute(image2, keypoints2, descriptors2);

    FlannBasedMatcher matcher(makePtr<flann::KDTreeIndexParams>(5), makePtr<flann::SearchParams>(32));

    // get k=2 best match that we can apply ratio test explained by D.Lowe
    std::vector<std::vector<DMatch>> matches_vector;
    matcher.knnMatch(descriptors1, descriptors2, matches_vector, 2);

    std::vector<Point2d> pts1, pts2;
    pts1.reserve(matches_vector.size()); pts2.reserve(matches_vector.size());
    for (const auto &m : matches_vector) {
        // compare best and second match using Lowe ratio test
        if (m[0].distance / m[1].distance < 0.75) {
            pts1.emplace_back(keypoints1[m[0].queryIdx].pt);
            pts2.emplace_back(keypoints2[m[0].trainIdx].pt);
        }
    }

    std::cout << "Number of points " << pts1.size() << '\n';

    Mat inliers;
    const auto begin_time = std::chrono::steady_clock::now();
    const Mat F = findFundamentalMat(pts1, pts2, RANSAC, 1., 0.99, 2000, inliers);
    std::cout << "RANSAC fundamental matrix time " << static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>
            (std::chrono::steady_clock::now() - begin_time).count()) << "\n";

    Mat points1 = Mat((int)pts1.size(), 2, CV_64F, pts1.data());
    Mat points2 = Mat((int)pts2.size(), 2, CV_64F, pts2.data());
    vconcat(points1.t(), Mat::ones(1, points1.rows, points1.type()), points1);
    vconcat(points2.t(), Mat::ones(1, points2.rows, points2.type()), points2);

    RNG rng;
    const int circle_sz = 3, line_sz = 1, max_lines = 300;
    std::vector<int> pts_shuffle (points1.cols);
    for (int i = 0; i < points1.cols; i++)
        pts_shuffle[i] = i;
    randShuffle(pts_shuffle);
    int plot_lines = 0, num_inliers = 0;
    double mean_err = 0;
    for (int pt : pts_shuffle) {
        if (inliers.at<uchar>(pt)) {
            const Scalar col (rng.uniform(0,256), rng.uniform(0,256), rng.uniform(0,256));
            const Mat l2 = F     * points1.col(pt);
            const Mat l1 = F.t() * points2.col(pt);
            double a1 = l1.at<double>(0), b1 = l1.at<double>(1), c1 = l1.at<double>(2);
            double a2 = l2.at<double>(0), b2 = l2.at<double>(1), c2 = l2.at<double>(2);
            const double mag1 = sqrt(a1*a1 + b1*b1), mag2 = (a2*a2 + b2*b2);
            a1 /= mag1; b1 /= mag1; c1 /= mag1; a2 /= mag2; b2 /= mag2; c2 /= mag2;
            if (plot_lines++ < max_lines) {
                line(image1, Point2d(0, -c1/b1),
                     Point2d((double)image1.cols, -(a1*image1.cols+c1)/b1), col, line_sz);
                line(image2, Point2d(0, -c2/b2),
                     Point2d((double)image2.cols, -(a2*image2.cols+c2)/b2), col, line_sz);
            }
            circle (image1, pts1[pt], circle_sz, col, -1);
            circle (image2, pts2[pt], circle_sz, col, -1);
            mean_err += (fabs(points1.col(pt).dot(l2)) / mag2 + fabs(points2.col(pt).dot(l1) / mag1)) / 2;
            num_inliers++;
        }
    }
    std::cout << "Mean distance from tentative inliers to epipolar lines " << mean_err/num_inliers
              << " number of inliers " << num_inliers << "\n";
    // concatenate two images
    hconcat(image1, image2, image1);
    const int new_img_size = 1200 * 800; // for example
    // resize with the same aspect ratio
    resize(image1, image1, Size((int) sqrt ((double) image1.cols * new_img_size / image1.rows),
                                (int)sqrt ((double) image1.rows * new_img_size / image1.cols)));

    // imshow("epipolar lines, image 1, 2", image1); // 注释掉图像显示
    imwrite("epipolar_lines.png", image1);
    printf("Result image saved as: epipolar_lines.png\n");
    // waitKey(0); // 注释掉等待按键

    return 0;
}

