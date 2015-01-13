#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <ctime>

using namespace std;
using namespace cv;

const float inlier_threshold = 2.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio
int main(void)
{
   Mat img1 = imread("../../data/graf1.png", IMREAD_GRAYSCALE);
   Mat img2 = imread("../../samples/data/graf3.png", IMREAD_GRAYSCALE);

   Mat homography;
   FileStorage fs("../../samples/data/H1to3p.xml", FileStorage::READ);

   fs.getFirstTopLevelNode() >> homography;

   vector<KeyPoint> kpts1, kpts2;
   Mat desc1, desc2;

   Ptr<xfeatures2d::SiftFeatureDetector> sift_detector = xfeatures2d::SiftFeatureDetector::create();
   Ptr<xfeatures2d::BriefDescriptorExtractor> brief = xfeatures2d::BriefDescriptorExtractor::create();

   sift_detector->detect(img1, kpts1);
   brief->compute(img1, kpts1, desc1);

   sift_detector->detect(img2, kpts2);
   brief->compute(img2, kpts2, desc2);

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
   imwrite("../../samples/data/brief_res.png", res);

   double inlier_ratio = inliers1.size() * 1.0 / matched1.size();
   cout << "BRIEF Matching Results" << endl;
   cout << "*******************************" << endl;
   cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
   cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
   cout << "# Matches:                            \t" << matched1.size() << endl;
   cout << "# Inliers:                            \t" << inliers1.size() << endl;
   cout << "# Inliers Ratio:                      \t" << inlier_ratio << endl;
   cout << endl;



   Ptr<xfeatures2d::RIBriefDescriptorExtractor> ri_brief = xfeatures2d::RIBriefDescriptorExtractor::create();

   kpts1.clear();
   kpts2.clear();

   Mat ri_desc1, ri_desc2;


   sift_detector->detect(img1, kpts1);
   ri_brief->compute(img1, kpts1, ri_desc1);

   sift_detector->detect(img2, kpts2);
   ri_brief->compute(img2, kpts2, ri_desc2);

   vector< vector<DMatch> > ri_nn_matches;
   matcher.knnMatch(ri_desc1, ri_desc2, ri_nn_matches, 2);

   vector<KeyPoint> ri_matched1, ri_matched2, ri_inliers1, ri_inliers2;
   vector<DMatch> ri_good_matches;
   for (size_t i = 0; i < ri_nn_matches.size(); i++) {
      DMatch first = ri_nn_matches[i][0];
      float dist1 = ri_nn_matches[i][0].distance;
      float dist2 = ri_nn_matches[i][1].distance;

      if (dist1 < nn_match_ratio * dist2) {
         ri_matched1.push_back(kpts1[first.queryIdx]);
         ri_matched2.push_back(kpts2[first.trainIdx]);
      }
   }

   for (unsigned i = 0; i < ri_matched1.size(); i++) {
      Mat col = Mat::ones(3, 1, CV_64F);
      col.at<double>(0) = ri_matched1[i].pt.x;
      col.at<double>(1) = ri_matched1[i].pt.y;

      col = homography * col;
      col /= col.at<double>(2);
      double dist = sqrt(pow(col.at<double>(0) - ri_matched2[i].pt.x, 2) +
         pow(col.at<double>(1) - ri_matched2[i].pt.y, 2));

      if (dist < inlier_threshold) {
         int new_i = static_cast<int>(ri_inliers1.size());
         ri_inliers1.push_back(ri_matched1[i]);
         ri_inliers2.push_back(ri_matched2[i]);
         ri_good_matches.push_back(DMatch(new_i, new_i, 0));
      }
   }

   Mat ri_res;
   drawMatches(img1, ri_inliers1, img2, ri_inliers2, ri_good_matches, ri_res);
   imwrite("../../samples/data/ri_brief_res.png", ri_res);

   double ri_inlier_ratio = ri_inliers1.size() * 1.0 / ri_matched1.size();
   cout << "RI_BRIEF Matching Results" << endl;
   cout << "*******************************" << endl;
   cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
   cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
   cout << "# Matches:                            \t" << ri_matched1.size() << endl;
   cout << "# Inliers:                            \t" << ri_inliers1.size() << endl;
   cout << "# Inliers Ratio:                      \t" << ri_inlier_ratio << endl;
   cout << endl;


   return 0;
}
