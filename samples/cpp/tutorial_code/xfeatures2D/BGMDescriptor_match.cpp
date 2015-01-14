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
   Mat img1 = imread("../../samples/data/graf1.png", IMREAD_GRAYSCALE);
   Mat img2 = imread("../../samples/data/graf3.png", IMREAD_GRAYSCALE);

   Mat homography;
   FileStorage fs("../../samples/data/H1to3p.xml", FileStorage::READ);

   fs.getFirstTopLevelNode() >> homography;

   
   //BGM
   vector<KeyPoint> kpts1, kpts2;
   Mat desc1, desc2;

   Ptr<xfeatures2d::SiftFeatureDetector> sift_detector = xfeatures2d::SiftFeatureDetector::create();
   
   xfeatures2d::BGMDescriptorExtractor BGMDescriptorExtractor_instance("../../../opencv_contrib/modules/xfeatures2d/src/bgm.bin");

   sift_detector->detect(img1, kpts1);
   BGMDescriptorExtractor_instance.compute(img1, kpts1, desc1);

   sift_detector->detect(img2, kpts2);
   BGMDescriptorExtractor_instance.compute(img2, kpts2, desc2);

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
   imwrite("../../samples/data/BGMDescriptorExtractor_res.png", res);

   double inlier_ratio = inliers1.size() * 1.0 / matched1.size();
   cout << "BGMDescriptorExtractor Matching Results" << endl;
   cout << "*******************************" << endl;
   cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
   cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
   cout << "# Matches:                            \t" << matched1.size() << endl;
   cout << "# Inliers:                            \t" << inliers1.size() << endl;
   cout << "# Inliers Ratio:                      \t" << inlier_ratio << endl;
   cout << endl;







   //LBGM
   xfeatures2d::LBGMDescriptorExtractor LBGMDescriptorExtractor_instance("../../../opencv_contrib/modules/xfeatures2d/src/lbgm.bin");

   kpts1.clear();
   kpts2.clear();

   Mat LBGM_desc1, LBGM_desc2;


   sift_detector->detect(img1, kpts1);
   LBGMDescriptorExtractor_instance.compute(img1, kpts1, LBGM_desc1);

   sift_detector->detect(img2, kpts2);
   LBGMDescriptorExtractor_instance.compute(img2, kpts2, LBGM_desc2);

   BFMatcher matcher_l2(NORM_L2);
   vector< vector<DMatch> > LBGM_nn_matches;
   matcher_l2.knnMatch(LBGM_desc1, LBGM_desc2, LBGM_nn_matches, 2);

   vector<KeyPoint> LBGM_matched1, LBGM_matched2, LBGM_inliers1, LBGM_inliers2;
   vector<DMatch> LBGM_good_matches;
   for (size_t i = 0; i < LBGM_nn_matches.size(); i++) {
	   DMatch first = LBGM_nn_matches[i][0];
	   float dist1 = LBGM_nn_matches[i][0].distance;
	   float dist2 = LBGM_nn_matches[i][1].distance;

      if (dist1 < nn_match_ratio * dist2) {
		  LBGM_matched1.push_back(kpts1[first.queryIdx]);
		  LBGM_matched2.push_back(kpts2[first.trainIdx]);
      }
   }

   for (unsigned i = 0; i < LBGM_matched1.size(); i++) {
      Mat col = Mat::ones(3, 1, CV_64F);
	  col.at<double>(0) = LBGM_matched1[i].pt.x;
	  col.at<double>(1) = LBGM_matched1[i].pt.y;

      col = homography * col;
      col /= col.at<double>(2);
	  double dist = sqrt(pow(col.at<double>(0) - LBGM_matched2[i].pt.x, 2) +
		  pow(col.at<double>(1) - LBGM_matched2[i].pt.y, 2));

      if (dist < inlier_threshold) {
		  int new_i = static_cast<int>(LBGM_inliers1.size());
		  LBGM_inliers1.push_back(LBGM_matched1[i]);
		  LBGM_inliers2.push_back(LBGM_matched2[i]);
		  LBGM_good_matches.push_back(DMatch(new_i, new_i, 0));
      }
   }

   Mat LBGM_res;
   drawMatches(img1, LBGM_inliers1, img2, LBGM_inliers2, LBGM_good_matches, LBGM_res);
   imwrite("../../samples/data/LBGMDescriptorExtractor_res.png", LBGM_res);

   double LBGM_inlier_ratio = LBGM_inliers1.size() * 1.0 / LBGM_matched1.size();
   cout << "LBGMDescriptorExtractor Matching Results" << endl;
   cout << "*******************************" << endl;
   cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
   cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
   cout << "# Matches:                            \t" << LBGM_matched1.size() << endl;
   cout << "# Inliers:                            \t" << LBGM_inliers1.size() << endl;
   cout << "# Inliers Ratio:                      \t" << LBGM_inlier_ratio << endl;
   cout << endl;












   //BinBoost
   xfeatures2d::BinBoostDescriptorExtractor BinBoostDescriptorExtractor_instance("../../../opencv_contrib/modules/xfeatures2d/src/binboost_256.bin");

   kpts1.clear();
   kpts2.clear();

   Mat BinBoost_desc1, BinBoost_desc2;


   sift_detector->detect(img1, kpts1);
   BinBoostDescriptorExtractor_instance.compute(img1, kpts1, BinBoost_desc1);

   sift_detector->detect(img2, kpts2);
   BinBoostDescriptorExtractor_instance.compute(img2, kpts2, BinBoost_desc2);

   vector< vector<DMatch> > BinBoost_nn_matches;
   matcher.knnMatch(BinBoost_desc1, BinBoost_desc2, BinBoost_nn_matches, 2);

   vector<KeyPoint> BinBoost_matched1, BinBoost_matched2, BinBoost_inliers1, BinBoost_inliers2;
   vector<DMatch> BinBoost_good_matches;
   for (size_t i = 0; i < BinBoost_nn_matches.size(); i++) {
	   DMatch first = BinBoost_nn_matches[i][0];
	   float dist1 = BinBoost_nn_matches[i][0].distance;
	   float dist2 = BinBoost_nn_matches[i][1].distance;

	   if (dist1 < nn_match_ratio * dist2) {
		   BinBoost_matched1.push_back(kpts1[first.queryIdx]);
		   BinBoost_matched2.push_back(kpts2[first.trainIdx]);
	   }
   }

   for (unsigned i = 0; i < BinBoost_matched1.size(); i++) {
	   Mat col = Mat::ones(3, 1, CV_64F);
	   col.at<double>(0) = BinBoost_matched1[i].pt.x;
	   col.at<double>(1) = BinBoost_matched1[i].pt.y;

	   col = homography * col;
	   col /= col.at<double>(2);
	   double dist = sqrt(pow(col.at<double>(0) - BinBoost_matched2[i].pt.x, 2) +
		   pow(col.at<double>(1) - BinBoost_matched2[i].pt.y, 2));

	   if (dist < inlier_threshold) {
		   int new_i = static_cast<int>(BinBoost_inliers1.size());
		   BinBoost_inliers1.push_back(BinBoost_matched1[i]);
		   BinBoost_inliers2.push_back(BinBoost_matched2[i]);
		   BinBoost_good_matches.push_back(DMatch(new_i, new_i, 0));
	   }
   }

   Mat BinBoost_res;
   drawMatches(img1, BinBoost_inliers1, img2, BinBoost_inliers2, BinBoost_good_matches, BinBoost_res);
   imwrite("../../samples/data/BinBoostDescriptorExtractor_res.png", BinBoost_res);

   double BinBoost_inlier_ratio = BinBoost_inliers1.size() * 1.0 / BinBoost_matched1.size();
   cout << "BinBoostDescriptorExtractor Matching Results" << endl;
   cout << "*******************************" << endl;
   cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
   cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
   cout << "# Matches:                            \t" << BinBoost_matched1.size() << endl;
   cout << "# Inliers:                            \t" << BinBoost_inliers1.size() << endl;
   cout << "# Inliers Ratio:                      \t" << BinBoost_inlier_ratio << endl;
   cout << endl;



   return 0;
}
