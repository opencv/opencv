/*
 * matching_test.cpp
 *
 *  Created on: Oct 17, 2010
 *      Author: ethan
 */
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <iostream>

using namespace cv;

using std::cout;
using std::cerr;
using std::endl;
using std::vector;

void help(char **av)
{
	   cerr << "usage: " << av[0] << " im1.jpg im2.jpg"
			   << "\n"
			   << "This program shows how to use BRIEF descriptor to match points in features2d\n"
			   << "It takes in two images, finds keypoints and matches them displaying matches and final homography warped results\n"
			   << endl;
}

//Copy (x,y) location of descriptor matches found from KeyPoint data structures into Point2f vectors
void matches2points(const vector<DMatch>& matches, const vector<KeyPoint>& kpts_train,
                    const vector<KeyPoint>& kpts_query, vector<Point2f>& pts_train, vector<Point2f>& pts_query)
{
  pts_train.clear();
  pts_query.clear();
  pts_train.reserve(matches.size());
  pts_query.reserve(matches.size());
  for (size_t i = 0; i < matches.size(); i++)
  {
    const DMatch& match = matches[i];
    pts_query.push_back(kpts_query[match.queryIdx].pt);
    pts_train.push_back(kpts_train[match.trainIdx].pt);
  }

}

double match(const vector<KeyPoint>& /*kpts_train*/, const vector<KeyPoint>& /*kpts_query*/, DescriptorMatcher& matcher,
            const Mat& train, const Mat& query, vector<DMatch>& matches)
{

  double t = (double)getTickCount();
  matcher.match(query, train, matches); //Using features2d
  return ((double)getTickCount() - t) / getTickFrequency();
}



int main(int ac, char ** av)
{
  if (ac != 3)
  {
	help(av);
    return 1;
  }
  string im1_name, im2_name;
  im1_name = av[1];
  im2_name = av[2];

  Mat im1 = imread(im1_name, CV_LOAD_IMAGE_GRAYSCALE);
  Mat im2 = imread(im2_name, CV_LOAD_IMAGE_GRAYSCALE);

  if (im1.empty() || im2.empty())
  {
    cerr << "could not open one of the images..." << endl;
    return 1;
  }

  double t = (double)getTickCount();

  FastFeatureDetector detector(50);
  BriefDescriptorExtractor extractor(32); //this is really 32 x 8 matches since they are binary matches packed into bytes

  vector<KeyPoint> kpts_1, kpts_2;
  detector.detect(im1, kpts_1);
  detector.detect(im2, kpts_2);

  t = ((double)getTickCount() - t) / getTickFrequency();

  cout << "found " << kpts_1.size() << " keypoints in " << im1_name << endl << "fount " << kpts_2.size()
      << " keypoints in " << im2_name << endl << "took " << t << " seconds." << endl;

  Mat desc_1, desc_2;

  cout << "computing descriptors..." << endl;

  t = (double)getTickCount();

  extractor.compute(im1, kpts_1, desc_1);
  extractor.compute(im2, kpts_2, desc_2);

  t = ((double)getTickCount() - t) / getTickFrequency();

  cout << "done computing descriptors... took " << t << " seconds" << endl;

  //Do matching with 2 methods using features2d
  cout << "matching with BruteForceMatcher<HammingLUT>" << endl;
  BruteForceMatcher<HammingLUT> matcher;
  vector<DMatch> matches_lut;
  float lut_time = (float)match(kpts_1, kpts_2, matcher, desc_1, desc_2, matches_lut);
  cout << "done BruteForceMatcher<HammingLUT> matching. took " << lut_time << " seconds" << endl;

  cout << "matching with BruteForceMatcher<Hamming>" << endl;
  BruteForceMatcher<Hamming> matcher_popcount;
  vector<DMatch> matches_popcount;
  double pop_time = match(kpts_1, kpts_2, matcher_popcount, desc_1, desc_2, matches_popcount);
  cout << "done BruteForceMatcher<Hamming> matching. took " << pop_time << " seconds" << endl;

  vector<Point2f> mpts_1, mpts_2;
  matches2points(matches_popcount, kpts_1, kpts_2, mpts_1, mpts_2); //Extract a list of the (x,y) location of the matches
  vector<uchar> outlier_mask;
  Mat H = findHomography(mpts_2, mpts_1, RANSAC, 1, outlier_mask);

  Mat outimg;
  drawMatches(im2, kpts_2, im1, kpts_1, matches_popcount, outimg, Scalar::all(-1), Scalar::all(-1),
              reinterpret_cast<const vector<char>&> (outlier_mask));
  imshow("matches - popcount - outliers removed", outimg);

  Mat warped;
  warpPerspective(im2, warped, H, im1.size());
  imshow("warped", warped);
  imshow("diff", im1 - warped);
  waitKey();
  return 0;
}
