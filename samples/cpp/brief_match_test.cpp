/*
 * matching_test.cpp
 *
 *  Created on: Oct 17, 2010
 *      Author: ethan
 */
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

//Copy (x,y) location of descriptor matches found from KeyPoint data structures into Point2f vectors
static void matches2points(const vector<DMatch>& matches, const vector<KeyPoint>& kpts_train,
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

static double match(const vector<KeyPoint>& /*kpts_train*/, const vector<KeyPoint>& /*kpts_query*/, DescriptorMatcher& matcher,
            const Mat& train, const Mat& query, vector<DMatch>& matches)
{

  double t = (double)getTickCount();
  matcher.match(query, train, matches); //Using features2d
  return ((double)getTickCount() - t) / getTickFrequency();
}

static void help()
{
       cout << "This program shows how to use BRIEF descriptor to match points in features2d" << endl <<
               "It takes in two images, finds keypoints and matches them displaying matches and final homography warped results" << endl <<
                "Usage: " << endl <<
                    "image1 image2 " << endl <<
                "Example: " << endl <<
                    "box.png box_in_scene.png " << endl;
}

const char* keys =
{
    "{1|  |box.png               |the first image}"
    "{2|  |box_in_scene.png|the second image}"
};

int main(int argc, const char ** argv)
{

  help();
  CommandLineParser parser(argc, argv, keys);
  string im1_name = parser.get<string>("1");
  string im2_name = parser.get<string>("2");

  Mat im1 = imread(im1_name, CV_LOAD_IMAGE_GRAYSCALE);
  Mat im2 = imread(im2_name, CV_LOAD_IMAGE_GRAYSCALE);

  if (im1.empty() || im2.empty())
  {
    cout << "could not open one of the images..." << endl;
    cout << "the cmd parameters have next current value: " << endl;
    parser.printParams();
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

  //Do matching using features2d
  cout << "matching with BruteForceMatcher<Hamming>" << endl;
  BFMatcher matcher_popcount(NORM_HAMMING);
  vector<DMatch> matches_popcount;
  double pop_time = match(kpts_1, kpts_2, matcher_popcount, desc_1, desc_2, matches_popcount);
  cout << "done BruteForceMatcher<Hamming> matching. took " << pop_time << " seconds" << endl;

  vector<Point2f> mpts_1, mpts_2;
  matches2points(matches_popcount, kpts_1, kpts_2, mpts_1, mpts_2); //Extract a list of the (x,y) location of the matches
  vector<char> outlier_mask;
  Mat H = findHomography(mpts_2, mpts_1, RANSAC, 1, outlier_mask);

  Mat outimg;
  drawMatches(im2, kpts_2, im1, kpts_1, matches_popcount, outimg, Scalar::all(-1), Scalar::all(-1), outlier_mask);
  imshow("matches - popcount - outliers removed", outimg);

  Mat warped;
  Mat diff;
  warpPerspective(im2, warped, H, im1.size());
  imshow("warped", warped);
  absdiff(im1,warped,diff);
  imshow("diff", diff);
  waitKey();
  return 0;
}
