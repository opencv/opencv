/*
 * video_homography.cpp
 *
 *  Created on: Oct 18, 2010
 *      Author: erublee
 */

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <list>
#include <vector>

using namespace std;
using namespace cv;

namespace
{
void drawMatchesRelative(const vector<KeyPoint>& train, const vector<KeyPoint>& query,
                         std::vector<cv::DMatch>& matches, Mat& img, const vector<unsigned char>& mask = vector<
                             unsigned char> ())
{
  for (int i = 0; i < (int)matches.size(); i++)
  {
    if (mask.empty() || mask[i])
    {
      Point2f pt_new = query[matches[i].queryIdx].pt;
      Point2f pt_old = train[matches[i].trainIdx].pt;
      Point2f dist = pt_new - pt_old;

      cv::line(img, pt_new, pt_old, Scalar(125, 255, 125), 1);
      cv::circle(img, pt_new, 2, Scalar(255, 0, 125), 1);

    }
  }
}

void keypoints2points(const vector<KeyPoint>& in, vector<Point2f>& out)
{
  out.clear();
  out.reserve(in.size());
  for (size_t i = 0; i < in.size(); ++i)
  {
    out.push_back(in[i].pt);
  }
}

void points2keypoints(const vector<Point2f>& in, vector<KeyPoint>& out)
{
  out.clear();
  out.reserve(in.size());
  for (size_t i = 0; i < in.size(); ++i)
  {
    out.push_back(KeyPoint(in[i], 1));
  }
}

void warpKeypoints(const Mat& H, const vector<KeyPoint>& in, vector<KeyPoint>& out)
{
  vector<Point2f> pts;
  keypoints2points(in, pts);
  vector<Point2f> pts_w(pts.size());
  Mat m_pts_w(pts_w);
  perspectiveTransform(Mat(pts), m_pts_w, H);
  points2keypoints(pts_w, out);
}

void matches2points(const vector<KeyPoint>& train, const vector<KeyPoint>& query,
                    const std::vector<cv::DMatch>& matches, std::vector<cv::Point2f>& pts_train,
                    std::vector<Point2f>& pts_query)
{

  pts_train.clear();
  pts_query.clear();
  pts_train.reserve(matches.size());
  pts_query.reserve(matches.size());

  size_t i = 0;

  for (; i < matches.size(); i++)
  {

    const DMatch & dmatch = matches[i];

    pts_query.push_back(query[dmatch.queryIdx].pt);
    pts_train.push_back(train[dmatch.trainIdx].pt);

  }

}

void resetH(Mat&H)
{
  H = Mat::eye(3, 3, CV_32FC1);
}
}
int main(int ac, char ** av)
{

  if (ac != 2)
  {
    cout << "usage: " << av[0] << " <video device number>" << endl;
    return 1;
  }

  BriefDescriptorExtractor brief(32);

  VideoCapture capture;
  capture.open(atoi(av[1]));
  if (!capture.isOpened())
  {
    cout << "capture device " << atoi(av[1]) << " failed to open!" << endl;
    return 1;
  }

  cout << "following keys do stuff:" << endl;
  cout << "t : grabs a reference frame to match against" << endl;
  cout << "l : makes the reference frame new every frame" << endl;
  cout << "q or escape: quit" << endl;

  Mat frame;

  vector<DMatch> matches;

  BruteForceMatcher<Hamming> desc_matcher;

  vector<Point2f> train_pts, query_pts;
  vector<KeyPoint> train_kpts, query_kpts;
  vector<unsigned char> match_mask;

  Mat gray;

  bool ref_live = true;

  Mat train_desc, query_desc;
  const int DESIRED_FTRS = 500;
  GridAdaptedFeatureDetector detector(new FastFeatureDetector(10, true), DESIRED_FTRS, 4, 4);

  Mat H_prev = Mat::eye(3, 3, CV_32FC1);
  for (;;)
  {
    capture >> frame;
    if (frame.empty())
      continue;

    cvtColor(frame, gray, CV_RGB2GRAY);

    detector.detect(gray, query_kpts);

    brief.compute(gray, query_kpts, query_desc);

    if (!train_kpts.empty())
    {

      vector<KeyPoint> test_kpts;
      warpKeypoints(H_prev.inv(), query_kpts, test_kpts);

      Mat mask = windowedMatchingMask(test_kpts, train_kpts, 25, 25);
      desc_matcher.match(query_desc, train_desc, matches, mask);
      drawKeypoints(frame, test_kpts, frame, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_OVER_OUTIMG);

      matches2points(train_kpts, query_kpts, matches, train_pts, query_pts);

      if (matches.size() > 5)
      {
        Mat H = findHomography(Mat(train_pts), Mat(query_pts), match_mask, RANSAC, 4);
        if (countNonZero(Mat(match_mask)) > 15)
        {
          H_prev = H;
        }
        else
          resetH(H_prev);
        drawMatchesRelative(train_kpts, query_kpts, matches, frame, match_mask);
      }
      else
        resetH(H_prev);

    }
    else
    {
      H_prev = Mat::eye(3, 3, CV_32FC1);
      Mat out;
      drawKeypoints(gray, query_kpts, out);
      frame = out;
    }

    imshow("frame", frame);

    if (ref_live)
    {
      train_kpts = query_kpts;
      query_desc.copyTo(train_desc);
    }
    char key = waitKey(2);
    switch (key)
    {
      case 'l':
        ref_live = true;
        resetH(H_prev);
        break;
      case 't':
        ref_live = false;
        train_kpts = query_kpts;
        query_desc.copyTo(train_desc);
        resetH(H_prev);
        break;
      case 27:
      case 'q':
        return 0;
        break;
    }

  }
  return 0;
}
