// system includes
#include <getopt.h>
#include <iostream>

// library includes
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <opencv2/debug_mode.hpp>
#include <opencv2/show_image.hpp>
#include <opencv2/filter.hpp>
#include <opencv2/dmatch.hpp>
#include <opencv2/final_show.hpp>


template<class T> std::string toString(const T& p_arg)
{
  std::stringstream ss;

  ss << p_arg;

  return ss.str();
}


void
usage()
{
  printf("usage: cvvt [-r WxH]\n");
  printf("-h       print this help\n");
  printf("-r WxH   change resolution to width W and height H\n");
}


int
main(int argc, char** argv)
{
  cv::Size* resolution = nullptr;

  // parse options
  const char* optstring = "hr:";
  int opt;
  while ((opt = getopt(argc, argv, optstring)) != -1) {
    switch (opt) {
    case 'h':
      usage();
      return 0;
      break;
    case 'r':
      {
        char dummych;
        resolution = new cv::Size();
        if (sscanf(optarg, "%d%c%d", &resolution->width, &dummych, &resolution->height) != 3) {
          printf("%s not a valid resolution\n", optarg);
          return 1;
        }
      }
      break;
    default: /* '?' */
      usage();
      return 2;
    }
  }

  // setup video capture
  cv::VideoCapture capture(0);
  if (!capture.isOpened()) {
    std::cout << "Could not open VideoCapture" << std::endl;
    return 3;
  }

  if (resolution) {
    printf("Setting resolution to %dx%d\n", resolution->width, resolution->height);
    capture.set(CV_CAP_PROP_FRAME_WIDTH, resolution->width);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, resolution->height);
  }


  cv::Mat prevImgGray;
  std::vector<cv::KeyPoint> prevKeypoints;
  cv::Mat prevDescriptors;

  int maxFeatureCount = 500;
  cv::ORB detector(maxFeatureCount);

  cv::BFMatcher matcher(cv::NORM_HAMMING);

  for (int imgId = 0; imgId < 10; imgId++) {
    // capture a frame
    cv::Mat imgRead;
    capture >> imgRead;
    printf("%d: image captured\n", imgId);

    std::string imgIdString{"imgRead"};
    imgIdString += toString(imgId);
		cvv::showImage(imgRead, CVVISUAL_LOCATION, imgIdString.c_str());

    // convert to grayscale
    cv::Mat imgGray;
    cv::cvtColor(imgRead, imgGray, CV_BGR2GRAY);
		cvv::debugFilter(imgRead, imgGray, CVVISUAL_LOCATION, "to gray");

    // detect ORB features
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector(imgGray, cv::noArray(), keypoints, descriptors);
    printf("%d: detected %zd keypoints\n", imgId, keypoints.size());

    // match them to previous image (if available)
    if (!prevImgGray.empty()) {
      std::vector<cv::DMatch> matches;
      matcher.match(prevDescriptors, descriptors, matches);
      printf("%d: all matches size=%zd\n", imgId, matches.size());
      std::string allMatchIdString{"all matches "};
      allMatchIdString += toString(imgId-1) + "<->" + toString(imgId);
      cvv::debugDMatch(prevImgGray, prevKeypoints, imgGray, keypoints, matches, CVVISUAL_LOCATION, allMatchIdString.c_str());

      // remove worst (as defined by match distance) bestRatio quantile
      double bestRatio = 0.8;
      std::sort(matches.begin(), matches.end());
      matches.resize(int(bestRatio * matches.size()));
      printf("%d: best matches size=%zd\n", imgId, matches.size());
      std::string bestMatchIdString{"best " + toString(bestRatio) + " matches "};
      bestMatchIdString += toString(imgId-1) + "<->" + toString(imgId);
      cvv::debugDMatch(prevImgGray, prevKeypoints, imgGray, keypoints, matches, CVVISUAL_LOCATION, bestMatchIdString.c_str());
    }

    prevImgGray = imgGray;
    prevKeypoints = keypoints;
    prevDescriptors = descriptors;
  }

  cvv::finalShow();

  return 0;
}
