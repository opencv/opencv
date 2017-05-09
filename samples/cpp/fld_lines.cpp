#include <iostream>

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
  std::string in;
  cv::CommandLineParser parser(argc, argv, "{@input|../data/building.jpg|input image}{help h||show help message}");
  if (parser.has("help"))
  {
    parser.printMessage();
    return 0;
  }
  in = parser.get<string>("@input");

  Mat image = imread(in, IMREAD_GRAYSCALE);

  if( image.empty() )
  {
    return -1;
  }
  // Because of some CPU's power strategy, it seems that the first running of an
  // algorithm took longer. So here we run both of the algorithmes 10 times to
  // see each algorithm's processing time with fully loaded CPU clock.
  for(int run_count = 0; run_count < 10; run_count++) {
    // Create LSD detector
    Ptr<LineSegmentDetector> lsd = createLineSegmentDetector();
    vector<Vec4f> lines_lsd;
    int64 start_lsd = getTickCount();
    lsd->detect(image, lines_lsd);
    // Detect the lines
    double freq = getTickFrequency();
    double duration_ms_lsd = double(getTickCount() - start_lsd) * 1000 / freq;
    std::cout << "Elapsed time for LSD: " << duration_ms_lsd << " ms." << std::endl;

    // Show found lines
    Mat line_image_lsd(image);
    lsd->drawSegments(line_image_lsd, lines_lsd);
    imshow("LSD result", line_image_lsd);

    // Create FLD detector
    Ptr<FastLineDetector> fld = createFastLineDetector();
    vector<Vec4f> lines_fld;
    int64 start = getTickCount();
    // Detect the lines
    fld->detect(image, lines_fld);
    double duration_ms = double(getTickCount() - start) * 1000 / freq;
    std::cout << "Ealpsed time for FLD " << duration_ms << " ms." << std::endl;

    // Show found lines
    Mat line_image_fld(image);
    fld->drawSegments(line_image_fld, lines_fld);
    imshow("FLD result", line_image_fld);
  }

  waitKey();
  return 0;
}
