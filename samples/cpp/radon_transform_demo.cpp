// Necessary headers
#include <math.h>
#include <stdlib.h>
#include <string>
#include <iostream>

#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

int main(int argc, char ** argv) {
  if(argc != 2) {
    cout<<"Usage: "<<argv[0]<<" </path/to/image>"<<endl;
    exit(1);
  }
  Mat img = imread(argv[1], 0), img2, rt;
  Mat img_clr = imread(argv[1]);

  img.convertTo(img2, CV_32F, 1.0/255);

  radonTransform(img2, rt);
  Mat rt_disp(rt.size(), CV_8UC1);
  for(int ii=0; ii<rt.rows; ++ii) {
    for(int jj=0; jj<rt.cols; ++jj) {
      rt_disp.at<uchar>(ii, jj) = static_cast<uchar>(255*rt.at<float>(ii, jj));
    }
  }
  imshow("img", img_clr);
  imshow("rt", rt_disp);

  waitKey(0);
  destroyAllWindows();
}
