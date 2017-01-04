#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using namespace testing;
using std::tr1::make_tuple;
using std::tr1::get;

typedef TestBaseWithParam< tr1::tuple<Size> > TestFill;

void fill(cv::Mat& img)
{
  for(int i = 0 ; i<100000 ; ++i)
  {
    cv::Size sz = img.size();
    cv::Rect rect(cv::Point(((int) theRNG())%sz.width, ((int) theRNG())%sz.height), cv::Size(((int) theRNG())%sz.width, ((int) theRNG())%sz.height));
    uchar colorComponent = theRNG();
    cv::Scalar color(colorComponent, colorComponent, colorComponent);
    cv::rectangle(img, rect, color, -1);
  }
}

PERF_TEST_P( TestFill, Fill,
                Values( Size(64, 64), Size(128, 128), Size(256, 256), Size(512, 512), Size(1024, 1024) )
)
{
    Size sz;
    sz         = get<0>(GetParam());

    Mat src(sz, CV_8UC3);
    Mat dst = src;
    declare.in(src, WARMUP_RNG).out(dst);
    TEST_CYCLE() fill(src);
    SANITY_CHECK(src);
}
