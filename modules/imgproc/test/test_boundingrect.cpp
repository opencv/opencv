#include "test_precomp.hpp"
#include <time.h>

using namespace cv;
using namespace std;

class CV_BoundingRectTest: public cvtest::ArrayTest
{
 public:
	CV_BoundingRectTest();
	~CV_BoundingRectTest();

 protected:
	void run (int);
	
 private:

};

CV_BoundingRectTest::CV_BoundingRectTest() {}
CV_BoundingRectTest::~CV_BoundingRectTest() {}

void CV_BoundingRectTest::run(int)
{
 const int MAX_WIDTH = 100;
 const int MAX_HEIGHT = 100;
 const int N = 100;

 RNG& rng = ts->get_rng();

 for (size_t i = 0; i < N; ++i)
 {
  int w = rng.next()%MAX_WIDTH + 1, h = rng.next()%MAX_HEIGHT + 1;
  cv::Mat src(h, w, CV_8U); cv::randu(src, Scalar_<bool>::all(false), Scalar_<bool>::all(true));
 }
}

TEST (Imgproc_BoundingRect, accuracy) { CV_BoundingRectTest test; test.safe_run(); }