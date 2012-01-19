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

}

TEST (Imgproc_BoundingRect, accuracy) { CV_BoundingRectTest test; test.safe_run(); }