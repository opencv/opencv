#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

typedef std::tr1::tuple<String, bool> VideoCapture_Reading_t;
typedef perf::TestBaseWithParam<VideoCapture_Reading_t> VideoCapture_Reading;

PERF_TEST_P(VideoCapture_Reading, ReadFile,
            testing::Combine( testing::Values( "highgui/video/big_buck_bunny.avi",
                                               "highgui/video/big_buck_bunny.mov",
                                               "highgui/video/big_buck_bunny.mp4",
                                               "highgui/video/big_buck_bunny.mpg",
                                               "highgui/video/big_buck_bunny.wmv" ),
                              testing::Values(true, true, true, true, true) ))
{
  string filename = getDataPath(get<0>(GetParam()));

  VideoCapture cap;

  TEST_CYCLE() cap.open(filename);

  SANITY_CHECK(cap.isOpened());
}
