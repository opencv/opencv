#include "perf_precomp.hpp"

#if BUILD_WITH_VIDEO_INPUT_SUPPORT

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;


typedef perf::TestBaseWithParam<std::string> VideoCapture_Reading;

#if defined(HAVE_MSMF)
// MPEG2 is not supported by Media Foundation yet
// http://social.msdn.microsoft.com/Forums/en-US/mediafoundationdevelopment/thread/39a36231-8c01-40af-9af5-3c105d684429
PERF_TEST_P(VideoCapture_Reading, ReadFile, testing::Values( "highgui/video/big_buck_bunny.avi",
                                               "highgui/video/big_buck_bunny.mov",
                                               "highgui/video/big_buck_bunny.mp4",
                                               "highgui/video/big_buck_bunny.wmv" ) )

#else
PERF_TEST_P(VideoCapture_Reading, ReadFile, testing::Values( "highgui/video/big_buck_bunny.avi",
                                               "highgui/video/big_buck_bunny.mov",
                                               "highgui/video/big_buck_bunny.mp4",
                                               "highgui/video/big_buck_bunny.mpg",
                                               "highgui/video/big_buck_bunny.wmv" ) )
#endif
{
  string filename = getDataPath(GetParam());

  VideoCapture cap;

  TEST_CYCLE() cap.open(filename);

  bool dummy = cap.isOpened();
  SANITY_CHECK(dummy);
}

#endif // BUILD_WITH_VIDEO_INPUT_SUPPORT
