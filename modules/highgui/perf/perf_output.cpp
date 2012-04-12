#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

typedef std::tr1::tuple<String, bool> VideoWriter_Writing_t;
typedef perf::TestBaseWithParam<VideoWriter_Writing_t> VideoWriter_Writing;

PERF_TEST_P(VideoWriter_Writing, WriteFrame,
            testing::Combine( testing::Values( "python/images/QCIF_00.bmp",
                                               "python/images/QCIF_01.bmp",
                                               "python/images/QCIF_02.bmp",
                                               "python/images/QCIF_03.bmp",
                                               "python/images/QCIF_04.bmp",
                                               "python/images/QCIF_05.bmp" ),
            testing::Bool()))
{
  string filename = getDataPath(get<0>(GetParam()));
  bool isColor = get<1>(GetParam());

  VideoWriter writer("perf_writer.avi", CV_FOURCC('X', 'V', 'I', 'D'), 25, cv::Size(640, 480), isColor);

  TEST_CYCLE() { Mat image = imread(filename, 1); writer << image; }

  SANITY_CHECK(writer.isOpened());
}
