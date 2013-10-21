#include "perf_precomp.hpp"

#if BUILD_WITH_VIDEO_OUTPUT_SUPPORT

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

typedef std::tr1::tuple<std::string, bool> VideoWriter_Writing_t;
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
  Mat image = imread(filename, 1);
#if defined(HAVE_MSMF) && !defined(HAVE_VFW) && !defined(HAVE_FFMPEG) // VFW has greater priority
  VideoWriter writer(cv::tempfile(".wmv"), VideoWriter::fourcc('W', 'M', 'V', '3'),
                            25, cv::Size(image.cols, image.rows), isColor);
#else
  VideoWriter writer(cv::tempfile(".avi"), VideoWriter::fourcc('X', 'V', 'I', 'D'),
                            25, cv::Size(image.cols, image.rows), isColor);
#endif

  TEST_CYCLE() { image = imread(filename, 1); writer << image; }

  bool dummy = writer.isOpened();
  SANITY_CHECK(dummy);
}

#endif // BUILD_WITH_VIDEO_OUTPUT_SUPPORT
