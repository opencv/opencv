#include "perf_precomp.hpp"

#ifdef HAVE_VIDEO_OUTPUT

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

typedef std::tr1::tuple<std::string, bool> VideoWriter_Writing_t;
typedef perf::TestBaseWithParam<VideoWriter_Writing_t> VideoWriter_Writing;

const string image_files[] = {
    "python/images/QCIF_00.bmp",
    "python/images/QCIF_01.bmp",
    "python/images/QCIF_02.bmp",
    "python/images/QCIF_03.bmp",
    "python/images/QCIF_04.bmp",
    "python/images/QCIF_05.bmp"
};

PERF_TEST_P(VideoWriter_Writing, WriteFrame,
            testing::Combine(
                testing::ValuesIn(image_files),
                testing::Bool()))
{
  const string filename = getDataPath(get<0>(GetParam()));
  const bool isColor = get<1>(GetParam());
  Mat image = imread(filename, 1);
#if defined(HAVE_MSMF) && !defined(HAVE_VFW) && !defined(HAVE_FFMPEG) // VFW has greater priority
  const string outfile = cv::tempfile(".wmv");
  const int fourcc = VideoWriter::fourcc('W', 'M', 'V', '3');
#else
  const string outfile = cv::tempfile(".avi");
  const int fourcc = VideoWriter::fourcc('X', 'V', 'I', 'D');
#endif

  VideoWriter writer(outfile, fourcc, 25, cv::Size(image.cols, image.rows), isColor);
  TEST_CYCLE_N(100) { writer << image; }
  SANITY_CHECK_NOTHING();
  remove(outfile.c_str());
}

#endif // HAVE_VIDEO_OUTPUT
