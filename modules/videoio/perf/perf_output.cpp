// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "perf_precomp.hpp"
#include "opencv2/videoio/registry.hpp"

namespace opencv_test
{
using namespace perf;

typedef tuple<std::string, bool> VideoWriter_Writing_t;
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
  Mat image = imread(filename, isColor ? IMREAD_COLOR : IMREAD_GRAYSCALE );
#if defined(HAVE_MSMF) && !defined(HAVE_FFMPEG)
  const string outfile = cv::tempfile(".wmv");
  const int fourcc = VideoWriter::fourcc('W', 'M', 'V', '3');
#else
  const string outfile = cv::tempfile(".avi");
  const int fourcc = VideoWriter::fourcc('X', 'V', 'I', 'D');
#endif

  VideoWriter writer(outfile, fourcc, 25, cv::Size(image.cols, image.rows), isColor);
  if (!writer.isOpened())
      throw SkipTestException("Video file can not be opened");

  TEST_CYCLE_N(100) { writer << image; }
  SANITY_CHECK_NOTHING();
  remove(outfile.c_str());
}

typedef tuple<Size, bool> VideoWriter_OpenCV_MJPEG_t;
typedef perf::TestBaseWithParam<VideoWriter_OpenCV_MJPEG_t> VideoWriter_OpenCV_MJPEG;

PERF_TEST_P(VideoWriter_OpenCV_MJPEG, WriteFrame,
            testing::Combine(
                testing::Values(szVGA, sz720p, sz1080p),
                testing::Bool()))
{
    if (!videoio_registry::hasBackend(CAP_OPENCV_MJPEG))
        throw SkipTestException("CAP_OPENCV_MJPEG is not available");

    const Size sz = get<0>(GetParam());
    const bool isColor = get<1>(GetParam());
    Mat image(sz, isColor ? CV_8UC3 : CV_8UC1);
    randu(image, 0, 256);

    const string outfile = cv::tempfile(".avi");
    const int fourcc = VideoWriter::fourcc('M', 'J', 'P', 'G');
    VideoWriter writer(outfile, CAP_OPENCV_MJPEG, fourcc, 25, sz, isColor);
    if (!writer.isOpened())
        throw SkipTestException("OpenCV Motion JPEG writer can not be opened");

    TEST_CYCLE() { writer << image; }
    SANITY_CHECK_NOTHING();
    remove(outfile.c_str());
}

} // namespace
