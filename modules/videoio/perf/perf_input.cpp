// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "perf_precomp.hpp"

#include "perf_camera.impl.hpp"

#include "opencv2/imgproc.hpp"
#include <fstream>
#include <sstream>
#include <thread>

namespace opencv_test
{
using namespace perf;

typedef perf::TestBaseWithParam<std::string> VideoCapture_Reading;

const string bunny_files[] = {
    "highgui/video/big_buck_bunny.avi",
    "highgui/video/big_buck_bunny.mov",
    "highgui/video/big_buck_bunny.mp4",
#ifndef HAVE_MSMF
    // MPEG2 is not supported by Media Foundation yet
    // http://social.msdn.microsoft.com/Forums/en-US/mediafoundationdevelopment/thread/39a36231-8c01-40af-9af5-3c105d684429
    "highgui/video/big_buck_bunny.mpg",
#endif
    "highgui/video/big_buck_bunny.wmv"
};

PERF_TEST_P(VideoCapture_Reading, ReadFile, testing::ValuesIn(bunny_files) )
{
  string filename = getDataPath(GetParam());

  VideoCapture cap;

  TEST_CYCLE() cap.open(filename);

  SANITY_CHECK_NOTHING();
}

class MemoryStream : public IStreamReader
{
public:
    MemoryStream(const std::string& data) : buf(data) {}

    long long read(char* buffer, long long size) CV_OVERRIDE
    {
        return buf.sgetn(buffer, size);
    }

    long long seek(long long offset, int way) CV_OVERRIDE
    {
        return buf.pubseekoff(offset, way == SEEK_SET ? std::ios_base::beg : (way == SEEK_END ? std::ios_base::end : std::ios_base::cur));
    }

private:
    std::stringbuf buf;
};

typedef perf::TestBaseWithParam<tuple<int, std::string>> VideoCapture_Prefetch;

PERF_TEST_P(VideoCapture_Prefetch, ReadStream,
            testing::Combine(testing::Values(0, 2, 4), testing::Values("none", "cpu", "wait")))
{
    const int depth = get<0>(GetParam());
    const std::string workload = get<1>(GetParam());

    std::ifstream f(getDataPath("highgui/video/big_buck_bunny.mp4"), std::ios::binary);
    const std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    ASSERT_FALSE(data.empty());

    Mat frame, out;
    while (next())
    {
        VideoCapture cap(makePtr<MemoryStream>(data), CAP_FFMPEG, {});
        if (!cap.isOpened())
            throw SkipTestException("FFmpeg can't open the stream");
        if (depth > 0)
        {
            ASSERT_TRUE(cap.set(CAP_PROP_PREFETCH_FRAMES, depth));
        }

        startTimer();
        while (cap.read(frame))
        {
            if (workload == "cpu")
                GaussianBlur(frame, out, Size(15, 15), 0);
            else if (workload == "wait")
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
