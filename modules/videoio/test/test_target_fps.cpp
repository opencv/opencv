// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/core/utils/filesystem.hpp"
#include "opencv2/imgcodecs.hpp"

using namespace std;

namespace opencv_test { namespace {

// target_fps drop-only frame-rate control, a VideoCapture constructor/open() parameter.
// big_buck_bunny.mp4 is mpeg4, 24/1 fps, 125 frames, no B-frames, so its per-frame timestamps are
// exact; findDataFile() throws SkipTestException itself if opencv_extra is unavailable.
static string targetFpsTestVideoPath()
{
    return findDataFile("video/big_buck_bunny.mp4");
}

static const double BBB_FPS = 24.0;
static const int BBB_FRAME_COUNT = 125;

// (target_fps, ratio) -- source frames between each kept frame. ratio=1 covers disabled, exactly
// native, and above native. 4.0/8.0 divide 24fps exactly but leave the last index (124) unaligned,
// so the end-of-stream path is reached mid-tick rather than on the last frame.
typedef tuple<double, int> TargetFps_Ratio;
typedef testing::TestWithParam<TargetFps_Ratio> videoio_target_fps;

TEST_P(videoio_target_fps, keeps_expected_frame_count)
{
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend was not found");

    const double target_fps = get<0>(GetParam());
    const int ratio = get<1>(GetParam());

    const string filename = targetFpsTestVideoPath();

    VideoCapture cap(filename, CAP_FFMPEG, target_fps);
    ASSERT_TRUE(cap.isOpened());

    int n = 0;
    Mat frame;
    while (cap.read(frame))
    {
        ASSERT_FALSE(frame.empty());
        n++;
    }
    cap.release();

    int expected = 0;
    for (int i = 0; i < BBB_FRAME_COUNT; i += ratio)
        expected++;
    EXPECT_EQ(expected, n);
}

const TargetFps_Ratio videoio_target_fps_params[] =
{
    make_tuple(0.0, 1),    // target_fps unset/disabled -> passthrough, every frame
    make_tuple(24.0, 1),   // == source's native fps -> every frame
    make_tuple(48.0, 1),   // above source's native fps -> degrades to native, every frame
    make_tuple(4.0, 6),    // keep indices 0,6,...,120 -> 21 frames
    make_tuple(8.0, 3),    // keep indices 0,3,...,123 -> 42 frames
};

inline static std::string videoio_target_fps_name_printer(const testing::TestParamInfo<videoio_target_fps::ParamType>& info)
{
    std::ostringstream os;
    os << "target_" << cvRound(get<0>(info.param) * 10) << "_ratio_" << get<1>(info.param);
    return os.str();
}

INSTANTIATE_TEST_CASE_P(videoio, videoio_target_fps, testing::ValuesIn(videoio_target_fps_params), videoio_target_fps_name_printer);

// End-of-stream reached mid-tick: get(CAP_PROP_POS_MSEC) must be the last emitted frame's own
// timestamp, not a backend-internal value unrelated to it.
TEST(videoio_target_fps, last_frame_pos_msec_matches_its_own_timestamp)
{
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend was not found");

    const double target_fps = 8.0;
    const string filename = targetFpsTestVideoPath();

    VideoCapture cap(filename, CAP_FFMPEG, target_fps);
    ASSERT_TRUE(cap.isOpened());

    double lastPosMsec = -1.0;
    int n = 0;
    Mat frame;
    while (cap.read(frame))
    {
        lastPosMsec = cap.get(CAP_PROP_POS_MSEC);
        n++;
    }
    cap.release();

    ASSERT_EQ(42, n); // indices 0,3,...,123
    const double srcFrameDurationMs = 1000.0 / BBB_FPS;
    EXPECT_NEAR(123 * srcFrameDurationMs, lastPosMsec, 1.0); // frame 123's own timestamp, not 0 and not a lookahead frame's
}

// Timing rather than quantity: successive kept frames should be exactly 1000/target_fps ms apart.
TEST(videoio_target_fps, emitted_frames_evenly_spaced)
{
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend was not found");

    const double target_fps = 8.0;
    const string filename = targetFpsTestVideoPath();

    VideoCapture cap(filename, CAP_FFMPEG, target_fps);
    ASSERT_TRUE(cap.isOpened());

    vector<double> posMsec;
    Mat frame;
    while (cap.read(frame))
        posMsec.push_back(cap.get(CAP_PROP_POS_MSEC));
    cap.release();

    ASSERT_GE(posMsec.size(), 5u);
    // Tight tolerance is deliberate: fpsControlGrab() already absorbs backend rounding noise.
    vector<double> diffs(posMsec.size());
    std::adjacent_difference(posMsec.begin(), posMsec.end(), diffs.begin());
    const double expectedStepMs = 1000.0 / target_fps;
    auto minIt = min_element(diffs.begin() + 1, diffs.end());
    auto maxIt = max_element(diffs.begin() + 1, diffs.end());
    EXPECT_NEAR(expectedStepMs, *minIt, 1.0);
    EXPECT_NEAR(expectedStepMs, *maxIt, 1.0);
}

// The default-argument path, guarding against declaration/definition signature drift breaking
// existing non-target_fps call sites.
TEST(videoio_target_fps, default_argument_is_disabled)
{
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend was not found");

    const string filename = targetFpsTestVideoPath();

    VideoCapture cap(filename, CAP_FFMPEG); // no target_fps argument at all
    ASSERT_TRUE(cap.isOpened());

    int n = 0;
    Mat frame;
    while (cap.read(frame))
        n++;
    cap.release();

    EXPECT_EQ(BBB_FRAME_COUNT, n);
}

// release() must reset fpsCtl, or a reopen through the params-vector overload inherits the previous
// open's clock and buffers.
TEST(videoio_target_fps, reopen_via_params_overload_resets_state)
{
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend was not found");

    const string filename = targetFpsTestVideoPath();

    VideoCapture cap(filename, CAP_FFMPEG, 8.0); // target_fps enabled, clock anchored to this open's PTS
    ASSERT_TRUE(cap.isOpened());
    Mat frame;
    ASSERT_TRUE(cap.read(frame)); // leave fpsCtl mid-schedule, not freshly reset

    // This overload never calls enableFpsControl(), so it only behaves correctly if the release()
    // it triggers internally resets fpsCtl. Reopening the same file is enough here.
    ASSERT_TRUE(cap.open(filename, CAP_FFMPEG, std::vector<int>()));
    ASSERT_TRUE(cap.isOpened());

    int n = 0;
    while (cap.read(frame))
        n++;
    cap.release();

    // target_fps isn't threaded through this overload by design, so the reopened capture should
    // be a plain passthrough: every native frame (125), not the previous open's schedule (42).
    EXPECT_EQ(BBB_FRAME_COUNT, n);
}

// Only channel 0 is buffered, so any other channel must fail rather than return channel 0's data.
TEST(videoio_target_fps, non_zero_channel_fails_under_fps_control)
{
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend was not found");

    const string filename = targetFpsTestVideoPath();

    VideoCapture cap(filename, CAP_FFMPEG, 8.0);
    ASSERT_TRUE(cap.isOpened());
    ASSERT_TRUE(cap.grab());

    Mat channel0;
    EXPECT_TRUE(cap.retrieve(channel0, 0)); // the only supported channel still works
    EXPECT_FALSE(channel0.empty());

    Mat channel1;
    EXPECT_FALSE(cap.retrieve(channel1, 1)); // must fail, not silently return channel 0's frame
    EXPECT_TRUE(channel1.empty());

    cap.release();
}

// Backend coverage beyond CAP_FFMPEG: the algorithm is backend-agnostic, so these only confirm each
// allow-listed backend's CAP_PROP_POS_MSEC is usable per-frame.

// A synthetic videotestsrc pipeline, as in test_gstreamer.cpp: num-buffers bounds it to a known
// count and the framerate caps give its buffer timestamps exact millisecond spacing.
static std::string targetFpsGstreamerPipeline(int srcFrameCount, double srcFps)
{
    std::ostringstream pipeline;
    pipeline << "videotestsrc pattern=ball num-buffers=" << srcFrameCount
             << " ! video/x-raw,framerate=" << cvRound(srcFps) << "/1 ! appsink";
    return pipeline.str();
}

TEST(videoio_target_fps, gstreamer_pipeline_keeps_expected_frame_count)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    const int srcFrameCount = 20;
    const double srcFps = 20.0;
    const double target_fps = 5.0; // ratio 4 -> keep indices 0,4,8,12,16 (5 kept); 20 % 4 == 0 but
                                    // the source's own last index (19) is unaligned to the ratio,
                                    // same deliberate mid-tick-at-EOF coverage as the FFmpeg params.

    VideoCapture cap;
    ASSERT_NO_THROW(cap.open(targetFpsGstreamerPipeline(srcFrameCount, srcFps), CAP_GSTREAMER, target_fps));
    ASSERT_TRUE(cap.isOpened());

    int n = 0;
    Mat frame;
    while (cap.read(frame))
    {
        ASSERT_FALSE(frame.empty());
        n++;
    }
    cap.release();

    EXPECT_EQ(5, n);
}

// emitted_frames_evenly_spaced against GStreamer's own buffer timestamp instead of a decoded PTS.
TEST(videoio_target_fps, gstreamer_pipeline_emitted_frames_evenly_spaced)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    const int srcFrameCount = 20;
    const double srcFps = 20.0;
    const double target_fps = 5.0;

    VideoCapture cap;
    ASSERT_NO_THROW(cap.open(targetFpsGstreamerPipeline(srcFrameCount, srcFps), CAP_GSTREAMER, target_fps));
    ASSERT_TRUE(cap.isOpened());

    vector<double> posMsec;
    Mat frame;
    while (cap.read(frame))
        posMsec.push_back(cap.get(CAP_PROP_POS_MSEC));
    cap.release();

    ASSERT_EQ(5u, posMsec.size());
    vector<double> diffs(posMsec.size());
    std::adjacent_difference(posMsec.begin(), posMsec.end(), diffs.begin());
    const double expectedStepMs = 1000.0 / target_fps;
    auto minIt = min_element(diffs.begin() + 1, diffs.end());
    auto maxIt = max_element(diffs.begin() + 1, diffs.end());
    // Looser than the FFmpeg test's 1ms since the appsink handoff precision is unverified here,
    // but still tight enough to catch an off-by-one-tick error.
    EXPECT_NEAR(expectedStepMs, *minIt, expectedStepMs * 0.01);
    EXPECT_NEAR(expectedStepMs, *maxIt, expectedStepMs * 0.01);
}

// Frame selection on CAP_OPENCV_MJPEG, whose CAP_PROP_POS_MSEC is derived rather than decoded.
TEST(videoio_target_fps, opencv_mjpeg_keeps_expected_frame_count)
{
    if (!videoio_registry::hasBackend(CAP_OPENCV_MJPEG))
        throw SkipTestException("CAP_OPENCV_MJPEG backend was not found");

    // This backend reads its own native AVI container, not the shared big_buck_bunny.mp4.
    const string filename = findDataFile("video/big_buck_bunny.mjpg.avi");

    const double target_fps = 8.0; // ratio 3 -> emitted source indices 0, 3, ..., 123
    VideoCapture cap(filename, CAP_OPENCV_MJPEG, target_fps);
    ASSERT_TRUE(cap.isOpened());
    ASSERT_EQ(CAP_OPENCV_MJPEG, static_cast<int>(cap.get(CAP_PROP_BACKEND)));
    EXPECT_EQ(target_fps, cap.get(CAP_PROP_FPS)) << "frame-rate control did not engage";

    // Timestamps are not asserted here: this backend derives CAP_PROP_POS_MSEC from the next
    // frame's index, so it reads one source frame ahead of the frame actually returned.
    int n = 0;
    Mat frame;
    while (cap.read(frame))
    {
        ASSERT_FALSE(frame.empty());
        n++;
    }
    cap.release();

    EXPECT_EQ(42, n);
}

// The allow-list in its refusing direction: CAP_IMAGES reports a constant CAP_PROP_POS_MSEC, which
// would make every drop comparison read as stale and emit only the last frame if it got through.
TEST(videoio_target_fps, unsupported_backend_falls_back_to_passthrough)
{
    if (!videoio_registry::hasBackend(CAP_IMAGES))
        throw SkipTestException("CAP_IMAGES backend was not found");

    const string dirname = cv::tempfile("opencv_test_target_fps_images");
    ASSERT_TRUE(cv::utils::fs::createDirectory(dirname));

    const int srcFrameCount = 5;
    for (int i = 0; i < srcFrameCount; i++)
    {
        // Distinct uniform brightness per image, so a dropped or reordered frame shows up as a
        // wrong index rather than only as a wrong count.
        const Mat img(32, 32, CV_8UC3, Scalar::all(i * 40));
        ASSERT_TRUE(imwrite(cv::format("%s/img%04d.png", dirname.c_str(), i), img));
    }

    const double target_fps = 2.0; // would be a reduction, if this backend were allowed
    VideoCapture cap(cv::format("%s/img%04d.png", dirname.c_str(), 0), CAP_IMAGES, target_fps);
    ASSERT_TRUE(cap.isOpened());
    ASSERT_EQ(CAP_IMAGES, static_cast<int>(cap.get(CAP_PROP_BACKEND)));

    vector<int> brightness;
    Mat frame;
    while (cap.read(frame))
    {
        ASSERT_FALSE(frame.empty());
        brightness.push_back(cvRound(mean(frame)[0]));
    }
    cap.release();
    cv::utils::fs::remove_all(dirname);

    ASSERT_EQ(static_cast<size_t>(srcFrameCount), brightness.size())
        << "target_fps must be refused for this backend, leaving an ordinary full-rate capture";
    for (int i = 0; i < srcFrameCount; i++)
        EXPECT_NEAR(i * 40, brightness[i], 5) << "image " << i;
}

// Every emitted frame, all position properties at once: POS_FRAMES - 1 must identify the frame just
// returned, POS_MSEC must be its own time, and the two must agree with each other.
TEST(videoio_target_fps, position_properties_describe_the_emitted_frame)
{
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend was not found");

    const double target_fps = 8.0; // ratio 3 -> emitted source indices 0, 3, 6, ...
    const int ratio = 3;
    VideoCapture cap(targetFpsTestVideoPath(), CAP_FFMPEG, target_fps);
    ASSERT_TRUE(cap.isOpened());

    const double srcFrameDurationMs = 1000.0 / BBB_FPS;
    int n = 0;
    Mat frame;
    while (cap.read(frame))
    {
        const double posMsec = cap.get(CAP_PROP_POS_MSEC);
        const double posFrames = cap.get(CAP_PROP_POS_FRAMES);

        const int expectedIndex = n * ratio;
        // Identifies the emitted frame, not the lookahead frame the drop decision had to read.
        EXPECT_EQ(expectedIndex + 1, cvRound(posFrames)) << "emitted frame " << n;
        // ...and the timestamp reported with it belongs to that same frame.
        EXPECT_NEAR(expectedIndex * srcFrameDurationMs, posMsec, 1.0) << "emitted frame " << n;
        // The two must agree with each other, which is the property that actually matters.
        EXPECT_NEAR((posFrames - 1) * srcFrameDurationMs, posMsec, 1.0) << "emitted frame " << n;
        n++;
    }
    cap.release();

    ASSERT_EQ(42, n);
}

// get(CAP_PROP_FPS) must be the emitted rate, since callers pass it straight to VideoWriter -- and
// must answer before the first read(), which is when a VideoWriter is normally constructed.
TEST(videoio_target_fps, reports_effective_output_fps)
{
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend was not found");

    const string filename = targetFpsTestVideoPath();

    {   // below native: the emitted rate is target_fps
        VideoCapture cap(filename, CAP_FFMPEG, 8.0);
        ASSERT_TRUE(cap.isOpened());
        EXPECT_EQ(8.0, cap.get(CAP_PROP_FPS)) << "before any read()";
        Mat frame;
        ASSERT_TRUE(cap.read(frame));
        EXPECT_EQ(8.0, cap.get(CAP_PROP_FPS)) << "after read()";
    }
    {   // above native: drop-only cannot emit faster than the source, so the native rate is the
        // honest answer -- reporting target_fps here would be a new inaccuracy, not a fix.
        VideoCapture cap(filename, CAP_FFMPEG, BBB_FPS * 2);
        ASSERT_TRUE(cap.isOpened());
        EXPECT_EQ(BBB_FPS, cap.get(CAP_PROP_FPS));
    }
    {   // disabled: untouched passthrough
        VideoCapture cap(filename, CAP_FFMPEG);
        ASSERT_TRUE(cap.isOpened());
        EXPECT_EQ(BBB_FPS, cap.get(CAP_PROP_FPS));
    }
}

// The get()->set() round trip POS_FRAMES exists for. Identity is checked on the pixels rather than
// by reading the position back, which would validate those values against themselves.
TEST(videoio_target_fps, bookmarked_frame_can_be_seeked_back_to)
{
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend was not found");

    const double target_fps = 8.0; // ratio 3 -> the 2nd emitted frame is source frame 3
    VideoCapture cap(targetFpsTestVideoPath(), CAP_FFMPEG, target_fps);
    ASSERT_TRUE(cap.isOpened());

    Mat frame;
    ASSERT_TRUE(cap.read(frame));
    ASSERT_TRUE(cap.read(frame));
    const Mat marked = frame.clone(); // the frame the caller believes it is bookmarking

    const int mark = cvRound(cap.get(CAP_PROP_POS_FRAMES)) - 1; // caller's cursor -> index idiom
    EXPECT_EQ(3, mark) << "bookmark does not name the frame that was actually handed over";

    if (!cap.set(CAP_PROP_POS_FRAMES, mark))
        throw SkipTestException("backend does not support seeking on this file");
    ASSERT_TRUE(cap.read(frame));
    ASSERT_FALSE(frame.empty());

    ASSERT_EQ(marked.size(), frame.size());
    ASSERT_EQ(marked.type(), frame.type());
    EXPECT_EQ(0.0, cv::norm(marked, frame, NORM_INF))
        << "seeking to the bookmarked index returned a different frame";
    cap.release();
}

// A seek must discard the clock and lookahead, or pre-seek frames get emitted and a backward seek
// drops frames until the source catches up to the stale clock.
TEST(videoio_target_fps, seek_resets_schedule_and_lookahead)
{
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend was not found");

    const double target_fps = 8.0; // ratio 3 -> 42 frames over the whole file
    const int expectedTotal = 42;
    const double srcFrameDurationMs = 1000.0 / BBB_FPS;
    VideoCapture cap(targetFpsTestVideoPath(), CAP_FFMPEG, target_fps);
    ASSERT_TRUE(cap.isOpened());

    Mat frame;
    for (int i = 0; i < 10; i++)
        ASSERT_TRUE(cap.read(frame)) << "priming read " << i;
    // Clock is now ~10 ticks in and two frames are buffered from around source frame 27.
    ASSERT_GT(cap.get(CAP_PROP_POS_MSEC), 1000.0);

    if (!cap.set(CAP_PROP_POS_FRAMES, 0))
        throw SkipTestException("backend does not support seeking on this file");

    // First frame after the seek must be the seek target itself, not a leftover pre-seek frame,
    // and not the result of burning frames to catch up to the stale clock.
    ASSERT_TRUE(cap.read(frame));
    ASSERT_FALSE(frame.empty());
    EXPECT_NEAR(0.0, cap.get(CAP_PROP_POS_MSEC), 1.0);
    EXPECT_EQ(1, cvRound(cap.get(CAP_PROP_POS_FRAMES)));

    // ...and the whole schedule restarts from there, rather than resuming mid-stride.
    int n = 1;
    while (cap.read(frame))
        n++;
    EXPECT_EQ(expectedTotal, n);

    // A forward seek re-anchors the same way: the clock starts from the new position, so the
    // remaining count follows the schedule from there rather than from the start of the file.
    ASSERT_TRUE(cap.set(CAP_PROP_POS_FRAMES, 0));
    ASSERT_TRUE(cap.read(frame));
    const int midIndex = 60;
    if (cap.set(CAP_PROP_POS_FRAMES, midIndex))
    {
        ASSERT_TRUE(cap.read(frame));
        EXPECT_NEAR(midIndex * srcFrameDurationMs, cap.get(CAP_PROP_POS_MSEC), srcFrameDurationMs);
        int m = 1;
        while (cap.read(frame))
            m++;
        // 65 frames remain from index 60 (60..124) at ratio 3 -> ceil(65/3) == 22.
        EXPECT_EQ(22, m);
    }
    cap.release();
}

// Same vivid virtual-device convention as test_v4l2.cpp. Being a live, unbounded capture there is no
// expected frame count -- what is tested is that a real kernel timestamp holds up against jitter.
TEST(videoio_target_fps, v4l2_vivid_respects_target_fps)
{
    if (!videoio_registry::hasBackend(CAP_V4L2))
        throw SkipTestException("V4L2 backend was not found");

    utils::Paths devs = utils::getConfigurationParameterPaths("OPENCV_TEST_V4L2_VIVID_DEVICE");
    if (devs.size() != 1)
        throw SkipTestException("OPENCV_TEST_V4L2_VIVID_DEVICE is not set");
    const string device = devs[0];

    // Well below vivid's default capture rate, leaving plenty of margin for real scheduling
    // jitter that a decoded-file/synthetic-pipeline PTS doesn't have to contend with.
    const double target_fps = 5.0;
    VideoCapture cap(device, CAP_V4L2, target_fps);
    ASSERT_TRUE(cap.isOpened());

    const int kFramesToRead = 5;
    vector<double> posMsec;
    Mat frame;
    for (int i = 0; i < kFramesToRead; i++)
    {
        ASSERT_TRUE(cap.read(frame));
        ASSERT_FALSE(frame.empty());
        posMsec.push_back(cap.get(CAP_PROP_POS_MSEC));
    }
    cap.release();

    const double expectedStepMs = 1000.0 / target_fps;
    for (size_t i = 1; i < posMsec.size(); i++)
    {
        const double stepMs = posMsec[i] - posMsec[i - 1];
        // 50% tolerance, vs. the file-based tests' ~1%: a live kernel capture-buffer timestamp is
        // subject to genuine scheduling jitter a decoded/synthetic PTS doesn't have. What matters
        // here is that frames are actually being dropped to hit ~200ms spacing, not sub-frame
        // precision.
        EXPECT_NEAR(expectedStepMs, stepMs, expectedStepMs * 0.5)
            << "step " << i << ": " << posMsec[i - 1] << " -> " << posMsec[i];
    }
}

}} // namespace
