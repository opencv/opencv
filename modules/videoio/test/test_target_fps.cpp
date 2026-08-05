// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

using namespace std;

namespace opencv_test { namespace {

// target_fps drop-only frame-rate control, exposed as a VideoCapture
// constructor/open() parameter. Ported from FFmpeg's own frame-rate
// reduction rule (libavfilter/vf_fps.c, write_frame()): buffer 2 frames,
// compare the second one's timestamp against a steadily-advancing output
// clock, and drop the first whenever it's already stale.
//
// The feature itself lives entirely in the backend-agnostic frontend
// (modules/videoio/src/cap.cpp) -- no backend-specific code was touched to
// implement it. These tests use the FFmpeg backend only as a convenient,
// precisely-controllable source of test video; the behavior under test is
// not FFmpeg-specific.
//
// Source video: opencv_extra's shared "video/big_buck_bunny.mp4" -- the
// same canonical asset already used throughout test_ffmpeg.cpp and
// perf_target_fps.cpp, rather than a synthetic, test-generated file.
// Confirmed properties (ffprobe): mpeg4 codec, 24/1 fps, 125 frames,
// has_b_frames=0 (I/P only, no reordering) -- so CAP_PROP_POS_MSEC on
// readback is exact per-frame, same guarantee the synthetic MJPG file used
// to provide. findDataFile() throws SkipTestException on its own if
// OPENCV_TEST_DATA_PATH/opencv_extra isn't available locally, matching the
// convention already used elsewhere in test_ffmpeg.cpp.
static string targetFpsTestVideoPath()
{
    return findDataFile("video/big_buck_bunny.mp4");
}

static const double BBB_FPS = 24.0;
static const int BBB_FRAME_COUNT = 125;

// (target_fps, ratio) -- ratio = how many source frames separate each kept
// frame (source_fps / target_fps when that divides evenly). ratio=1 covers
// three distinct cases that should all behave identically: disabled (0),
// exactly the source's native rate, and requesting a rate *above* native
// (degrades to native rate rather than duplicating -- see CLAUDE.md's
// "Degenerate cases" note).
//
// 4.0/8.0 are chosen as exact divisors of the source's 24fps (ratios 6 and 3)
// so the expected counts below are unambiguous; both also leave the file's
// true last frame index (124) unaligned to the ratio (124 % 6 == 4,
// 124 % 3 == 1) -- deliberately, so these cases still exercise
// fpsControlGrab()'s end-of-stream path: the schedule is reached mid-tick
// rather than landing exactly on the last frame.
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

// The last kept frame here (index 123, of 125 total) is exactly the case that used to break:
// end-of-stream is reached with the schedule mid-tick, and get(CAP_PROP_POS_MSEC) used to
// report a backend-specific value unrelated to the frame actually returned (observed 0.0)
// instead of that frame's own timestamp.
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

// Separate from the count check above: confirms the *timing* is right, not
// just the quantity -- successive kept frames should be spaced exactly
// 1000/target_fps ms apart, matching FFmpeg's own output-clock behavior.
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
    // get(CAP_PROP_POS_MSEC) reports the emitted frame's own timestamp (fpsCtl.pendingPts),
    // including the final read at end-of-stream -- no reading needs to be excluded here.
    vector<double> diffs(posMsec.size());
    std::adjacent_difference(posMsec.begin(), posMsec.end(), diffs.begin());
    const double expectedStepMs = 1000.0 / target_fps;
    // A small epsilon, not a full frame-duration: fpsControlGrab()'s own boundary comparison
    // tolerates backend floating-point rounding noise (observed on the order of 1e-13 ms) via
    // kEpsMs, so every gap here should land on the exact scheduled step.
    auto minIt = min_element(diffs.begin() + 1, diffs.end());
    auto maxIt = max_element(diffs.begin() + 1, diffs.end());
    EXPECT_NEAR(expectedStepMs, *minIt, 1.0);
    EXPECT_NEAR(expectedStepMs, *maxIt, 1.0);
}

// Exercises the plain C++ default-argument path (no target_fps passed at all)
// to guard against a signature drift between the header declaration and the
// .cpp definition silently breaking the pre-existing, non-target_fps call sites.
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

// Regression test: release() must reset fpsCtl, or a capture reopened through the
// general params-vector overload (which never calls enableFpsControl()) inherits stale
// clock/buffer state from the previous open and can silently return false forever.
TEST(videoio_target_fps, reopen_via_params_overload_resets_state)
{
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend was not found");

    const string filename = targetFpsTestVideoPath();

    VideoCapture cap(filename, CAP_FFMPEG, 8.0); // target_fps enabled, clock anchored to this open's PTS
    ASSERT_TRUE(cap.isOpened());
    Mat frame;
    ASSERT_TRUE(cap.read(frame)); // leave fpsCtl mid-schedule, not freshly reset

    // General params-vector overload -- never calls enableFpsControl(), so this only behaves
    // correctly if release() (called internally since cap.isOpened()) resets fpsCtl itself.
    // Reopening the SAME file here (rather than a second, distinct one) is enough: what's under
    // test is whether stale fpsCtl state survives the reopen, not which file is read.
    ASSERT_TRUE(cap.open(filename, CAP_FFMPEG, std::vector<int>()));
    ASSERT_TRUE(cap.isOpened());

    int n = 0;
    while (cap.read(frame))
        n++;
    cap.release();

    // target_fps isn't threaded through this overload (by design -- see CLAUDE.md), so the
    // reopened capture should behave as a plain, unrestricted passthrough: every native frame
    // (125), not silently zero (the pre-fix symptom) and not the previous open's leftover
    // 8fps/ratio-3 schedule (42).
    EXPECT_EQ(BBB_FRAME_COUNT, n);
}

// Regression test: target_fps only buffers/decodes channel 0 (fpsControlReadOne() always calls
// retrieveFrame(0, ...)). Requesting any other channel must fail loudly rather than silently
// handing back channel 0's data -- this matters for multi-head sources (stereo camera, Kinect).
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

}} // namespace
