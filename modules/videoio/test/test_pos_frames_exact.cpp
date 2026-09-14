// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

using namespace std;

namespace opencv_test { namespace {

// big_buck_bunny.mp4: mpeg4, 24fps, 125 frames, key frame every 12 frames (0,12,...,120).
static string posFramesExactTestVideoPath()
{
    return findDataFile("video/big_buck_bunny.mp4");
}

// FFmpeg's own seek-then-decode-forward already lands exactly; the generic layer's job here is
// just to confirm that and not add overhead or false negatives on top of it.
TEST(videoio_pos_frames_exact, ffmpeg_non_raw_seek_is_always_exact)
{
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend was not found");

    VideoCapture cap(posFramesExactTestVideoPath(), CAP_FFMPEG);
    ASSERT_TRUE(cap.isOpened());

    for (int target : {0, 1, 5, 12, 15, 20, 24, 62, 100, 124})
    {
        ASSERT_TRUE(cap.set(CAP_PROP_POS_FRAMES, target)) << "target " << target;
        EXPECT_EQ(1, cvRound(cap.get(CAP_PROP_POS_FRAMES_IS_EXACT))) << "target " << target;
        Mat frame;
        ASSERT_TRUE(cap.read(frame)) << "target " << target;
    }
}

// RAW mode has no codec to decode forward with, so a seek can only be exact when the requested
// frame is itself a key frame (0, 12, 24, ...); everything else lands short and must say so.
TEST(videoio_pos_frames_exact, ffmpeg_raw_mode_seek_is_exact_only_on_key_frames)
{
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend was not found");

    VideoCapture cap(posFramesExactTestVideoPath(), CAP_FFMPEG, {CAP_PROP_FORMAT, -1});
    ASSERT_TRUE(cap.isOpened());

    struct { int target; int landed; bool exact; } cases[] = {
        {0, 0, true}, {12, 12, true}, {24, 24, true},    // requested frame is itself a key frame
        {5, 0, false}, {15, 12, false}, {20, 12, false}, // between key frames 0 and 12
        {62, 60, false}, {100, 96, false}, {124, 120, false},
    };
    for (const auto& c : cases)
    {
        ASSERT_TRUE(cap.set(CAP_PROP_POS_FRAMES, c.target)) << "target " << c.target;
        // Checked before read(): a raw-mode grab right after a seek can leave position unchanged
        // (it consumes the seek's own internal lookahead rather than advancing), so this is the
        // moment that actually reflects where the seek landed, matching what set() itself verified.
        EXPECT_EQ(c.landed, cvRound(cap.get(CAP_PROP_POS_FRAMES))) << "target " << c.target;
        EXPECT_EQ(c.exact ? 1 : 0, cvRound(cap.get(CAP_PROP_POS_FRAMES_IS_EXACT))) << "target " << c.target;
        Mat raw;
        ASSERT_TRUE(cap.read(raw)) << "target " << c.target;
    }
}

// The documented contract ("seek to key frame k <= i") is something the generic layer assumes each
// backend honors, not something it can independently enforce -- grabFrame() only ever moves forward,
// so an overshoot has no way to be corrected. This sweeps every single frame in the file, in both
// modes, to check that assumption actually holds rather than trusting it on a handful of samples.
TEST(videoio_pos_frames_exact, ffmpeg_seek_never_lands_past_the_target)
{
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend was not found");

    for (bool raw : {false, true})
    {
        VideoCapture cap = raw
            ? VideoCapture(posFramesExactTestVideoPath(), CAP_FFMPEG, {CAP_PROP_FORMAT, -1})
            : VideoCapture(posFramesExactTestVideoPath(), CAP_FFMPEG);
        ASSERT_TRUE(cap.isOpened()) << "raw=" << raw;

        for (int target = 0; target < 125; target++)
        {
            ASSERT_TRUE(cap.set(CAP_PROP_POS_FRAMES, target)) << "raw=" << raw << " target=" << target;
            double landed = cap.get(CAP_PROP_POS_FRAMES);
            ASSERT_NE(static_cast<double>(CAP_PROP_UNKNOWN), landed) << "raw=" << raw << " target=" << target;
            EXPECT_LE(cvRound(landed), target) << "raw=" << raw << " target=" << target;
            Mat frame;
            ASSERT_TRUE(cap.read(frame)) << "raw=" << raw << " target=" << target;
        }
    }
}

// Intra-only codec: every frame is its own key frame, so this is CAP_IMAGES-like -- always exact.
TEST(videoio_pos_frames_exact, opencv_mjpeg_seek_is_always_exact)
{
    if (!videoio_registry::hasBackend(CAP_OPENCV_MJPEG))
        throw SkipTestException("CAP_OPENCV_MJPEG backend was not found");

    VideoCapture cap(findDataFile("video/big_buck_bunny.mjpg.avi"), CAP_OPENCV_MJPEG);
    ASSERT_TRUE(cap.isOpened());

    for (int target : {0, 1, 5, 12, 24, 62, 100, 124})
    {
        ASSERT_TRUE(cap.set(CAP_PROP_POS_FRAMES, target)) << "target " << target;
        EXPECT_EQ(1, cvRound(cap.get(CAP_PROP_POS_FRAMES_IS_EXACT))) << "target " << target;
        Mat frame;
        ASSERT_TRUE(cap.read(frame)) << "target " << target;
    }
}

// A synthetic pipeline avoids the real container's demuxer/qtdemux specifics, but not the
// preroll-timing instability below -- position querying on this source is consistently
// unverifiable in practice (confirmed by repeated runs), even when a seek is accepted.
static std::string posFramesExactGstreamerPipeline(int srcFrameCount, double srcFps)
{
    std::ostringstream pipeline;
    pipeline << "videotestsrc pattern=ball num-buffers=" << srcFrameCount
             << " ! video/x-raw,framerate=" << cvRound(srcFps) << "/1 ! appsink";
    return pipeline.str();
}

// Same honesty contract as the real-file test below, on a synthetic source: a seek this backend
// accepts but can't verify must report -1 (unknown), never a false 1, and one it does verify must
// actually match where it landed.
TEST(videoio_pos_frames_exact, gstreamer_synthetic_pipeline_seek_stays_honest_either_way)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    VideoCapture cap;
    ASSERT_NO_THROW(cap.open(posFramesExactGstreamerPipeline(30, 30.0), CAP_GSTREAMER));
    ASSERT_TRUE(cap.isOpened());

    Mat frame;
    ASSERT_TRUE(cap.read(frame)); // prime: matches the preroll state a real caller would be in

    for (int target : {0, 5, 10, 20, 29})
    {
        bool ok = cap.set(CAP_PROP_POS_FRAMES, target);
        double exact = cap.get(CAP_PROP_POS_FRAMES_IS_EXACT);
        if (!ok)
        {
            EXPECT_NE(1, cvRound(exact)) << "target " << target;
            continue;
        }
        if (cvRound(exact) == 1)
        {
            EXPECT_EQ(target, cvRound(cap.get(CAP_PROP_POS_FRAMES))) << "target " << target;
        }
        ASSERT_TRUE(cap.read(frame)) << "target " << target;
    }
}

// #10324: CAP_PROP_POS_FRAMES seeking on a real container (qtdemux here) is unreliable in ways the
// generic layer cannot fix -- isPosFramesSupported is decided once, during pipeline preroll, and
// that outcome is sensitive to exact call sequencing. This test does not assert a specific result;
// it documents that whichever way it goes, the contract stays honest: a failed seek must not be
// misreported as exact, and a successful one must land where it says it did.
TEST(videoio_pos_frames_exact, gstreamer_real_file_seek_stays_honest_either_way)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    VideoCapture cap(posFramesExactTestVideoPath(), CAP_GSTREAMER);
    ASSERT_TRUE(cap.isOpened());

    for (int target : {0, 5, 15, 62, 100})
    {
        bool ok = cap.set(CAP_PROP_POS_FRAMES, target);
        double exact = cap.get(CAP_PROP_POS_FRAMES_IS_EXACT);
        // Checked before read(): GStreamer's own position reporting can become unavailable again
        // after a subsequent read(), which is a symptom of the same instability, not something a
        // read() should be able to retroactively invalidate about the seek that already happened.
        if (!ok)
        {
            // A failed seek must never claim to be exact.
            EXPECT_NE(1, cvRound(exact)) << "target " << target;
        }
        else if (cvRound(exact) == 1)
        {
            EXPECT_EQ(target, cvRound(cap.get(CAP_PROP_POS_FRAMES))) << "target " << target;
        }
        if (ok)
        {
            Mat frame;
            ASSERT_TRUE(cap.read(frame)) << "target " << target;
        }
    }
}

}} // namespace
