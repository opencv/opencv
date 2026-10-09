// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

#if !defined(OPENCV_DISABLE_THREAD_SUPPORT)
#include <thread>
#endif

namespace opencv_test { namespace {

const Point2f SHIFT(1.7f, -1.1f);

// A blurred noise texture translated by SHIFT per frame.
static std::vector<Mat> makeSequence(Size size, int count, int seed = 0x5eed)
{
    Mat base(size.height * 2, size.width * 2, CV_8U);
    RNG(seed).fill(base, RNG::UNIFORM, 0, 256);
    GaussianBlur(base, base, Size(9, 9), 3, 3);

    std::vector<Mat> frames;
    for( int i = 0; i < count; i++ )
    {
        Mat shift = (Mat_<double>(2, 3) << 1, 0, SHIFT.x * i, 0, 1, SHIFT.y * i);
        Mat warped;
        warpAffine(base, warped, shift, base.size(), INTER_LINEAR);
        frames.push_back(warped(Rect(size.width / 2, size.height / 2,
                                     size.width, size.height)).clone());
    }
    return frames;
}

static Ptr<FarnebackOpticalFlow> makeFlow(bool reuse)
{
    Ptr<FarnebackOpticalFlow> algo = FarnebackOpticalFlow::create(3, 0.5, false, 9, 3);
    algo->setReuseExpansion(reuse);
    return algo;
}

static Mat flowWithoutReuse(const Mat& prev, const Mat& next)
{
    Mat flow;
    makeFlow(false)->calc(prev, next, flow);
    return flow;
}

static void expectSameFlow(const Mat& expected, const Mat& actual, const std::string& what)
{
    ASSERT_EQ(expected.size(), actual.size()) << what;
    ASSERT_EQ(expected.type(), actual.type()) << what;
    ASSERT_GT(countNonZero(expected.reshape(1) != 0.0f), 0) << what;
    const size_t rowBytes = (size_t)expected.cols * expected.elemSize();
    for( int y = 0; y < expected.rows; y++ )
        ASSERT_EQ(0, memcmp(expected.ptr(y), actual.ptr(y), rowBytes)) << what << ", row " << y;
}

TEST(DenseOpticalFlow_Farneback, Accuracy)
{
    const std::vector<Mat> frames = makeSequence(Size(320, 240), 4);
    Ptr<FarnebackOpticalFlow> algo = FarnebackOpticalFlow::create();
    algo->setReuseExpansion(true);

    for( size_t i = 0; i + 1 < frames.size(); i++ )
    {
        Mat flow;
        algo->calc(frames[i], frames[i + 1], flow);
        std::vector<Mat> xy;
        split(flow(Rect(16, 16, flow.cols - 32, flow.rows - 32)), xy);
        for( int c = 0; c < 2; c++ )
        {
            Mat v = xy[c].reshape(1, 1).clone();
            std::nth_element(v.begin<float>(), v.begin<float>() + v.cols / 2, v.end<float>());
            EXPECT_NEAR(c == 0 ? SHIFT.x : SHIFT.y, v.at<float>(v.cols / 2), 0.1)
                << "pair " << i << ", channel " << c;
        }
    }
}

TEST(DenseOpticalFlow_Farneback, ReuseMatchesNoReuseOnSequence)
{
    EXPECT_FALSE(FarnebackOpticalFlow::create()->getReuseExpansion());

    const std::vector<Mat> frames = makeSequence(Size(160, 120), 6);
    Ptr<FarnebackOpticalFlow> algo = makeFlow(true);
    ASSERT_TRUE(algo->getReuseExpansion());
    for( size_t i = 0; i + 1 < frames.size(); i++ )
    {
        Mat flow;
        algo->calc(frames[i], frames[i + 1], flow);
        expectSameFlow(flowWithoutReuse(frames[i], frames[i + 1]), flow, format("pair %d", (int)i));
    }
}

// Calls that must not reuse the held expansion: changed content, parameters or frame size.
TEST(DenseOpticalFlow_Farneback, ReuseMissesOnChangedContentOrParameters)
{
    const std::vector<Mat> frames = makeSequence(Size(160, 120), 3);
    const std::vector<Mat> other = makeSequence(Size(160, 120), 2, 0x77);

    // Same buffer refilled between calls: only the content differs.
    {
        Ptr<FarnebackOpticalFlow> algo = makeFlow(true);
        Mat buf = frames[1].clone(), flow;
        algo->calc(frames[0], buf, flow);
        other[0].copyTo(buf);
        algo->calc(buf, other[1], flow);
        expectSameFlow(flowWithoutReuse(other[0], other[1]), flow, "refilled buffer");
    }

    typedef std::function<void(const Ptr<FarnebackOpticalFlow>&)> Tweak;
    const std::pair<std::string, Tweak> tweaks[] = {
        std::make_pair("polyN", Tweak([](const Ptr<FarnebackOpticalFlow>& f) {
            f->setPolyN(7); })),
        std::make_pair("polySigma", Tweak([](const Ptr<FarnebackOpticalFlow>& f) {
            f->setPolySigma(1.3); })),
        std::make_pair("pyrScale", Tweak([](const Ptr<FarnebackOpticalFlow>& f) {
            f->setPyrScale(0.45); })),  // same pyramid depth as 0.5 at this size
    };
    for( size_t t = 0; t < sizeof(tweaks) / sizeof(tweaks[0]); t++ )
    {
        Ptr<FarnebackOpticalFlow> algo = makeFlow(true), fresh = makeFlow(false);
        Mat flow, expected;
        algo->calc(frames[0], frames[1], flow);
        tweaks[t].second(algo);
        tweaks[t].second(fresh);
        algo->calc(frames[1], frames[2], flow);
        fresh->calc(frames[1], frames[2], expected);
        expectSameFlow(expected, flow, tweaks[t].first);
    }

    // Frame size change.
    {
        const std::vector<Mat> small = makeSequence(Size(96, 72), 2);
        Ptr<FarnebackOpticalFlow> algo = makeFlow(true);
        Mat flow;
        algo->calc(frames[0], frames[1], flow);
        algo->calc(small[0], small[1], flow);
        expectSameFlow(flowWithoutReuse(small[0], small[1]), flow, "size change");
    }
}

// Non-continuous ROIs, and one frame arriving strided as next and then continuous as prev.
TEST(DenseOpticalFlow_Farneback, ReuseWithRoiInput)
{
    const Size size(96, 72);
    const Rect roi(6, 4, size.width, size.height);
    const std::vector<Mat> frames = makeSequence(Size(size.width + 19, size.height + 13), 4);

    Ptr<FarnebackOpticalFlow> algo = makeFlow(true);
    for( size_t i = 0; i + 1 < frames.size(); i++ )
    {
        const Mat prev = frames[i](roi), next = frames[i + 1](roi);
        ASSERT_FALSE(prev.isContinuous());
        Mat flow;
        algo->calc(i == 1 ? prev.clone() : prev, next, flow);
        expectSameFlow(flowWithoutReuse(prev, next), flow, format("pair %d", (int)i));
    }
}

TEST(DenseOpticalFlow_Farneback, ReuseReleaseKeepsResult)
{
    const std::vector<Mat> frames = makeSequence(Size(96, 72), 5);
    Ptr<FarnebackOpticalFlow> algo = makeFlow(true);
    Mat flow;

    algo->calc(frames[0], frames[1], flow);
    algo->collectGarbage();
    algo->calc(frames[1], frames[2], flow);
    expectSameFlow(flowWithoutReuse(frames[1], frames[2]), flow, "after collectGarbage");

    algo->setReuseExpansion(false);
    EXPECT_FALSE(algo->getReuseExpansion());
    algo->calc(frames[2], frames[3], flow);
    expectSameFlow(flowWithoutReuse(frames[2], frames[3]), flow, "reuse off");

    algo->setReuseExpansion(true);
    algo->calc(frames[3], frames[4], flow);
    expectSameFlow(flowWithoutReuse(frames[3], frames[4]), flow, "reuse back on");
}

#if !defined(OPENCV_DISABLE_THREAD_SUPPORT)
TEST(DenseOpticalFlow_Farneback, ReuseConcurrentCalls)
{
    const int threads = 4, pairs = 6;
    std::vector<std::vector<Mat> > frames(threads), actual(threads);
    for( int t = 0; t < threads; t++ )
    {
        frames[t] = makeSequence(Size(64, 48), pairs + 1, 0x900 + t);
        actual[t].resize(pairs);
    }

    Ptr<FarnebackOpticalFlow> algo = makeFlow(true);
    std::vector<std::thread> workers;
    for( int t = 0; t < threads; t++ )
        workers.push_back(std::thread([&, t]() {
            for( int i = 0; i < pairs; i++ )
                algo->calc(frames[t][i], frames[t][i + 1], actual[t][i]);
        }));
    for( size_t t = 0; t < workers.size(); t++ )
        workers[t].join();

    for( int t = 0; t < threads; t++ )
        for( int i = 0; i < pairs; i++ )
            expectSameFlow(flowWithoutReuse(frames[t][i], frames[t][i + 1]), actual[t][i],
                           format("thread %d, pair %d", t, i));
}
#endif

}} // namespace
