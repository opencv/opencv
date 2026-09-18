// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

#if !defined(OPENCV_DISABLE_THREAD_SUPPORT)
#include <thread>
#endif

namespace opencv_test { namespace {

// FarnebackOpticalFlow keeps the polynomial expansion of a call's second image and reuses
// it on the next call, where in a video that image arrives as the first one. Every case
// here asserts the same property: whatever the caller does in between, a reused instance
// returns bit for bit what a fresh instance returns.
//
// Nothing counts cache hits, so a case that needs a hit asserts its premise instead --
// that the buffer really was recycled, that the input really is strided, that two frames
// really do differ.

static void expectSameFlow(const Mat& fresh, const Mat& reused, const std::string& what,
                           bool mayBeZero = false)
{
    ASSERT_EQ(fresh.size(), reused.size()) << what;
    ASSERT_EQ(fresh.type(), reused.type()) << what;
    ASSERT_FALSE(fresh.empty()) << what;
    // An all-zero field would make the comparison below vacuous.
    if( !mayBeZero )
    {
        ASSERT_GT(countNonZero(fresh.reshape(1) != 0.0f), 0)
            << what << ": flow is entirely zero";
    }

    // Bytes, not ==: the claim is identical bits, and == calls +0.0f and -0.0f equal and
    // a NaN unequal to itself.
    int differingRows = 0;
    const size_t rowBytes = (size_t)fresh.cols * fresh.elemSize();
    for( int y = 0; y < fresh.rows; y++ )
    {
        if( memcmp(fresh.ptr(y), reused.ptr(y), rowBytes) != 0 )
            differingRows++;
    }
    EXPECT_EQ(0, differingRows) << what << ": " << differingRows << " of " << fresh.rows
                                << " flow rows differ";
}

// A translated texture, so the flow is a real field rather than the near-zero one a pair
// of independent noise images produces.
static std::vector<Mat> makeSequence(Size size, int count, int seed = 0x5eed)
{
    Mat base(size.height * 2, size.width * 2, CV_8U);
    RNG(seed).fill(base, RNG::UNIFORM, 0, 256);
    GaussianBlur(base, base, Size(9, 9), 3, 3);

    std::vector<Mat> frames;
    for( int i = 0; i < count; i++ )
    {
        Mat shift = (Mat_<double>(2, 3) << 1, 0, 1.7 * i, 0, 1, -1.1 * i);
        Mat warped;
        warpAffine(base, warped, shift, base.size(), INTER_LINEAR);
        frames.push_back(warped(Rect(size.width / 2, size.height / 2,
                                     size.width, size.height)).clone());
    }
    return frames;
}

typedef std::function<Ptr<FarnebackOpticalFlow>()> FlowFactory;

// Cheaper than the defaults, so these tests can afford many calls.
static Ptr<FarnebackOpticalFlow> makeFlow()
{
    Ptr<FarnebackOpticalFlow> algo = FarnebackOpticalFlow::create();
    algo->setNumLevels(3);
    algo->setWinSize(9);
    algo->setNumIters(3);
    return algo;
}

// Drives frame pairs through one reused instance and through a fresh instance per pair,
// and requires the two to agree on every pair.
static void checkPairs(const std::vector<std::pair<Mat, Mat> >& pairs,
                       const FlowFactory& make,
                       const std::string& what)
{
    ASSERT_FALSE(pairs.empty());

    Ptr<FarnebackOpticalFlow> reused = make();
    for( size_t i = 0; i < pairs.size(); i++ )
    {
        Mat flowFresh, flowReused;
        make()->calc(pairs[i].first, pairs[i].second, flowFresh);
        reused->calc(pairs[i].first, pairs[i].second, flowReused);
        expectSameFlow(flowFresh, flowReused, what + ", pair " + std::to_string(i));
    }
}

static std::vector<std::pair<Mat, Mat> > consecutivePairs(const std::vector<Mat>& frames)
{
    std::vector<std::pair<Mat, Mat> > pairs;
    for( size_t i = 0; i + 1 < frames.size(); i++ )
        pairs.push_back(std::make_pair(frames[i], frames[i + 1]));
    return pairs;
}

typedef testing::TestWithParam<Size> DenseOpticalFlow_Farneback;

// The case the reuse is for: consecutive pairs of one sequence, forwards and then
// backwards, so every call after the first finds its first image already expanded.
TEST_P(DenseOpticalFlow_Farneback, ReusedInstanceMatchesFreshOverSequence)
{
    std::vector<Mat> frames = makeSequence(GetParam(), 6);
    checkPairs(consecutivePairs(frames), makeFlow, "forward sequence");
    std::reverse(frames.begin(), frames.end());
    checkPairs(consecutivePairs(frames), makeFlow, "reversed sequence");
}

// Two unrelated sources on one instance, as a caller multiplexing cameras would do.
TEST_P(DenseOpticalFlow_Farneback, ReusedInstanceMatchesFreshWhenSourcesInterleave)
{
    const std::vector<Mat> a = makeSequence(GetParam(), 4, 0x11);
    const std::vector<Mat> b = makeSequence(GetParam(), 4, 0x22);

    std::vector<std::pair<Mat, Mat> > pairs;
    for( size_t i = 0; i + 1 < a.size(); i++ )
    {
        pairs.push_back(std::make_pair(a[i], a[i + 1]));
        pairs.push_back(std::make_pair(b[i], b[i + 1]));
    }
    checkPairs(pairs, makeFlow, "interleaved sources");
}

// Unrelated pairs agreeing on size, type and step and on nothing else, so weakening the
// frame comparison to those makes this fail.
TEST_P(DenseOpticalFlow_Farneback, ReusedInstanceMatchesFreshOnUnrelatedPairs)
{
    std::vector<std::pair<Mat, Mat> > pairs;
    for( int i = 0; i < 4; i++ )
    {
        const std::vector<Mat> frames = makeSequence(GetParam(), 2, 0x100 + i * 7);
        pairs.push_back(std::make_pair(frames[0], frames[1]));
    }
    checkPairs(pairs, makeFlow, "unrelated pairs");
}

// The ordinary in-place caller: one pair of Mats refilled per frame, so address, size and
// step are constant and only the content moves.
TEST_P(DenseOpticalFlow_Farneback, ReusedInstanceMatchesFreshWhenBuffersAreOverwritten)
{
    const Size size = GetParam();
    const std::vector<Mat> frames = makeSequence(size, 6);

    Mat prev(size, CV_8U), next(size, CV_8U);
    Ptr<FarnebackOpticalFlow> reused = makeFlow();

    for( size_t i = 0; i + 1 < frames.size(); i++ )
    {
        // Shuffled, so no call is handed as its first image what the previous call was
        // handed as its second: nothing here may be reused.
        Mat leftBehind;
        if( i > 0 )
            next.copyTo(leftBehind);
        frames[(i + 4) % frames.size()].copyTo(prev);
        frames[i + 1].copyTo(next);
        if( i > 0 )
        {
            ASSERT_GT(countNonZero(leftBehind != prev), 0)
                << "pair " << i << ": the shuffle must not line up into a hit";
        }

        Mat flowFresh, flowReused;
        makeFlow()->calc(prev, next, flowFresh);
        reused->calc(prev, next, flowReused);
        expectSameFlow(flowFresh, flowReused, "overwritten buffers, pair " + std::to_string(i));
    }
}

// Two frame buffers swapping roles, so this call's first image sits in the buffer the
// previous call was handed as its second, refilled in between. A key on the data pointer,
// or a retained reference instead of a copy, matches here while the pixels do not.
TEST_P(DenseOpticalFlow_Farneback, ReusedInstanceMatchesFreshWhenBuffersAlternate)
{
    const Size size = GetParam();
    const std::vector<Mat> frames = makeSequence(size, 8);

    Mat buf[2] = { Mat(size, CV_8U), Mat(size, CV_8U) };
    Ptr<FarnebackOpticalFlow> reused = makeFlow();
    const uchar* heldAddress = NULL;

    for( size_t i = 0; i + 1 < frames.size(); i++ )
    {
        Mat& first = buf[i % 2];
        Mat& second = buf[(i + 1) % 2];

        Mat heldContent;
        if( i > 0 )
            first.copyTo(heldContent);

        // The stride of 3 keeps the two images of a call distinct, and this call's first
        // image distinct from the previous call's second.
        frames[i].copyTo(first);
        frames[(i + 3) % frames.size()].copyTo(second);

        // Without these the case could pass while testing nothing.
        if( i > 0 )
        {
            ASSERT_EQ(heldAddress, first.data) << "pair " << i << ": buffers must alternate";
            ASSERT_GT(countNonZero(heldContent != first), 0)
                << "pair " << i << ": the alternating buffer must be refilled";
        }
        heldAddress = second.data;

        Mat flowFresh, flowReused;
        makeFlow()->calc(first, second, flowFresh);
        reused->calc(first, second, flowReused);
        expectSameFlow(flowFresh, flowReused, "alternating buffers, pair " + std::to_string(i));
    }
}

INSTANTIATE_TEST_CASE_P(FullSet, DenseOpticalFlow_Farneback,
                        testing::Values(szODD, szQVGA));

// Pyramid geometries the parameterised sizes do not reach: a scale close to 1 goes deep,
// and the last three sizes truncate a requested depth of 8 to one or two levels.
TEST(DenseOpticalFlow_Farneback, PyramidDepthRangeKeepsResult)
{
    const struct { Size size; double pyrScale; int numLevels; } cases[] = {
        { Size(320, 240), 0.8, 5 },
        { Size(320, 240), 0.5, 5 },
        { Size(200, 200), 0.9, 6 },
        { Size(40, 34),   0.5, 8 },
        { Size(70, 70),   0.5, 8 },
        { Size(320, 61),  0.5, 8 },
    };

    for( size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++ )
    {
        const double pyrScale = cases[c].pyrScale;
        const int numLevels = cases[c].numLevels;
        const FlowFactory make = [pyrScale, numLevels]()
        {
            Ptr<FarnebackOpticalFlow> algo = FarnebackOpticalFlow::create();
            algo->setPyrScale(pyrScale);
            algo->setNumLevels(numLevels);
            algo->setWinSize(9);
            algo->setNumIters(2);
            return algo;
        };

        checkPairs(consecutivePairs(makeSequence(cases[c].size, 4)), make,
                   format("%dx%d at pyrScale %.1f, numLevels %d",
                          cases[c].size.width, cases[c].size.height, pyrScale, numLevels));
    }
}

// calc() asserts one channel and equal sizes but not a depth, so wider images reach it.
TEST(DenseOpticalFlow_Farneback, NonByteDepthKeepsResult)
{
    const int depths[] = { CV_16U, CV_32F };
    const std::vector<Mat> frames8u = makeSequence(Size(96, 72), 5);

    for( size_t d = 0; d < sizeof(depths) / sizeof(depths[0]); d++ )
    {
        std::vector<std::pair<Mat, Mat> > pairs;
        for( size_t i = 0; i + 1 < frames8u.size(); i++ )
        {
            Mat a, b;
            frames8u[i].convertTo(a, depths[d], 257.0);
            frames8u[i + 1].convertTo(b, depths[d], 257.0);
            pairs.push_back(std::make_pair(a, b));
        }
        checkPairs(pairs, makeFlow, format("depth %d", depths[d]));
    }
}

// A 16-bit difference confined to the far end of every row. A row comparison sized from
// the column count instead of the element size reads only the low half of each row, misses
// the difference and hands these frames the wrong expansion.
TEST(DenseOpticalFlow_Farneback, LateRowDifferenceIsDetected)
{
    const std::vector<Mat> frames = makeSequence(Size(96, 72), 3);

    Mat first, held, other;
    frames[0].convertTo(first, CV_16U, 257.0);
    frames[1].convertTo(held, CV_16U, 257.0);
    frames[2].convertTo(other, CV_16U, 257.0);

    Mat altered = held.clone();
    const Rect tail(held.cols / 2, 0, held.cols - held.cols / 2, held.rows);
    other(tail).copyTo(altered(tail));
    ASSERT_EQ(0, countNonZero(altered.colRange(0, held.cols / 2)
                              != held.colRange(0, held.cols / 2)));
    ASSERT_GT(countNonZero(altered != held), 0);

    std::vector<std::pair<Mat, Mat> > pairs;
    pairs.push_back(std::make_pair(first, held));
    pairs.push_back(std::make_pair(altered, other));
    checkPairs(pairs, makeFlow, "late row difference");
}

// Strided input: a whole sequence of ROIs into larger frames, and then the same pixels
// arriving continuous on one call and strided on the next, which is still a hit.
TEST(DenseOpticalFlow_Farneback, StridedInputKeepsResult)
{
    const Size size(96, 72);
    const std::vector<Mat> frames = makeSequence(Size(size.width + 19, size.height + 13), 4);
    const Rect roi(6, 4, size.width, size.height);

    std::vector<std::pair<Mat, Mat> > pairs;
    for( size_t i = 0; i + 1 < frames.size(); i++ )
    {
        Mat a = frames[i](roi), b = frames[i + 1](roi);
        ASSERT_FALSE(a.isContinuous());
        pairs.push_back(std::make_pair(a, b));
    }
    checkPairs(pairs, makeFlow, "strided sequence");

    const Mat firstStrided = frames[0](roi);
    const Mat heldStrided = frames[1](roi);
    const Mat nextStrided = frames[2](roi);
    const Mat heldContinuous = heldStrided.clone();
    ASSERT_TRUE(heldContinuous.isContinuous());

    Ptr<FarnebackOpticalFlow> reused = makeFlow();
    Mat flowFresh, flowReused;

    makeFlow()->calc(firstStrided, heldContinuous, flowFresh);
    reused->calc(firstStrided, heldContinuous, flowReused);
    expectSameFlow(flowFresh, flowReused, "continuity change, first call");

    makeFlow()->calc(heldStrided, nextStrided, flowFresh);
    reused->calc(heldStrided, nextStrided, flowReused);
    expectSameFlow(flowFresh, flowReused, "continuity change, second call");
}

// One image against itself, which answers zero and both finds and leaves an expansion of
// that image.
TEST(DenseOpticalFlow_Farneback, SelfPairKeepsResult)
{
    const std::vector<Mat> frames = makeSequence(Size(96, 72), 3);
    Ptr<FarnebackOpticalFlow> reused = makeFlow();

    const std::pair<Mat, Mat> pairs[] = {
        std::make_pair(frames[0], frames[1]),
        std::make_pair(frames[1], frames[1]),
        std::make_pair(frames[1], frames[2]),
        std::make_pair(frames[2], frames[2]),
    };

    for( size_t i = 0; i < sizeof(pairs) / sizeof(pairs[0]); i++ )
    {
        Mat flowFresh, flowReused;
        makeFlow()->calc(pairs[i].first, pairs[i].second, flowFresh);
        reused->calc(pairs[i].first, pairs[i].second, flowReused);
        const bool selfPair = pairs[i].first.data == pairs[i].second.data;
        expectSameFlow(flowFresh, flowReused, "self pair, pair " + std::to_string(i), selfPair);
    }
}

// The depth a call arrives at, changed with a warm cache: at 320x240 numLevels 1 keeps two
// sets of planes and numLevels 3 asks for three, so the second call indexes past what the
// first kept. Only the length of the held vector notices.
TEST(DenseOpticalFlow_Farneback, EffectiveDepthChangeBetweenCallsKeepsResult)
{
    const std::vector<Mat> frames = makeSequence(Size(320, 240), 3);

    Ptr<FarnebackOpticalFlow> reused = makeFlow();
    reused->setNumLevels(1);
    Ptr<FarnebackOpticalFlow> shallow = makeFlow();
    shallow->setNumLevels(1);

    Mat flowFresh, flowReused;
    shallow->calc(frames[0], frames[1], flowFresh);
    reused->calc(frames[0], frames[1], flowReused);
    expectSameFlow(flowFresh, flowReused, "one level, first pair");

    reused->setNumLevels(3);
    Ptr<FarnebackOpticalFlow> deep = makeFlow();
    deep->calc(frames[1], frames[2], flowFresh);
    reused->calc(frames[1], frames[2], flowReused);
    expectSameFlow(flowFresh, flowReused, "three levels after one level");

    // The depth does move the answer, so a call served by the shallower expansion could
    // not have passed above.
    Mat flowShallow;
    shallow->calc(frames[1], frames[2], flowShallow);
    ASSERT_GT(countNonZero(flowShallow.reshape(1) != flowFresh.reshape(1)), 0);
}

// pyrScale changed with a warm cache between two values that truncate to the same number
// of levels, 0.5 and 0.45 at this size, so the length of the held vector cannot tell them
// apart and pyrScale is the only term that can.
TEST(DenseOpticalFlow_Farneback, PyrScaleChangeBetweenCallsKeepsResult)
{
    const std::vector<Mat> frames = makeSequence(Size(320, 240), 3);

    Ptr<FarnebackOpticalFlow> reused = makeFlow();
    Mat flowFresh, flowReused;
    makeFlow()->calc(frames[0], frames[1], flowFresh);
    reused->calc(frames[0], frames[1], flowReused);
    expectSameFlow(flowFresh, flowReused, "pyrScale 0.5, first pair");

    reused->setPyrScale(0.45);
    Ptr<FarnebackOpticalFlow> rescaled = makeFlow();
    rescaled->setPyrScale(0.45);
    Mat flowRescaled;
    rescaled->calc(frames[1], frames[2], flowRescaled);
    reused->calc(frames[1], frames[2], flowReused);
    expectSameFlow(flowRescaled, flowReused, "pyrScale 0.45 after 0.5");

    makeFlow()->calc(frames[1], frames[2], flowFresh);
    ASSERT_GT(countNonZero(flowRescaled.reshape(1) != flowFresh.reshape(1)), 0);
}

// A size change with a warm cache, both ways round: the held planes are sized for the
// previous frame, so carrying them over would give the update step short rows to read.
TEST(DenseOpticalFlow_Farneback, FrameSizeChangeBetweenCallsKeepsResult)
{
    const Size sizes[] = { Size(320, 240), Size(96, 72), Size(127, 61), Size(320, 240) };
    const int count = (int)(sizeof(sizes) / sizeof(sizes[0]));

    Ptr<FarnebackOpticalFlow> reused = makeFlow();
    for( int i = 0; i < count; i++ )
    {
        // Three frames per size, so the second call at each size chains.
        const std::vector<Mat> frames = makeSequence(sizes[i], 3);
        for( size_t j = 0; j + 1 < frames.size(); j++ )
        {
            Mat flowFresh, flowReused;
            makeFlow()->calc(frames[j], frames[j + 1], flowFresh);
            reused->calc(frames[j], frames[j + 1], flowReused);
            expectSameFlow(flowFresh, flowReused,
                           format("size change to %dx%d, pair %d",
                                  sizes[i].width, sizes[i].height, (int)j));
        }
    }
}

// poly_n and poly_sigma size the expansion kernel and the flags pick the update step.
// calc() constrains none of them to the documented values, so the range it accepts has to
// keep working; 160x120 gives a pyramid rather than a single level.
TEST(DenseOpticalFlow_Farneback, ExpansionParameterRangeKeepsResult)
{
    const std::vector<Mat> frames = makeSequence(Size(160, 120), 4);
    const int polyNs[] = { 5, 6, 7 };
    const int flagValues[] = { 0, OPTFLOW_FARNEBACK_GAUSSIAN };

    for( size_t n = 0; n < sizeof(polyNs) / sizeof(polyNs[0]); n++ )
    {
        for( size_t f = 0; f < sizeof(flagValues) / sizeof(flagValues[0]); f++ )
        {
            const int polyN = polyNs[n], flags = flagValues[f];
            const double polySigma = polyN <= 5 ? 1.1 : 1.5;
            const FlowFactory make = [polyN, polySigma, flags]()
            {
                Ptr<FarnebackOpticalFlow> algo = makeFlow();
                algo->setPolyN(polyN);
                algo->setPolySigma(polySigma);
                algo->setFlags(flags);
                return algo;
            };

            checkPairs(consecutivePairs(frames), make,
                       format("poly_n %d, flags %d", polyN, flags));
        }
    }
}

// Every setter, changed one at a time between two chaining calls. Those that feed the
// expansion have to invalidate what is held, the rest may keep it, and either way the
// answer cannot move. polyN and polySigma get a row each as well as a row together: they
// are the key terms that leave no trace in the shape or type of a held plane.
TEST(DenseOpticalFlow_Farneback, ParameterChangeBetweenCallsKeepsResult)
{
    const std::vector<Mat> frames = makeSequence(Size(96, 72), 3);

    typedef std::function<void(const Ptr<FarnebackOpticalFlow>&)> Tweak;
    std::vector<std::pair<std::string, Tweak> > tweaks;
    tweaks.push_back(std::make_pair("numLevels",
        [](const Ptr<FarnebackOpticalFlow>& f) { f->setNumLevels(2); }));
    tweaks.push_back(std::make_pair("pyrScale",
        [](const Ptr<FarnebackOpticalFlow>& f) { f->setPyrScale(0.4); }));
    tweaks.push_back(std::make_pair("polyN",
        [](const Ptr<FarnebackOpticalFlow>& f) { f->setPolyN(7); }));
    tweaks.push_back(std::make_pair("polyN and polySigma",
        [](const Ptr<FarnebackOpticalFlow>& f) { f->setPolyN(7); f->setPolySigma(1.5); }));
    tweaks.push_back(std::make_pair("polySigma",
        [](const Ptr<FarnebackOpticalFlow>& f) { f->setPolySigma(1.3); }));
    tweaks.push_back(std::make_pair("winSize",
        [](const Ptr<FarnebackOpticalFlow>& f) { f->setWinSize(13); }));
    tweaks.push_back(std::make_pair("numIters",
        [](const Ptr<FarnebackOpticalFlow>& f) { f->setNumIters(5); }));
    tweaks.push_back(std::make_pair("flags",
        [](const Ptr<FarnebackOpticalFlow>& f) { f->setFlags(OPTFLOW_FARNEBACK_GAUSSIAN); }));
    tweaks.push_back(std::make_pair("fastPyramids",
        [](const Ptr<FarnebackOpticalFlow>& f) { f->setFastPyramids(true); }));

    for( size_t t = 0; t < tweaks.size(); t++ )
    {
        Ptr<FarnebackOpticalFlow> reused = makeFlow();
        Mat ignored;
        reused->calc(frames[0], frames[1], ignored);
        tweaks[t].second(reused);

        Ptr<FarnebackOpticalFlow> fresh = makeFlow();
        tweaks[t].second(fresh);

        Mat flowFresh, flowReused;
        fresh->calc(frames[1], frames[2], flowFresh);
        reused->calc(frames[1], frames[2], flowReused);
        expectSameFlow(flowFresh, flowReused, "after changing " + tweaks[t].first);
    }
}

// OPTFLOW_USE_INITIAL_FLOW feeds each call the previous flow field, so both instances have
// to be fed the same one to stay comparable.
TEST(DenseOpticalFlow_Farneback, InitialFlowSequenceKeepsResult)
{
    const std::vector<Mat> frames = makeSequence(szQVGA, 5);

    Ptr<FarnebackOpticalFlow> reused = makeFlow();
    reused->setFlags(OPTFLOW_USE_INITIAL_FLOW);

    Mat carried = Mat::zeros(frames[0].size(), CV_32FC2);
    for( size_t i = 0; i + 1 < frames.size(); i++ )
    {
        Mat flowFresh = carried.clone(), flowReused = carried.clone();

        Ptr<FarnebackOpticalFlow> fresh = makeFlow();
        fresh->setFlags(OPTFLOW_USE_INITIAL_FLOW);
        fresh->calc(frames[i], frames[i + 1], flowFresh);
        reused->calc(frames[i], frames[i + 1], flowReused);

        expectSameFlow(flowFresh, flowReused, "initial flow, pair " + std::to_string(i));
        carried = flowFresh;
    }
}

// collectGarbage() drops whatever is held; the next call must still be right.
TEST(DenseOpticalFlow_Farneback, CollectGarbageBetweenCallsKeepsResult)
{
    const std::vector<Mat> frames = makeSequence(Size(96, 72), 5);
    Ptr<FarnebackOpticalFlow> reused = makeFlow();

    for( size_t i = 0; i + 1 < frames.size(); i++ )
    {
        Mat flowFresh, flowReused;
        makeFlow()->calc(frames[i], frames[i + 1], flowFresh);
        reused->calc(frames[i], frames[i + 1], flowReused);
        expectSameFlow(flowFresh, flowReused, "after collectGarbage, pair " + std::to_string(i));
        if( i % 2 == 0 )
            reused->collectGarbage();
    }
}

#if !defined(OPENCV_DISABLE_THREAD_SUPPORT)

// Several threads driving one instance, each over its own sequence. Whether a thread finds
// the held expansion is a race; what it gets back is not. The small size keeps calls short
// and publishes frequent, the larger one has a pyramid deep enough for a torn snapshot to
// have levels to be torn between.
static void runConcurrentCalls(Size size, int threads, int pairs)
{
    std::vector<std::vector<Mat> > frames(threads);
    std::vector<std::vector<Mat> > expected(threads);
    for( int t = 0; t < threads; t++ )
    {
        frames[t] = makeSequence(size, pairs + 1, 0x900 + t * 13);
        for( int i = 0; i < pairs; i++ )
        {
            Mat flow;
            makeFlow()->calc(frames[t][i], frames[t][i + 1], flow);
            expected[t].push_back(flow);
        }
    }

    Ptr<FarnebackOpticalFlow> shared = makeFlow();
    std::vector<std::vector<Mat> > actual(threads);
    std::vector<std::thread> workers;
    for( int t = 0; t < threads; t++ )
    {
        actual[t].resize(pairs);
        workers.push_back(std::thread([&, t]()
        {
            for( int i = 0; i < pairs; i++ )
                shared->calc(frames[t][i], frames[t][i + 1], actual[t][i]);
        }));
    }
    for( size_t t = 0; t < workers.size(); t++ )
        workers[t].join();

    for( int t = 0; t < threads; t++ )
        for( int i = 0; i < pairs; i++ )
            expectSameFlow(expected[t][i], actual[t][i],
                           format("%dx%d, concurrent thread %d, pair %d",
                                  size.width, size.height, t, i));
}

TEST(DenseOpticalFlow_Farneback, ConcurrentCallsOnOneInstanceKeepResult)
{
    runConcurrentCalls(Size(64, 48), 12, 20);
    runConcurrentCalls(Size(160, 120), 8, 12);
}

#endif

// The free function creates an instance per call and keeps nothing.
TEST(DenseOpticalFlow_Farneback, FreeFunctionMatchesFreshInstance)
{
    const std::vector<Mat> frames = makeSequence(Size(96, 72), 4);
    for( size_t i = 0; i + 1 < frames.size(); i++ )
    {
        Mat flowFunc, flowInstance;
        calcOpticalFlowFarneback(frames[i], frames[i + 1], flowFunc, 0.5, 3, 9, 3, 5, 1.1, 0);
        makeFlow()->calc(frames[i], frames[i + 1], flowInstance);
        expectSameFlow(flowInstance, flowFunc, "free function, pair " + std::to_string(i));
    }
}

}} // namespace
