// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020-2022 Intel Corporation

#include "gapi_ocv_stateful_kernel_test_utils.hpp"
#include <opencv2/gapi/cpu/core.hpp>
#include <opencv2/gapi/streaming/cap.hpp>

#include <opencv2/core.hpp>
#include <opencv2/core/cvstd.hpp>
#ifdef HAVE_OPENCV_VIDEO
#include <opencv2/video.hpp>
#endif

#include <memory> // required by std::shared_ptr

namespace opencv_test
{
    struct BackSubStateParams
    {
        std::string method;
    };

    struct CountStateSetupsParams
    {
        std::shared_ptr<int> pSetupsCount;
    };
} // namespace opencv_test

namespace cv
{
    namespace detail
    {
        template<> struct CompileArgTag<opencv_test::BackSubStateParams>
        {
            static const char* tag()
            {
                return "org.opencv.test.background_substractor_state_params";
            }
        };

        template<> struct CompileArgTag<opencv_test::CountStateSetupsParams>
        {
            static const char* tag()
            {
                return "org.opencv.test.count_state_setups_params";
            }
        };
    } // namespace detail
} // namespace cv

namespace opencv_test
{
//TODO: test OT, Background Subtractor, Kalman with 3rd version of API
//----------------------------------------------- Simple tests ------------------------------------------------
namespace
{
    G_TYPED_KERNEL(GCountCalls, <cv::GOpaque<int>(GMat)>, "org.opencv.test.count_calls")
    {
        static GOpaqueDesc outMeta(GMatDesc /* in */) { return empty_gopaque_desc(); }
    };

    GAPI_OCV_KERNEL_ST(GOCVCountCalls, GCountCalls, int)
    {
        static void setup(const cv::GMatDesc &/* in */, std::shared_ptr<int> &state)
        {
            state.reset(new int{  });
        }

        static void run(const cv::Mat &/* in */, int &out, int& state)
        {
            out = ++state;
        }
    };

    G_TYPED_KERNEL(GIsStateUpToDate, <cv::GOpaque<bool>(GMat)>,
                   "org.opencv.test.is_state_up-to-date")
    {
        static GOpaqueDesc outMeta(GMatDesc /* in */) { return empty_gopaque_desc(); }
    };

    GAPI_OCV_KERNEL_ST(GOCVIsStateUpToDate, GIsStateUpToDate, cv::Size)
    {
        static void setup(const cv::GMatDesc &in, std::shared_ptr<cv::Size> &state)
        {
            state.reset(new cv::Size(in.size));
        }

        static void run(const cv::Mat &in , bool &out, cv::Size& state)
        {
            out = in.size() == state;
        }
    };

    G_TYPED_KERNEL(GStInvalidResize, <GMat(GMat,Size,double,double,int)>,
                   "org.opencv.test.st_invalid_resize")
    {
         static GMatDesc outMeta(GMatDesc in, Size, double, double, int) { return in; }
    };

    GAPI_OCV_KERNEL_ST(GOCVStInvalidResize, GStInvalidResize, int)
    {
        static void setup(const cv::GMatDesc, cv::Size, double, double, int,
                          std::shared_ptr<int> &/* state */)
        {  }

        static void run(const cv::Mat& in, cv::Size sz, double fx, double fy, int interp,
                        cv::Mat &out, int& /* state */)
        {
            cv::resize(in, out, sz, fx, fy, interp);
        }
    };

    G_TYPED_KERNEL(GBackSub, <GMat(GMat)>, "org.opencv.test.background_substractor")
    {
         static GMatDesc outMeta(GMatDesc in) { return in.withType(CV_8U, 1); }
    };
#ifdef HAVE_OPENCV_VIDEO
    GAPI_OCV_KERNEL_ST(GOCVBackSub, GBackSub, cv::BackgroundSubtractor)
    {
        static void setup(const cv::GMatDesc &/* desc */,
                          std::shared_ptr<BackgroundSubtractor> &state,
                          const cv::GCompileArgs &compileArgs)
        {
            auto sbParams = cv::gapi::getCompileArg<BackSubStateParams>(compileArgs)
                                .value_or(BackSubStateParams { });

            if (sbParams.method == "knn")
                state = createBackgroundSubtractorKNN();
            else if (sbParams.method == "mog2")
                state = createBackgroundSubtractorMOG2();

            GAPI_Assert(state);
        }

        static void run(const cv::Mat& in, cv::Mat &out, BackgroundSubtractor& state)
        {
            state.apply(in, out, -1);
        }
    };
#endif

    G_TYPED_KERNEL(GCountStateSetups, <cv::GOpaque<bool>(GMat)>,
                   "org.opencv.test.count_state_setups")
    {
        static GOpaqueDesc outMeta(GMatDesc /* in */) { return empty_gopaque_desc(); }
    };

    GAPI_OCV_KERNEL_ST(GOCVCountStateSetups, GCountStateSetups, int)
    {
        static void setup(const cv::GMatDesc &, std::shared_ptr<int> &,
                          const cv::GCompileArgs &compileArgs)
        {
            auto params = cv::gapi::getCompileArg<CountStateSetupsParams>(compileArgs)
                .value_or(CountStateSetupsParams { });
            if (params.pSetupsCount != nullptr) {
                (*params.pSetupsCount)++;
            }
        }

        static void run(const cv::Mat & , bool &out, int &)
        {
            out = true;
        }
    };
};

TEST(StatefulKernel, StateInitOnceInRegularMode)
{
    cv::GMat in;
    cv::GOpaque<bool> out = GCountStateSetups::on(in);
    cv::GComputation c(cv::GIn(in), cv::GOut(out));

    // Input mat:
    cv::Mat inputData(1080, 1920, CV_8UC1);
    cv::randu(inputData, cv::Scalar::all(1), cv::Scalar::all(128));

    // variable to update when state is initialized in the kernel
    CountStateSetupsParams params;
    params.pSetupsCount.reset(new int(0));

    // Testing for 100 frames
    bool result { };
    for (int i = 0; i < 100; ++i) {
        c.apply(cv::gin(inputData), cv::gout(result),
                cv::compile_args(cv::gapi::kernels<GOCVCountStateSetups>(), params));
        EXPECT_TRUE(result);
        EXPECT_TRUE(params.pSetupsCount != nullptr);
        EXPECT_EQ(1, *params.pSetupsCount);
    }
};

struct StateInitOnce : public ::testing::TestWithParam<bool>{};
TEST_P(StateInitOnce, StreamingCompiledWithMeta)
{
    bool compileWithMeta = GetParam();
    cv::GMat in;
    cv::GOpaque<bool> out = GCountStateSetups::on(in);
    cv::GComputation c(cv::GIn(in), cv::GOut(out));

    // Input mat:
    cv::Mat inputData(1080, 1920, CV_8UC1);
    cv::randu(inputData, cv::Scalar::all(1), cv::Scalar::all(128));

    // variable to update when state is initialized in the kernel
    CountStateSetupsParams params;
    params.pSetupsCount.reset(new int(0));

    // Compilation & testing
    auto ccomp = (compileWithMeta)
        ? c.compileStreaming(cv::descr_of(inputData),
              cv::compile_args(cv::gapi::kernels<GOCVCountStateSetups>(),
                               params))
        : c.compileStreaming(
              cv::compile_args(cv::gapi::kernels<GOCVCountStateSetups>(),
                               params));

    ccomp.setSource(cv::gin(inputData));

    ccomp.start();
    EXPECT_TRUE(ccomp.running());

    int counter { };
    bool result;
    // Process mat 100 times
    while (ccomp.pull(cv::gout(result)) && (counter++ < 100)) {
        EXPECT_TRUE(params.pSetupsCount != nullptr);
        EXPECT_EQ(1, *params.pSetupsCount);
    }

    ccomp.stop();
    EXPECT_FALSE(ccomp.running());
}

INSTANTIATE_TEST_CASE_P(StatefulKernel, StateInitOnce, ::testing::Bool());

TEST(StatefulKernel, StateIsMutableInRuntime)
{
    constexpr int expectedCallsCount = 10;

    cv::Mat dummyIn { 1, 1, CV_8UC1 };
    int actualCallsCount = 0;

    // Declaration of G-API expression
    GMat in;
    GOpaque<int> out = GCountCalls::on(in);
    cv::GComputation comp(cv::GIn(in), cv::GOut(out));

    const auto pkg = cv::gapi::kernels<GOCVCountCalls>();

    // Compilation of G-API expression
    auto callsCounter = comp.compile(cv::descr_of(dummyIn), cv::compile_args(pkg));

    // Simulating video stream: call GCompiled multiple times
    for (int i = 0; i < expectedCallsCount; i++)
    {
        callsCounter(cv::gin(dummyIn), cv::gout(actualCallsCount));
        EXPECT_EQ(i + 1, actualCallsCount);
    }

    // End of "video stream"
    EXPECT_EQ(expectedCallsCount, actualCallsCount);

    // User asks G-API to prepare for a new stream
    callsCounter.prepareForNewStream();
    callsCounter(cv::gin(dummyIn), cv::gout(actualCallsCount));
    EXPECT_EQ(1, actualCallsCount);

}

TEST(StateIsResetOnNewStream, RegularMode)
{
    cv::GMat in;
    cv::GOpaque<bool> out = GCountStateSetups::on(in);
    cv::GComputation c(cv::GIn(in), cv::GOut(out));

    // Input mat:
    cv::Mat inputData(1080, 1920, CV_8UC1);
    cv::randu(inputData, cv::Scalar::all(1), cv::Scalar::all(128));

    // variable to update when state is initialized in the kernel
    CountStateSetupsParams params;
    params.pSetupsCount.reset(new int(0));

    auto setupsCounter = c.compile(cv::descr_of(inputData),
                                   cv::compile_args(cv::gapi::kernels<GOCVCountStateSetups>(),
                                                    params));

    bool result { };
    for (int i = 0; i < 2; ++i) {
        setupsCounter(cv::gin(inputData), cv::gout(result));
        EXPECT_TRUE(params.pSetupsCount != nullptr);
        EXPECT_EQ(1, *params.pSetupsCount);
    }

    EXPECT_TRUE(params.pSetupsCount != nullptr);
    EXPECT_EQ(1, *params.pSetupsCount);
    setupsCounter.prepareForNewStream();

    for (int i = 0; i < 2; ++i) {
        setupsCounter(cv::gin(inputData), cv::gout(result));
        EXPECT_TRUE(params.pSetupsCount != nullptr);
        EXPECT_EQ(2, *params.pSetupsCount);
    }
}

TEST(StateIsResetOnNewStream, StreamingMode)
{
    cv::GMat in;
    cv::GOpaque<bool> out = GIsStateUpToDate::on(in);
    cv::GComputation c(cv::GIn(in), cv::GOut(out));

    const auto pkg = cv::gapi::kernels<GOCVIsStateUpToDate>();

    // Compilation & testing
    auto ccomp = c.compileStreaming(cv::compile_args(pkg));

    auto path = findDataFile("cv/video/768x576.avi");
    try {
        ccomp.setSource(gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(path));
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }
    ccomp.start();
    EXPECT_TRUE(ccomp.running());

    // Process the full video
    bool isStateUpToDate = false;
    while (ccomp.pull(cv::gout(isStateUpToDate))) {
        EXPECT_TRUE(isStateUpToDate);
    }
    EXPECT_FALSE(ccomp.running());

    path = findDataFile("cv/video/1920x1080.avi");
    try {
        ccomp.setSource(gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(path));
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }
    ccomp.start();
    EXPECT_TRUE(ccomp.running());

    while (ccomp.pull(cv::gout(isStateUpToDate))) {
        EXPECT_TRUE(isStateUpToDate);
    }
    EXPECT_FALSE(ccomp.running());
}

TEST(StatefulKernel, InvalidReallocatingKernel)
{
    cv::GMat in, out;
    cv::Mat in_mat(500, 500, CV_8UC1), out_mat;
    out = GStInvalidResize::on(in, cv::Size(300, 300), 0.0, 0.0, cv::INTER_LINEAR);

    const auto pkg = cv::gapi::kernels<GOCVStInvalidResize>();
    cv::GComputation comp(cv::GIn(in), cv::GOut(out));

    EXPECT_THROW(comp.apply(in_mat, out_mat, cv::compile_args(pkg)), std::logic_error);
}

#ifdef HAVE_OPENCV_VIDEO
namespace
{
    void compareBackSubResults(const cv::Mat &actual, const cv::Mat &expected,
                               const int diffPercent)
    {
        GAPI_Assert(actual.size() == expected.size());
        int allowedNumDiffPixels = actual.size().area() * diffPercent / 100;

        cv::Mat diff;
        cv::absdiff(actual, expected, diff);

        cv::Mat hist(256, 1, CV_32FC1, cv::Scalar(0));
        const float range[] { 0, 256 };
        const float *histRange { range };
        calcHist(&diff, 1, 0, Mat(), hist, 1, &hist.rows, &histRange, true, false);
        for (int i = 2; i < hist.rows; ++i)
        {
            hist.at<float>(i) += hist.at<float>(i - 1);
        }

        int numDiffPixels = static_cast<int>(hist.at<float>(255));

        EXPECT_GT(allowedNumDiffPixels, numDiffPixels);
    }
} // anonymous namespace

TEST(StatefulKernel, StateIsInitViaCompArgs)
{
    cv::Mat frame(1080, 1920, CV_8UC3),
            gapiForeground,
            ocvForeground;

    cv::randu(frame, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    // G-API code
    cv::GMat in;
    cv::GMat out = GBackSub::on(in);
    cv::GComputation c(cv::GIn(in), cv::GOut(out));

    const auto pkg = cv::gapi::kernels<GOCVBackSub>();

    auto gapiBackSub = c.compile(cv::descr_of(frame),
                                 cv::compile_args(pkg, BackSubStateParams { "knn" }));

    gapiBackSub(cv::gin(frame), cv::gout(gapiForeground));

    // OpenCV code
    auto pOcvBackSub = createBackgroundSubtractorKNN();
    pOcvBackSub->apply(frame, ocvForeground);

    // Comparison
    // Allowing 1% difference of all pixels between G-API and OpenCV results
    compareBackSubResults(gapiForeground, ocvForeground, 1);

    // Additionally, test the case where state is reset
    gapiBackSub.prepareForNewStream();
    gapiBackSub(cv::gin(frame), cv::gout(gapiForeground));
    pOcvBackSub->apply(frame, ocvForeground);
    compareBackSubResults(gapiForeground, ocvForeground, 1);
}
#endif

#ifdef HAVE_OPENCV_VIDEO
namespace
{
    void testBackSubInStreaming(cv::GStreamingCompiled gapiBackSub, const int diffPercent)
    {
        cv::Mat frame,
                gapiForeground,
                ocvForeground;

        gapiBackSub.start();
        EXPECT_TRUE(gapiBackSub.running());

        // OpenCV reference substractor
        auto pOCVBackSub = createBackgroundSubtractorKNN();

        // Comparison of G-API and OpenCV substractors
        std::size_t frames = 0u;
        while (gapiBackSub.pull(cv::gout(frame, gapiForeground))) {
            pOCVBackSub->apply(frame, ocvForeground, -1);

            compareBackSubResults(gapiForeground, ocvForeground, diffPercent);

            frames++;
        }
        EXPECT_LT(0u, frames);
        EXPECT_FALSE(gapiBackSub.running());
    }
} // anonymous namespace

TEST(StatefulKernel, StateIsInitViaCompArgsInStreaming)
{
    // G-API graph declaration
    cv::GMat in;
    cv::GMat out = GBackSub::on(in);
    // Preserving 'in' in output to have possibility to compare with OpenCV reference
    cv::GComputation c(cv::GIn(in), cv::GOut(cv::gapi::copy(in), out));

    // G-API compilation of graph for streaming mode
    const auto pkg = cv::gapi::kernels<GOCVBackSub>();
    auto gapiBackSub = c.compileStreaming(
                           cv::compile_args(pkg, BackSubStateParams { "knn" }));

    // Testing G-API Background Substractor in streaming mode
    auto path = findDataFile("cv/video/768x576.avi");
    try {
        gapiBackSub.setSource(gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(path));
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }
    // Allowing 1% difference of all pixels between G-API and reference OpenCV results
    testBackSubInStreaming(gapiBackSub, 1);

    path = findDataFile("cv/video/1920x1080.avi");
    try {
        // Additionally, test the case when the new stream happens
        gapiBackSub.setSource(gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(path));
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }
    // Allowing 5% difference of all pixels between G-API and reference OpenCV results
    testBackSubInStreaming(gapiBackSub, 5);
}

TEST(StatefulKernel, StateIsChangedViaCompArgsOnReshape)
{
    cv::GMat in;
    cv::GComputation comp(in, GBackSub::on(in));

    const auto pkg = cv::gapi::kernels<GOCVBackSub>();

    // OpenCV reference substractor
    auto pOCVBackSubKNN = createBackgroundSubtractorKNN();
    auto pOCVBackSubMOG2 = createBackgroundSubtractorMOG2();

    const auto run = [&](const std::string& videoPath, const std::string& method) {
        auto path = findDataFile(videoPath);
        cv::gapi::wip::IStreamSource::Ptr source;
        try {
            source = gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(path);
        } catch(...) {
            throw SkipTestException("Video file can not be opened");
        }
        cv::Mat inMat, gapiForeground, ocvForeground;

        for (int i = 0; i < 10; i++) {
            cv::gapi::wip::Data inData;
            source->pull(inData);
            inMat = cv::util::get<cv::Mat>(inData);
            comp.apply(inMat, gapiForeground,
                       cv::compile_args(pkg, BackSubStateParams{method}));

            if (method == "knn") {
                pOCVBackSubKNN->apply(inMat, ocvForeground, -1);
                // Allowing 1% difference among all pixels
                compareBackSubResults(gapiForeground, ocvForeground, 1);
            } else if (method == "mog2") {
                pOCVBackSubMOG2->apply(inMat, ocvForeground, -1);
                compareBackSubResults(gapiForeground, ocvForeground, 5);
            } else {
                CV_Assert(false && "Unknown BackSub method");
            }
        }
    };

    run("cv/video/768x576.avi", "knn");
    run("cv/video/1920x1080.avi", "mog2");
}

TEST(StatefulKernel, StateIsResetOnceOnReshapeInStreaming)
{
    cv::GMat in;
    cv::GOpaque<bool> out = GCountStateSetups::on(in);
    cv::GComputation c(cv::GIn(in), cv::GOut(out));

    // variable to update when state is initialized in the kernel
    CountStateSetupsParams params;
    params.pSetupsCount.reset(new int(0));

    auto ccomp = c.compileStreaming(
        cv::compile_args(cv::gapi::kernels<GOCVCountStateSetups>(), params));

    auto run = [&ccomp, &params](const std::string& videoPath, int expectedSetupsCount) {
        auto path = findDataFile(videoPath);
        try {
            ccomp.setSource<cv::gapi::wip::GCaptureSource>(path);
        } catch(...) {
            throw SkipTestException("Video file can not be opened");
        }
        ccomp.start();

        int frames = 0;
        bool result = false;
        while (ccomp.pull(cv::gout(result)) && (frames++ < 10)) {
            EXPECT_TRUE(result);
            EXPECT_TRUE(params.pSetupsCount != nullptr);
            EXPECT_EQ(expectedSetupsCount, *params.pSetupsCount);
        }
        ccomp.stop();
    };

    run("cv/video/768x576.avi", 1);
    // FIXME: it should be 2, not 3 for expectedSetupsCount here.
    // With current implemention both GCPUExecutable reshape() and
    // handleNewStream() call setupKernelStates()
    run("cv/video/1920x1080.avi", 3);
}
#endif

TEST(StatefulKernel, StateIsAutoResetOnReshape)
{
    cv::GMat in;
    cv::GOpaque<bool> up_to_date = GIsStateUpToDate::on(in);
    cv::GOpaque<int>  calls_count = GCountCalls::on(in);
    cv::GComputation comp(cv::GIn(in), cv::GOut(up_to_date, calls_count));

    auto run = [&comp](const cv::Mat& in_mat) {
        const auto pkg = cv::gapi::kernels<GOCVIsStateUpToDate, GOCVCountCalls>();
        bool stateIsUpToDate = false;
        int callsCount = 0;
        for (int i = 0; i < 3; i++) {
            comp.apply(cv::gin(in_mat), cv::gout(stateIsUpToDate, callsCount),
                       cv::compile_args(pkg));
            EXPECT_TRUE(stateIsUpToDate);
            EXPECT_EQ(i+1, callsCount);
        }
    };

    cv::Mat in_mat1(32, 32, CV_8UC1);
    run(in_mat1);

    cv::Mat in_mat2(16, 16, CV_8UC1);
    run(in_mat2);
}

//-------------------------------------------------------------------------------------------------------------


//------------------------------------------- Typed tests on setup() ------------------------------------------
namespace
{
template<typename Tuple>
struct SetupStateTypedTest : public ::testing::Test
{
    using StateT = typename std::tuple_element<0, Tuple>::type;
    using SetupT = typename std::tuple_element<1, Tuple>::type;

    G_TYPED_KERNEL(GReturnState, <cv::GOpaque<StateT>(GMat)>, "org.opencv.test.return_state")
    {
        static GOpaqueDesc outMeta(GMatDesc /* in */) { return empty_gopaque_desc(); }
    };

    GAPI_OCV_KERNEL_ST(GOCVReturnState, GReturnState, StateT)
    {
        static void setup(const cv::GMatDesc &/* in */, std::shared_ptr<StateT> &state)
        {
            // Don't use input cv::GMatDesc intentionally
            state.reset(new StateT(SetupT::value()));
        }

        static void run(const cv::Mat &/* in */, StateT &out, StateT& state)
        {
            out = state;
        }
    };
};

TYPED_TEST_CASE_P(SetupStateTypedTest);
} // namespace


TYPED_TEST_P(SetupStateTypedTest, ReturnInitializedState)
{
    using StateType = typename TestFixture::StateT;
    using SetupType = typename TestFixture::SetupT;

    cv::Mat dummyIn { 1, 1, CV_8UC1 };
    StateType retState { };

    GMat in;
    auto out = TestFixture::GReturnState::on(in);
    cv::GComputation comp(cv::GIn(in), cv::GOut(out));

    const auto pkg = cv::gapi::kernels<typename TestFixture::GOCVReturnState>();
    comp.apply(cv::gin(dummyIn), cv::gout(retState), cv::compile_args(pkg));

    EXPECT_EQ(SetupType::value(), retState);
}

REGISTER_TYPED_TEST_CASE_P(SetupStateTypedTest,
                           ReturnInitializedState);


DEFINE_INITIALIZER(CharValue, char, 'z');
DEFINE_INITIALIZER(IntValue, int, 7);
DEFINE_INITIALIZER(FloatValue, float, 42.f);
DEFINE_INITIALIZER(UcharPtrValue, uchar*, nullptr);
namespace
{
using Std3IntArray = std::array<int, 3>;
}
DEFINE_INITIALIZER(StdArrayValue, Std3IntArray, { 1, 2, 3 });
DEFINE_INITIALIZER(UserValue, UserStruct, { 5, 7.f });

using TypesToVerify = ::testing::Types<std::tuple<char, CharValue>,
                                       std::tuple<int, IntValue>,
                                       std::tuple<float, FloatValue>,
                                       std::tuple<uchar*, UcharPtrValue>,
                                       std::tuple<std::array<int, 3>, StdArrayValue>,
                                       std::tuple<UserStruct, UserValue>>;

INSTANTIATE_TYPED_TEST_CASE_P(SetupStateTypedInst, SetupStateTypedTest, TypesToVerify);
//-------------------------------------------------------------------------------------------------------------

} // opencv_test
