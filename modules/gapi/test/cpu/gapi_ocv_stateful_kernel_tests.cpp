// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "gapi_ocv_stateful_kernel_test_utils.hpp"
#include <opencv2/gapi/cpu/core.hpp>
#include <opencv2/gapi/streaming/cap.hpp>

#include <opencv2/core.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/video.hpp>


namespace opencv_test
{
    struct BackSubStateParams
    {
        std::string method;
    };
}

namespace cv
{
    namespace detail
    {
        template<> struct CompileArgTag<opencv_test::BackSubStateParams>
        {
            static const char* tag()
            {
                return "org.opencv.test..background_substractor_state_params";
            }
        };
    }
}

namespace opencv_test
{
//TODO: test OT, Background Subtractor, Kalman with 3rd version of API
//----------------------------------------------- Simple tests ------------------------------------------------
namespace
{
    inline void initTestDataPath()
    {
#ifndef WINRT
        static bool initialized = false;
        if (!initialized)
        {
            // Since G-API has no own test data (yet), it is taken from the common space
            const char* testDataPath = getenv("OPENCV_TEST_DATA_PATH");
            GAPI_Assert(testDataPath != nullptr);

            cvtest::addDataSearchPath(testDataPath);
            initialized = true;
        }
#endif // WINRT
    }

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
};

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

TEST(StatefulKernel, StateIsAutoResetForNewStream)
{
    initTestDataPath();

    cv::GMat in;
    cv::GOpaque<bool> out = GIsStateUpToDate::on(in);
    cv::GComputation c(cv::GIn(in), cv::GOut(out));

    const auto pkg = cv::gapi::kernels<GOCVIsStateUpToDate>();

    // Compilation & testing
    auto ccomp = c.compileStreaming(cv::compile_args(pkg));

    ccomp.setSource(gapi::wip::make_src<cv::gapi::wip::GCaptureSource>
                               (findDataFile("cv/video/768x576.avi")));
    ccomp.start();
    EXPECT_TRUE(ccomp.running());

    // Process the full video
    bool isStateUpToDate = false;
    while (ccomp.pull(cv::gout(isStateUpToDate))) {
        EXPECT_TRUE(isStateUpToDate);
    }
    EXPECT_FALSE(ccomp.running());

    ccomp.setSource(gapi::wip::make_src<cv::gapi::wip::GCaptureSource>
                               (findDataFile("cv/video/1920x1080.avi")));
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

    // Additionally, test the case where state is resetted
    gapiBackSub.prepareForNewStream();
    gapiBackSub(cv::gin(frame), cv::gout(gapiForeground));
    pOcvBackSub->apply(frame, ocvForeground);
    compareBackSubResults(gapiForeground, ocvForeground, 1);
}

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
    initTestDataPath();

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
    gapiBackSub.setSource(gapi::wip::make_src<cv::gapi::wip::GCaptureSource>
                               (findDataFile("cv/video/768x576.avi")));
    // Allowing 1% difference of all pixels between G-API and reference OpenCV results
    testBackSubInStreaming(gapiBackSub, 1);

    // Additionally, test the case when the new stream happens
    gapiBackSub.setSource(gapi::wip::make_src<cv::gapi::wip::GCaptureSource>
                               (findDataFile("cv/video/1920x1080.avi")));
    // Allowing 5% difference of all pixels between G-API and reference OpenCV results
    testBackSubInStreaming(gapiBackSub, 5);
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
