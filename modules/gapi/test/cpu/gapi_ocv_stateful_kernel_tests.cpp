// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "gapi_ocv_stateful_kernel_test_utils.hpp"
#include <opencv2/gapi/cpu/core.hpp>
#include <opencv2/gapi/streaming/cap.hpp>

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
        static void setup(const cv::GMatDesc &in,
                          std::shared_ptr<cv::Size> &state)
        {
            state.reset(new cv::Size(in.size));
        }

        static void run(const cv::Mat &in , bool &out, cv::Size& state)
        {
            out = in.size() == state;
        }
    };

    G_TYPED_KERNEL(GStInvalidResize, <GMat(GMat,Size,double,double,int)>, "org.opencv.test.st_invalid_resize")
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
    GOpaque<bool> out = GIsStateUpToDate::on(in);
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
//-------------------------------------------------------------------------------------------------------------


//------------------------------------------- Typed tests on setup() ------------------------------------------
namespace
{
template<typename Tuple>
struct SetupStateTypedTest : public ::testing::Test
{
    using StateT = typename std::tuple_element<0, Tuple>::type;
    using SetupT  = typename std::tuple_element<1, Tuple>::type;

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
