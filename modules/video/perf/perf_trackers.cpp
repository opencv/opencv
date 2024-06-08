// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test { namespace {
using namespace perf;

typedef tuple<string, int, Rect> TrackingParams_t;

std::vector<TrackingParams_t> getTrackingParams()
{
    std::vector<TrackingParams_t> params {
        TrackingParams_t("david/data/david.webm", 300, Rect(163,62,47,56)),
        TrackingParams_t("dudek/data/dudek.webm", 1, Rect(123,87,132,176)),
        TrackingParams_t("faceocc2/data/faceocc2.webm", 1, Rect(118,57,82,98))
    };
    return params;
}

class Tracking : public perf::TestBaseWithParam<TrackingParams_t>
{
public:
    template<typename ROI_t = Rect2d, typename Tracker>
    void runTrackingTest(const Ptr<Tracker>& tracker, const TrackingParams_t& params);
};

template<typename ROI_t, typename Tracker>
void Tracking::runTrackingTest(const Ptr<Tracker>& tracker, const TrackingParams_t& params)
{
    const int N = 10;
    string video = get<0>(params);
    int startFrame = get<1>(params);
    //int endFrame = startFrame + N;
    Rect boundingBox = get<2>(params);

    string videoPath = findDataFile(std::string("cv/tracking/") + video);

    VideoCapture c;
    c.open(videoPath);
    if (!c.isOpened())
        throw SkipTestException("Can't open video file");
#if 0
    // c.set(CAP_PROP_POS_FRAMES, startFrame);
#else
    if (startFrame)
        std::cout << "startFrame = " << startFrame << std::endl;
    for (int i = 0; i < startFrame; i++)
    {
        Mat dummy_frame;
        c >> dummy_frame;
        ASSERT_FALSE(dummy_frame.empty()) << i << ": " << videoPath;
    }
#endif

    // decode frames into memory (don't measure decoding performance)
    std::vector<Mat> frames;
    for (int i = 0; i < N; ++i)
    {
        Mat frame;
        c >> frame;
        ASSERT_FALSE(frame.empty()) << "i=" << i;
        frames.push_back(frame);
    }

    std::cout << "frame size = " << frames[0].size() << std::endl;

    PERF_SAMPLE_BEGIN();
    {
        tracker->init(frames[0], (ROI_t)boundingBox);
        for (int i = 1; i < N; ++i)
        {
            ROI_t rc;
            tracker->update(frames[i], rc);
            ASSERT_FALSE(rc.empty());
        }
    }
    PERF_SAMPLE_END();

    SANITY_CHECK_NOTHING();
}


//==================================================================================================

PERF_TEST_P(Tracking, MIL, testing::ValuesIn(getTrackingParams()))
{
    auto tracker = TrackerMIL::create();
    runTrackingTest<Rect>(tracker, GetParam());
}

PERF_TEST_P(Tracking, GOTURN, testing::ValuesIn(getTrackingParams()))
{
    std::string model = cvtest::findDataFile("dnn/gsoc2016-goturn/goturn.prototxt");
    std::string weights = cvtest::findDataFile("dnn/gsoc2016-goturn/goturn.caffemodel", false);
    TrackerGOTURN::Params params;
    params.modelTxt = model;
    params.modelBin = weights;
    auto tracker = TrackerGOTURN::create(params);
    runTrackingTest<Rect>(tracker, GetParam());
}

}} // namespace
