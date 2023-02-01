// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include "../test/common/gapi_tests_common.hpp"

#include "../src/streaming/gstreamer/gstreamer_pipeline_facade.hpp"
#include "../src/streaming/gstreamer/gstreamerptr.hpp"

#include <opencv2/ts.hpp>

#include <thread>

#ifdef HAVE_GSTREAMER
#include <gst/app/gstappsink.h>

namespace opencv_test
{

TEST(GStreamerPipelineFacadeTest, GetElsByFactoryNameUnitTest)
{
    auto comparator = [](GstElement* element, const std::string& factoryName) {
        cv::gapi::wip::gst::GStreamerPtr<gchar> name(
            gst_object_get_name(GST_OBJECT(gst_element_get_factory(element))));
        return name && (0 == strcmp(name, factoryName.c_str()));
    };

    cv::gapi::wip::gst::GStreamerPipelineFacade
        pipelineFacade("videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                       "videorate ! videoscale ! "
                       "video/x-raw,width=1920,height=1080,framerate=3/1 ! "
                       "appsink name=sink1 "
                       "videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                       "videorate ! videoscale ! "
                       "video/x-raw,width=1920,height=1080,framerate=3/1 ! "
                       "appsink name=sink2");

    auto videotestsrcs = pipelineFacade.getElementsByFactoryName("videotestsrc");
    EXPECT_EQ(2u, videotestsrcs.size());
    for (auto&& src : videotestsrcs) {
        EXPECT_TRUE(comparator(src, "videotestsrc"));
    }

    auto appsinks = pipelineFacade.getElementsByFactoryName("appsink");
    EXPECT_EQ(2u, appsinks.size());
    for (auto&& sink : appsinks) {
        EXPECT_TRUE(comparator(sink, "appsink"));
    }
}

TEST(GStreamerPipelineFacadeTest, GetElByNameUnitTest)
{
    auto comparator = [](GstElement* element, const std::string& elementName) {
        cv::gapi::wip::gst::GStreamerPtr<gchar> name(gst_element_get_name(element));
        return name && (0 == strcmp(name, elementName.c_str()));
    };

    cv::gapi::wip::gst::GStreamerPipelineFacade
        pipelineFacade("videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                       "videorate ! videoscale ! "
                       "video/x-raw,width=1920,height=1080,framerate=3/1 ! "
                       "appsink name=sink1 "
                       "videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                       "videorate ! videoscale ! "
                       "video/x-raw,width=1920,height=1080,framerate=3/1 ! "
                       "appsink name=sink2");

    auto appsink1 = pipelineFacade.getElementByName("sink1");
    GAPI_Assert(appsink1 != nullptr);
    EXPECT_TRUE(comparator(appsink1, "sink1"));
    auto appsink2 = pipelineFacade.getElementByName("sink2");
    GAPI_Assert(appsink2 != nullptr);
    EXPECT_TRUE(comparator(appsink2, "sink2"));
}

TEST(GStreamerPipelineFacadeTest, CompletePrerollUnitTest)
{
    cv::gapi::wip::gst::GStreamerPipelineFacade
        pipelineFacade("videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                       "videorate ! videoscale ! "
                       "video/x-raw,width=1920,height=1080,framerate=3/1 ! "
                       "appsink name=sink1 "
                       "videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                       "videorate ! videoscale ! "
                       "video/x-raw,width=1920,height=1080,framerate=3/1 ! "
                       "appsink name=sink2");

    auto appsink = pipelineFacade.getElementByName("sink1");
    pipelineFacade.completePreroll();

    cv::gapi::wip::gst::GStreamerPtr<GstSample> prerollSample(
#if GST_VERSION_MINOR >= 10
            gst_app_sink_try_pull_preroll(GST_APP_SINK(appsink), 5 * GST_SECOND)
#else // GST_VERSION_MINOR < 10
            // TODO: This function may cause hang with some pipelines, need to check whether these
            // pipelines are really wrong or not?
            gst_app_sink_pull_preroll(GST_APP_SINK(appsink))
#endif // GST_VERSION_MINOR >= 10
                                                             );
    GAPI_Assert(prerollSample != nullptr);
}

TEST(GStreamerPipelineFacadeTest, PlayUnitTest)
{
    cv::gapi::wip::gst::GStreamerPipelineFacade
        pipelineFacade("videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                       "videorate ! videoscale ! "
                       "video/x-raw,width=1920,height=1080,framerate=3/1 ! "
                       "appsink name=sink1 "
                       "videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                       "videorate ! videoscale ! "
                       "video/x-raw,width=1920,height=1080,framerate=3/1 ! "
                       "appsink name=sink2");

    auto appsink = pipelineFacade.getElementByName("sink2");

    pipelineFacade.play();

    cv::gapi::wip::gst::PipelineState state;
    GstStateChangeReturn status =
        gst_element_get_state(appsink, &state.current, &state.pending, 5 * GST_SECOND);
    EXPECT_EQ(GST_STATE_CHANGE_SUCCESS, status);
    EXPECT_EQ(GST_STATE_PLAYING, state.current);
    EXPECT_EQ(GST_STATE_VOID_PENDING, state.pending);
}

TEST(GStreamerPipelineFacadeTest, IsPlayingUnitTest)
{
    cv::gapi::wip::gst::GStreamerPipelineFacade
        pipelineFacade("videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                       "videorate ! videoscale ! "
                       "video/x-raw,width=1920,height=1080,framerate=3/1 ! "
                       "appsink name=sink1 "
                       "videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                       "videorate ! videoscale ! "
                       "video/x-raw,width=1920,height=1080,framerate=3/1 ! "
                       "appsink name=sink2");

    EXPECT_FALSE(pipelineFacade.isPlaying());
    pipelineFacade.play();
    EXPECT_TRUE(pipelineFacade.isPlaying());
}

TEST(GStreamerPipelineFacadeTest, MTSafetyUnitTest)
{
    cv::gapi::wip::gst::GStreamerPipelineFacade
        pipelineFacade("videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                       "videorate ! videoscale ! "
                       "video/x-raw,width=1920,height=1080,framerate=3/1 ! "
                       "appsink name=sink1 "
                       "videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                       "videorate ! videoscale ! "
                       "video/x-raw,width=1920,height=1080,framerate=3/1 ! "
                       "appsink name=sink2");

    auto prerollRoutine = [&pipelineFacade](){ pipelineFacade.completePreroll(); };
    auto playRoutine = [&pipelineFacade](){ pipelineFacade.play(); };
    auto isPlayingRoutine = [&pipelineFacade](){ pipelineFacade.isPlaying(); };

    using f = std::function<void()>;

    auto routinesLauncher = [](const f& r1, const f& r2, const f& r3) {
        std::vector<f> routines { r1, r2, r3 };
        std::vector<std::thread> threads { };

        for (auto&& r : routines) {
            threads.emplace_back(r);
        }

        for (auto&& t : threads) {
            t.join();
        }
    };

    routinesLauncher(prerollRoutine, playRoutine, isPlayingRoutine);
    routinesLauncher(prerollRoutine, isPlayingRoutine, playRoutine);
    routinesLauncher(isPlayingRoutine, prerollRoutine, playRoutine);
    routinesLauncher(playRoutine, prerollRoutine, isPlayingRoutine);
    routinesLauncher(playRoutine, isPlayingRoutine, prerollRoutine);
    routinesLauncher(isPlayingRoutine, playRoutine, prerollRoutine);

    EXPECT_TRUE(true);
}
} // namespace opencv_test

#endif // HAVE_GSTREAMER
