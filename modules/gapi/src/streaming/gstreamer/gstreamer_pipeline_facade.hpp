// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_GSTREAMER_GSTREAMER_PIPELINE_FACADE_HPP
#define OPENCV_GAPI_STREAMING_GSTREAMER_GSTREAMER_PIPELINE_FACADE_HPP

#include "gstreamerptr.hpp"

#include <string>
#include <atomic>
#include <mutex>

#ifdef HAVE_GSTREAMER
#include <gst/gst.h>

namespace cv {
namespace gapi {
namespace wip {
namespace gst {

// GAPI_EXPORTS here is only for testing purposes.
struct GAPI_EXPORTS PipelineState
{
    GstState current = GST_STATE_NULL;
    GstState pending = GST_STATE_NULL;
};

// This class represents facade for pipeline GstElement and related functions.
// Class restricts pipeline to only move forward in its state machine:
// NULL -> READY -> PAUSED -> PLAYING.
// There is no possibility to pause and resume pipeline, it can be only once played.
// GAPI_EXPORTS here is only for testing purposes.
class GAPI_EXPORTS GStreamerPipelineFacade
{
public:
    // Strong exception guarantee.
    explicit GStreamerPipelineFacade(const std::string& pipeline);

    // The destructors are noexcept by default. (since C++11)
    ~GStreamerPipelineFacade();

    // Elements getters are not guarded with mutexes because elements order is not supposed
    // to change in the pipeline.
    std::vector<GstElement*> getElementsByFactoryName(const std::string& factoryName);
    GstElement* getElementByName(const std::string& elementName);

    // Pipeline state modifiers: can be called only once, MT-safe, mutually exclusive.
    void completePreroll();
    void play();

    // Pipeline state checker: MT-safe.
    bool isPlaying();

private:
    std::string m_pipelineDesc;

    GStreamerPtr<GstElement> m_pipeline;

    std::atomic<bool> m_isPrerolled;
    std::atomic<bool> m_isPlaying;
    // Mutex to guard state(paused, playing) from changes from multiple threads
    std::mutex m_stateChangeMutex;

private:
    // This constructor is needed only to make public constructor as delegating constructor
    // and allow it to throw exceptions.
    GStreamerPipelineFacade();

    // Elements getter is not guarded with mutex because elements order is not supposed
    // to change in the pipeline.
    std::vector<GstElement*> getElements(std::function<bool(GstElement*)> comparator);

    // Getters, modifiers, verifiers are not MT-safe, because they are called from
    // MT-safe mutually exclusive public functions.
    PipelineState queryState();
    void setState(GstState state);
    void verifyStateChange(GstStateChangeReturn status);
    void checkBusMessages() const;
};

} // namespace gst
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_GSTREAMER
#endif // OPENCV_GAPI_STREAMING_GSTREAMER_GSTREAMER_PIPELINE_FACADE_HPP
