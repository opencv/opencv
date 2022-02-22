// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_GSTREAMER_GSTREAMERSOURCE_PRIV_HPP
#define OPENCV_GAPI_STREAMING_GSTREAMER_GSTREAMERSOURCE_PRIV_HPP

#include "gstreamerptr.hpp"
#include "gstreamer_pipeline_facade.hpp"
#include <opencv2/gapi/streaming/gstreamer/gstreamersource.hpp>

#include <string>

#ifdef HAVE_GSTREAMER
#include <gst/gst.h>
#include <gst/video/video-frame.h>
#endif // HAVE_GSTREAMER

namespace cv {
namespace gapi {
namespace wip {
namespace gst {

#ifdef HAVE_GSTREAMER

class GStreamerSource::Priv
{
public:
    Priv(const std::string& pipeline, const GStreamerSource::OutputType outputType);

    Priv(std::shared_ptr<GStreamerPipelineFacade> pipeline, const std::string& appsinkName,
         const GStreamerSource::OutputType outputType);

    bool pull(cv::gapi::wip::Data& data);

    // non-const in difference with GStreamerSource, because contains delayed meta initialization
    GMetaArg descr_of() noexcept;

    virtual ~Priv();

protected:
    // Shares:
    std::shared_ptr<GStreamerPipelineFacade> m_pipeline;

    // Owns:
    GStreamerPtr<GstElement> m_appsink;
    GStreamerPtr<GstSample> m_sample;
    GstBuffer* m_buffer = nullptr; // Actual frame memory holder
    GstVideoInfo m_videoInfo; // Information about Video frame

    GStreamerSource::OutputType m_outputType = GStreamerSource::OutputType::MAT;

    GMatDesc m_matMeta;
    GFrameDesc m_mediaFrameMeta;

    bool m_isMetaPrepared = false;
    bool m_isPipelinePlaying = false;

    int64_t m_frameId = 0L;
    size_t m_type = 0; //Gstreamer video format type

protected:
    void configureAppsink();
    void prepareVideoMeta();

    int64_t computeTimestamp();

    bool pullBuffer();
    bool retrieveFrame(cv::Mat& data);
    bool retrieveFrame(cv::MediaFrame& data);
};

#else // HAVE_GSTREAMER

class GStreamerSource::Priv
{
public:
    Priv(const std::string& pipeline, const GStreamerSource::OutputType outputType);
    Priv(std::shared_ptr<GStreamerPipelineFacade> pipeline, const std::string& appsinkName,
         const GStreamerSource::OutputType outputType);
    bool pull(cv::gapi::wip::Data& data);
    GMetaArg descr_of() const noexcept;
    virtual ~Priv();
};

#endif // HAVE_GSTREAMER

} // namespace gst
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_STREAMING_GSTREAMER_GSTREAMERSOURCE_PRIV_HPP
