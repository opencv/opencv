// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_GSTREAMER_GSTREAMERPIPELINE_PRIV_HPP
#define OPENCV_GAPI_STREAMING_GSTREAMER_GSTREAMERPIPELINE_PRIV_HPP

#include <opencv2/gapi/streaming/gstreamer/gstreamerpipeline.hpp>

#include <string>
#include <unordered_map>

namespace cv {
namespace gapi {
namespace wip {
namespace gst {

#ifdef HAVE_GSTREAMER

class GStreamerPipeline::Priv
{
public:
    explicit Priv(const std::string& pipeline);

    IStreamSource::Ptr getStreamingSource(const std::string& appsinkName,
                                          const GStreamerSource::OutputType outputType);

    virtual ~Priv();

protected:
    std::shared_ptr<GStreamerPipelineFacade> m_pipeline;
    std::unordered_map<std::string, bool> m_appsinkNamesToUse;
};

#else // HAVE_GSTREAMER

class GStreamerPipeline::Priv
{
public:
    explicit Priv(const std::string& pipeline);

    IStreamSource::Ptr getStreamingSource(const std::string& appsinkName,
                                          const GStreamerSource::OutputType outputType);

    virtual ~Priv();
};

#endif // HAVE_GSTREAMER

} // namespace gst
} // namespace wip
} // namespace gapi
} // namespace cv


#endif // OPENCV_GAPI_STREAMING_GSTREAMER_GSTREAMERPIPELINE_PRIV_HPP
