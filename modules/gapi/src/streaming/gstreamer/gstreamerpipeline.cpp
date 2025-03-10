// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include "gstreamer_pipeline_facade.hpp"
#include "gstreamerpipeline_priv.hpp"
#include <opencv2/gapi/streaming/gstreamer/gstreamerpipeline.hpp>

#ifdef HAVE_GSTREAMER
#include <gst/app/gstappsink.h>
#endif // HAVE_GSTREAMER

namespace cv {
namespace gapi {
namespace wip {
namespace gst {

#ifdef HAVE_GSTREAMER

GStreamerPipeline::Priv::Priv(const std::string& pipelineDesc):
    m_pipeline(std::make_shared<GStreamerPipelineFacade>(pipelineDesc))
{
    std::vector<GstElement*> appsinks =
        m_pipeline->getElementsByFactoryName("appsink");

    for (std::size_t i = 0ul; i < appsinks.size(); ++i)
    {
        auto* appsink = appsinks[i];
        GAPI_Assert(appsink != nullptr);
        GStreamerPtr<gchar> name(gst_element_get_name(appsink));
        auto result = m_appsinkNamesToUse.insert({ name.get(), true /* free */ });
        GAPI_Assert(std::get<1>(result) && "Each appsink name must be unique!");
    }
}

IStreamSource::Ptr GStreamerPipeline::Priv::getStreamingSource(
    const std::string& appsinkName, const GStreamerSource::OutputType outputType)
{
    auto appsinkNameIt = m_appsinkNamesToUse.find(appsinkName);
    if (appsinkNameIt == m_appsinkNamesToUse.end())
    {
        cv::util::throw_error(std::logic_error(std::string("There is no appsink element in the "
            "pipeline with the name '") + appsinkName + "'."));
    }

    if (!appsinkNameIt->second)
    {
        cv::util::throw_error(std::logic_error(std::string("appsink element with the name '") +
            appsinkName + "' has been already used to create a GStreamerSource!"));
    }

    m_appsinkNamesToUse[appsinkName] = false /* not free */;

    IStreamSource::Ptr src;
    try {
        src = cv::gapi::wip::make_src<cv::gapi::wip::GStreamerSource>(m_pipeline, appsinkName,
                                                                      outputType);
    }
    catch(...) {
        m_appsinkNamesToUse[appsinkName] = true; /* free */
        cv::util::throw_error(std::runtime_error(std::string("Error during creation of ") +
            "GStreamerSource on top of '" + appsinkName + "' appsink element!"));
    }

    return src;
}

GStreamerPipeline::Priv::~Priv() { }

#else // HAVE_GSTREAMER

GStreamerPipeline::Priv::Priv(const std::string&)
{
    GAPI_Error("Built without GStreamer support!");
}

IStreamSource::Ptr GStreamerPipeline::Priv::getStreamingSource(const std::string&,
                                                               const GStreamerSource::OutputType)
{
    // No need an assert here. The assert raise C4702 warning. Constructor have already got assert.
    return nullptr;
}

GStreamerPipeline::Priv::~Priv()
{
    // No need an assert here. The assert raise C4722 warning. Constructor have already got assert.
}

#endif // HAVE_GSTREAMER

GStreamerPipeline::GStreamerPipeline(const std::string& pipelineDesc):
    m_priv(new Priv(pipelineDesc)) { }

IStreamSource::Ptr GStreamerPipeline::getStreamingSource(
    const std::string& appsinkName, const GStreamerSource::OutputType outputType)
{
    return m_priv->getStreamingSource(appsinkName, outputType);
}

GStreamerPipeline::~GStreamerPipeline()
{ }

GStreamerPipeline::GStreamerPipeline(std::unique_ptr<Priv> priv):
    m_priv(std::move(priv))
{ }

} // namespace gst
} // namespace wip
} // namespace gapi
} // namespace cv
