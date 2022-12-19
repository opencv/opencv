// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include "gstreamer_buffer_utils.hpp"

#include "gstreamer_media_adapter.hpp"

#include "gstreamersource_priv.hpp"
#include <opencv2/gapi/streaming/gstreamer/gstreamersource.hpp>

#include <opencv2/gapi/streaming/meta.hpp>

#include <logger.hpp>

#include <opencv2/imgproc.hpp>

#ifdef HAVE_GSTREAMER
#include <gst/app/gstappsink.h>
#include <gst/gstbuffer.h>
#include <gst/video/video-frame.h>
#endif // HAVE_GSTREAMER

namespace cv {
namespace gapi {
namespace wip {
namespace gst {

#ifdef HAVE_GSTREAMER

constexpr char ALLOWED_CAPS_STRING[] =
    "video/x-raw,format=(string){NV12, GRAY8};video/x-raw(memory:DMABuf),format=(string){NV12, GRAY8}";


namespace {
GstPadProbeReturn appsinkQueryCallback(GstPad*, GstPadProbeInfo* info, gpointer)
{
    GstQuery *query = GST_PAD_PROBE_INFO_QUERY(info);

    if (GST_QUERY_TYPE(query) != GST_QUERY_ALLOCATION)
        return GST_PAD_PROBE_OK;

    gst_query_add_allocation_meta(query, GST_VIDEO_META_API_TYPE, NULL);

    return GST_PAD_PROBE_HANDLED;
}
} // anonymous namespace

GStreamerSource::Priv::Priv(const std::string& pipelineDesc,
                            const GStreamerSource::OutputType outputType) :
    m_pipeline(std::make_shared<GStreamerPipelineFacade>(pipelineDesc)),
    m_outputType(outputType)
{
    GAPI_Assert((m_outputType == GStreamerSource::OutputType::FRAME ||
                 m_outputType == GStreamerSource::OutputType::MAT)
                && "Unsupported output type for GStreamerSource!");

    auto appsinks = m_pipeline->getElementsByFactoryName("appsink");
    GAPI_Assert(1ul == appsinks.size() &&
        "GStreamerSource can accept pipeline with only 1 appsink element inside!\n"
        "Please, note, that amount of sink elements of other than appsink type is not limited.\n");

    m_appsink = GST_ELEMENT(gst_object_ref(appsinks[0]));

    configureAppsink();
}

GStreamerSource::Priv::Priv(std::shared_ptr<GStreamerPipelineFacade> pipeline,
                            const std::string& appsinkName,
                            const GStreamerSource::OutputType outputType) :
    m_pipeline(pipeline),
    m_outputType(outputType)
{
    GAPI_Assert((m_outputType == GStreamerSource::OutputType::FRAME ||
                 m_outputType == GStreamerSource::OutputType::MAT)
                && "Unsupported output type for GStreamerSource!");

    m_appsink = (GST_ELEMENT(gst_object_ref(m_pipeline->getElementByName(appsinkName))));
    configureAppsink();
}

bool GStreamerSource::Priv::pull(cv::gapi::wip::Data& data)
{
    bool result = false;
    switch(m_outputType) {
        case GStreamerSource::OutputType::FRAME: {
            cv::MediaFrame frame;
            result = retrieveFrame(frame);
            if (result) {
                data = frame;
            }
            break;
        }
        case GStreamerSource::OutputType::MAT: {
            cv::Mat mat;
            result = retrieveFrame(mat);
            if (result) {
                data = mat;
            }
            break;
        }
    }

    if (result) {
        data.meta[cv::gapi::streaming::meta_tag::timestamp] = computeTimestamp();
        data.meta[cv::gapi::streaming::meta_tag::seq_id]    = m_frameId++;
    }

    return result;
}

GMetaArg GStreamerSource::Priv::descr_of() noexcept
{
    // Prepare frame metadata if it wasn't prepared yet.
    prepareVideoMeta();

    switch(m_outputType) {
        case GStreamerSource::OutputType::FRAME: {
            return GMetaArg { m_mediaFrameMeta };
        }
        case GStreamerSource::OutputType::MAT: {
            return GMetaArg { m_matMeta };
        }
    }

    return GMetaArg { };
}

void GStreamerSource::Priv::configureAppsink() {
    // NOTE: appsink element should be configured before pipeline launch.
    GAPI_Assert(!m_pipeline->isPlaying());

    // TODO: is 1 single buffer really high enough?
    gst_app_sink_set_max_buffers(GST_APP_SINK(m_appsink.get()), 1);

    // Do not emit signals: all calls will be synchronous and blocking.
    gst_app_sink_set_emit_signals(GST_APP_SINK(m_appsink.get()), FALSE);

    GStreamerPtr<GstCaps> gstCaps(gst_caps_from_string(ALLOWED_CAPS_STRING));

    GStreamerPtr<GstPad> appsinkPad(gst_element_get_static_pad(m_appsink, "sink"));
    GStreamerPtr<GstCaps> peerCaps(gst_pad_peer_query_caps(appsinkPad, NULL));
    if (!gst_caps_can_intersect(peerCaps, gstCaps)) {
        cv::util::throw_error(
            std::logic_error("appsink element can only consume video-frame in NV12 or GRAY8 format in "
                             "GStreamerSource"));
    }

    gst_app_sink_set_caps(GST_APP_SINK(m_appsink.get()), gstCaps);

    gst_pad_add_probe(appsinkPad, GST_PAD_PROBE_TYPE_QUERY_DOWNSTREAM, appsinkQueryCallback,
                      NULL, NULL);
}

void GStreamerSource::Priv::prepareVideoMeta()
{
    if (!m_isMetaPrepared) {
        m_pipeline->completePreroll();

        GStreamerPtr<GstSample> prerollSample(
#if GST_VERSION_MINOR >= 10
            gst_app_sink_try_pull_preroll(GST_APP_SINK(m_appsink.get()), 5 * GST_SECOND));
#else // GST_VERSION_MINOR < 10
            // TODO: This function may cause hang with some pipelines, need to check whether these
            // pipelines are really wrong or not?
            gst_app_sink_pull_preroll(GST_APP_SINK(m_appsink.get())));
#endif // GST_VERSION_MINOR >= 10
        GAPI_Assert(prerollSample != nullptr);

        GstCaps* prerollCaps(gst_sample_get_caps(prerollSample));
        GAPI_Assert(prerollCaps != nullptr);

        const GstStructure* structure = gst_caps_get_structure(prerollCaps, 0);

        // Width and height held in GstCaps structure are format-independent width and height
        // of the frame. So the actual height of the output buffer in NV12 format will be
        // height * 3/2.
        int width = 0;
        int height = 0;
        if (!gst_structure_get_int(structure, "width", &width) ||
            !gst_structure_get_int(structure, "height", &height))
        {
            cv::util::throw_error(std::logic_error("Cannot query video width/height."));
        }

        // Fill GstVideoInfo structure to work further with GstVideoFrame class.
        if (!gst_video_info_from_caps(&m_videoInfo, prerollCaps)) {
            cv::util::throw_error(std::logic_error("preroll sample has invalid caps."));
        }
        m_type = GST_VIDEO_INFO_FORMAT(&m_videoInfo);
        switch(m_outputType) {
            case GStreamerSource::OutputType::FRAME: {
                // Construct metadata for media frame.
                switch (m_type) {
                    case GST_VIDEO_FORMAT_NV12: {
                        m_mediaFrameMeta = GFrameDesc{ cv::MediaFormat::NV12, cv::Size(width, height) };
                        GAPI_Assert(GST_VIDEO_INFO_N_PLANES(&m_videoInfo) == 2);
                        break;
                    }
                    case GST_VIDEO_FORMAT_GRAY8: {
                        m_mediaFrameMeta = GFrameDesc{ cv::MediaFormat::GRAY, cv::Size(width, height) };
                        GAPI_Assert(GST_VIDEO_INFO_N_PLANES(&m_videoInfo) == 1);
                        break;
                    }
                    default: {
                        GAPI_Error("Unsupported GStreamerSource FRAME type.");
                    }
                }
                break;
            }
            case GStreamerSource::OutputType::MAT: {
                // Construct metadata for BGR mat.
                m_matMeta = GMatDesc { CV_8U, 3, cv::Size(width, height), false };
                break;
            }
        }

        m_isMetaPrepared = true;
    }
}

int64_t GStreamerSource::Priv::computeTimestamp()
{
    GAPI_Assert(m_buffer && "Pulled buffer is nullptr!");

    int64_t timestamp { };

    GstClockTime pts = GST_BUFFER_PTS(m_buffer);
    if (GST_CLOCK_TIME_IS_VALID(pts)) {
        timestamp = GST_TIME_AS_USECONDS(pts);
    } else {
        const auto now = std::chrono::system_clock::now();
        const auto dur = std::chrono::duration_cast<std::chrono::microseconds>
            (now.time_since_epoch());
        timestamp = int64_t{dur.count()};
    }

    return timestamp;
}

bool GStreamerSource::Priv::pullBuffer()
{
    // Start the pipeline if it is not in the playing state yet
    if (!m_isPipelinePlaying) {
        m_pipeline->play();
        m_isPipelinePlaying = true;
    }

    // Bail out if EOS
    if (gst_app_sink_is_eos(GST_APP_SINK(m_appsink.get())))
    {
        return false;
    }

    m_sample = gst_app_sink_pull_sample(GST_APP_SINK(m_appsink.get()));

    // TODO: GAPI_Assert because of already existed check for EOS?
    if (m_sample == nullptr)
    {
        return false;
    }

    m_buffer = gst_sample_get_buffer(m_sample);
    GAPI_Assert(m_buffer && "Grabbed sample has no buffer!");

    return true;
}

bool GStreamerSource::Priv::retrieveFrame(cv::Mat& data)
{
    // Prepare metadata if it isn't prepared yet.
    prepareVideoMeta();

    bool result = pullBuffer();
    if (!result)
    {
        return false;
    }

    // TODO: Use RAII for map/unmap
    GstVideoFrame videoFrame;
    gstreamer_utils::mapBufferToFrame(*m_buffer, m_videoInfo, videoFrame, GST_MAP_READ);

    try
    {
        switch (m_type) {
            case GST_VIDEO_FORMAT_NV12: {
                // m_matMeta holds width and height for 8U BGR frame, but actual
                // frame m_buffer we request from GStreamer pipeline has 8U NV12 format.
                // Constructing y and uv cv::Mat-s from such a m_buffer:
                GAPI_Assert((uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&videoFrame, 1) ==
                    (uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&videoFrame, 0) +
                    GST_VIDEO_FRAME_PLANE_OFFSET(&videoFrame, 1));
                GAPI_Assert(GST_VIDEO_INFO_N_PLANES(&m_videoInfo) == 2);

                cv::Mat y(m_matMeta.size, CV_8UC1,
                    (uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&videoFrame, 0) +
                    GST_VIDEO_FRAME_PLANE_OFFSET(&videoFrame, 0),
                    GST_VIDEO_FRAME_PLANE_STRIDE(&videoFrame, 0));
                cv::Mat uv(m_matMeta.size / 2, CV_8UC2,
                    (uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&videoFrame, 0) +
                    GST_VIDEO_FRAME_PLANE_OFFSET(&videoFrame, 1),
                    GST_VIDEO_FRAME_PLANE_STRIDE(&videoFrame, 1));

                cv::cvtColorTwoPlane(y, uv, data, cv::COLOR_YUV2BGR_NV12);
                break;
            }
            case GST_VIDEO_FORMAT_GRAY8: {
                GAPI_Assert(GST_VIDEO_INFO_N_PLANES(&m_videoInfo) == 1);
                cv::Mat y(m_matMeta.size, CV_8UC1,
                    (uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&videoFrame, 0) +
                    GST_VIDEO_FRAME_PLANE_OFFSET(&videoFrame, 0),
                    GST_VIDEO_FRAME_PLANE_STRIDE(&videoFrame, 0));
                cv::cvtColor(y, data, cv::COLOR_GRAY2BGR);
                break;
            }
            default: {
                GAPI_Error("retrieveFrame - unsupported GStreamerSource FRAME type.");
            }
        }
    }
    catch (...)
    {
        gst_video_frame_unmap(&videoFrame);
        cv::util::throw_error(std::runtime_error("NV12 or GRAY8 buffer conversion to BGR is failed!"));
    }
    gst_video_frame_unmap(&videoFrame);

    return true;
}

bool GStreamerSource::Priv::retrieveFrame(cv::MediaFrame& data)
{
    // Prepare metadata if it isn't prepared yet.
    prepareVideoMeta();

    bool result = pullBuffer();
    if (!result)
    {
        return false;
    }

    data = cv::MediaFrame::Create<GStreamerMediaAdapter>(m_mediaFrameMeta, &m_videoInfo,
                                                         m_buffer);

    return true;
}

GStreamerSource::Priv::~Priv() { }

#else // HAVE_GSTREAMER

GStreamerSource::Priv::Priv(const std::string&, const GStreamerSource::OutputType)
{
    GAPI_Error("Built without GStreamer support!");
}

GStreamerSource::Priv::Priv(std::shared_ptr<GStreamerPipelineFacade>, const std::string&,
                            const GStreamerSource::OutputType)
{
    GAPI_Error("Built without GStreamer support!");
}

bool GStreamerSource::Priv::pull(cv::gapi::wip::Data&)
{
    // No need an assert here. Constructor have already got assert.
    return false;
}

GMetaArg GStreamerSource::Priv::descr_of() const noexcept
{
    // No need an assert here. The assert raise C4702 warning. Constructor have already got assert.
    return GMetaArg{};
}

GStreamerSource::Priv::~Priv()
{
    // No need an assert here. The assert raise C4722 warning. Constructor have already got assert.
}

#endif // HAVE_GSTREAMER

GStreamerSource::GStreamerSource(const std::string& pipeline,
                                 const GStreamerSource::OutputType outputType):
    m_priv(new Priv(pipeline, outputType)) { }

GStreamerSource::GStreamerSource(std::shared_ptr<GStreamerPipelineFacade> pipeline,
                                 const std::string& appsinkName,
                                 const GStreamerSource::OutputType outputType):
    m_priv(new Priv(pipeline, appsinkName, outputType)) { }

bool GStreamerSource::pull(cv::gapi::wip::Data& data)
{
    return m_priv->pull(data);
}

GMetaArg GStreamerSource::descr_of() const
{
    return m_priv->descr_of();
}

GStreamerSource::~GStreamerSource()
{ }

GStreamerSource::GStreamerSource(std::unique_ptr<Priv> priv):
    m_priv(std::move(priv))
{ }

} // namespace gst
} // namespace wip
} // namespace gapi
} // namespace cv
