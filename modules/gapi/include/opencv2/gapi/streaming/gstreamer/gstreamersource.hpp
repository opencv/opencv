// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_GSTREAMER_GSTREAMERSOURCE_HPP
#define OPENCV_GAPI_STREAMING_GSTREAMER_GSTREAMERSOURCE_HPP

#include <opencv2/gapi/streaming/source.hpp>
#include <opencv2/gapi/garg.hpp>

#include <memory>

namespace cv {
namespace gapi {
namespace wip {
namespace gst {

/**
 * @brief OpenCV's GStreamer streaming source.
 *        Streams cv::Mat-s/cv::MediaFrame from passed GStreamer pipeline.
 *
 * This class implements IStreamSource interface.
 *
 * To create GStreamerSource instance you need to pass 'pipeline' and, optionally, 'outputType'
 * arguments into constructor.
 * 'pipeline' should represent GStreamer pipeline in form of textual description.
 * Almost any custom pipeline is supported which can be successfully ran via gst-launch.
 * The only two limitations are:
 *      - there should be __one__ appsink element in the pipeline to pass data to OpenCV app.
 *        Pipeline can actually contain many sink elements, but it must have one and only one
 *        appsink among them.
 *
 *      - data passed to appsink should be video-frame in NV12 format.
 *
 * 'outputType' is used to select type of output data to produce: 'cv::MediaFrame' or 'cv::Mat'.
 * To produce 'cv::MediaFrame'-s you need to pass 'GStreamerSource::OutputType::FRAME' and,
 * correspondingly, 'GStreamerSource::OutputType::MAT' to produce 'cv::Mat'-s.
 * Please note, that in the last case, output 'cv::Mat' will be of BGR format, internal conversion
 * from NV12 GStreamer data will happen.
 * Default value for 'outputType' is 'GStreamerSource::OutputType::MAT'.
 *
 * @note Stream sources are passed to G-API via shared pointers, so please use gapi::make_src<>
 *       to create objects and ptr() to pass a GStreamerSource to cv::gin().
 *
 * @note You need to build OpenCV with GStreamer support to use this class.
 */

class GStreamerPipelineFacade;

class GAPI_EXPORTS GStreamerSource : public IStreamSource
{
public:
    class Priv;

    // Indicates what type of data should be produced by GStreamerSource: cv::MediaFrame or cv::Mat
    enum class OutputType {
        FRAME,
        MAT
    };

    GStreamerSource(const std::string& pipeline,
                    const GStreamerSource::OutputType outputType =
                        GStreamerSource::OutputType::MAT);
    GStreamerSource(std::shared_ptr<GStreamerPipelineFacade> pipeline,
                    const std::string& appsinkName,
                    const GStreamerSource::OutputType outputType =
                        GStreamerSource::OutputType::MAT);

    bool pull(cv::gapi::wip::Data& data) override;
    GMetaArg descr_of() const override;
    ~GStreamerSource() override;

protected:
    explicit GStreamerSource(std::unique_ptr<Priv> priv);

    std::unique_ptr<Priv> m_priv;
};

} // namespace gst

using GStreamerSource = gst::GStreamerSource;

} // namespace wip
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_STREAMING_GSTREAMER_GSTREAMERSOURCE_HPP
