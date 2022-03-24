#ifndef OPENCV_GAPI_PIPELINE_MODELING_TOOL_DUMMY_SOURCE_HPP
#define OPENCV_GAPI_PIPELINE_MODELING_TOOL_DUMMY_SOURCE_HPP

#include <thread>
#include <memory>
#include <chrono>

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/streaming/cap.hpp> // cv::gapi::wip::IStreamSource

#include "utils.hpp"

class DummySource final: public cv::gapi::wip::IStreamSource {
public:
    using Ptr = std::shared_ptr<DummySource>;
    DummySource(const double       latency,
                const OutputDescr& output,
                const bool         drop_frames);
    bool pull(cv::gapi::wip::Data& data) override;
    cv::GMetaArg descr_of() const override;

private:
    double  m_latency;
    cv::Mat m_mat;
    bool    m_drop_frames;
    double  m_next_tick_ts = -1;
    int64_t m_curr_seq_id  = 0;
};

DummySource::DummySource(const double       latency,
                         const OutputDescr& output,
                         const bool         drop_frames)
    : m_latency(latency), m_drop_frames(drop_frames) {
    utils::createNDMat(m_mat, output.dims, output.precision);
    utils::generateRandom(m_mat);
}

bool DummySource::pull(cv::gapi::wip::Data& data) {
    using namespace std::chrono;
    using namespace cv::gapi::streaming;

    // NB: Wait m_latency before return the first frame.
    if (m_next_tick_ts == -1) {
        m_next_tick_ts = utils::timestamp<milliseconds>() + m_latency;
    }

    int64_t curr_ts = utils::timestamp<milliseconds>();
    if (curr_ts < m_next_tick_ts) {
        /*
         *            curr_ts
         *               |
         *    ------|----*-----|------->
         *                     ^
         *               m_next_tick_ts
         *
         *
         * NB: New frame will be produced at the m_next_tick_ts point.
         */
        utils::sleep(m_next_tick_ts - curr_ts);
    } else {
        /*
         *                                       curr_ts
         *                         +1         +2    |
         *    |----------|----------|----------|----*-----|------->
         *               ^                     ^
         *         m_next_tick_ts ------------->
         *
         *
         *  NB: Shift m_next_tick_ts to the nearest tick before curr_ts and
         *  update current seq_id correspondingly.
         *
         *  if drop_frames is enabled, wait for the next tick, otherwise
         *  return last written frame (+2 at the picture above) immediately.
         */
        int64_t num_frames =
            static_cast<int64_t>((curr_ts - m_next_tick_ts) / m_latency);
        m_curr_seq_id  += num_frames;
        m_next_tick_ts += num_frames * m_latency;
        if (m_drop_frames) {
            m_next_tick_ts += m_latency;
            ++m_curr_seq_id;
            utils::sleep(m_next_tick_ts - curr_ts);
        }
    }

    // NB: Just increase reference counter not to release mat memory
    // after assigning it to the data.
    cv::Mat mat = m_mat;

    data.meta[meta_tag::timestamp] = utils::timestamp<milliseconds>();
    data.meta[meta_tag::seq_id] = m_curr_seq_id++;
    data = mat;
    m_next_tick_ts += m_latency;

    return true;
}

cv::GMetaArg DummySource::descr_of() const {
    return cv::GMetaArg{cv::descr_of(m_mat)};
}

#endif // OPENCV_GAPI_PIPELINE_MODELING_TOOL_DUMMY_SOURCE_HPP
