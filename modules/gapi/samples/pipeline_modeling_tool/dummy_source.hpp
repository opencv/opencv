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
                const OutputDescr& output);
    bool pull(cv::gapi::wip::Data& data) override;
    cv::GMetaArg descr_of() const override;

private:
    double  m_latency;
    cv::Mat m_mat;
    using TimePoint =
        std::chrono::time_point<std::chrono::high_resolution_clock>;
    cv::optional<TimePoint> m_prev_pull_tp;
};

DummySource::DummySource(const double       latency,
                         const OutputDescr& output)
    : m_latency(latency) {
    utils::createNDMat(m_mat, output.dims, output.precision);
    utils::generateRandom(m_mat);
}

bool DummySource::pull(cv::gapi::wip::Data& data) {
    using namespace std::chrono;
    using namespace cv::gapi::streaming;
    // NB: In case it's the first pull.
    if (!m_prev_pull_tp) {
        m_prev_pull_tp = cv::util::make_optional(high_resolution_clock::now());
    }
    // NB: Just increase reference counter not to release mat memory
    // after assigning it to the data.
    cv::Mat mat = m_mat;
    auto end = high_resolution_clock::now();
    auto elapsed =
        duration_cast<duration<double, std::milli>>(end - *m_prev_pull_tp).count();
    auto delta = m_latency - elapsed;
    if (delta > 0) {
        utils::sleep(delta);
    }
    data.meta[meta_tag::timestamp] = int64_t{utils::timestamp<milliseconds>()};
    data = mat;
    m_prev_pull_tp = cv::util::make_optional(high_resolution_clock::now());
    return true;
}

cv::GMetaArg DummySource::descr_of() const {
    return cv::GMetaArg{cv::descr_of(m_mat)};
}

#endif // OPENCV_GAPI_PIPELINE_MODELING_TOOL_DUMMY_SOURCE_HPP
