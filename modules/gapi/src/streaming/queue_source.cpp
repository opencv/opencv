// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation

#include <chrono>
#include <atomic>

#include <opencv2/gapi/streaming/queue_source.hpp>
#include <opencv2/gapi/streaming/meta.hpp>

#include "executor/conc_queue.hpp"

namespace cv {
namespace gapi {
namespace wip {

class QueueSourceBase::Priv {
public:
    explicit Priv(const cv::GMetaArg &meta) {
        m = meta;
        halted = false;
    }

    cv::GMetaArg m;
    cv::gapi::own::concurrent_bounded_queue<cv::GRunArg> q;
    int64_t c = 0;
    std::atomic<bool> halted;
};

QueueSourceBase::QueueSourceBase(const cv::GMetaArg &m)
    : m_priv(new Priv(m)) {
}

void QueueSourceBase::push(Data &&data) {

    // Tag data with seq_id/ts
    const auto now = std::chrono::system_clock::now();
    const auto dur = std::chrono::duration_cast<std::chrono::microseconds>
        (now.time_since_epoch());
    data.meta[cv::gapi::streaming::meta_tag::timestamp] = int64_t{dur.count()};
    data.meta[cv::gapi::streaming::meta_tag::seq_id]    = int64_t{m_priv->c++};

    m_priv->q.push(data);
}

bool QueueSourceBase::pull(Data &data) {
    m_priv->q.pop(data);

    if (m_priv->halted) {
        return false;
    }
    return true;
}

void QueueSourceBase::halt() {
    m_priv->halted.store(true);
    m_priv->q.push(cv::GRunArg{});
}

cv::GMetaArg QueueSourceBase::descr_of() const {
    return m_priv->m;
}

} // wip
} // gapi
} // cv
