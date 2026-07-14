// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_EXECUTOR_LAST_VALUE_HPP
#define OPENCV_GAPI_EXECUTOR_LAST_VALUE_HPP

#include <mutex>
#include <condition_variable>

#include <opencv2/gapi/util/optional.hpp>
#include <opencv2/gapi/own/assert.hpp>

namespace cv {
namespace gapi {
namespace own {

// This class implements a "Last Written Value" thing.  Writer threads
// (in our case, it is just one) can write as many values there as it
// can.
//
// The reader thread gets only a value it gets at the time (or blocks
// if there was no value written since the last read).
//
// Again, the implementation is highly inefficient right now.
template<class T>
class last_written_value {
    cv::util::optional<T> m_data;

    std::mutex m_mutex;
    std::condition_variable m_cond_empty;

    void unsafe_pop(T &t);

public:
    last_written_value() {}
    last_written_value(const last_written_value<T> &cc)
        : m_data(cc.m_data) {
        // FIXME: what to do with all that locks, etc?
    }
    last_written_value(last_written_value<T> &&cc)
        : m_data(std::move(cc.m_data)) {
        // FIXME: what to do with all that locks, etc?
    }

    // FIXME: && versions
    void push(const T &t);
    void pop(T &t);
    bool try_pop(T &t);

    // Not thread-safe
    void clear();
};

// Internal: do shared pop things assuming the lock is already there
template<typename T>
void last_written_value<T>::unsafe_pop(T &t) {
    GAPI_Assert(m_data.has_value());
    t = std::move(m_data.value());
    m_data.reset();
}

// Push an element to the queue. Blocking if there's no space left
template<typename T>
void last_written_value<T>::push(const T& t) {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_data = cv::util::make_optional(t);
    lock.unlock();
    m_cond_empty.notify_one();
}

// Pop an element from the queue. Blocking if there's no items
template<typename T>
void last_written_value<T>::pop(T &t) {
    std::unique_lock<std::mutex> lock(m_mutex);
    if (!m_data.has_value()) {
        // if there is no data, wait
        m_cond_empty.wait(lock, [&](){return m_data.has_value();});
    }
    unsafe_pop(t);
}

// Try pop an element from the queue. Returns false if queue is empty
template<typename T>
bool last_written_value<T>::try_pop(T &t) {
    std::unique_lock<std::mutex> lock(m_mutex);
    if (!m_data.has_value()) {
        // if there is no data, return
        return false;
    }
    unsafe_pop(t);
    return true;
}

// Clear the value holder. This method is not thread-safe.
template<typename T>
void last_written_value<T>::clear() {
    m_data.reset();
}

}}} // namespace cv::gapi::own

#endif //  OPENCV_GAPI_EXECUTOR_CONC_QUEUE_HPP
