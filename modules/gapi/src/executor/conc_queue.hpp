// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#ifndef OPENCV_GAPI_EXECUTOR_CONC_QUEUE_HPP
#define OPENCV_GAPI_EXECUTOR_CONC_QUEUE_HPP

#include <queue>
#include <mutex>
#include <condition_variable>

#include <opencv2/gapi/own/assert.hpp>

namespace cv {
namespace gapi {
namespace own {

// This class implements a bare minimum interface of TBB's
// concurrent_bounded_queue with only std:: stuff to make streaming
// API work without TBB.
//
// Highly inefficient, please use it as a last resort if TBB is not
// available in the build.
template<class T>
class concurrent_bounded_queue {
    std::queue<T> m_data;
    std::size_t m_capacity;

    std::mutex m_mutex;
    std::condition_variable m_cond_empty;
    std::condition_variable m_cond_full;

    void unsafe_pop(T &t);

public:
    concurrent_bounded_queue() : m_capacity(0) {}
    concurrent_bounded_queue(const concurrent_bounded_queue<T> &cc)
        : m_data(cc.m_data), m_capacity(cc.m_capacity) {
        // FIXME: what to do with all that locks, etc?
    }
    concurrent_bounded_queue(concurrent_bounded_queue<T> &&cc)
        : m_data(std::move(cc.m_data)), m_capacity(cc.m_capacity) {
        // FIXME: what to do with all that locks, etc?
    }

    // FIXME: && versions
    void push(const T &t);
    void pop(T &t);
    bool try_pop(T &t);

    void set_capacity(std::size_t capacity);

    // Not thread-safe - as in TBB
    void clear();
};

// Internal: do shared pop things assuming the lock is already there
template<typename T>
void concurrent_bounded_queue<T>::unsafe_pop(T &t) {
    GAPI_Assert(!m_data.empty());
    t = m_data.front();
    m_data.pop();
}

// Push an element to the queue. Blocking if there's no space left
template<typename T>
void concurrent_bounded_queue<T>::push(const T& t) {
    std::unique_lock<std::mutex> lock(m_mutex);

    if (m_capacity && m_capacity == m_data.size()) {
        // if there is a limit and it is reached, wait
        m_cond_full.wait(lock, [&](){return m_capacity > m_data.size();});
        GAPI_Assert(m_capacity > m_data.size());
    }
    m_data.push(t);
    lock.unlock();
    m_cond_empty.notify_one();
}

// Pop an element from the queue. Blocking if there's no items
template<typename T>
void concurrent_bounded_queue<T>::pop(T &t) {
    std::unique_lock<std::mutex> lock(m_mutex);
    if (m_data.empty()) {
        // if there is no data, wait
        m_cond_empty.wait(lock, [&](){return !m_data.empty();});
    }
    unsafe_pop(t);
    lock.unlock();
    m_cond_full.notify_one();
}

// Try pop an element from the queue. Returns false if queue is empty
template<typename T>
bool concurrent_bounded_queue<T>::try_pop(T &t) {
    std::unique_lock<std::mutex> lock(m_mutex);
    if (m_data.empty()) {
        // if there is no data, return
        return false;
    }
    unsafe_pop(t);
    lock.unlock();
    m_cond_full.notify_one();
    return true;
}

// Specify the upper limit to the queue. Assumed to be called after
// queue construction but before any real use, any other case is UB
template<typename T>
void concurrent_bounded_queue<T>::set_capacity(std::size_t capacity) {
    GAPI_Assert(m_data.empty());
    GAPI_Assert(m_capacity == 0u);
    GAPI_Assert(capacity != 0u);
    m_capacity = capacity;
}

// Clear the queue. Similar to the TBB version, this method is not
// thread-safe.
template<typename T>
void concurrent_bounded_queue<T>::clear() {
    m_data = std::queue<T>{};
}

}}} // namespace cv::gapi::own

#endif //  OPENCV_GAPI_EXECUTOR_CONC_QUEUE_HPP
