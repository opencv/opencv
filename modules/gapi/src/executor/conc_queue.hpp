// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#ifndef OPENCV_GAPI_EXECUTOR_CONC_QUEUE_HPP
#define OPENCV_GAPI_EXECUTOR_CONC_QUEUE_HPP
#include <iostream>
#include <queue>
#include <mutex>
#include <thread>
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
    concurrent_bounded_queue() : m_capacity(0), incoming_counter(0),
                                 outgoing_counter(0){}
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
    bool try_push(const T &t);
    void pop(T &t);
    bool try_pop(T &t);
    size_t size();

    void set_capacity(std::size_t capacity);

    // Not thread-safe - as in TBB
    void clear();

    std::atomic<size_t> incoming_counter;
    std::atomic<size_t> outgoing_counter;
};

// Internal: do shared pop things assuming the lock is already there
template<typename T>
void concurrent_bounded_queue<T>::unsafe_pop(T &t) {
    GAPI_Assert(!m_data.empty());
    t = m_data.front();
    m_data.pop();
    outgoing_counter.fetch_add(1);
}

// Push an element to the queue. Blocking if there's no space left
template<typename T>
void concurrent_bounded_queue<T>::push(const T& t) {
    size_t thread_id = incoming_counter.fetch_add(1);
    static std::atomic<size_t> waiters(0);

    std::unique_lock<std::mutex> lock(m_mutex);
    waiters++;
    bool cond_true = false;
    while (m_capacity && (m_capacity == m_data.size() || waiters.load() != 1) && !cond_true) {
        // if there is a limit and it is reached, wait
        // if there had got overload and new producer appeared (waiters++), queue thread & wait
        m_cond_full.wait(lock, [&]() {
            if(m_capacity > m_data.size()) {
                // somebody consumes data then awake the oldest waiting thread
                if ((thread_id <= outgoing_counter.load() +  m_data.size())) {
                    // when consumer pop more than 1 element from queue
                    // we ensure that threads will awaked starting from oldest `thread_id`:
                    // when consumer pop element then `outgoing_counter` is increased but
                    // `m_data.size()` is decreased on the same value - so its `sum` stays
                    // unchanged. Independent from how much elements had been consumed
                    // `sum` stays constant and the oldest `thread_id` pass the if-condition
                    // When it passed if-condition then it would increase `m_data.size() + 1`
                    // after own push but `outgoing_counter` would stay unchanged, so we would
                    // got `sum + 1` in result which would awake the next `thread_id + 1`
                    // and so on
                    cond_true = true;
                    return true;
                }
            }
            return false;
        });
    }
    waiters--;
    m_data.push(t);
    //m_cond_full.notify_all();
    lock.unlock();
    m_cond_empty.notify_one();
}

template<typename T>
bool concurrent_bounded_queue<T>::try_push(const T &t) {
    std::unique_lock<std::mutex> lock(m_mutex);

    if (m_capacity && m_capacity == m_data.size()) {
        // if there is a limit and it is reached, then fail
        return false;
    }
    m_data.push(t);
    incoming_counter.fetch_add(1);
    lock.unlock();
    m_cond_empty.notify_one();
    return true;
}

template<typename T>
size_t concurrent_bounded_queue<T>::size() {
    std::unique_lock<std::mutex> lock(m_mutex);
    return m_data.size();
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
    m_cond_full.notify_all();
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
    m_cond_full.notify_all();
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
    incoming_counter.store(0);
    outgoing_counter.store(0);
}

}}} // namespace cv::gapi::own

#endif //  OPENCV_GAPI_EXECUTOR_CONC_QUEUE_HPP
