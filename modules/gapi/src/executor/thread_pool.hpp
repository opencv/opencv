// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2024 Intel Corporation

#ifndef OPENCV_GAPI_THREAD_POOL_HPP
#define OPENCV_GAPI_THREAD_POOL_HPP

#include <functional>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

#include <opencv2/gapi/own/exports.hpp> // GAPI_EXPORTS

#if defined(HAVE_TBB)
#  include <tbb/concurrent_queue.h> // FIXME: drop it from here!
template<typename T> using QueueClass = tbb::concurrent_bounded_queue<T>;
#else
#  include "executor/conc_queue.hpp"
template<typename T> using QueueClass = cv::gapi::own::concurrent_bounded_queue<T>;
#endif // TBB

namespace cv {
namespace gapi {
namespace own {

// NB: Only for tests
class GAPI_EXPORTS Latch {
public:
    explicit Latch(const uint64_t expected);

    Latch(const Latch&) = delete;
    Latch& operator=(const Latch&) = delete;

    void count_down();
    void wait();

private:
    uint64_t                m_expected;
    std::mutex              m_mutex;
    std::condition_variable m_all_done;
};

// NB: Only for tests
class GAPI_EXPORTS ThreadPool {
public:
    using Task = std::function<void()>;
    explicit ThreadPool(const uint32_t num_workers);

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void schedule(Task&& task);
    ~ThreadPool();

private:
    static void worker(QueueClass<Task>& queue);
    void shutdown();

private:
    std::vector<std::thread> m_workers;
    QueueClass<Task>         m_queue;
};

}}} // namespace cv::gapi::own

#endif // OPENCV_GAPI_THREAD_POOL_HPP
