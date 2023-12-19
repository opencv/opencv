// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation

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

class WaitGroup {
public:
    void add();
    void done();
    void wait();

private:
    std::atomic<uint64_t>   task_counter{0u};
    std::mutex              m;
    std::condition_variable all_done;
};

class GAPI_EXPORTS ThreadPool {
public:
    explicit ThreadPool(const uint32_t num_workers);
    using Task = std::function<void()>;

    void start();
    void schedule(Task task);
    void wait();
    void stop();

    ~ThreadPool();

private:
    void worker();

private:
    uint32_t                 m_num_workers;
    std::vector<std::thread> m_workers;
    QueueClass<Task>         m_queue;
    WaitGroup                m_wg;
};

}}} // namespace cv::gapi::own

#endif // OPENCV_GAPI_THREAD_POOL_HPP
