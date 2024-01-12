// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2024 Intel Corporation


#include "thread_pool.hpp"

#include <opencv2/gapi/util/throw.hpp>

void cv::gapi::own::WaitGroup::add() {
    std::lock_guard<std::mutex> lk{m};
    ++task_counter;
}

void cv::gapi::own::WaitGroup::done() {
    std::lock_guard<std::mutex> lk{m};
    --task_counter;
    if (task_counter == 0u) {
        all_done.notify_one();
    }
}

void cv::gapi::own::WaitGroup::wait() {
    std::unique_lock<std::mutex> lk{m};
    while (task_counter != 0u) {
        all_done.wait(lk);
    }
}

cv::gapi::own::ThreadPool::ThreadPool(const uint32_t num_workers)
    : m_num_workers(num_workers) {
}

void cv::gapi::own::ThreadPool::start() {
    for (uint32_t i = 0; i < m_num_workers; ++i) {
        m_workers.emplace_back([this](){ worker(); });
    }
}

static thread_local cv::gapi::own::ThreadPool* current;

cv::gapi::own::ThreadPool* cv::gapi::own::ThreadPool::get() {
    if (!current) {
        cv::util::throw_error(
            std::logic_error("ThreadPool::get() must not be accessed"
                             " from the main thread!"));
    }
    return current;
}

void cv::gapi::own::ThreadPool::worker() {
    current = this;
    while (true) {
        cv::gapi::own::ThreadPool::Task task;
        m_queue.pop(task);
        if (!task) {
            break;
        }
        task();
        m_wg.done();
    }
}

void cv::gapi::own::ThreadPool::schedule(cv::gapi::own::ThreadPool::Task task) {
    m_wg.add();
    m_queue.push(std::move(task));
};

void cv::gapi::own::ThreadPool::wait() {
    m_wg.wait();
}

void cv::gapi::own::ThreadPool::stop() {
    wait();
    for (uint32_t i = 0; i < m_num_workers; ++i) {
        m_queue.push({});
    }
    for (auto& worker : m_workers) {
        worker.join();
    }
    m_workers.clear();
}

cv::gapi::own::ThreadPool::~ThreadPool() {
    stop();
}
