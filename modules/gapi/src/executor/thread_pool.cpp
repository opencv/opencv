// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation

#include "thread_pool.hpp"

void cv::gapi::own::WaitGroup::add() {
    task_counter.fetch_add(1u);
}

void cv::gapi::own::WaitGroup::done() {
    if (task_counter.fetch_sub(1u) == 1u) {
        m.lock();
        m.unlock();
        all_done.notify_one();
    }
}

void cv::gapi::own::WaitGroup::wait() {
    while (task_counter.load() != 0u) {
        std::unique_lock<std::mutex> lk{m};
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

void cv::gapi::own::ThreadPool::worker() {
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
