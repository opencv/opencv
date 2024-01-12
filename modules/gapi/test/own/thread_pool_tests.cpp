// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2024 Intel Corporation

#include "../test_precomp.hpp"

#include <chrono>
#include <thread>

#include "executor/thread_pool.hpp"

namespace opencv_test
{

using namespace cv::gapi;

TEST(ThreadPool, SingleTaskMultipleThreads)
{
    own::ThreadPool tp(4u);
    uint32_t counter = 0u;

    tp.start();
    tp.schedule([&counter](){counter++;});
    tp.wait();

    EXPECT_EQ(1u, counter);
}

TEST(ThreadPool, WaitOnEmptyQueue)
{
    own::ThreadPool tp(4u);
    tp.wait();
}

TEST(ThreadPool, WaitForTheCompletion)
{
    own::ThreadPool tp(4u);
    bool done = false;

    tp.start();
    tp.schedule([&done]() {
      std::this_thread::sleep_for(std::chrono::milliseconds{500u});
      done = true;
    });
    tp.wait();

    EXPECT_TRUE(done);
}

TEST(ThreadPool, MultipleTasks)
{
    own::ThreadPool tp(4u);
    tp.start();

    const uint32_t kNumTasks = 100u;
    std::atomic<uint32_t> completed{0u};
    for (uint32_t i = 0; i < kNumTasks; ++i) {
        tp.schedule([&]() {
            ++completed;
        });
    }
    tp.wait();

    EXPECT_EQ(kNumTasks, completed.load());
}

TEST(ThreadPool, OrderedExecution)
{
    own::ThreadPool tp(4u);
    tp.start();

    const uint32_t kNumTasks = 100u;
    std::atomic<uint32_t> completed{0u};
    for (uint32_t i = 0; i < kNumTasks; ++i) {
        tp.schedule([&]() {
            ++completed;
        });
        tp.wait();
        EXPECT_EQ(i+1, completed.load());
    }
}

static void doRecursive(std::atomic<uint32_t> &scheduled,
                        std::atomic<uint32_t> &executed,
                        const uint32_t        limit) {
    // NB: Protects function to be executed more than limit number of times
    if (scheduled.fetch_add(1u) >= limit) {
        return;
    }
    // NB: This is the execution part
    ++executed;
    // NB: Schedule the new one recursively
    own::ThreadPool::get()->schedule([&scheduled, &executed, limit](){
        doRecursive(scheduled, executed, limit);
    });
}

TEST(ThreadPool, ScheduleRecursively)
{
    const int kNumThreads = 1u;
    own::ThreadPool tp(kNumThreads);
    tp.start();

    const uint32_t kNumTasks = 10000u;
    std::atomic<uint32_t> scheduled{0u};
    std::atomic<uint32_t> executed {0u};

    // NB: Run initial task which will repeat
    // until executed is equal to limit
    tp.schedule([&](){
        doRecursive(scheduled, executed, kNumTasks);
    });
    tp.wait();

    EXPECT_EQ(kNumTasks, executed.load());
}

TEST(ThreadPool, ScheduleFromDifferentThreads)
{
    const int kNumThreads = 10u;
    own::ThreadPool tp(kNumThreads);
    tp.start();

    const uint32_t kNumTasks = 10000u;
    std::atomic<uint32_t> scheduled{0u};
    std::atomic<uint32_t> executed {0u};

    // NB: Run multiple initial tasks that will be
    // executed in different threads recursively
    for (uint32_t i = 0; i < kNumThreads; ++i) {
        tp.schedule([&](){
            doRecursive(scheduled, executed, kNumTasks);
        });
    }
    tp.wait();

    EXPECT_EQ(kNumTasks, executed.load());
}

TEST(ThreadPool, ExecutionIsParallel)
{
    const uint32_t kNumThreads = 4u;
    own::ThreadPool tp(kNumThreads);
    std::atomic<uint32_t> counter{0};

    auto start = std::chrono::high_resolution_clock::now();

    tp.start();
    for (uint32_t i = 0; i < kNumThreads; ++i) {
      tp.schedule([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds{800u});
        ++counter;
      });
    }
    tp.wait();

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_GE(1000u, elapsed);
    EXPECT_EQ(kNumThreads, counter.load());
}

} // namespace opencv_test
