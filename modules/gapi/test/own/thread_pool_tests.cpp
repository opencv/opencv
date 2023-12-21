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

TEST(ThreadPool, WaitForTheCompletion)
{
    own::ThreadPool tp(4u);
    bool done = false;

    tp.start();
    tp.schedule([&done]() {
      std::this_thread::sleep_for(std::chrono::seconds{1u});
      done = true;
    });
    tp.wait();

    EXPECT_TRUE(done);
}

TEST(ThreadPool, MultipleWait)
{
    own::ThreadPool tp(1u);
    tp.start();

    const uint32_t kNumIters = 5u;
    for (uint32_t i = 0; i < kNumIters; ++i) {
        bool done = false;

        tp.schedule([&]() {
            std::this_thread::sleep_for(std::chrono::seconds{1u});
            done = true;
        });
        tp.wait();

        EXPECT_TRUE(done);
    }
}

TEST(ThreadPool, MultipleTasks)
{
    own::ThreadPool tp(4u);
    const uint32_t kNumTasks = 42u;
    std::atomic<uint32_t> counter{0u};

    tp.start();
    for (uint32_t i = 0; i < kNumTasks; ++i) {
        tp.schedule([&](){++counter;});
    }
    tp.wait();

    EXPECT_EQ(kNumTasks, counter.load());
}

TEST(ThreadPool, ParallelTasks)
{
    own::ThreadPool tp(4u);
    std::atomic<uint32_t> counter{0u};

    tp.start();
    tp.schedule([&]() {
        std::this_thread::sleep_for(std::chrono::seconds{1u});
        ++counter;
    });
    tp.schedule([&]() { ++counter; });

    std::this_thread::sleep_for(std::chrono::milliseconds{100u});
    EXPECT_EQ(1u, counter.load());

    tp.wait();
    EXPECT_EQ(2u, counter.load());
}

TEST(ThreadPool, InnerScheduleAfterWait)
{
    own::ThreadPool tp{4};
    bool done = false;

    tp.start();
    tp.schedule([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds{500u});
        tp.schedule([&](){
            std::this_thread::sleep_for(std::chrono::milliseconds{500u});
            done = true;
        });
    });
    tp.wait();

    EXPECT_TRUE(done);
}

TEST(ThreadPool, ParallelBenchmark)
{
    own::ThreadPool tp(4u);
    std::atomic<uint32_t> counter{0};

    auto start = std::chrono::high_resolution_clock::now();

    tp.start();
    for (uint32_t i = 0; i < 4; ++i) {
      tp.schedule([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds{800u});
        ++counter;
      });
    }
    tp.wait();

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_GE(1000u, elapsed);
    EXPECT_EQ(4u, counter.load());
}

} // namespace opencv_test
