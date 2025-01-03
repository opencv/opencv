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

TEST(ThreadPool, ScheduleNotBlock)
{
    own::Latch latch(1u);
    std::atomic<uint32_t> counter{0u};

    own::ThreadPool tp(4u);
    tp.schedule([&](){
        std::this_thread::sleep_for(std::chrono::milliseconds{500u});
        counter++;
        latch.count_down();
    });

    EXPECT_EQ(0u, counter);
    latch.wait();
    EXPECT_EQ(1u, counter);
}

TEST(ThreadPool, MultipleTasks)
{
    const uint32_t kNumTasks = 100u;
    own::Latch latch(kNumTasks);
    std::atomic<uint32_t> completed{0u};

    own::ThreadPool tp(4u);
    for (uint32_t i = 0; i < kNumTasks; ++i) {
        tp.schedule([&]() {
            ++completed;
            latch.count_down();
        });
    }
    latch.wait();

    EXPECT_EQ(kNumTasks, completed.load());
}

struct ExecutionState {
    ExecutionState(const uint32_t num_threads,
                   const uint32_t num_tasks)
        : guard(0u),
          critical(0u),
          limit(num_tasks),
          latch(num_threads),
          tp(num_threads) {
    }

    std::atomic<uint32_t> guard;
    std::atomic<uint32_t> critical;
    const uint32_t        limit;
    own::Latch            latch;
    own::ThreadPool       tp;
};

static void doRecursive(ExecutionState& state) {
    // NB: Protects function to be executed no more than limit number of times
    if (state.guard.fetch_add(1u) >= state.limit) {
        state.latch.count_down();
        return;
    }
    // NB: This simulates critical section
    std::this_thread::sleep_for(std::chrono::milliseconds{50});
    ++state.critical;
    // NB: Schedule the new one recursively
    state.tp.schedule([&](){ doRecursive(state); });
}

TEST(ThreadPool, ScheduleRecursively)
{
    const int kNumThreads = 5u;
    const uint32_t kNumTasks = 100u;

    ExecutionState state(kNumThreads, kNumTasks);
    for (uint32_t i = 0; i < kNumThreads; ++i) {
        state.tp.schedule([&](){
            doRecursive(state);
        });
    }
    state.latch.wait();

    EXPECT_EQ(kNumTasks, state.critical.load());
}

TEST(ThreadPool, ExecutionIsParallel)
{
    const uint32_t kNumThreads = 4u;
    std::atomic<uint32_t> counter{0};
    own::Latch latch{kNumThreads};

    own::ThreadPool tp(kNumThreads);
    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < kNumThreads; ++i) {
      tp.schedule([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds{800u});
        ++counter;
        latch.count_down();
      });
    }
    latch.wait();

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_GE(1000u, elapsed);
    EXPECT_EQ(kNumThreads, counter.load());
}

} // namespace opencv_test
