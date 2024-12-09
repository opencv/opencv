// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

// Deliberately include .cpp file instead of header as we use non exported function (execute)
#include <executor/gtbbexecutor.cpp>

#ifdef HAVE_TBB
#include <tbb/tbb.h>
#include <tbb/task.h>
#if TBB_INTERFACE_VERSION < 12000

#include <tbb/task_arena.h>

#include "../test_precomp.hpp"
#include <thread>

namespace {
    tbb::task_arena create_task_arena(int max_concurrency  = tbb::task_arena::automatic /* set to 1 for single thread */) {
        unsigned int reserved_for_master_threads = 1;
        if (max_concurrency == 1) {
            // Leave no room for TBB worker threads, by reserving all to masters.
            // TBB runtime guarantees that no worker threads will join the arena
            // if max_concurrency is equal to reserved_for_master_threads
            // except 1:1 + use of enqueued tasks for safety guarantee.
            // So deliberately make it 2:2 to force TBB not to create extra thread.
            //
            // N.B. one slot will left empty as only one master thread(one that
            // calls root->wait_for_all()) will join the arena.

            // FIXME: strictly speaking master can take any free slot, not the first one.
            // However at the moment master seems to pick 0 slot all the time.
            max_concurrency = 2;
            reserved_for_master_threads = 2;
        }
        return tbb::task_arena{max_concurrency, reserved_for_master_threads};
    }
}

namespace opencv_test {

TEST(TBBExecutor, Basic) {
    using namespace cv::gimpl::parallel;
    bool executed = false;
    prio_items_queue_t q;
    tile_node n([&]() {
        executed = true;
    });
    q.push(&n);
    execute(q);
    EXPECT_TRUE(executed);
}

TEST(TBBExecutor, SerialExecution) {
    using namespace cv::gimpl::parallel;
    const int n = 10;
    prio_items_queue_t q;
    std::vector<tile_node> nodes; nodes.reserve(n+1);
    std::vector<std::thread::id> thread_id(n);
    for (int i=0; i <n; i++) {
        nodes.push_back(tile_node([&, i]() {
                thread_id[i] = std::this_thread::get_id();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));

        }));
        q.push(&nodes.back());
    }

    auto serial_arena = create_task_arena(1);
    execute(q, serial_arena);
    auto print_thread_ids = [&] {
        std::stringstream str;
        for (auto& i : thread_id) { str << i <<" ";}
        return str.str();
    };
    EXPECT_NE(thread_id[0], std::thread::id{}) << print_thread_ids();
    EXPECT_EQ(thread_id.size(), static_cast<size_t>(std::count(thread_id.begin(), thread_id.end(), thread_id[0])))
        << print_thread_ids();
}

TEST(TBBExecutor, AsyncBasic) {
    using namespace cv::gimpl::parallel;

    std::atomic<bool> callback_ready {false};
    std::function<void()> callback;

    std::atomic<bool> callback_called   {false};
    std::atomic<bool> master_is_waiting {true};
    std::atomic<bool> master_was_blocked_until_callback_called {false};

    auto async_thread = std::thread([&] {
            bool slept = false;
            while (!callback_ready) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                slept = true;
            }
            if (!slept) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            callback_called = true;
            master_was_blocked_until_callback_called = (master_is_waiting == true);
            callback();
    });

    auto async_task_body = [&](std::function<void()>&& cb, size_t /*total_order_index*/) {
        callback = std::move(cb);
        callback_ready = true;
    };
    tile_node n(async, std::move(async_task_body));

    prio_items_queue_t q;
    q.push(&n);
    execute(q);
    master_is_waiting = false;

    async_thread.join();

    EXPECT_TRUE(callback_called);
    EXPECT_TRUE(master_was_blocked_until_callback_called);
}

TEST(TBBExecutor, Dependencies) {
    using namespace cv::gimpl::parallel;
    const int n = 10;
    bool serial = true;
    std::atomic<int> counter {0};
    prio_items_queue_t q;
    std::vector<tile_node> nodes; nodes.reserve(n+1);
    const int invalid_order = -10;
    std::vector<int> tiles_exec_order(n, invalid_order);

    auto add_dependency_to = [](tile_node& node, tile_node& dependency) {
        dependency.dependants.push_back(&node);
        node.dependencies++;
        node.dependency_count.fetch_add(1);
    };
    for (int i=0; i <n; i++) {
        nodes.push_back(tile_node([&, i]() {
                tiles_exec_order[i] = counter++;
                if (!serial) {
                    //sleep gives a better chance for other threads to take part in the execution
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
        }));
        if (i >0) {
            auto last_node = nodes.end() - 1;
            add_dependency_to(*last_node, *(last_node -1));
        }
    }

    q.push(&nodes.front());

    auto arena = serial ? create_task_arena(1) : create_task_arena();
    execute(q, arena);
    auto print_execution_order = [&] {
        std::stringstream str;
        for (auto& i : tiles_exec_order) { str << i <<" ";}
        return str.str();
    };
    ASSERT_EQ(0, std::count(tiles_exec_order.begin(), tiles_exec_order.end(), invalid_order))
        << "Not all " << n << " task executed ?\n"
        <<" execution order : " << print_execution_order();

    for (size_t i=0; i <nodes.size(); i++) {
        auto node_exec_order = tiles_exec_order[i];
        for (auto* dependee : nodes[i].dependants) {
            auto index = std::distance(&nodes.front(), dependee);
            auto dependee_execution_order = tiles_exec_order[index];
            ASSERT_LT(node_exec_order, dependee_execution_order) << "node number " << index << " is executed earlier than it's dependency " << i;
        }
    }
}
} // namespace opencv_test

#endif //TBB_INTERFACE_VERSION
#endif //HAVE_TBB
