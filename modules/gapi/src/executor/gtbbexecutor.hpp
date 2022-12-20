// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_TBB_EXECUTOR_HPP
#define OPENCV_GAPI_TBB_EXECUTOR_HPP

#if !defined(GAPI_STANDALONE)
#include <opencv2/cvconfig.h>
#endif

#ifdef HAVE_TBB
#ifndef TBB_SUPPRESS_DEPRECATED_MESSAGES
#define TBB_SUPPRESS_DEPRECATED_MESSAGES 1
#endif
#include <tbb/tbb.h>
#include <tbb/task.h>
#if TBB_INTERFACE_VERSION < 12000
// TODO: TBB task API has been deprecated and removed in 12000

#include <atomic>
#include <vector>
#include <functional>
#include <iosfwd>

#include <tbb/concurrent_priority_queue.h>
#include <tbb/task_arena.h>

#include <opencv2/gapi/util/variant.hpp>

namespace cv { namespace gimpl { namespace parallel {

// simple wrapper to allow copies of std::atomic
template<typename  count_t>
struct atomic_copyable_wrapper {
    std::atomic<count_t> value;

    atomic_copyable_wrapper(count_t val) : value(val) {}
    atomic_copyable_wrapper(atomic_copyable_wrapper const& lhs) : value (lhs.value.load(std::memory_order_relaxed)) {}

    atomic_copyable_wrapper& operator=(count_t val) {
        value.store(val, std::memory_order_relaxed);
        return *this;
    }

    count_t fetch_sub(count_t val) {
        return value.fetch_sub(val);
    }

    count_t fetch_add(count_t val) {
        return value.fetch_add(val);
    }
};

struct async_tag {};
constexpr async_tag async;

// Class describing a piece of work in the node in the tasks graph.
// Most of the fields are set only once during graph compilation and never changes.
// (However at the moment they can not be made const due to two phase initialization
// of the tile_node objects)
// FIXME: refactor the code to make the const?
struct tile_node {
    // place in totally ordered queue of tasks to execute. Inverse to priority, i.e.
    // lower index means higher priority
    size_t                                          total_order_index = 0;

    // FIXME: use templates here instead of std::function
    struct sync_task_body {
        std::function<void()> body;
    };
    struct async_task_body {
        std::function<void(std::function<void()>&& callback, size_t total_order_index)> body;
    };

    util::variant<sync_task_body, async_task_body>  task_body;

    // number of dependencies according to a dependency graph (i.e. number of "input" edges).
    size_t                                          dependencies     = 0;

    // number of unsatisfied dependencies. When drops to zero task is ready for execution.
    // Initially equal to "dependencies"
    atomic_copyable_wrapper<size_t>                 dependency_count = 0;

    std::vector<tile_node*>                         dependants;

    tile_node(decltype(sync_task_body::body)&& f) : task_body(sync_task_body{std::move(f)}) {}
    tile_node(async_tag, decltype(async_task_body::body)&& f) : task_body(async_task_body{std::move(f)}) {}
};

std::ostream& operator<<(std::ostream& o, tile_node const& n);

struct tile_node_indirect_priority_comparator {
    bool operator()(tile_node const * lhs, tile_node const * rhs) const {
        return lhs->total_order_index > rhs->total_order_index;
    }
};

using prio_items_queue_t = tbb::concurrent_priority_queue<tile_node*, tile_node_indirect_priority_comparator>;

void execute(prio_items_queue_t& q);
void execute(prio_items_queue_t& q, tbb::task_arena& arena);

}}} // namespace cv::gimpl::parallel

#endif // TBB_INTERFACE_VERSION
#endif // HAVE_TBB

#endif // OPENCV_GAPI_TBB_EXECUTOR_HPP
