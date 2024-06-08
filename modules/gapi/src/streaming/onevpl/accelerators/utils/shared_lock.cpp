// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <thread>
#include "streaming/onevpl/accelerators/utils/shared_lock.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

SharedLock::SharedLock() {
    exclusive_lock.store(false);
    shared_counter.store(0);
}

size_t SharedLock::shared_lock() {
    size_t prev = 0;
    bool in_progress = false;
    bool pred_excl = exclusive_lock.load();
    do {
        if (!pred_excl) {
            // if no exclusive lock then start shared lock transaction
            prev = shared_counter.fetch_add(1);
            in_progress = true; // transaction is in progress
        } else {
            if (in_progress) {
                in_progress = false;
                shared_counter.fetch_sub(1);
            }
            std::this_thread::yield();
        }

        // test if exclusive lock happened before
        pred_excl = exclusive_lock.load();
    } while (pred_excl || !in_progress);

    return prev;
}

size_t SharedLock::unlock_shared() {
    return shared_counter.fetch_sub(1);
}

void SharedLock::lock() {
    bool in_progress = false;
    size_t prev_shared = shared_counter.load();
    do {
        if (prev_shared == 0) {
            bool expected = false;
            while (!exclusive_lock.compare_exchange_strong(expected, true)) {
                expected = false;
                std::this_thread::yield();
            }
            in_progress = true;
        } else {
            if (in_progress) {
                in_progress = false;
                exclusive_lock.store(false);
            }
            std::this_thread::yield();
        }
        prev_shared = shared_counter.load();
    } while (prev_shared != 0 || !in_progress);
}

bool SharedLock::try_lock() {
    if (shared_counter.load() != 0) {
        return false;
    }

    bool expected = false;
    if (exclusive_lock.compare_exchange_strong(expected, true)) {
        if (shared_counter.load() == 0) {
            return true;
        } else {
            exclusive_lock.store(false);
        }
    }
    return false;
}

void SharedLock::unlock() {
    exclusive_lock.store(false);
}
bool SharedLock::owns() const {
    return exclusive_lock.load();
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
