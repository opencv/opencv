// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include "streaming/onevpl/accelerators/utils/shared_lock.hpp"


namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

size_t SharedLock::shared_lock() {
    int curr_value;
    do {
        do {
            // acquire if no writer, multiple readers are allowed
            curr_value = counter.load();
        } while (EXCLUSIVE_ACCESS == curr_value);
    } while (!counter.compare_exchange_weak(curr_value, curr_value + 1));

    //return prev value
    return curr_value;
}

size_t SharedLock::unlock_shared() {
    return counter.fetch_sub(1);
}

void SharedLock::lock() {
    int curr_value;
    do {
        // acquire if no readers only
        curr_value = 0;
    } while (!counter.compare_exchange_weak(curr_value, EXCLUSIVE_ACCESS));
}

bool SharedLock::try_lock() {
    int curr_value = 0;
    return counter.compare_exchange_strong(curr_value, EXCLUSIVE_ACCESS);
}

void SharedLock::unlock() {
    counter.store(0);
}

bool SharedLock::owns() const {
    return (counter.load() == EXCLUSIVE_ACCESS);
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
