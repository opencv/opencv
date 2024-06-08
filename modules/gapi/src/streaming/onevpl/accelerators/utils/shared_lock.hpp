// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_ACCELERATORS_SURFACE_SHARED_LOCK_HPP
#define GAPI_STREAMING_ONEVPL_ACCELERATORS_SURFACE_SHARED_LOCK_HPP

#include <atomic>
#include <memory>

#include "opencv2/gapi/own/exports.hpp" // GAPI_EXPORTS

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

class GAPI_EXPORTS SharedLock {
public:
    SharedLock();
    ~SharedLock() = default;

    size_t shared_lock();
    size_t unlock_shared();

    void lock();
    bool try_lock();
    void unlock();

    bool owns() const;
private:
    SharedLock(const SharedLock&) = delete;
    SharedLock& operator= (const SharedLock&) = delete;
    SharedLock(SharedLock&&) = delete;
    SharedLock& operator== (SharedLock&&) = delete;

    std::atomic<bool> exclusive_lock;
    std::atomic<size_t> shared_counter;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // GAPI_STREAMING_ONEVPL_ACCELERATORS_SURFACE_SHARED_LOCK_HPP
