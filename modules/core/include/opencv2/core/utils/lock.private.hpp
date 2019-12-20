// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_UTILS_LOCK_HPP
#define OPENCV_UTILS_LOCK_HPP

namespace cv { namespace utils {


/** @brief A simple scoped lock (RAII-style locking for exclusive/write access).
 *
 * Emulate std::lock_guard (C++11), partially std::unique_lock (C++11),
 */
template <class _Mutex>
class lock_guard {
public:
    typedef _Mutex Mutex;

    explicit inline lock_guard(Mutex &m) : mutex_(&m) { mutex_->lock(); }

    inline ~lock_guard() { if (mutex_) mutex_->unlock(); }

    inline void release()
    {
        CV_DbgAssert(mutex_);
        mutex_->unlock();
        mutex_ = NULL;
    }

private:
    Mutex* mutex_;

private:
    lock_guard(const lock_guard&); // disabled
    lock_guard& operator=(const lock_guard&); // disabled
};


/** @brief A shared scoped lock (RAII-style locking for shared/reader access).
 *
 * Emulate boost::shared_lock_guard, subset of std::shared_lock (C++14),
 */
template <class _Mutex>
class shared_lock_guard {
public:
    typedef _Mutex Mutex;

    explicit inline shared_lock_guard(Mutex &m) : mutex_(&m) { mutex_->lock_shared(); }

    inline ~shared_lock_guard() { if (mutex_) mutex_->unlock_shared(); }

    inline void release()
    {
        CV_DbgAssert(mutex_);
        mutex_->unlock_shared();
        mutex_ = NULL;
    }

protected:
    Mutex* mutex_;

private:
    shared_lock_guard(const shared_lock_guard&); // disabled
    shared_lock_guard& operator=(const shared_lock_guard&); // disabled
};


/** @brief An optional simple scoped lock (RAII-style locking for exclusive/write access).
 *
 * Doesn't lock if mutex pointer is NULL.
 *
 * @sa lock_guard
 */
template <class _Mutex>
class optional_lock_guard {
public:
    typedef _Mutex Mutex;

    explicit inline optional_lock_guard(Mutex* m) : mutex_(m) { if (mutex_) mutex_->lock(); }

    inline ~optional_lock_guard() { if (mutex_) mutex_->unlock(); }

private:
    Mutex* mutex_;

private:
    optional_lock_guard(const optional_lock_guard&); // disabled
    optional_lock_guard& operator=(const optional_lock_guard&); // disabled
};


/** @brief An optional shared scoped lock (RAII-style locking for shared/reader access).
 *
 * Doesn't lock if mutex pointer is NULL.
 *
 * @sa shared_lock_guard
 */
template <class _Mutex>
class optional_shared_lock_guard {
public:
    typedef _Mutex Mutex;

    explicit inline optional_shared_lock_guard(Mutex* m) : mutex_(m) { if (mutex_) mutex_->lock_shared(); }

    inline ~optional_shared_lock_guard() { if (mutex_) mutex_->unlock_shared(); }

protected:
    Mutex* mutex_;

private:
    optional_shared_lock_guard(const optional_shared_lock_guard&); // disabled
    optional_shared_lock_guard& operator=(const optional_shared_lock_guard&); // disabled
};


}} // namespace

#endif // OPENCV_UTILS_LOCK_HPP
