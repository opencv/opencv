// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_ACCELERATORS_UTILS_ELASTIC_BARRIER_HPP
#define GAPI_STREAMING_ONEVPL_ACCELERATORS_UTILS_ELASTIC_BARRIER_HPP
#include <atomic>

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

template<typename Impl>
class elastic_barrier {
public:
    using self_t = Impl;
    elastic_barrier() :
        incoming_requests(),
        outgoing_requests(),
        pending_requests(),
        reinit(false) {
    }

    self_t* get_self() {
        return static_cast<self_t*>(this);
    }

    template<typename ...Args>
    void visit_in (Args&& ...args) {
        on_lock(std::forward<Args>(args)...);
    }

    template<typename ...Args>
    void visit_out (Args&& ...args) {
        on_unlock(std::forward<Args>(args)...);
    }

protected:
    ~elastic_barrier() = default;

private:
    std::atomic<size_t> incoming_requests;
    std::atomic<size_t> outgoing_requests;
    std::atomic<size_t> pending_requests;
    std::atomic<bool> reinit;

    template<typename ...Args>
    void on_first_in(Args&& ...args) {
        get_self()->on_first_in_impl(std::forward<Args>(args)...);
    }

    template<typename ...Args>
    void on_last_out(Args&& ...args) {
        get_self()->on_last_out_impl(std::forward<Args>(args)...);
    }

    template<typename ...Args>
    void on_lock(Args&& ...args) {
        // Read access is more complex
        // each `incoming` request must check in before acquire resource
        size_t thread_id = incoming_requests.fetch_add(1);
        if (thread_id == 0) {
            /*
             * only one `incoming` request is allowable to init resource
             * at first time
             * let's filter out the first one by `thread_id`
             *
             * The first one `incoming` request becomes main `incoming` request
             * */
            if (outgoing_requests.load() == 0) {
                get_self()->on_first_in(std::forward<Args>(args)...);
                /*
                 * The main `incoming` request finished resource initialization
                 * and became `outgoing`
                 *
                 * Non empty `outgoing` count means that
                 * other further `incoming` (or busy-wait) requests
                 * are getting on with its job without resource initialization,
                 * because main `incoming` request has already initialized it at here
                 * */
                outgoing_requests.fetch_add(1);
                return;
            }
            return;
        } else {
            /*
             * CASE 1)
             *
             * busy wait for others `incoming` requests for resource initialization
             * besides main `incoming` request which are getting on
             * resource initialization at this point
             *
             * */

            // OR

            /*
             * CASE 2)
             *
             * busy wait for ALL `incoming` request for resource initialization
             * including main `incoming` request. It will happen if
             * new `incoming` requests had came here while resource was getting on deinit
             * in `on_unlock` in another processing thread.
             * In this case no actual main `incoming` request is available and
             * all `incoming` requests must be in busy-wait stare
             *
             * */

            // Each `incoming` request became `busy-wait` request
            size_t busy_thread_id = pending_requests.fetch_add(1);

            /*
             * CASE 1)
             *
             * Non empty `outgoing` requests count means that other further `incoming` or
             * `busy-wait` request are getting on with its job
             * without resource initialization because
             * main thread has already initialized it at here
             * */
            while (outgoing_requests.load() == 0) {

                // OR

                /*
                 * CASE 2)
                 *
                 * In case of NO master `incoming `request is available and doesn't
                 * provide resource initialization. All `incoming` requests must be in
                 * busy-wait state.
                 * If it is not true then CASE 1) is going on
                 *
                 * OR
                 *
                 * `on_unlock` is in deinitialization phase in another thread.
                 * Both cases mean busy-wait state here
                 * */
                if (pending_requests.load() == incoming_requests.load()) {
                    /*
                     * CASE 2) ONLY
                     *
                     * It will happen if 'on_unlock` in another thread
                     * finishes its execution only
                     *
                     * `on_unlock` in another thread might finished with either
                     * deinitialization action or without deinitialization action
                     * (the call off deinitialization case)
                     *
                     * We must not continue at here (without reinit)
                     * if deinitialization happens in `on_unlock` in another thread.
                     * So try it on
                     * */

                    // only single `busy-wait` request must make sure about possible
                    // deinitialization. So first `busy-wait` request becomes
                    // main `busy-wait` request
                    if (busy_thread_id == 0) {
                        bool expected_reinit = true;
                        if (!reinit.compare_exchange_strong(expected_reinit, false)) {
                            /*
                             * deinitialization called off in `on_unlock`
                             * because new `incoming` request had appeared at here before
                             * `on_unlock` started deinit procedure in another thread.
                             * So no reinit required because no deinit had happened
                             *
                             * main `busy-wait` request must break busy-wait state
                             * and become `outgoing` request.
                             * Non empty `outgoing` count means that other
                             * further `incoming` requests or
                             * `busy-wait` requests are getting on with its job
                             * without resource initialization/reinitialization
                             * because no deinit happened in `on_unlock`
                             * in another thread
                             * */
                            break; //just quit busy loop
                        } else {
                            /* Deinitialization had happened in `on_unlock`
                             * in another thread right before
                             * new `incoming` requests appeared.
                             * So main `busy-wait` request must start reinit procedure
                             */
                            get_self()->on_first_in(std::forward<Args>(args)...);

                            /*
                             * Main `busy-wait` request has finished reinit procedure
                             * and becomes `outgong` request.
                             * Non empty `outgoing` count means that other
                             * further `incoming` requests or
                             * `busy-wait` requests are getting on with its job
                             * without resource initialization because
                             * main `busy-wait` request
                             * has already re-initialized it at here
                             */
                            outgoing_requests.fetch_add(1);
                            pending_requests.fetch_sub(1);
                            return;
                        }
                    }
                }
            }

            // All non main requests became `outgoing` and look at on initialized resource
            outgoing_requests++;

            // Each `busy-wait` request are not busy-wait now
            pending_requests.fetch_sub(1);
        }
        return;
    }

    template<typename ...Args>
    void on_unlock(Args&& ...args) {
        // Read unlock
        /*
        * Each released `outgoing` request checks out to doesn't use resource anymore.
        * The last `outgoing` request becomes main `outgoing` request and
        * must deinitialize resource if no `incoming` or `busy-wait` requests
        * are waiting for it
        */
        size_t thread_id = outgoing_requests.fetch_sub(1);
        if (thread_id == 1) {
            /*
            * Make sure that no another `incoming` (including `busy-wait)
            * exists.
            * But beforehand its must make sure that no `incoming` or `pending`
            * requests are exist.
            *
            * The main `outgoing` request is an one of `incoming` request
            * (it is the oldest one in the current `incoming` bunch) and still
            * holds resource in initialized state (thus we compare with 1).
            * We must not deinitialize resource before decrease
            * `incoming` requests counter because
            * after it has got 0 value in `on_lock` another thread
            * will start initialize resource procedure which will get conflict
            * with current deinitialize procedure
            *
            * From this point, all `on_lock` request in another thread would
            * become `busy-wait` without reaching main `incoming` state (CASE 2)
            * */
            if (incoming_requests.load() == 1) {
                /*
                * The main `outgoing` request is ready to deinit shared resource
                * in unconflicting manner.
                *
                * This is a critical section for single thread for main `outgoing`
                * request
                *
                * CASE 2 only available in `on_lock` thread
                * */
                get_self()->on_last_out(std::forward<Args>(args)...);

                /*
                * Before main `outgoinq` request become released it must notify
                * subsequent `busy-wait` requests in `on_lock` in another thread
                * that main `busy-wait` must start reinit resource procedure
                * */
                reinit.store(true);

                /*
                * Deinitialize procedure is finished and main `outgoing` request
                * (it is the oldest one in `incoming` request) must become released
                *
                * Right after when we decrease `incoming` counter
                * the condition for equality
                * `busy-wait` and `incoming` counter will become true (CASE 2 only)
                * in `on_lock` in another threads. After that
                * a main `busy-wait` request would check `reinit` condition
                * */
                incoming_requests.fetch_sub(1);
                return;
            }

            /*
            * At this point we have guarantee that new `incoming` requests
            * had became increased in `on_lock` in another thread right before
            * current thread deinitialize resource.
            *
            * So call off deinitialization procedure here
            * */
        }
        incoming_requests.fetch_sub(1);
    }

    elastic_barrier(const elastic_barrier&) = delete;
    elastic_barrier(elastic_barrier&&) = delete;
    elastic_barrier& operator() (const elastic_barrier&) = delete;
    elastic_barrier& operator() (elastic_barrier&&) = delete;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // GAPI_STREAMING_ONEVPL_ACCELERATORS_UTILS_ELASTIC_BARRIER_HPP
