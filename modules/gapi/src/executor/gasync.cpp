// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#include "opencv2/gapi/gcomputation_async.hpp"
#include "opencv2/gapi/gcomputation.hpp"
#include "opencv2/gapi/gcompiled_async.hpp"
#include "opencv2/gapi/gcompiled.hpp"

#include <condition_variable>

#include <future>
#include <condition_variable>
#include <stdexcept>
#include <queue>

namespace {
    //This is a tool to move initialize captures of a lambda in C++11
    template<typename T>
    struct move_through_copy{
       T value;
       move_through_copy(T&& g) : value(std::move(g)) {}
       move_through_copy(move_through_copy&&) = default;
       move_through_copy(move_through_copy const& lhs) : move_through_copy(std::move(const_cast<move_through_copy&>(lhs))) {}
    };
}

namespace cv {
namespace gapi {
namespace wip  {

namespace impl{

class async_service {

    std::mutex mtx;
    std::condition_variable cv;
    std::queue<std::function<void()>> q;
    std::atomic<bool> exiting           = {false};
    std::atomic<bool> thread_started    = {false};

    std::thread thrd;

public:
    async_service() = default ;

    void add_task(std::function<void()>&& t){
        if (!thread_started)
        {
            //thread has not been started yet, so start it
            //try to Compare And Swap the flag, false -> true
            //If there are multiple threads - only single one will succeed in changing the value.
            bool expected = false;
            if (thread_started.compare_exchange_strong(expected, true))
            {
                //have won (probable) race - so actually start the thread
                thrd = std::thread {[this](){
                    //move the whole queue into local instance in order to minimize time the guarding lock is held
                    decltype(q) second_q;
                    while (!exiting){
                        std::unique_lock<std::mutex> lck{mtx};
                        if (q.empty())
                        {
                            //block current thread until arrival of exit request or new elements
                            cv.wait(lck, [&](){ return exiting || !q.empty();});
                        }
                        //usually swap for std::queue is plain pointers exchange, so relatively cheap
                        q.swap(second_q);
                        lck.unlock();

                        while (!second_q.empty())
                        {
                            auto& f = second_q.front();
                            f();
                            second_q.pop();
                        }
                    }
                }};
            }
        }
        std::unique_lock<std::mutex> lck{mtx};
        bool first_task = q.empty();
        q.push(std::move(t));
        lck.unlock();

        if (first_task)
        {
            //as the queue was empty before adding the task,
            //the thread might be sleeping, so wake it up
            cv.notify_one();
        }
    }
    ~async_service(){
        if (thread_started && thrd.joinable())
        {
            exiting = true;
            mtx.lock();
            mtx.unlock();
            cv.notify_one();
            thrd.join();
        }
    }
};

async_service the_ctx;
}

namespace {
template<typename f_t>
std::exception_ptr call_and_catch(f_t&& f){
    std::exception_ptr eptr;
    try {
        std::forward<f_t>(f)();
    } catch(...) {
        eptr = std::current_exception();
    }

    return eptr;
}

template<typename f_t, typename callback_t>
void call_with_callback(f_t&& f, callback_t&& cb){
    auto eptr = call_and_catch(std::forward<f_t>(f));
    std::forward<callback_t>(cb)(eptr);
}

template<typename f_t>
void call_with_futute(f_t&& f, std::promise<void>& p){
    auto eptr = call_and_catch(std::forward<f_t>(f));
    if (eptr){
        p.set_exception(eptr);
    }
    else {
        p.set_value();
    }
}
}//namespace

//For now these async functions are simply wrapping serial version of apply/operator() into a functor.
//These functors are then serialized into single queue, which is processed by a devoted background thread.
void async_apply(GComputation& gcomp, std::function<void(std::exception_ptr)>&& callback, GRunArgs &&ins, GRunArgsP &&outs, GCompileArgs &&args){
    //TODO: use move_through_copy for all args except gcomp
    auto l = [=]() mutable {
        auto apply_l = [&](){
            gcomp.apply(std::move(ins), std::move(outs), std::move(args));
        };

        call_with_callback(apply_l,std::move(callback));
    };
    impl::the_ctx.add_task(l);
}

std::future<void> async_apply(GComputation& gcomp, GRunArgs &&ins, GRunArgsP &&outs, GCompileArgs &&args){
    move_through_copy<std::promise<void>> prms{{}};
    auto f = prms.value.get_future();
    auto l = [=]() mutable {
        auto apply_l = [&](){
            gcomp.apply(std::move(ins), std::move(outs), std::move(args));
        };

        call_with_futute(apply_l, prms.value);
    };

    impl::the_ctx.add_task(l);
    return f;
}

void async(GCompiled& gcmpld, std::function<void(std::exception_ptr)>&& callback, GRunArgs &&ins, GRunArgsP &&outs){
    auto l = [=]() mutable {
        auto apply_l = [&](){
            gcmpld(std::move(ins), std::move(outs));
        };

        call_with_callback(apply_l,std::move(callback));
    };

    impl::the_ctx.add_task(l);
}

std::future<void> async(GCompiled& gcmpld, GRunArgs &&ins, GRunArgsP &&outs){
    move_through_copy<std::promise<void>> prms{{}};
    auto f = prms.value.get_future();
    auto l = [=]() mutable {
        auto apply_l = [&](){
            gcmpld(std::move(ins), std::move(outs));
        };

        call_with_futute(apply_l, prms.value);
    };

    impl::the_ctx.add_task(l);
    return f;

}
}}} //namespace wip namespace gapi namespace cv
