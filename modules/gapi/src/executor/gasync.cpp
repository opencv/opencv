// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#include <opencv2/gapi/gcomputation_async.hpp>
#include <opencv2/gapi/gcomputation.hpp>
#include <opencv2/gapi/gcompiled_async.hpp>
#include <opencv2/gapi/gcompiled.hpp>
#include <opencv2/gapi/gasync_context.hpp>

#include <opencv2/gapi/util/copy_through_move.hpp>

#include <condition_variable>

#include <future>
#include <condition_variable>
#include <stdexcept>
#include <queue>


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

    async_service() = default ;

public:
    // singleton
    static async_service& instance()
    {
        static async_service the_ctx;
        return the_ctx;
    }

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

protected:
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

}

namespace {
template<typename f_t, typename context_t>
std::exception_ptr call_and_catch(f_t&& f, context_t&& ctx){
    if (std::forward<context_t>(ctx).isCanceled()){
        return std::make_exception_ptr(GAsyncCanceled{});
    }

    std::exception_ptr eptr;
    try {
        std::forward<f_t>(f)();
    } catch(...) {
        eptr = std::current_exception();
    }

    return eptr;
}

struct DummyContext {
    bool isCanceled() const {
        return false;
    }
};

template<typename f_t, typename callback_t, typename context_t>
void call_with_callback(f_t&& f, callback_t&& cb, context_t&& ctx){
    auto eptr =  call_and_catch(std::forward<f_t>(f), std::forward<context_t>(ctx));
    std::forward<callback_t>(cb)(eptr);
}

template<typename f_t, typename context_t>
void call_with_future(f_t&& f, std::promise<void>& p, context_t&& ctx){
    auto eptr =  call_and_catch(std::forward<f_t>(f), std::forward<context_t>(ctx));
    if (eptr){
        p.set_exception(eptr);
    }
    else {
        p.set_value();
    }
}
}//namespace

bool GAsyncContext::cancel(){
    bool expected = false;
    bool updated  = cancelation_requested.compare_exchange_strong(expected, true);
    return updated;
}

bool GAsyncContext::isCanceled() const {
    return cancelation_requested.load();
}

const char* GAsyncCanceled::what() const noexcept {
    return "GAPI asynchronous operation was canceled";
}

//For now these async functions are simply wrapping serial version of apply/operator() into a functor.
//These functors are then serialized into single queue, which is processed by a devoted background thread.
void async_apply(GComputation& gcomp, std::function<void(std::exception_ptr)>&& callback, GRunArgs &&ins, GRunArgsP &&outs, GCompileArgs &&args){
    //TODO: use copy_through_move_t for all args except gcomp
    //TODO: avoid code duplication between versions of "async" functions
    auto l = [=]() mutable {
        auto apply_l = [&](){
            gcomp.apply(std::move(ins), std::move(outs), std::move(args));
        };

        call_with_callback(apply_l,std::move(callback), DummyContext{});
    };
    impl::async_service::instance().add_task(l);
}

std::future<void> async_apply(GComputation& gcomp, GRunArgs &&ins, GRunArgsP &&outs, GCompileArgs &&args){
    util::copy_through_move_t<std::promise<void>> prms{{}};
    auto f = prms.value.get_future();
    auto l = [=]() mutable {
        auto apply_l = [&](){
            gcomp.apply(std::move(ins), std::move(outs), std::move(args));
        };

        call_with_future(apply_l, prms.value, DummyContext{});
    };

    impl::async_service::instance().add_task(l);
    return f;
}

void async_apply(GComputation& gcomp, std::function<void(std::exception_ptr)>&& callback, GRunArgs &&ins, GRunArgsP &&outs, GCompileArgs &&args, GAsyncContext& ctx){
    //TODO: use copy_through_move_t for all args except gcomp
    auto l = [=, &ctx]() mutable {
        auto apply_l = [&](){
            gcomp.apply(std::move(ins), std::move(outs), std::move(args));
        };

        call_with_callback(apply_l,std::move(callback), ctx);
    };
    impl::async_service::instance().add_task(l);
}

std::future<void> async_apply(GComputation& gcomp, GRunArgs &&ins, GRunArgsP &&outs, GCompileArgs &&args, GAsyncContext& ctx){
    util::copy_through_move_t<std::promise<void>> prms{{}};
    auto f = prms.value.get_future();
    auto l = [=, &ctx]() mutable {
        auto apply_l = [&](){
            gcomp.apply(std::move(ins), std::move(outs), std::move(args));
        };

        call_with_future(apply_l, prms.value, ctx);
    };

    impl::async_service::instance().add_task(l);
    return f;

}

void async(GCompiled& gcmpld, std::function<void(std::exception_ptr)>&& callback, GRunArgs &&ins, GRunArgsP &&outs){
    auto l = [=]() mutable {
        auto apply_l = [&](){
            gcmpld(std::move(ins), std::move(outs));
        };

        call_with_callback(apply_l,std::move(callback), DummyContext{});
    };

    impl::async_service::instance().add_task(l);
}

void async(GCompiled& gcmpld, std::function<void(std::exception_ptr)>&& callback, GRunArgs &&ins, GRunArgsP &&outs, GAsyncContext& ctx){
    auto l = [=, &ctx]() mutable {
        auto apply_l = [&](){
            gcmpld(std::move(ins), std::move(outs));
        };

        call_with_callback(apply_l,std::move(callback), ctx);
    };

    impl::async_service::instance().add_task(l);
}

std::future<void> async(GCompiled& gcmpld, GRunArgs &&ins, GRunArgsP &&outs){
    util::copy_through_move_t<std::promise<void>> prms{{}};
    auto f = prms.value.get_future();
    auto l = [=]() mutable {
        auto apply_l = [&](){
            gcmpld(std::move(ins), std::move(outs));
        };

        call_with_future(apply_l, prms.value, DummyContext{});
    };

    impl::async_service::instance().add_task(l);
    return f;

}
std::future<void> async(GCompiled& gcmpld, GRunArgs &&ins, GRunArgsP &&outs, GAsyncContext& ctx){
    util::copy_through_move_t<std::promise<void>> prms{{}};
    auto f = prms.value.get_future();
    auto l = [=, &ctx]() mutable {
        auto apply_l = [&](){
            gcmpld(std::move(ins), std::move(outs));
        };

        call_with_future(apply_l, prms.value, ctx);
    };

    impl::async_service::instance().add_task(l);
    return f;

}
}}} //namespace wip namespace gapi namespace cv
