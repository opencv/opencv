// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#include "test_precomp.hpp"
#include "opencv2/gapi/gcomputation_async.hpp"
#include "opencv2/gapi/gcompiled_async.hpp"

#include <condition_variable>
#include <stdexcept>

namespace opencv_test
{
//Main idea behind these tests is to have the same test script that is parameterized in order to test all setups (GCompiled vs apply, callback vs future).
//So these differences are factored into devoted helper classes (mixins) which are then used by the common test script by help of CRTP.
//Actual GAPI Computation with parameters to run on is mixed into test via CRTP as well.

struct SumOfSum2x2 {
    cv::GComputation sum_of_sum;
    SumOfSum2x2() : sum_of_sum([]{
        cv::GMat in;
        cv::GScalar out = cv::gapi::sum(in + in);
        return GComputation{in, out};
    })
    {}

    const cv::Size sz{2, 2};
    cv::Mat in_mat{sz, CV_8U, cv::Scalar(1)};
    cv::Scalar out_sc;

    cv::GCompiled compile(){
        return sum_of_sum.compile(descr_of(in_mat));
    }

    cv::GComputation& computation(){
        return sum_of_sum;
    }

    cv::GCompileArgs compile_args(){
        return {};
    }

    cv::GRunArgs in_args(){
        return cv::gin(in_mat);
    }

    cv::GRunArgsP out_args(){
        return cv::gout(out_sc);
    }

    void verify(){
        EXPECT_EQ(8, out_sc[0]);
    }
};

namespace {
    G_TYPED_KERNEL(GThrow, <GMat(GMat)>, "org.opencv.test.throw")
    {
        static GMatDesc outMeta(GMatDesc in) { return in;  }

    };

    struct gthrow_exception : std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    GAPI_OCV_KERNEL(GThrowImpl, GThrow)
    {
        static void run(const cv::Mat& in, cv::Mat&)
        {
            //this condition is needed to avoid "Unreachable code" warning on windows inside OCVCallHelper
            if (!in.empty())
            {
                throw gthrow_exception{"test"};
            }
        }
    };
}

struct ExceptionOnExecution {
    cv::GComputation throwing_gcomp;
    ExceptionOnExecution() : throwing_gcomp([]{
        cv::GMat in;
        auto gout = GThrow::on(in);
        return GComputation{in, gout};
    })
    {}


    const cv::Size sz{2, 2};
    cv::Mat in_mat{sz, CV_8U, cv::Scalar(1)};
    cv::Mat out;

    cv::GCompiled compile(){
        return throwing_gcomp.compile(descr_of(in_mat), compile_args());
    }

    cv::GComputation& computation(){
        return throwing_gcomp;
    }

    cv::GRunArgs in_args(){
        return cv::gin(in_mat);
    }

    cv::GRunArgsP out_args(){
        return cv::gout(out);
    }

    cv::GCompileArgs compile_args(){
        auto pkg = cv::gapi::kernels<GThrowImpl>();
        return cv::compile_args(pkg);
    }

};

template<typename crtp_final_t>
struct crtp_cast {
    template<typename crtp_base_t>
    static crtp_final_t* crtp_cast_(crtp_base_t* this_)
    {
        return  static_cast<crtp_final_t*>(this_);
    }
};

//Test Mixin, hiding details of callback based notification
template<typename crtp_final_t>
struct CallBack: crtp_cast<crtp_final_t> {
    std::atomic<bool> callback_called = {false};
    std::mutex mtx;
    std::exception_ptr ep;

    std::condition_variable cv;

    std::function<void(std::exception_ptr)> callback(){
        return [&](std::exception_ptr ep_){
            ep = ep_;
            callback_called = true;
            mtx.lock();
            mtx.unlock();
            cv.notify_one();
        };
    };

    template<typename... Args >
    void start_async(Args&&... args){
        this->crtp_cast_(this)->async(callback(), std::forward<Args>(args)...);
    }

    void wait_for_result()
    {
        std::unique_lock<std::mutex> lck{mtx};
        cv.wait(lck,[&]{return callback_called == true;});
        if (ep)
        {
            std::rethrow_exception(ep);
        }
    }
};

//Test Mixin, hiding details of future based notification
template<typename crtp_final_t>
struct Future: crtp_cast<crtp_final_t> {
    std::future<void> f;

    template<typename... Args >
    void start_async(Args&&... args){
        f = this->crtp_cast_(this)->async(std::forward<Args>(args)...);
    }

    void wait_for_result()
    {
        f.get();
    }
};

//Test Mixin, hiding details of using compiled GAPI object
template<typename crtp_final_t>
struct AsyncCompiled  : crtp_cast<crtp_final_t>{

    template<typename... Args>
    auto async(Args&&... args) -> decltype(cv::gapi::wip::async(std::declval<cv::GCompiled&>(), std::forward<Args>(args)...)){
        auto gcmpld = this->crtp_cast_(this)->compile();
        return cv::gapi::wip::async(gcmpld, std::forward<Args>(args)...);
    }
};

//Test Mixin, hiding details of calling apply (async_apply) on GAPI Computation object
template<typename crtp_final_t>
struct AsyncApply : crtp_cast<crtp_final_t> {

    template<typename... Args>
    auto async(Args&&... args) ->decltype(cv::gapi::wip::async_apply(std::declval<cv::GComputation&>(), std::forward<Args>(args)...)) {
        return cv::gapi::wip::async_apply(this->crtp_cast_(this)->computation(), std::forward<Args>(args)..., this->crtp_cast_(this)->compile_args());
    }
};


template<typename case_t>
struct normal: ::testing::Test, case_t{};

TYPED_TEST_CASE_P(normal);

TYPED_TEST_P(normal, basic){
    //Normal scenario:  start function asynchronously and wait for the result, and verify it
    this->start_async(this->in_args(), this->out_args());
    this->wait_for_result();

    this->verify();
}

REGISTER_TYPED_TEST_CASE_P(normal,
        basic
);

template<typename case_t>
struct exception: ::testing::Test, case_t{};
TYPED_TEST_CASE_P(exception);

TYPED_TEST_P(exception, basic){
    //Exceptional scenario:  start function asynchronously and make sure exception is passed to the user
    this->start_async(this->in_args(), this->out_args());
    EXPECT_THROW(this->wait_for_result(), gthrow_exception);
}

REGISTER_TYPED_TEST_CASE_P(exception,
        basic
);

template<typename case_t>
struct stress : ::testing::Test{};
TYPED_TEST_CASE_P(stress);

TYPED_TEST_P(stress, test){
    //Some stress testing: use a number of threads to start a bunch of async requests
    const std::size_t request_per_thread = 10;
    const std::size_t number_of_threads  = 4;

    auto thread_body = [&](){
        std::vector<TypeParam> requests{request_per_thread};
        for (auto&& r : requests){
            r.start_async(r.in_args(), r.out_args());
        }

        for (auto&& r : requests){
            r.wait_for_result();
            r.verify();
        }
    };

    std::vector<std::thread> pool {number_of_threads};
    for (auto&& t : pool){
        t = std::thread{thread_body};
    }

    for (auto&& t : pool){
        t.join();
    }
}
REGISTER_TYPED_TEST_CASE_P(stress, test);

//little helpers to match up all combinations of setups
template<typename compute_fixture_t,template <typename> class callback_or_future_t, template <typename> class compiled_or_apply_t>
struct Case
        : compute_fixture_t,
          callback_or_future_t<Case<compute_fixture_t,callback_or_future_t,compiled_or_apply_t>>,
          compiled_or_apply_t <Case<compute_fixture_t,callback_or_future_t,compiled_or_apply_t>>
{};

template<typename computation_t>
using cases = ::testing::Types<
            Case<computation_t, CallBack, AsyncCompiled>,
            Case<computation_t, CallBack, AsyncApply>,
            Case<computation_t, Future,   AsyncCompiled>,
            Case<computation_t, Future,   AsyncApply>
            >;
INSTANTIATE_TYPED_TEST_CASE_P(AsyncAPINormalFlow_,        normal,     cases<SumOfSum2x2>);
INSTANTIATE_TYPED_TEST_CASE_P(AsyncAPIExceptionHandling_, exception,  cases<ExceptionOnExecution>);

INSTANTIATE_TYPED_TEST_CASE_P(AsyncAPIStress,             stress,     cases<SumOfSum2x2>);

TEST(AsyncAPI, Sample){
    cv::GComputation self_mul([]{
        cv::GMat in;
        cv::GMat out = cv::gapi::mul(in, in);
        return GComputation{in, out};
    });

    const cv::Size sz{2, 2};
    cv::Mat in_mat{sz, CV_8U, cv::Scalar(1)};
    cv::Mat out;

    auto f = cv::gapi::wip::async_apply(self_mul,cv::gin(in_mat), cv::gout(out));
    f.wait();
}
} // namespace opencv_test
