// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#include "test_precomp.hpp"
#include <opencv2/gapi/gcomputation_async.hpp>
#include <opencv2/gapi/gcompiled_async.hpp>
#include <opencv2/gapi/gasync_context.hpp>


#include <condition_variable>
#include <stdexcept>
#include <thread>

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


    //TODO: unify with callback helper code
    struct cancel_struct {
        std::atomic<int> num_tasks_to_spawn;

        cv::gapi::wip::GAsyncContext ctx;

        cancel_struct(int tasks_to_spawn) : num_tasks_to_spawn(tasks_to_spawn) {}
    };

    G_TYPED_KERNEL(GCancelationAdHoc, <GMat(GMat, cancel_struct*)>, "org.opencv.test.cancel_ad_hoc")
    {
        static GMatDesc outMeta(GMatDesc in, cancel_struct* ) { return in;  }

    };

    GAPI_OCV_KERNEL(GCancelationAdHocImpl, GCancelationAdHoc)
    {
        static void run(const cv::Mat& , cancel_struct* cancel_struct_p, cv::Mat&)        {
            auto& cancel_struct_ = * cancel_struct_p;
            auto num_tasks_to_spawn =  -- cancel_struct_.num_tasks_to_spawn;
            cancel_struct_.ctx.cancel();
            EXPECT_GT(num_tasks_to_spawn, 0)<<"Incorrect Test setup - to small number of tasks to feed the queue \n";
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

struct SelfCanceling {
    cv::GComputation self_cancel;
    SelfCanceling(cancel_struct* cancel_struct_p) : self_cancel([cancel_struct_p]{
        cv::GMat in;
        cv::GMat out = GCancelationAdHoc::on(in, cancel_struct_p);
        return GComputation{in, out};
    })
    {}

    const cv::Size sz{2, 2};
    cv::Mat in_mat{sz, CV_8U, cv::Scalar(1)};
    cv::Mat out_mat;

    cv::GCompiled compile(){
        return self_cancel.compile(descr_of(in_mat), compile_args());
    }

    cv::GComputation& computation(){
        return self_cancel;
    }

    cv::GRunArgs in_args(){
        return cv::gin(in_mat);
    }

    cv::GRunArgsP out_args(){
        return cv::gout(out_mat);
    }

    cv::GCompileArgs compile_args(){
        auto pkg = cv::gapi::kernels<GCancelationAdHocImpl>();
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
    }

    template<typename... Args >
    void start_async(Args&&... args){
        this->crtp_cast_(this)->async(callback(), std::forward<Args>(args)...);
    }

    template<typename... Args >
    void start_async(cv::gapi::wip::GAsyncContext& ctx, Args&&... args){
        this->crtp_cast_(this)->async(ctx, callback(), std::forward<Args>(args)...);
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

    template<typename... Args>
    auto async(cv::gapi::wip::GAsyncContext& ctx, Args&&... args) ->
        decltype(cv::gapi::wip::async(std::declval<cv::GCompiled&>(), std::forward<Args>(args)..., std::declval<cv::gapi::wip::GAsyncContext&>()))
    {
        auto gcmpld = this->crtp_cast_(this)->compile();
        return cv::gapi::wip::async(gcmpld, std::forward<Args>(args)..., ctx);
    }
};

//Test Mixin, hiding details of calling apply (async_apply) on GAPI Computation object
template<typename crtp_final_t>
struct AsyncApply : crtp_cast<crtp_final_t> {

    template<typename... Args>
    auto async(Args&&... args) ->
         decltype(cv::gapi::wip::async_apply(std::declval<cv::GComputation&>(), std::forward<Args>(args)..., std::declval<cv::GCompileArgs>()))
    {
        return cv::gapi::wip::async_apply(
                this->crtp_cast_(this)->computation(), std::forward<Args>(args)..., this->crtp_cast_(this)->compile_args()
        );
    }

    template<typename... Args>
    auto async(cv::gapi::wip::GAsyncContext& ctx, Args&&... args) ->
         decltype(cv::gapi::wip::async_apply(std::declval<cv::GComputation&>(), std::forward<Args>(args)... , std::declval<cv::GCompileArgs>(), std::declval<cv::gapi::wip::GAsyncContext&>()))
    {
        return cv::gapi::wip::async_apply(
                this->crtp_cast_(this)->computation(), std::forward<Args>(args)..., this->crtp_cast_(this)->compile_args(), ctx
        );
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
        std::vector<TypeParam> requests(request_per_thread);
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

template<typename case_t>
struct cancel : ::testing::Test{};
TYPED_TEST_CASE_P(cancel);

TYPED_TEST_P(cancel, basic)
{
#if defined(__GNUC__) && __GNUC__ >= 11
    // std::vector<TypeParam> requests can't handle type with ctor parameter (SelfCanceling)
    FAIL() << "Test code is not available due to compilation error with GCC 11";
#else
    constexpr int num_tasks = 100;
    cancel_struct cancel_struct_ {num_tasks};
    std::vector<TypeParam> requests; requests.reserve(num_tasks);

    for (auto i = num_tasks; i>0; i--){
        requests.emplace_back(&cancel_struct_);
    }
    for (auto&& r : requests){
        //first request will cancel other on it's execution
        r.start_async(cancel_struct_.ctx, r.in_args(), r.out_args());
    }

    unsigned int canceled = 0 ;
    for (auto&& r : requests){
        try {
            r.wait_for_result();
        }catch (cv::gapi::wip::GAsyncCanceled&){
            ++canceled;
        }
    }
    ASSERT_GT(canceled, 0u);
#endif
}

namespace {
    GRunArgs deep_copy_out_args(const GRunArgsP& args ){
        GRunArgs result; result.reserve(args.size());
        for (auto&& arg : args){
            //FIXME: replace this switch with use of visit() on variant, when it will be available
            switch (arg.index()){
                case GRunArgP::index_of<cv::UMat*>()                :   result.emplace_back(*util::get<cv::UMat*>(arg));    break;
                case GRunArgP::index_of<cv::Mat*>()                 :   result.emplace_back(*util::get<cv::Mat*>(arg));     break;
                case GRunArgP::index_of<cv::Scalar*>()              :   result.emplace_back(*util::get<cv::Scalar*>           (arg));   break;
                case GRunArgP::index_of<cv::detail::VectorRef>()    :   result.emplace_back(util::get<cv::detail::VectorRef>  (arg));   break;
                default : ;
            }
        }
        return result;
    }

    GRunArgsP args_p_from_args(GRunArgs& args){
        GRunArgsP result; result.reserve(args.size());
        for (auto&& arg : args){
            switch (arg.index()){
                case GRunArg::index_of<cv::Mat>()                 :   result.emplace_back(&util::get<cv::Mat>(arg));     break;
                case GRunArg::index_of<cv::UMat>()                :   result.emplace_back(&util::get<cv::UMat>(arg));    break;
                case GRunArg::index_of<cv::Scalar>()              :   result.emplace_back(&util::get<cv::Scalar>           (arg));   break;
                case GRunArg::index_of<cv::detail::VectorRef>()   :   result.emplace_back(util::get<cv::detail::VectorRef> (arg));   break;
                default : ;
            }
        }
        return result;
    }
}

REGISTER_TYPED_TEST_CASE_P(cancel, basic);

template<typename case_t>
struct output_args_lifetime : ::testing::Test{
    static constexpr const int num_of_requests = 20;
};
TYPED_TEST_CASE_P(output_args_lifetime);
//There are intentionally no actual checks (asserts and verify) in output_args_lifetime tests.
//They are more of example use-cases than real tests. (ASAN/valgrind can still catch issues here)
TYPED_TEST_P(output_args_lifetime, callback){

    std::atomic<int> active_requests = {0};

    for (int i=0; i<this->num_of_requests; i++)
    {
        TypeParam r;

        //As output arguments are __captured by reference__  calling code
        //__must__ ensure they live long enough to complete asynchronous activity.
        //(i.e. live at least until callback is called)
        auto out_args_ptr =  std::make_shared<cv::GRunArgs>(deep_copy_out_args(r.out_args()));

        //Extend lifetime of out_args_ptr content by capturing it into a callback
        auto cb =  [&active_requests, out_args_ptr](std::exception_ptr ){
            --active_requests;
        };

        ++active_requests;

        r.async(cb, r.in_args(), args_p_from_args(*out_args_ptr));
    }


   while(active_requests){
       std::this_thread::sleep_for(std::chrono::milliseconds{2});
   }
}


TYPED_TEST_P(output_args_lifetime, future){

    std::vector<std::future<void>>                      fs(this->num_of_requests);
    std::vector<std::shared_ptr<cv::GRunArgs>>    out_ptrs(this->num_of_requests);

    for (int i=0; i<this->num_of_requests; i++)
    {
        TypeParam r;

        //As output arguments are __captured by reference__  calling code
        //__must__ ensure they live long enough to complete asynchronous activity.
        //(i.e. live at least until future.get()/wait() is returned)
        auto out_args_ptr =  std::make_shared<cv::GRunArgs>(deep_copy_out_args(r.out_args()));

        //Extend lifetime of out_args_ptr content
        out_ptrs[i] = out_args_ptr;

        fs[i] = r.async(r.in_args(), args_p_from_args(*out_args_ptr));
    }

    for (auto const& ftr : fs ){
        ftr.wait();
    }
}
REGISTER_TYPED_TEST_CASE_P(output_args_lifetime, callback, future);

//little helpers to match up all combinations of setups
template<typename compute_fixture_t, template<typename> class... args_t>
struct Case
        : compute_fixture_t,
          args_t<Case<compute_fixture_t, args_t...>> ...
{
    template<typename... Args>
    Case(Args&&... args) : compute_fixture_t(std::forward<Args>(args)...) { }
    Case(Case const &  ) = default;
    Case(Case &&  ) = default;

    Case() = default;
};

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

INSTANTIATE_TYPED_TEST_CASE_P(AsyncAPICancelation,        cancel,     cases<SelfCanceling>);

template<typename computation_t>
using explicit_wait_cases = ::testing::Types<
            Case<computation_t, AsyncCompiled>,
            Case<computation_t, AsyncApply>,
            Case<computation_t, AsyncCompiled>,
            Case<computation_t, AsyncApply>
            >;

INSTANTIATE_TYPED_TEST_CASE_P(AsyncAPIOutArgsLifetTime,   output_args_lifetime,     explicit_wait_cases<SumOfSum2x2>);

} // namespace opencv_test
