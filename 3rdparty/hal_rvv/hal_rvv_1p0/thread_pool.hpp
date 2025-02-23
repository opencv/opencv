// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_THREAD_POOL_HPP_INCLUDED
#define OPENCV_HAL_RVV_THREAD_POOL_HPP_INCLUDED

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <tuple>

namespace cv { namespace cv_hal_rvv {
    class ThreadPool
    {
    private:
        std::queue<std::tuple<int, int, std::function<int(int, int)> > > jobs;
        std::queue<int> vals;
        std::condition_variable cv_jobs, cv_vals;
        std::mutex m_jobs, m_vals;

        ThreadPool()
        {
            int num_threads = std::thread::hardware_concurrency();
            for (int i = 0; i < num_threads; i++)
            {
                std::thread(child_thread).detach();
            }
        }

        static ThreadPool& Instance()
        {
            static ThreadPool tp;
            return tp;
        }

        static void child_thread()
        {
            ThreadPool& tp = ThreadPool::Instance();
            while (true)
            {
                std::remove_reference<decltype(jobs.front())>::type job;
                {
                    std::unique_lock<std::mutex> lk(tp.m_jobs);
                    tp.cv_jobs.wait(lk, [&]{ return !tp.jobs.empty(); });
                    job = tp.jobs.front();
                    tp.jobs.pop();
                }
                int start = std::get<0>(job), end = std::get<1>(job);
                int val = start < end ? std::get<2>(job)(start, end) : CV_HAL_ERROR_OK;
                {
                    std::unique_lock<std::mutex> lk(tp.m_vals);
                    tp.vals.push(val);
                }
                tp.cv_vals.notify_one();
            }
        }

    public:
        template<typename... Args>
        static int parallel_for(int length, double fstripe, std::function<int(int, int, Args...)> func, Args&&... args)
        {
            ThreadPool& tp = ThreadPool::Instance();
            int num_threads = std::thread::hardware_concurrency();
            int stripe = fstripe < 0 ? num_threads : std::min(std::max(static_cast<int>(std::round(fstripe)), 1), length);
            int step = (length + stripe - 1) / stripe;

            {
                std::unique_lock<std::mutex> lk(tp.m_jobs);
                for (int i = 0; i < stripe; i++)
                {
                    tp.jobs.emplace(std::min(length, i * step), std::min(length, (i + 1) * step), std::bind(func, std::placeholders::_1, std::placeholders::_2, std::forward<Args>(args)...));
                }
            }
            tp.cv_jobs.notify_all();

            {
                std::unique_lock<std::mutex> lk(tp.m_vals);
                tp.cv_vals.wait(lk, [&]{ return tp.vals.size() == (size_t)stripe; });
                while (!tp.vals.empty())
                {
                    int val = tp.vals.front();
                    if (val != CV_HAL_ERROR_OK)
                    {
                        std::queue<int>().swap(tp.vals);
                        return val;
                    }
                    tp.vals.pop();
                }
            }

            return CV_HAL_ERROR_OK;
        }
    };
}}

#endif
