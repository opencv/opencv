#ifndef _OPENCV_GREENTEA_BENCHMARK_HPP_
#define _OPENCV_GREENTEA_BENCHMARK_HPP_
#include "../../precomp.hpp"
#include "common.hpp"

namespace greentea {

#ifdef HAVE_OPENCL
class Timer {
    public:
        Timer();
        virtual ~Timer();
        virtual void Start();
        virtual void Stop();
        virtual float MilliSeconds();
        virtual float MicroSeconds();
        virtual float Seconds();

        inline bool initted() { return initted_; }
        inline bool running() { return running_; }
        inline bool has_run_at_least_once() { return has_run_at_least_once_; }

    protected:
        void Init();

        bool initted_;
        bool running_;
        bool has_run_at_least_once_;
        cl_event start_gpu_cl_;
        cl_event stop_gpu_cl_;
        float elapsed_milliseconds_;
        float elapsed_microseconds_;
};
#endif

}  // namespace greentea

#endif
