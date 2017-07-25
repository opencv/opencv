/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (c) 2016-2017 Fabian David Tschopp, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#ifndef _OPENCV_LIBDNN_BENCHMARK_HPP_
#define _OPENCV_LIBDNN_BENCHMARK_HPP_
#include "../../precomp.hpp"
#include "common.hpp"

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
#endif
