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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
#ifndef __OPENCV_TEST_PRECOMP_HPP__
#define __OPENCV_TEST_PRECOMP_HPP__

#include "opencv2/ts.hpp"
#include "opencv2/ts/ts_perf.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"

#include "opencv2/dnn.hpp"
#include "test_common.hpp"

namespace cv {
namespace dnn {
CV__DNN_EXPERIMENTAL_NS_BEGIN

static inline void PrintTo(const cv::dnn::Backend& v, std::ostream* os)
{
    switch (v) {
    case DNN_BACKEND_DEFAULT: *os << "DNN_BACKEND_DEFAULT"; return;
    case DNN_BACKEND_HALIDE: *os << "DNN_BACKEND_HALIDE"; return;
    case DNN_BACKEND_INFERENCE_ENGINE: *os << "DNN_BACKEND_INFERENCE_ENGINE"; return;
    case DNN_BACKEND_OPENCV: *os << "DNN_BACKEND_OPENCV"; return;
    } // don't use "default:" to emit compiler warnings
    *os << "DNN_BACKEND_UNKNOWN(" << v << ")";
}

static inline void PrintTo(const cv::dnn::Target& v, std::ostream* os)
{
    switch (v) {
    case DNN_TARGET_CPU: *os << "DNN_TARGET_CPU"; return;
    case DNN_TARGET_OPENCL: *os << "DNN_TARGET_OPENCL"; return;
    case DNN_TARGET_OPENCL_FP16: *os << "DNN_TARGET_OPENCL_FP16"; return;
    case DNN_TARGET_MYRIAD: *os << "DNN_TARGET_MYRIAD"; return;
    } // don't use "default:" to emit compiler warnings
    *os << "DNN_TARGET_UNKNOWN(" << v << ")";
}

CV__DNN_EXPERIMENTAL_NS_END
}} // namespace

namespace opencv_test {
using namespace cv::dnn;

static testing::internal::ParamGenerator<Target> availableDnnTargets()
{
    static std::vector<Target> targets;
    if (targets.empty())
    {
        targets.push_back(DNN_TARGET_CPU);
#ifdef HAVE_OPENCL
        if (cv::ocl::useOpenCL())
            targets.push_back(DNN_TARGET_OPENCL);
#endif
    }
    return testing::ValuesIn(targets);
}

static testing::internal::ParamGenerator<tuple<Backend, Target> > dnnBackendsAndTargets()
{
    static const tuple<Backend, Target> testCases[] = {
    #ifdef HAVE_INF_ENGINE
        tuple<Backend, Target>(DNN_BACKEND_INFERENCE_ENGINE, DNN_TARGET_CPU),
        tuple<Backend, Target>(DNN_BACKEND_INFERENCE_ENGINE, DNN_TARGET_OPENCL),
        tuple<Backend, Target>(DNN_BACKEND_INFERENCE_ENGINE, DNN_TARGET_OPENCL_FP16),
        tuple<Backend, Target>(DNN_BACKEND_INFERENCE_ENGINE, DNN_TARGET_MYRIAD),
    #endif
        tuple<Backend, Target>(DNN_BACKEND_OPENCV, DNN_TARGET_CPU),
        tuple<Backend, Target>(DNN_BACKEND_OPENCV, DNN_TARGET_OPENCL),
        tuple<Backend, Target>(DNN_BACKEND_OPENCV, DNN_TARGET_OPENCL_FP16)
    };
    return testing::ValuesIn(testCases);
}

class DNNTestLayer : public TestWithParam<tuple<Backend, Target> >
{
public:
    dnn::Backend backend;
    dnn::Target target;
    double default_l1, default_lInf;

    DNNTestLayer()
    {
        backend = (dnn::Backend)(int)get<0>(GetParam());
        target = (dnn::Target)(int)get<1>(GetParam());
        getDefaultThresholds(backend, target, &default_l1, &default_lInf);
    }

   static void getDefaultThresholds(int backend, int target, double* l1, double* lInf)
   {
       if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD)
       {
           *l1 = 4e-3;
           *lInf = 2e-2;
       }
       else
       {
           *l1 = 1e-5;
           *lInf = 1e-4;
       }
   }

   static void checkBackend(int backend, int target, Mat* inp = 0, Mat* ref = 0)
   {
       if (backend == DNN_BACKEND_OPENCV && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
       {
#ifdef HAVE_OPENCL
           if (!cv::ocl::useOpenCL())
#endif
           {
               throw SkipTestException("OpenCL is not available/disabled in OpenCV");
           }
       }
       if (backend == DNN_BACKEND_INFERENCE_ENGINE && target == DNN_TARGET_MYRIAD)
       {
           if (!checkMyriadTarget())
           {
               throw SkipTestException("Myriad is not available/disabled in OpenCV");
           }
           if (inp && ref && inp->size[0] != 1)
           {
               // Myriad plugin supports only batch size 1. Slice a single sample.
               if (inp->size[0] == ref->size[0])
               {
                   std::vector<cv::Range> range(inp->dims, Range::all());
                   range[0] = Range(0, 1);
                   *inp = inp->operator()(range);

                   range = std::vector<cv::Range>(ref->dims, Range::all());
                   range[0] = Range(0, 1);
                   *ref = ref->operator()(range);
               }
               else
                   throw SkipTestException("Myriad plugin supports only batch size 1");
           }
       }
   }

protected:
    void checkBackend(Mat* inp = 0, Mat* ref = 0)
    {
        checkBackend(backend, target, inp, ref);
    }
};

} // namespace
#endif
