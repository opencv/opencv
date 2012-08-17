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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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

#include "precomp.hpp"

#if !defined HAVE_TBB && !defined HAVE_OPENMP && !defined HAVE_GCD && !defined HAVE_CONCURRENCY

#ifdef __APPLE__
#define HAVE_GCD
#elif defined __MSC_VER && __MSC_VER >= 1600
#define HAVE_CONCURRENCY
#endif

#endif

#ifdef HAVE_CONCURRENCY
#  include <ppl.h>
#elif defined HAVE_OPENMP
#  include <omp.h>
#elif defined HAVE_GCD
#  include <dispatch/dispatch.h>
#elif defined HAVE_TBB
#  include "tbb/tbb_stddef.h"
#  if TBB_VERSION_MAJOR*100 + TBB_VERSION_MINOR >= 202
#    include "tbb/tbb.h"
#    include "tbb/task.h"
#    undef min
#    undef max
#  else
#    undef HAVE_TBB
#  endif // end TBB version
#endif // HAVE_CONCURRENCY

/*
    HAVE_TBB - using TBB
    HAVE_GCD - using GCD
    HAVE_OPENMP - using OpenMP
    HAVE_CONCURRENCY - using visual studio 2010 concurrency
*/

namespace cv
{
    ParallelLoopBody::~ParallelLoopBody() { }

#ifdef HAVE_TBB
    class TbbProxyLoopBody
    {
    public:
        TbbProxyLoopBody(const ParallelLoopBody& _body) :
            body(&_body)
        { }

        void operator ()(const tbb::blocked_range<int>& range) const
        {
            body->operator()(Range(range.begin(), range.end()));
        }

    private:
        const ParallelLoopBody* body;
    };
#endif // end HAVE_TBB

#ifdef HAVE_GCD
    static
    void block_function(void* context, size_t index)
    {
        ParallelLoopBody* ptr_body = static_cast<ParallelLoopBody*>(context);
        ptr_body->operator()(Range(index, index + 1));
    }
#endif // HAVE_GCD

    void parallel_for_(const Range& range, const ParallelLoopBody& body)
    {
#ifdef HAVE_TBB

        tbb::parallel_for(tbb::blocked_range<int>(range.start, range.end), TbbProxyLoopBody(body));

#elif defined HAVE_CONCURRENCY

        Concurrency::parallel_for(range.start, range.end, body);

#elif defined HAVE_OPENMP

#pragma omp parallel for schedule(dynamic)
        for (int i = range.start; i < range.end; ++i)
            body(Range(i, i + 1));

#elif defined (HAVE_GCD)

        dispatch_queue_t concurrent_queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
        dispatch_apply_f(range.end - range.start, concurrent_queue, &const_cast<ParallelLoopBody&>(body), block_function);

#else

        body(range);

#endif // end HAVE_TBB
    }

} // namespace cv
