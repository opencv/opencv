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

#ifndef __OPENCV_PARALLEL_TOOL_HPP__
#define __OPENCV_PARALLEL_TOOL_HPP__

#ifdef HAVE_CVCONFIG_H
# include <cvconfig.h>
#endif // HAVE_CVCONFIG_H

/*
    HAVE_TBB - using TBB
    HAVE_GCD - using GCD
    HAVE_OPENMP - using OpenMP
    HAVE_CONCURRENCY - using visual studio 2010 concurrency
*/

#ifdef HAVE_TBB
#  include "tbb/tbb_stddef.h"
#  if TBB_VERSION_MAJOR*100 + TBB_VERSION_MINOR >= 202
#    include "tbb/tbb.h"
#    include "tbb/task.h"
#    undef min
#    undef max
#  else
#    undef HAVE_TBB
#   endif // end TBB version
#endif // HAVE_TBB

#ifdef __cplusplus

namespace cv
{
    // a base body class
    class CV_EXPORTS ParallelLoopBody
    {
    public:
        virtual void operator() (const Range& range) const = 0;
        virtual ~ParallelLoopBody();
    };

    CV_EXPORTS void parallel_for_(const Range& range, const ParallelLoopBody& body);

    template <typename Iterator, typename Body> inline
    CV_EXPORTS void parallel_do_(Iterator first, Iterator last, const Body& body)
    {
#ifdef HAVE_TBB
        tbb::parallel_do(first, last, body);
#else
        for ( ; first != last; ++first)
            body(*first);
#endif // HAVE_TBB
    }

    template <typename Body> inline
    CV_EXPORTS void parallel_reduce_(const Range& range, Body& body)
    {
#ifdef HAVE_TBB
        tbb::parallel_reduce(tbb::blocked_range<int>(range.start, range.end), body);
#else
        body(range);
#endif // end HAVE_TBB
    }

} // namespace cv

#endif // __cplusplus

#endif // __OPENCV_PARALLEL_TOOL_HPP__
