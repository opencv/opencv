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
// Copyright (C) 2008-2012, Willow Garage Inc., all rights reserved.
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
//     and / or other materials provided with the distribution.
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

#ifndef __SFT_RANDOM_HPP__
#define __SFT_RANDOM_HPP__

#if defined(_MSC_VER) && _MSC_VER >= 1600
# include <random>

namespace cv { namespace softcascade { namespace internal
{

struct Random
{
    typedef std::mt19937 engine;
    typedef std::uniform_int<int> uniform;
};

}}}

#elif (__GNUC__) && __GNUC__ > 3 && __GNUC_MINOR__ > 1 && !defined(__ANDROID__)
# if defined (__cplusplus) && __cplusplus > 201100L
#  include <random>

namespace cv { namespace softcascade { namespace internal
{

struct Random
{
    typedef std::mt19937 engine;
    typedef std::uniform_int<int> uniform;
};
}}}

# else
#  include <tr1/random>

namespace cv { namespace softcascade { namespace internal
{

struct Random
{
    typedef std::tr1::mt19937 engine;
    typedef std::tr1::uniform_int<int> uniform;
};

}}}
# endif

#else
# include <opencv2/core/core.hpp>

namespace cv { namespace softcascade { namespace internal
{
namespace rnd {

typedef cv::RNG engine;

template<typename T>
struct uniform_int
{
    uniform_int(const int _min, const int _max) : min(_min), max(_max) {}
    T operator() (engine& eng, const int bound) const
    {
        return (T)eng.uniform(min, bound);
    }

    T operator() (engine& eng) const
    {
        return (T)eng.uniform(min, max);
    }

private:
    int min;
    int max;
};

}

struct Random
{
    typedef rnd::engine engine;
    typedef rnd::uniform_int<int> uniform;
};

}}}

#endif

#if defined _WIN32 && (_WIN32 || _WIN64)
# if _WIN64
#  define USE_LONG_SEEDS
# endif
#endif
#if defined (__GNUC__) &&__GNUC__
# if defined(__x86_64__) || defined(__ppc64__)
#  define USE_LONG_SEEDS
# endif
#endif

#if defined USE_LONG_SEEDS
# define FEATURE_RECT_SEED      8854342234LU
# define INDEX_ENGINE_SEED      764224349868LU
#else
# define FEATURE_RECT_SEED      88543422LU
# define INDEX_ENGINE_SEED      76422434LU
#endif
#undef USE_LONG_SEEDS

#define DCHANNELS_SEED         314152314LU
#define DX_DY_SEED             65633343LU

#endif