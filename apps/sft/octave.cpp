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

#include <sft/octave.hpp>
#include <sft/random.hpp>

#if defined VISUALIZE_GENERATION
# include <opencv2/highgui/highgui.hpp>
# define show(a, b)  \
    do {             \
    cv::imshow(a,b); \
    cv::waitkey(0);  \
    } while(0)
#else
# define show(a, b)
#endif

// ============ Octave ============ //
sft::Octave::Octave(){}

sft::Octave::~Octave(){}

bool sft::Octave::train( const cv::Mat& trainData, const cv::Mat& responses, const cv::Mat& varIdx,
       const cv::Mat& sampleIdx, const cv::Mat& varType, const cv::Mat& missingDataMask)
{
    bool update = false;
    return cv::Boost::train(trainData, CV_COL_SAMPLE, responses, varIdx, sampleIdx, varType, missingDataMask, params,
    update);
}

// ========= FeaturePool ========= //
sft::FeaturePool::FeaturePool(cv::Size m, int n) : model(m), nfeatures(n)
{
    CV_Assert(m != cv::Size() && n > 0);
    fill(nfeatures);
}

sft::FeaturePool::~FeaturePool(){}


void sft::FeaturePool::fill(int desired)
{

    int mw = model.width;
    int mh = model.height;

    int maxPoolSize = (mw -1) * mw / 2 * (mh - 1) * mh / 2 * N_CHANNELS;

    nfeatures = std::min(desired, maxPoolSize);

    pool.reserve(nfeatures);

    sft::Random::engine eng(seed);
    sft::Random::engine eng_ch(seed);

    sft::Random::uniform chRand(0, N_CHANNELS - 1);

    sft::Random::uniform xRand(0, model.width  - 2);
    sft::Random::uniform yRand(0, model.height - 2);

    sft::Random::uniform wRand(1, model.width  - 1);
    sft::Random::uniform hRand(1, model.height - 1);

    while (pool.size() < size_t(nfeatures))
    {
        int x = xRand(eng);
        int y = yRand(eng);

        int w = 1 + wRand(eng, model.width  - x - 1);
        int h = 1 + hRand(eng, model.height - y - 1);

        CV_Assert(w > 0);
        CV_Assert(h > 0);

        CV_Assert(w + x < model.width);
        CV_Assert(h + y < model.height);

        int ch = chRand(eng_ch);

        sft::ICF f(x, y, w, h, ch);

        if (std::find(pool.begin(), pool.end(),f) == pool.end())
            pool.push_back(f);
    }
}