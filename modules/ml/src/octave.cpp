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

#include "precomp.hpp"
#include <queue>

//#define WITH_DEBUG_OUT

#if defined WITH_DEBUG_OUT
# include <stdio.h>
# define dprintf(format, ...) \
            do { printf(format, ##__VA_ARGS__); } while (0)
#else
# define dprintf(format, ...)
#endif

#if defined(_MSC_VER) && _MSC_VER >= 1600

# include <random>
namespace sft {
struct Random
{
    typedef std::mt19937 engine;
    typedef std::uniform_int<int> uniform;
};
}

#elif (__GNUC__) && __GNUC__ > 3 && __GNUC_MINOR__ > 1 && !defined(__ANDROID__)

# if defined (__cplusplus) && __cplusplus > 201100L
#  include <random>
namespace sft {
struct Random
{
    typedef std::mt19937 engine;
    typedef std::uniform_int<int> uniform;
};
}
# else
#   include <tr1/random>

namespace sft {
struct Random
{
    typedef std::tr1::mt19937 engine;
    typedef std::tr1::uniform_int<int> uniform;
};
}
# endif

#else
#include <opencv2/core/core.hpp>
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

namespace sft {
struct Random
{
    typedef rnd::engine engine;
    typedef rnd::uniform_int<int> uniform;
};
}

#endif

cv::FeaturePool::~FeaturePool(){}
cv::Dataset::~Dataset(){}

cv::Octave::Octave(cv::Rect bb, int np, int nn, int ls, int shr)
: logScale(ls), boundingBox(bb), npositives(np), nnegatives(nn), shrinkage(shr)
{
    int maxSample = npositives + nnegatives;
    responses.create(maxSample, 1, CV_32FC1);

    CvBoostParams _params;
    {
        // tree params
        _params.max_categories       = 10;
        _params.max_depth            = 2;
        _params.cv_folds             = 0;
        _params.truncate_pruned_tree = false;
        _params.use_surrogates       = false;
        _params.use_1se_rule         = false;
        _params.regression_accuracy  = 1.0e-6;

        // boost params
        _params.boost_type           = CvBoost::GENTLE;
        _params.split_criteria       = CvBoost::SQERR;
        _params.weight_trim_rate     = 0.95;

        // simple defaults
        _params.min_sample_count     = 2;
        _params.weak_count           = 1;
    }

    params = _params;
}

cv::Octave::~Octave(){}

bool cv::Octave::train( const cv::Mat& _trainData, const cv::Mat& _responses, const cv::Mat& varIdx,
       const cv::Mat& sampleIdx, const cv::Mat& varType, const cv::Mat& missingDataMask)
{
    bool update = false;
    return cv::Boost::train(_trainData, CV_COL_SAMPLE, _responses, varIdx, sampleIdx, varType, missingDataMask, params,
    update);
}

void cv::Octave::setRejectThresholds(cv::OutputArray _thresholds)
{
    dprintf("set thresholds according to DBP strategy\n");

    // labels desided by classifier
    cv::Mat desisions(responses.cols, responses.rows, responses.type());
    float* dptr = desisions.ptr<float>(0);

    // mask of samples satisfying the condition
    cv::Mat ppmask(responses.cols, responses.rows, CV_8UC1);
    uchar* mptr = ppmask.ptr<uchar>(0);

    int nsamples = npositives + nnegatives;

    cv::Mat stab;

    for (int si = 0; si < nsamples; ++si)
    {
        float decision = dptr[si] = predict(trainData.col(si), stab, false, false);
        mptr[si] = cv::saturate_cast<uchar>((uint)( (responses.ptr<float>(si)[0] == 1.f) && (decision == 1.f)));
    }

    int weaks = weak->total;
    _thresholds.create(1, weaks, CV_64FC1);
    cv::Mat& thresholds = _thresholds.getMatRef();
    double* thptr = thresholds.ptr<double>(0);

    cv::Mat traces(weaks, nsamples, CV_64FC1, cv::Scalar::all(FLT_MAX));

    for (int w = 0; w < weaks; ++w)
    {
        double* rptr = traces.ptr<double>(w);
        for (int si = 0; si < nsamples; ++si)
        {
            cv::Range curr(0, w + 1);
            if (mptr[si])
            {
                float trace = predict(trainData.col(si), curr);
                rptr[si] = trace;
            }
        }
        double mintrace = 0.;
        cv::minMaxLoc(traces.row(w), &mintrace);
        thptr[w] = mintrace;
    }
}

void cv::Octave::processPositives(const Dataset* dataset, const FeaturePool* pool)
{
    int w = boundingBox.width;
    int h = boundingBox.height;

    integrals.create(pool->size(), (w / shrinkage + 1) * (h / shrinkage * 10 + 1), CV_32SC1);

    int total = 0;
    for (int curr = 0; curr < dataset->available( Dataset::POSITIVE); ++curr)
    {
        cv::Mat sample = dataset->get( Dataset::POSITIVE, curr);

        cv::Mat channels = integrals.row(total).reshape(0, h / shrinkage * 10 + 1);
        sample = sample(boundingBox);

        pool->preprocess(sample, channels);
        responses.ptr<float>(total)[0] = 1.f;

        if (++total >= npositives) break;
    }

    dprintf("Processing positives finished:\n\trequested %d positives, collected %d samples.\n", npositives, total);

    npositives  = total;
    nnegatives = cvRound(nnegatives * total / (double)npositives);
}

void cv::Octave::generateNegatives(const Dataset* dataset, const FeaturePool* pool)
{
    // ToDo: set seed, use offsets
    sft::Random::engine eng(65633343L);
    sft::Random::engine idxEng(764224349868L);

    int h = boundingBox.height;

    int nimages = dataset->available(Dataset::NEGATIVE);
    sft::Random::uniform iRand(0, nimages - 1);

    int total = 0;
    Mat sum;
    for (int i = npositives; i < nnegatives + npositives; ++total)
    {
        int curr = iRand(idxEng);

        Mat frame = dataset->get(Dataset::NEGATIVE, curr);

        int maxW = frame.cols - 2 * boundingBox.x - boundingBox.width;
        int maxH = frame.rows - 2 * boundingBox.y - boundingBox.height;

        sft::Random::uniform wRand(0, maxW -1);
        sft::Random::uniform hRand(0, maxH -1);

        int dx = wRand(eng);
        int dy = hRand(eng);

        frame = frame(cv::Rect(dx, dy, boundingBox.width, boundingBox.height));

        cv::Mat channels = integrals.row(i).reshape(0, h / shrinkage * 10 + 1);
        pool->preprocess(frame, channels);

        dprintf("generated %d %d\n", dx, dy);
        // // if (predict(sum))
        {
            responses.ptr<float>(i)[0] = 0.f;
            ++i;
        }
    }

    dprintf("Processing negatives finished:\n\trequested %d negatives, viewed %d samples.\n", nnegatives, total);
}


template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

void cv::Octave::traverse(const CvBoostTree* tree, cv::FileStorage& fs, int& nfeatures, int* used, const double* th) const
{
    std::queue<const CvDTreeNode*> nodes;
    nodes.push( tree->get_root());
    const CvDTreeNode* tempNode;
    int leafValIdx = 0;
    int internalNodeIdx = 1;
    float* leafs = new float[(int)pow(2.f, get_params().max_depth)];

    fs << "{";
    fs << "treeThreshold" << *th;
    fs << "internalNodes" << "[";
    while (!nodes.empty())
    {
        tempNode = nodes.front();
        CV_Assert( tempNode->left );
        if ( !tempNode->left->left && !tempNode->left->right)
        {
            leafs[-leafValIdx] = (float)tempNode->left->value;
            fs << leafValIdx-- ;
        }
        else
        {
            nodes.push( tempNode->left );
            fs << internalNodeIdx++;
        }
        CV_Assert( tempNode->right );
        if ( !tempNode->right->left && !tempNode->right->right)
        {
            leafs[-leafValIdx] = (float)tempNode->right->value;
            fs << leafValIdx--;
        }
        else
        {
            nodes.push( tempNode->right );
            fs << internalNodeIdx++;
        }

        int fidx = tempNode->split->var_idx;
        fs << nfeatures;
        used[nfeatures++] = fidx;

        fs << tempNode->split->ord.c;

        nodes.pop();
    }
    fs << "]";

    fs << "leafValues" << "[";
    for (int ni = 0; ni < -leafValIdx; ni++)
        fs << leafs[ni];
    fs << "]";


    fs << "}";
}

void cv::Octave::write( cv::FileStorage &fso, const FeaturePool* pool, InputArray _thresholds) const
{
    CV_Assert(!_thresholds.empty());
    cv::Mat used( 1, weak->total * (pow(2, params.max_depth) - 1), CV_32SC1);
    int* usedPtr = used.ptr<int>(0);
    int nfeatures = 0;
    cv::Mat thresholds = _thresholds.getMat();
    fso << "{"
        << "scale" << logScale
        << "weaks" << weak->total
        << "trees" << "[";
        // should be replased with the H.L. one
        CvSeqReader reader;
        cvStartReadSeq( weak, &reader);

        for(int i = 0; i < weak->total; i++ )
        {
            CvBoostTree* tree;
            CV_READ_SEQ_ELEM( tree, reader );

            traverse(tree, fso, nfeatures, usedPtr, thresholds.ptr<double>(0) + i);
        }
    fso << "]";
    // features

    fso << "features" << "[";
    for (int i = 0; i < nfeatures; ++i)
        pool->write(fso, usedPtr[i]);
    fso << "]"
        << "}";
}

void cv::Octave::initial_weights(double (&p)[2])
{
    double n = data->sample_count;
    p[0] =  n / (2. * (double)(nnegatives));
    p[1] =  n / (2. * (double)(npositives));
}

bool cv::Octave::train(const Dataset* dataset, const FeaturePool* pool, int weaks, int treeDepth)
{
    CV_Assert(treeDepth == 2);
    CV_Assert(weaks > 0);

    params.max_depth  = treeDepth;
    params.weak_count = weaks;

    // 1. fill integrals and classes
    processPositives(dataset, pool);
    generateNegatives(dataset, pool);

    // 2. only sumple case (all features used)
    int nfeatures = pool->size();
    cv::Mat varIdx(1, nfeatures, CV_32SC1);
    int* ptr = varIdx.ptr<int>(0);

    for (int x = 0; x < nfeatures; ++x)
        ptr[x] = x;

    // 3. only sumple case (all samples used)
    int nsamples = npositives + nnegatives;
    cv::Mat sampleIdx(1, nsamples, CV_32SC1);
    ptr = sampleIdx.ptr<int>(0);

    for (int x = 0; x < nsamples; ++x)
        ptr[x] = x;

    // 4. ICF has an orderable responce.
    cv::Mat varType(1, nfeatures + 1, CV_8UC1);
    uchar* uptr = varType.ptr<uchar>(0);
    for (int x = 0; x < nfeatures; ++x)
        uptr[x] = CV_VAR_ORDERED;
    uptr[nfeatures] = CV_VAR_CATEGORICAL;

    trainData.create(nfeatures, nsamples, CV_32FC1);
    for (int fi = 0; fi < nfeatures; ++fi)
    {
        float* dptr = trainData.ptr<float>(fi);
        for (int si = 0; si < nsamples; ++si)
        {
            dptr[si] = pool->apply(fi, si, integrals);
        }
    }

    cv::Mat missingMask;

    bool ok = train(trainData, responses, varIdx, sampleIdx, varType, missingMask);
    if (!ok)
        CV_Error(CV_StsInternal, "ERROR: tree can not be trained");
    return ok;

}

float cv::Octave::predict( cv::InputArray _sample, cv::InputArray _votes, bool raw_mode, bool return_sum ) const
{
    cv::Mat sample = _sample.getMat();
    CvMat csample = sample;
    if (_votes.empty())
        return CvBoost::predict(&csample, 0, 0, CV_WHOLE_SEQ, raw_mode, return_sum);
    else
    {
        cv::Mat votes = _votes.getMat();
        CvMat cvotes = votes;
        return CvBoost::predict(&csample, 0, &cvotes, CV_WHOLE_SEQ, raw_mode, return_sum);
    }
}

float cv::Octave::predict( const Mat& _sample, const cv::Range range) const
{
    CvMat sample = _sample;
    return CvBoost::predict(&sample, 0, 0, range, false, true);
}

void cv::Octave::write( CvFileStorage* fs, string name) const
{
    CvBoost::write(fs, name.c_str());
}
