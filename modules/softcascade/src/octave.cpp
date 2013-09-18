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
// Copyright (C) 2008-2013, Willow Garage Inc., all rights reserved.
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

#include "precomp.hpp"
#include "opencv2/ml.hpp"
#include <queue>

using cv::InputArray;
using cv::OutputArray;
using cv::Mat;

using cv::softcascade::Octave;
using cv::softcascade::FeaturePool;
using cv::softcascade::Dataset;
using cv::softcascade::ChannelFeatureBuilder;

FeaturePool::~FeaturePool(){}
Dataset::~Dataset(){}

namespace {

class BoostedSoftCascadeOctave : public cv::Boost, public Octave
{
public:

    BoostedSoftCascadeOctave(cv::Rect boundingBox = cv::Rect(), int npositives = 0, int nnegatives = 0, int logScale = 0,
        int shrinkage = 1, cv::Ptr<ChannelFeatureBuilder> builder = ChannelFeatureBuilder::create("HOG6MagLuv"));
    virtual ~BoostedSoftCascadeOctave();
    virtual cv::AlgorithmInfo* info() const;
    virtual bool train(const Dataset* dataset, const FeaturePool* pool, int weaks, int treeDepth);
    virtual void setRejectThresholds(OutputArray thresholds);
    virtual void write( cv::FileStorage &fs, const FeaturePool* pool, InputArray thresholds) const;
    virtual void write( CvFileStorage* fs, cv::String name) const;
protected:
    virtual float predict( InputArray _sample, InputArray _votes, bool raw_mode, bool return_sum ) const;
    virtual bool train( const cv::Mat& trainData, const cv::Mat& responses, const cv::Mat& varIdx=cv::Mat(),
       const cv::Mat& sampleIdx=cv::Mat(), const cv::Mat& varType=cv::Mat(), const cv::Mat& missingDataMask=cv::Mat());

    void processPositives(const Dataset* dataset);
    void generateNegatives(const Dataset* dataset);

    float predict( const Mat& _sample, const cv::Range range) const;
private:
    void traverse(const CvBoostTree* tree, cv::FileStorage& fs, int& nfeatures, int* used, const double* th) const;
    virtual void initialize_weights(double (&p)[2]);

    int logScale;
    cv::Rect boundingBox;

    int npositives;
    int nnegatives;

    int shrinkage;

    Mat integrals;
    Mat responses;

    CvBoostParams params;

    Mat trainData;

    cv::Ptr<ChannelFeatureBuilder> builder;
};

BoostedSoftCascadeOctave::BoostedSoftCascadeOctave(cv::Rect bb, int np, int nn, int ls, int shr,
    cv::Ptr<ChannelFeatureBuilder> _builder)
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
        _params.regression_accuracy  = 0;

        // boost params
        _params.boost_type           = CvBoost::GENTLE;
        _params.split_criteria       = CvBoost::SQERR;
        _params.weight_trim_rate     = 0.95;

        // simple defaults
        _params.min_sample_count     = 0;
        _params.weak_count           = 1;
    }

    params = _params;

    builder = _builder;

    int w = boundingBox.width;
    int h = boundingBox.height;

    integrals.create(npositives + nnegatives, (w / shrinkage + 1) * (h / shrinkage * builder->totalChannels() + 1), CV_32SC1);
}

BoostedSoftCascadeOctave::~BoostedSoftCascadeOctave(){}

bool BoostedSoftCascadeOctave::train( const cv::Mat& _trainData, const cv::Mat& _responses, const cv::Mat& varIdx,
       const cv::Mat& sampleIdx, const cv::Mat& varType, const cv::Mat& missingDataMask)
{
    bool update = false;
    return cv::Boost::train(_trainData, CV_COL_SAMPLE, _responses, varIdx, sampleIdx, varType, missingDataMask, params,
    update);
}

void BoostedSoftCascadeOctave::setRejectThresholds(cv::OutputArray _thresholds)
{
    // labels decided by classifier
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
        mptr[si] = cv::saturate_cast<uchar>((unsigned int)( (responses.ptr<float>(si)[0] == 1.f) && (decision == 1.f)));
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

void BoostedSoftCascadeOctave::processPositives(const Dataset* dataset)
{
    int h = boundingBox.height;

    ChannelFeatureBuilder& _builder = *builder;

    int total = 0;
    for (int curr = 0; curr < dataset->available( Dataset::POSITIVE); ++curr)
    {
        cv::Mat sample = dataset->get( Dataset::POSITIVE, curr);

        cv::Mat channels = integrals.row(total).reshape(0, h / shrinkage * builder->totalChannels() + 1);
        sample = sample(boundingBox);

        _builder(sample, channels);
        responses.ptr<float>(total)[0] = 1.f;

        if (++total >= npositives) break;
    }
    npositives  = total;
    nnegatives = cvRound(nnegatives * total / (double)npositives);
}

void BoostedSoftCascadeOctave::generateNegatives(const Dataset* dataset)
{
    using namespace cv::softcascade::internal;
    // ToDo: set seed, use offsets
    Random::engine eng(DX_DY_SEED);
    Random::engine idxEng((Random::seed_type)INDEX_ENGINE_SEED);

    int h = boundingBox.height;

    int nimages = dataset->available(Dataset::NEGATIVE);
    Random::uniform iRand(0, nimages - 1);

    int total = 0;
    Mat sum;

    ChannelFeatureBuilder& _builder = *builder;
    for (int i = npositives; i < nnegatives + npositives; ++total)
    {
        int curr = iRand(idxEng);

        Mat frame = dataset->get(Dataset::NEGATIVE, curr);

        int maxW = frame.cols - 2 * boundingBox.x - boundingBox.width;
        int maxH = frame.rows - 2 * boundingBox.y - boundingBox.height;

        Random::uniform wRand(0, maxW -1);
        Random::uniform hRand(0, maxH -1);

        int dx = wRand(eng);
        int dy = hRand(eng);

        frame = frame(cv::Rect(dx, dy, boundingBox.width, boundingBox.height));

        cv::Mat channels = integrals.row(i).reshape(0, h / shrinkage * builder->totalChannels() + 1);
        _builder(frame, channels);

        // // if (predict(sum))
        {
            responses.ptr<float>(i)[0] = 0.f;
            ++i;
        }
    }
}


template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

void BoostedSoftCascadeOctave::traverse(const CvBoostTree* tree, cv::FileStorage& fs, int& nfeatures, int* used, const double* th) const
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

    delete [] leafs;
}

void BoostedSoftCascadeOctave::write( cv::FileStorage &fso, const FeaturePool* pool, InputArray _thresholds) const
{
    CV_Assert(!_thresholds.empty());
    cv::Mat used( 1, weak->total * ( (int)pow(2.f, params.max_depth) - 1), CV_32SC1);
    int* usedPtr = used.ptr<int>(0);
    int nfeatures = 0;
    cv::Mat thresholds = _thresholds.getMat();
    fso << "{"
        << "scale" << logScale
        << "weaks" << weak->total
        << "trees" << "[";
        // should be replaced with the H.L. one
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

void BoostedSoftCascadeOctave::initialize_weights(double (&p)[2])
{
    double n = data->sample_count;
    p[0] =  n / (2. * (double)(nnegatives));
    p[1] =  n / (2. * (double)(npositives));
}

bool BoostedSoftCascadeOctave::train(const Dataset* dataset, const FeaturePool* pool, int weaks, int treeDepth)
{
    CV_Assert(treeDepth == 2);
    CV_Assert(weaks > 0);

    params.max_depth  = treeDepth;
    params.weak_count = weaks;

    // 1. fill integrals and classes
    processPositives(dataset);
    generateNegatives(dataset);

    // 2. only simple case (all features used)
    int nfeatures = pool->size();
    cv::Mat varIdx(1, nfeatures, CV_32SC1);
    int* ptr = varIdx.ptr<int>(0);

    for (int x = 0; x < nfeatures; ++x)
        ptr[x] = x;

    // 3. only simple case (all samples used)
    int nsamples = npositives + nnegatives;
    cv::Mat sampleIdx(1, nsamples, CV_32SC1);
    ptr = sampleIdx.ptr<int>(0);

    for (int x = 0; x < nsamples; ++x)
        ptr[x] = x;

    // 4. ICF has an ordered response.
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

float BoostedSoftCascadeOctave::predict( cv::InputArray _sample, cv::InputArray _votes, bool raw_mode, bool return_sum ) const
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

float BoostedSoftCascadeOctave::predict( const Mat& _sample, const cv::Range range) const
{
    CvMat sample = _sample;
    return CvBoost::predict(&sample, 0, 0, range, false, true);
}

void BoostedSoftCascadeOctave::write( CvFileStorage* fs, cv::String _name) const
{
    CvBoost::write(fs, _name.c_str());
}

}

CV_INIT_ALGORITHM(BoostedSoftCascadeOctave, "Octave.BoostedSoftCascadeOctave", );

Octave::~Octave(){}

cv::Ptr<Octave> Octave::create(cv::Rect boundingBox, int npositives, int nnegatives,
        int logScale, int shrinkage, cv::Ptr<ChannelFeatureBuilder> builder)
{
    cv::Ptr<Octave> octave(
        new BoostedSoftCascadeOctave(boundingBox, npositives, nnegatives, logScale, shrinkage, builder));
    return octave;
}
