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
# define show(a, b)  \
    do {             \
    cv::imshow(a,b); \
    cv::waitkey(0);  \
    } while(0)
#else
# define show(a, b)
#endif

#include <glob.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// ============ Octave ============ //
sft::Octave::Octave(cv::Rect bb, int np, int nn, int ls, int shr)
: logScale(ls), boundingBox(bb), npositives(np), nnegatives(nn), shrinkage(shr)
{
    int maxSample = npositives + nnegatives;
    responses.create(maxSample, 1, CV_32FC1);
}

sft::Octave::~Octave(){}

bool sft::Octave::train( const cv::Mat& trainData, const cv::Mat& _responses, const cv::Mat& varIdx,
       const cv::Mat& sampleIdx, const cv::Mat& varType, const cv::Mat& missingDataMask)
{
    CvBoostParams _params;
    {
        // tree params
        _params.max_categories       = 10;
        _params.max_depth            = 2;
        _params.cv_folds             = 0;
        _params.truncate_pruned_tree = false;
        _params.use_surrogates       = false;
        _params.use_1se_rule         = false;
        _params.regression_accuracy  = 0.0;

        // boost params
        _params.boost_type           = CvBoost::GENTLE;
        _params.split_criteria       = CvBoost::SQERR;
        _params.weight_trim_rate     = 0.95;


        /// ToDo: move to params
        _params.min_sample_count     = 2;
        _params.weak_count           = 1;
    }

    std::cout << "WARNING: " << sampleIdx << std::endl;

    bool update = false;
    return cv::Boost::train(trainData, CV_COL_SAMPLE, _responses, varIdx, sampleIdx, varType, missingDataMask, _params,
    update);
}

namespace {
using namespace sft;
class Preprocessor
{
public:
    Preprocessor(int shr) : shrinkage(shr) {}

    void apply(const Mat& frame, Mat& integrals)
    {
        CV_Assert(frame.type() == CV_8UC3);

        int h = frame.rows;
        int w = frame.cols;

        cv::Mat channels, gray;

        channels.create(h * BINS, w, CV_8UC1);
        channels.setTo(0);

        cvtColor(frame, gray, CV_BGR2GRAY);

        cv::Mat df_dx, df_dy, mag, angle;
        cv::Sobel(gray, df_dx, CV_32F, 1, 0);
        cv::Sobel(gray, df_dy, CV_32F, 0, 1);

        cv::cartToPolar(df_dx, df_dy, mag, angle, true);
        mag *= (1.f / (8 * sqrt(2.f)));

        cv::Mat nmag;
        mag.convertTo(nmag, CV_8UC1);

        angle *=  6 / 360.f;

        for (int y = 0; y < h; ++y)
        {
            uchar* magnitude = nmag.ptr<uchar>(y);
            float* ang = angle.ptr<float>(y);

            for (int x = 0; x < w; ++x)
            {
                channels.ptr<uchar>(y + (h * (int)ang[x]))[x] = magnitude[x];
            }
        }

        cv::Mat luv, shrunk;
        cv::cvtColor(frame, luv, CV_BGR2Luv);

        std::vector<cv::Mat> splited;
        for (int i = 0; i < 3; ++i)
            splited.push_back(channels(cv::Rect(0, h * (7 + i), w, h)));
        split(luv, splited);

        cv::resize(channels, shrunk, cv::Size(), 1.0 / shrinkage, 1.0 / shrinkage, CV_INTER_AREA);
        cv::integral(shrunk, integrals, cv::noArray(), CV_32S);
    }

    int shrinkage;
    enum {BINS = 10};
};
}

// ToDo: parallelize it
// ToDo: sunch model size and shrinced model size usage/ Now model size mean already shrinked model
void sft::Octave::processPositives(const Dataset& dataset, const FeaturePool& pool)
{
    Preprocessor prepocessor(shrinkage);

    int w =  64 * pow(2, logScale) /shrinkage;
    int h = 128 * pow(2, logScale) /shrinkage * 10;

    integrals.create(pool.size(), (w + 1) * (h + 1), CV_32SC1);

    int total = 0;

    for (svector::const_iterator it = dataset.pos.begin(); it != dataset.pos.end(); ++it)
    {
        const string& curr = *it;

        dprintf("Process candidate positive image %s\n", curr.c_str());

        cv::Mat sample   = cv::imread(curr);
        cv::Mat channels = integrals.row(total).reshape(0, h + 1);
        prepocessor.apply(sample, channels);

        responses.ptr<float>(total)[0] = 1.f;

        if (++total >= npositives) break;
    }

    dprintf("Processing positives finished:\n\trequested %d positives, collected %d samples.\n", npositives, total);

    npositives  = total;
    nnegatives = cvRound(nnegatives * total / (double)npositives);
}

void sft::Octave::generateNegatives(const Dataset& dataset)
{
    // ToDo: set seed, use offsets
    sft::Random::engine eng;
    sft::Random::engine idxEng;

    int w =  64 * pow(2, logScale) /shrinkage;
    int h = 128 * pow(2, logScale) /shrinkage * 10;

    Preprocessor prepocessor(shrinkage);

    int nimages = (int)dataset.neg.size();
    sft::Random::uniform iRand(0, nimages - 1);

    int total = 0;
    Mat sum;
    for (int i = npositives; i < nnegatives + npositives; ++total)
    {
        int curr = iRand(idxEng);

        dprintf("View %d-th sample\n", curr);
        dprintf("Process %s\n", dataset.neg[curr].c_str());

        Mat frame = cv::imread(dataset.neg[curr]);
        prepocessor.apply(frame, sum);

        std::cout << "WARNING: " << frame.cols << " " << frame.rows << std::endl;
        std::cout << "WARNING: " << frame.cols / shrinkage << " " << frame.rows / shrinkage << std::endl;

        int maxW = frame.cols / shrinkage - 2 * boundingBox.x - boundingBox.width;
        int maxH = frame.rows / shrinkage - 2 * boundingBox.y - boundingBox.height;

        std::cout << "WARNING: " << maxW << " " << maxH << std::endl;

        sft::Random::uniform wRand(0, maxW -1);
        sft::Random::uniform hRand(0, maxH -1);

        int dx = wRand(eng);
        int dy = hRand(eng);

        std::cout << "WARNING: " << dx << " " << dy << std::endl;
        std::cout << "WARNING: " << dx + boundingBox.width + 1 << " " << dy + boundingBox.height + 1 << std::endl;
        std::cout << "WARNING: " << sum.cols << " " << sum.rows << std::endl;

        sum = sum(cv::Rect(dx, dy, boundingBox.width + 1, boundingBox.height * 10 + 1));

        dprintf("generated %d %d\n", dx, dy);

        // if (predict(sum))
        {
            responses.ptr<float>(i)[0] = 0.f;
            // sum = sum.reshape(0, 1);
            sum.copyTo(integrals.row(i).reshape(0, h + 1));
            ++i;
        }
    }

    dprintf("Processing negatives finished:\n\trequested %d negatives, viewed %d samples.\n", nnegatives, total);
}

bool sft::Octave::train(const Dataset& dataset, const FeaturePool& pool)
{
    // 1. fill integrals and classes
    processPositives(dataset, pool);
    generateNegatives(dataset);

    // 2. only sumple case (all features used)
    int nfeatures = pool.size();
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

    cv::Mat trainData(nfeatures, nsamples, CV_32FC1);
    for (int fi = 0; fi < nfeatures; ++fi)
    {
        float* dptr = trainData.ptr<float>(fi);
        for (int si = 0; si < nsamples; ++si)
        {
            dptr[si] = pool.apply(fi, si, integrals);
        }
    }

    cv::Mat missingMask;

    bool ok = train(trainData, responses, varIdx, sampleIdx, varType, missingMask);
    if (!ok)
        std::cout << "ERROR:tree couldnot be trained" << std::endl;
    return ok;

}

// ========= FeaturePool ========= //
sft::FeaturePool::FeaturePool(cv::Size m, int n) : model(m), nfeatures(n)
{
    CV_Assert(m != cv::Size() && n > 0);
    fill(nfeatures);
}

sft::FeaturePool::~FeaturePool(){}

float sft::FeaturePool::apply(int fi, int si, const Mat& integrals) const
{
    return 0.f;
}


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

// ============ Dataset ============ //
namespace {
using namespace sft;

string itoa(long i)
{
    char s[65];
    sprintf(s, "%ld", i);
    return std::string(s);
}

void glob(const string& path, svector& ret)
{
    glob_t glob_result;
    glob(path.c_str(), GLOB_TILDE, 0, &glob_result);

    ret.clear();
    ret.reserve(glob_result.gl_pathc);

    for(uint i = 0; i < glob_result.gl_pathc; ++i)
    {
        ret.push_back(std::string(glob_result.gl_pathv[i]));
        dprintf("%s\n", ret[i].c_str());
    }

    globfree(&glob_result);
}
}
// in the default case data folders should be alligned as following:
// 1. positives: <train or test path>/octave_<octave number>/pos/*.png
// 2. negatives: <train or test path>/octave_<octave number>/neg/*.png
Dataset::Dataset(const string& path, const int oct)
{
    dprintf("%s\n", "get dataset file names...");

    dprintf("%s\n", "Positives globbing...");
    glob(path + "/pos/octave_" + itoa(oct) + "/*.png", pos);

    dprintf("%s\n", "Negatives globbing...");
    glob(path + "/neg/octave_" + itoa(oct) + "/*.png", neg);

    // Check: files not empty
    CV_Assert(pos.size() != size_t(0));
    CV_Assert(neg.size() != size_t(0));
}