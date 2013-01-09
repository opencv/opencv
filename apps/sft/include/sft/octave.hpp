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

#ifndef __SFT_OCTAVE_HPP__
#define __SFT_OCTAVE_HPP__

#include <opencv2/ml/ml.hpp>
#include <sft/common.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
namespace sft
{

class Preprocessor
{
public:
    Preprocessor() {}

    void apply(const cv::Mat& frame, cv::Mat& integrals) const
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

        float shrinkage = static_cast<float>(integrals.cols - 1) / channels.cols;

        CV_Assert(shrinkage == 0.25);

        cv::resize(channels, shrunk, cv::Size(), shrinkage, shrinkage, CV_INTER_AREA);
        cv::integral(shrunk, integrals, cv::noArray(), CV_32S);
    }

    enum {BINS = 10};
};

struct ICF
{
    ICF(int x, int y, int w, int h, int ch) : bb(cv::Rect(x, y, w, h)), channel(ch) {}

    bool operator ==(ICF b)
    {
        return bb == b.bb && channel == b.channel;
    }

    bool operator !=(ICF b)
    {
        return bb != b.bb || channel != b.channel;
    }


    float operator() (const cv::Mat& integrals, const cv::Size& model) const
    {
        int step = model.width + 1;

        const int* ptr = integrals.ptr<int>(0) + (model.height * channel + bb.y) * step + bb.x;

        int a = ptr[0];
        int b = ptr[bb.width];

        ptr += bb.height * step;

        int c = ptr[bb.width];
        int d = ptr[0];

        return (float)(a - b + c - d);
    }

private:
    cv::Rect bb;
    int channel;

    friend void write(cv::FileStorage& fs, const std::string&, const ICF& f);
    friend std::ostream& operator<<(std::ostream& out, const ICF& f);
};

void write(cv::FileStorage& fs, const std::string&, const ICF& f);
std::ostream& operator<<(std::ostream& out, const ICF& m);

class ICFFeaturePool : public cv::FeaturePool
{
public:
    ICFFeaturePool(cv::Size model, int nfeatures);

    virtual int size() const { return (int)pool.size(); }
    virtual float apply(int fi, int si, const cv::Mat& integrals) const;
    virtual void preprocess(const cv::Mat& frame, cv::Mat& integrals) const;
    virtual void write( cv::FileStorage& fs, int index) const;

    virtual ~ICFFeaturePool();

private:

    void fill(int desired);

    cv::Size model;
    int nfeatures;

    std::vector<ICF> pool;

    static const unsigned int seed = 0;

    Preprocessor preprocessor;

    enum { N_CHANNELS = 10 };
};


using cv::FeaturePool;



class Dataset
{
public:
    typedef enum {POSITIVE = 1, NEGATIVE = 2} SampleType;
    Dataset(const sft::string& path, const int octave);

    cv::Mat get(SampleType type, int idx) const;
    int available(SampleType type) const;

private:
    svector pos;
    svector neg;
};

// used for traning single octave scale
class Octave : cv::Boost
{
public:

    enum
    {
        // Direct backward pruning. (Cha Zhang and Paul Viola)
        DBP = 1,
        // Multiple instance pruning. (Cha Zhang and Paul Viola)
        MIP = 2,
        // Originally proposed by L. bourdev and J. brandt
        HEURISTIC = 4
    };

    Octave(cv::Rect boundingBox, int npositives, int nnegatives, int logScale, int shrinkage);
    virtual ~Octave();

    virtual bool train(const Dataset& dataset, const FeaturePool* pool, int weaks, int treeDepth);

    virtual float predict( const Mat& _sample, Mat& _votes, bool raw_mode, bool return_sum ) const;
    virtual void setRejectThresholds(cv::Mat& thresholds);
    virtual void write( CvFileStorage* fs, string name) const;

    virtual void write( cv::FileStorage &fs, const FeaturePool* pool, const Mat& thresholds) const;

    int logScale;

protected:
    virtual bool train( const cv::Mat& trainData, const cv::Mat& responses, const cv::Mat& varIdx=cv::Mat(),
       const cv::Mat& sampleIdx=cv::Mat(), const cv::Mat& varType=cv::Mat(), const cv::Mat& missingDataMask=cv::Mat());

    void processPositives(const Dataset& dataset, const FeaturePool* pool);
    void generateNegatives(const Dataset& dataset, const FeaturePool* pool);

    float predict( const Mat& _sample, const cv::Range range) const;
private:
    void traverse(const CvBoostTree* tree, cv::FileStorage& fs, int& nfeatures, int* used, const double* th) const;
    virtual void initial_weights(double (&p)[2]);

    cv::Rect boundingBox;

    int npositives;
    int nnegatives;

    int shrinkage;

    Mat integrals;
    Mat responses;

    CvBoostParams params;

    Mat trainData;
};

}

#endif