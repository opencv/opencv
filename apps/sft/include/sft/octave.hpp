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

namespace sft
{

class Dataset
{
public:
    Dataset(const sft::string& path, const int octave);

// private:
    svector pos;
    svector neg;
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


    float operator() (const Mat& integrals, const cv::Size& model) const
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

    friend std::ostream& operator<<(std::ostream& out, const ICF& m);
};

std::ostream& operator<<(std::ostream& out, const ICF& m);

class FeaturePool
{
public:
    FeaturePool(cv::Size model, int nfeatures);

    int size() const { return (int)pool.size(); }
    float apply(int fi, int si, const Mat& integrals) const;

private:
    void fill(int desired);

    cv::Size model;
    int nfeatures;

    Icfvector pool;

    static const unsigned int seed = 0;

    enum { N_CHANNELS = 10 };
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

    virtual bool train(const Dataset& dataset, const FeaturePool& pool, int weaks, int treeDepth);
    virtual void write( CvFileStorage* fs, string name) const;
    virtual float predict( const Mat& _sample, Mat& _votes, bool raw_mode, bool return_sum ) const;
    virtual void setRejectThresholds(cv::Mat& thresholds);

    virtual void write( cv::FileStorage &fs, const Mat& thresholds = Mat()) const;

    int logScale;

protected:
    virtual bool train( const cv::Mat& trainData, const cv::Mat& responses, const cv::Mat& varIdx=cv::Mat(),
       const cv::Mat& sampleIdx=cv::Mat(), const cv::Mat& varType=cv::Mat(), const cv::Mat& missingDataMask=cv::Mat());

    void processPositives(const Dataset& dataset, const FeaturePool& pool);
    void generateNegatives(const Dataset& dataset);

    float predict( const Mat& _sample, const cv::Range range) const;
private:
    void traverse(const CvBoostTree* tree, cv::FileStorage& fs, const float* th = 0) const;

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