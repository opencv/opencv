/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
// This file originates from the openFABMAP project:
// [http://code.google.com/p/openfabmap/]
//
// For published work which uses all or part of OpenFABMAP, please cite:
// [http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6224843]
//
// Original Algorithm by Mark Cummins and Paul Newman:
// [http://ijr.sagepub.com/content/27/6/647.short]
// [http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=5613942]
// [http://ijr.sagepub.com/content/30/9/1100.abstract]
//
//                           License Agreement
//
// Copyright (C) 2012 Arren Glover [aj.glover@qut.edu.au] and
//                    Will Maddern [w.maddern@qut.edu.au], all rights reserved.
//
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
#include "opencv2/contrib/openfabmap.hpp"


/*
    Calculate the sum of two log likelihoods
*/
namespace cv {

namespace of2 {

static double logsumexp(double a, double b) {
    return a > b ? log(1 + exp(b - a)) + a : log(1 + exp(a - b)) + b;
}

FabMap::FabMap(const Mat& _clTree, double _PzGe,
        double _PzGNe, int _flags, int _numSamples) :
    clTree(_clTree), PzGe(_PzGe), PzGNe(_PzGNe), flags(
            _flags), numSamples(_numSamples) {

    CV_Assert(flags & MEAN_FIELD || flags & SAMPLED);
    CV_Assert(flags & NAIVE_BAYES || flags & CHOW_LIU);
    if (flags & NAIVE_BAYES) {
        PzGL = &FabMap::PzqGL;
    } else {
        PzGL = &FabMap::PzqGzpqL;
    }

    //check for a valid Chow-Liu tree
    CV_Assert(clTree.type() == CV_64FC1);
    cv::checkRange(clTree.row(0), false, NULL, 0, clTree.cols);
    cv::checkRange(clTree.row(1), false, NULL, DBL_MIN, 1);
    cv::checkRange(clTree.row(2), false, NULL, DBL_MIN, 1);
    cv::checkRange(clTree.row(3), false, NULL, DBL_MIN, 1);

    // TODO: Add default values for member variables
    Pnew = 0.9;
    sFactor = 0.99;
    mBias = 0.5;
}

FabMap::~FabMap() {
}

const std::vector<cv::Mat>& FabMap::getTrainingImgDescriptors() const {
    return trainingImgDescriptors;
}

const std::vector<cv::Mat>& FabMap::getTestImgDescriptors() const {
    return testImgDescriptors;
}

void FabMap::addTraining(const Mat& queryImgDescriptor) {
    CV_Assert(!queryImgDescriptor.empty());
    vector<Mat> queryImgDescriptors;
    for (int i = 0; i < queryImgDescriptor.rows; i++) {
        queryImgDescriptors.push_back(queryImgDescriptor.row(i));
    }
    addTraining(queryImgDescriptors);
}

void FabMap::addTraining(const vector<Mat>& queryImgDescriptors) {
    for (size_t i = 0; i < queryImgDescriptors.size(); i++) {
        CV_Assert(!queryImgDescriptors[i].empty());
        CV_Assert(queryImgDescriptors[i].rows == 1);
        CV_Assert(queryImgDescriptors[i].cols == clTree.cols);
        CV_Assert(queryImgDescriptors[i].type() == CV_32F);
        trainingImgDescriptors.push_back(queryImgDescriptors[i]);
    }
}

void FabMap::add(const cv::Mat& queryImgDescriptor) {
    CV_Assert(!queryImgDescriptor.empty());
    vector<Mat> queryImgDescriptors;
    for (int i = 0; i < queryImgDescriptor.rows; i++) {
        queryImgDescriptors.push_back(queryImgDescriptor.row(i));
    }
    add(queryImgDescriptors);
}

void FabMap::add(const std::vector<cv::Mat>& queryImgDescriptors) {
    for (size_t i = 0; i < queryImgDescriptors.size(); i++) {
        CV_Assert(!queryImgDescriptors[i].empty());
        CV_Assert(queryImgDescriptors[i].rows == 1);
        CV_Assert(queryImgDescriptors[i].cols == clTree.cols);
        CV_Assert(queryImgDescriptors[i].type() == CV_32F);
        testImgDescriptors.push_back(queryImgDescriptors[i]);
    }
}

void FabMap::compare(const Mat& queryImgDescriptor,
            vector<IMatch>& matches, bool addQuery,
            const Mat& mask) {
    CV_Assert(!queryImgDescriptor.empty());
    vector<Mat> queryImgDescriptors;
    for (int i = 0; i < queryImgDescriptor.rows; i++) {
        queryImgDescriptors.push_back(queryImgDescriptor.row(i));
    }
    compare(queryImgDescriptors,matches,addQuery,mask);
}

void FabMap::compare(const Mat& queryImgDescriptor,
            const Mat& testImgDescriptor, vector<IMatch>& matches,
            const Mat& mask) {
    CV_Assert(!queryImgDescriptor.empty());
    vector<Mat> queryImgDescriptors;
    for (int i = 0; i < queryImgDescriptor.rows; i++) {
        queryImgDescriptors.push_back(queryImgDescriptor.row(i));
    }

    CV_Assert(!testImgDescriptor.empty());
    vector<Mat> _testImgDescriptors;
    for (int i = 0; i < testImgDescriptor.rows; i++) {
        _testImgDescriptors.push_back(testImgDescriptor.row(i));
    }
    compare(queryImgDescriptors,_testImgDescriptors,matches,mask);

}

void FabMap::compare(const Mat& queryImgDescriptor,
        const vector<Mat>& _testImgDescriptors,
        vector<IMatch>& matches, const Mat& mask) {
    CV_Assert(!queryImgDescriptor.empty());
    vector<Mat> queryImgDescriptors;
    for (int i = 0; i < queryImgDescriptor.rows; i++) {
        queryImgDescriptors.push_back(queryImgDescriptor.row(i));
    }
    compare(queryImgDescriptors,_testImgDescriptors,matches,mask);
}

void FabMap::compare(const vector<Mat>& queryImgDescriptors,
                     vector<IMatch>& matches, bool addQuery, const Mat& /*mask*/) {

    // TODO: add first query if empty (is this necessary)

    for (size_t i = 0; i < queryImgDescriptors.size(); i++) {
        CV_Assert(!queryImgDescriptors[i].empty());
        CV_Assert(queryImgDescriptors[i].rows == 1);
        CV_Assert(queryImgDescriptors[i].cols == clTree.cols);
        CV_Assert(queryImgDescriptors[i].type() == CV_32F);

        // TODO: add mask

        compareImgDescriptor(queryImgDescriptors[i],
                             (int)i, testImgDescriptors, matches);
        if (addQuery)
            add(queryImgDescriptors[i]);
    }
}

void FabMap::compare(const vector<Mat>& queryImgDescriptors,
        const vector<Mat>& _testImgDescriptors,
        vector<IMatch>& matches, const Mat& /*mask*/) {

    CV_Assert(!(flags & MOTION_MODEL));
    for (size_t i = 0; i < _testImgDescriptors.size(); i++) {
        CV_Assert(!_testImgDescriptors[i].empty());
        CV_Assert(_testImgDescriptors[i].rows == 1);
        CV_Assert(_testImgDescriptors[i].cols == clTree.cols);
        CV_Assert(_testImgDescriptors[i].type() == CV_32F);
    }

    for (size_t i = 0; i < queryImgDescriptors.size(); i++) {
        CV_Assert(!queryImgDescriptors[i].empty());
        CV_Assert(queryImgDescriptors[i].rows == 1);
        CV_Assert(queryImgDescriptors[i].cols == clTree.cols);
        CV_Assert(queryImgDescriptors[i].type() == CV_32F);

        // TODO: add mask

        compareImgDescriptor(queryImgDescriptors[i],
                (int)i, _testImgDescriptors, matches);
    }
}

void FabMap::compareImgDescriptor(const Mat& queryImgDescriptor,
        int queryIndex, const vector<Mat>& _testImgDescriptors,
        vector<IMatch>& matches) {

    vector<IMatch> queryMatches;
    queryMatches.push_back(IMatch(queryIndex,-1,
        getNewPlaceLikelihood(queryImgDescriptor),0));
    getLikelihoods(queryImgDescriptor,_testImgDescriptors,queryMatches);
    normaliseDistribution(queryMatches);
    for (size_t j = 1; j < queryMatches.size(); j++) {
        queryMatches[j].queryIdx = queryIndex;
    }
    matches.insert(matches.end(), queryMatches.begin(), queryMatches.end());
}

void FabMap::getLikelihoods(const Mat& /*queryImgDescriptor*/,
        const vector<Mat>& /*testImgDescriptors*/, vector<IMatch>& /*matches*/) {

}

double FabMap::getNewPlaceLikelihood(const Mat& queryImgDescriptor) {
    if (flags & MEAN_FIELD) {
        double logP = 0;
        bool zq, zpq;
        if(flags & NAIVE_BAYES) {
            for (int q = 0; q < clTree.cols; q++) {
                zq = queryImgDescriptor.at<float>(0,q) > 0;

                logP += log(Pzq(q, false) * PzqGeq(zq, false) +
                        Pzq(q, true) * PzqGeq(zq, true));
            }
        } else {
            for (int q = 0; q < clTree.cols; q++) {
                zq = queryImgDescriptor.at<float>(0,q) > 0;
                zpq = queryImgDescriptor.at<float>(0,pq(q)) > 0;

                double alpha, beta, p;
                alpha = Pzq(q, zq) * PzqGeq(!zq, false) * PzqGzpq(q, !zq, zpq);
                beta = Pzq(q, !zq) * PzqGeq(zq, false) * PzqGzpq(q, zq, zpq);
                p = Pzq(q, false) * beta / (alpha + beta);

                alpha = Pzq(q, zq) * PzqGeq(!zq, true) * PzqGzpq(q, !zq, zpq);
                beta = Pzq(q, !zq) * PzqGeq(zq, true) * PzqGzpq(q, zq, zpq);
                p += Pzq(q, true) * beta / (alpha + beta);

                logP += log(p);
            }
        }
        return logP;
    }

    if (flags & SAMPLED) {
        CV_Assert(!trainingImgDescriptors.empty());
        CV_Assert(numSamples > 0);

        vector<Mat> sampledImgDescriptors;

        // TODO: this method can result in the same sample being added
        // multiple times. Is this desired?

        for (int i = 0; i < numSamples; i++) {
            int index = rand() % trainingImgDescriptors.size();
            sampledImgDescriptors.push_back(trainingImgDescriptors[index]);
        }

        vector<IMatch> matches;
        getLikelihoods(queryImgDescriptor,sampledImgDescriptors,matches);

        double averageLogLikelihood = -DBL_MAX + matches.front().likelihood + 1;
        for (int i = 0; i < numSamples; i++) {
            averageLogLikelihood =
                logsumexp(matches[i].likelihood, averageLogLikelihood);
        }

        return averageLogLikelihood - log((double)numSamples);
    }
    return 0;
}

void FabMap::normaliseDistribution(vector<IMatch>& matches) {
    CV_Assert(!matches.empty());

    if (flags & MOTION_MODEL) {

        matches[0].match = matches[0].likelihood + log(Pnew);

        if (priorMatches.size() > 2) {
            matches[1].match = matches[1].likelihood;
            matches[1].match += log(
                (2 * (1-mBias) * priorMatches[1].match +
                priorMatches[1].match +
                2 * mBias * priorMatches[2].match) / 3);
            for (size_t i = 2; i < priorMatches.size()-1; i++) {
                matches[i].match = matches[i].likelihood;
                matches[i].match += log(
                    (2 * (1-mBias) * priorMatches[i-1].match +
                    priorMatches[i].match +
                    2 * mBias * priorMatches[i+1].match)/3);
            }
            matches[priorMatches.size()-1].match =
                matches[priorMatches.size()-1].likelihood;
            matches[priorMatches.size()-1].match += log(
                (2 * (1-mBias) * priorMatches[priorMatches.size()-2].match +
                priorMatches[priorMatches.size()-1].match +
                2 * mBias * priorMatches[priorMatches.size()-1].match)/3);

            for(size_t i = priorMatches.size(); i < matches.size(); i++) {
                matches[i].match = matches[i].likelihood;
            }
        } else {
            for(size_t i = 1; i < matches.size(); i++) {
                matches[i].match = matches[i].likelihood;
            }
        }

        double logsum = -DBL_MAX + matches.front().match + 1;

        //calculate the normalising constant
        for (size_t i = 0; i < matches.size(); i++) {
            logsum = logsumexp(logsum, matches[i].match);
        }

        //normalise
        for (size_t i = 0; i < matches.size(); i++) {
            matches[i].match = exp(matches[i].match - logsum);
        }

        //smooth final probabilities
        for (size_t i = 0; i < matches.size(); i++) {
            matches[i].match = sFactor*matches[i].match +
            (1 - sFactor)/matches.size();
        }

        //update our location priors
        priorMatches = matches;

    } else {

        double logsum = -DBL_MAX + matches.front().likelihood + 1;

        for (size_t i = 0; i < matches.size(); i++) {
            logsum = logsumexp(logsum, matches[i].likelihood);
        }
        for (size_t i = 0; i < matches.size(); i++) {
            matches[i].match = exp(matches[i].likelihood - logsum);
        }
        for (size_t i = 0; i < matches.size(); i++) {
            matches[i].match = sFactor*matches[i].match +
            (1 - sFactor)/matches.size();
        }
    }
}

int FabMap::pq(int q) {
    return (int)clTree.at<double>(0,q);
}

double FabMap::Pzq(int q, bool zq) {
    return (zq) ? clTree.at<double>(1,q) : 1 - clTree.at<double>(1,q);
}

double FabMap::PzqGzpq(int q, bool zq, bool zpq) {
    if (zpq) {
        return (zq) ? clTree.at<double>(2,q) : 1 - clTree.at<double>(2,q);
    } else {
        return (zq) ? clTree.at<double>(3,q) : 1 - clTree.at<double>(3,q);
    }
}

double FabMap::PzqGeq(bool zq, bool eq) {
    if (eq) {
        return (zq) ? PzGe : 1 - PzGe;
    } else {
        return (zq) ? PzGNe : 1 - PzGNe;
    }
}

double FabMap::PeqGL(int q, bool Lzq, bool eq) {
    double alpha, beta;
    alpha = PzqGeq(Lzq, true) * Pzq(q, true);
    beta = PzqGeq(Lzq, false) * Pzq(q, false);

    if (eq) {
        return alpha / (alpha + beta);
    } else {
        return 1 - (alpha / (alpha + beta));
    }
}

double FabMap::PzqGL(int q, bool zq, bool /*zpq*/, bool Lzq) {
    return PeqGL(q, Lzq, false) * PzqGeq(zq, false) +
        PeqGL(q, Lzq, true) * PzqGeq(zq, true);
}


double FabMap::PzqGzpqL(int q, bool zq, bool zpq, bool Lzq) {
    double p;
    double alpha, beta;

    alpha = Pzq(q,  zq) * PzqGeq(!zq, false) * PzqGzpq(q, !zq, zpq);
    beta  = Pzq(q, !zq) * PzqGeq( zq, false) * PzqGzpq(q,  zq, zpq);
    p = PeqGL(q, Lzq, false) * beta / (alpha + beta);

    alpha = Pzq(q,  zq) * PzqGeq(!zq, true) * PzqGzpq(q, !zq, zpq);
    beta  = Pzq(q, !zq) * PzqGeq( zq, true) * PzqGzpq(q,  zq, zpq);
    p += PeqGL(q, Lzq, true) * beta / (alpha + beta);

    return p;
}


FabMap1::FabMap1(const Mat& _clTree, double _PzGe, double _PzGNe, int _flags,
        int _numSamples) : FabMap(_clTree, _PzGe, _PzGNe, _flags,
                _numSamples) {
}

FabMap1::~FabMap1() {
}

void FabMap1::getLikelihoods(const Mat& queryImgDescriptor,
        const vector<Mat>& testImageDescriptors, vector<IMatch>& matches) {

    for (size_t i = 0; i < testImageDescriptors.size(); i++) {
        bool zq, zpq, Lzq;
        double logP = 0;
        for (int q = 0; q < clTree.cols; q++) {

            zq = queryImgDescriptor.at<float>(0,q) > 0;
            zpq = queryImgDescriptor.at<float>(0,pq(q)) > 0;
            Lzq = testImageDescriptors[i].at<float>(0,q) > 0;

            logP += log((this->*PzGL)(q, zq, zpq, Lzq));

        }
        matches.push_back(IMatch(0,(int)i,logP,0));
    }
}

FabMapLUT::FabMapLUT(const Mat& _clTree, double _PzGe, double _PzGNe,
        int _flags, int _numSamples, int _precision) :
FabMap(_clTree, _PzGe, _PzGNe, _flags, _numSamples), precision(_precision) {

    int nWords = clTree.cols;
    double precFactor = (double)pow(10.0, precision);

    table = new int[nWords][8];

    for (int q = 0; q < nWords; q++) {
        for (unsigned char i = 0; i < 8; i++) {

            bool Lzq = (bool) ((i >> 2) & 0x01);
            bool zq = (bool) ((i >> 1) & 0x01);
            bool zpq = (bool) (i & 1);

            table[q][i] = -(int)(log((this->*PzGL)(q, zq, zpq, Lzq))
                    * precFactor);
        }
    }
}

FabMapLUT::~FabMapLUT() {
    delete[] table;
}

void FabMapLUT::getLikelihoods(const Mat& queryImgDescriptor,
        const vector<Mat>& testImageDescriptors, vector<IMatch>& matches) {

    double precFactor = (double)pow(10.0, -precision);

    for (size_t i = 0; i < testImageDescriptors.size(); i++) {
        unsigned long long int logP = 0;
        for (int q = 0; q < clTree.cols; q++) {
            logP += table[q][(queryImgDescriptor.at<float>(0,pq(q)) > 0) +
            ((queryImgDescriptor.at<float>(0, q) > 0) << 1) +
            ((testImageDescriptors[i].at<float>(0,q) > 0) << 2)];
        }
        matches.push_back(IMatch(0,(int)i,-precFactor*(double)logP,0));
    }
}

FabMapFBO::FabMapFBO(const Mat& _clTree, double _PzGe, double _PzGNe,
        int _flags, int _numSamples, double _rejectionThreshold,
        double _PsGd, int _bisectionStart, int _bisectionIts) :
FabMap(_clTree, _PzGe, _PzGNe, _flags, _numSamples), PsGd(_PsGd),
    rejectionThreshold(_rejectionThreshold), bisectionStart(_bisectionStart),
        bisectionIts(_bisectionIts) {
}


FabMapFBO::~FabMapFBO() {
}

void FabMapFBO::getLikelihoods(const Mat& queryImgDescriptor,
        const vector<Mat>& testImageDescriptors, vector<IMatch>& matches) {

    std::multiset<WordStats> wordData;
    setWordStatistics(queryImgDescriptor, wordData);

    vector<int> matchIndices;
    vector<IMatch> queryMatches;

    for (size_t i = 0; i < testImageDescriptors.size(); i++) {
        queryMatches.push_back(IMatch(0,(int)i,0,0));
        matchIndices.push_back((int)i);
    }

    double currBest  = -DBL_MAX;
    double bailedOut = DBL_MAX;

    for (std::multiset<WordStats>::iterator wordIter = wordData.begin();
            wordIter != wordData.end(); wordIter++) {
        bool zq = queryImgDescriptor.at<float>(0,wordIter->q) > 0;
        bool zpq = queryImgDescriptor.at<float>(0,pq(wordIter->q)) > 0;

        currBest = -DBL_MAX;

        for (size_t i = 0; i < matchIndices.size(); i++) {
            bool Lzq =
                testImageDescriptors[matchIndices[i]].at<float>(0,wordIter->q) > 0;
            queryMatches[matchIndices[i]].likelihood +=
                log((this->*PzGL)(wordIter->q,zq,zpq,Lzq));
            currBest =
                std::max(queryMatches[matchIndices[i]].likelihood, currBest);
        }

        if (matchIndices.size() == 1)
            continue;

        double delta = std::max(limitbisection(wordIter->V, wordIter->M),
            -log(rejectionThreshold));

        vector<int>::iterator matchIter = matchIndices.begin();
        while (matchIter != matchIndices.end()) {
            if (currBest - queryMatches[*matchIter].likelihood > delta) {
                queryMatches[*matchIter].likelihood = bailedOut;
                matchIter = matchIndices.erase(matchIter);
            } else {
                matchIter++;
            }
        }
    }

    for (size_t i = 0; i < queryMatches.size(); i++) {
        if (queryMatches[i].likelihood == bailedOut) {
            queryMatches[i].likelihood = currBest + log(rejectionThreshold);
        }
    }
    matches.insert(matches.end(), queryMatches.begin(), queryMatches.end());

}

void FabMapFBO::setWordStatistics(const Mat& queryImgDescriptor,
        std::multiset<WordStats>& wordData) {
    //words are sorted according to information = -ln(P(zq|zpq))
    //in non-log format this is lowest probability first
    for (int q = 0; q < clTree.cols; q++) {
        wordData.insert(WordStats(q,PzqGzpq(q,
                queryImgDescriptor.at<float>(0,q) > 0,
                queryImgDescriptor.at<float>(0,pq(q)) > 0)));
    }

    double d = 0, V = 0, M = 0;
    bool zq, zpq;

    for (std::multiset<WordStats>::reverse_iterator wordIter =
            wordData.rbegin();
            wordIter != wordData.rend(); wordIter++) {

        zq = queryImgDescriptor.at<float>(0,wordIter->q) > 0;
        zpq = queryImgDescriptor.at<float>(0,pq(wordIter->q)) > 0;

        d = log((this->*PzGL)(wordIter->q, zq, zpq, true)) -
            log((this->*PzGL)(wordIter->q, zq, zpq, false));

        V += pow(d, 2.0) * 2 *
            (Pzq(wordIter->q, true) - pow(Pzq(wordIter->q, true), 2.0));
        M = std::max(M, fabs(d));

        wordIter->V = V;
        wordIter->M = M;
    }
}

double FabMapFBO::limitbisection(double v, double m) {
    double midpoint, left_val, mid_val;
    double left = 0, right = bisectionStart;

    left_val = bennettInequality(v, m, left) - PsGd;

    for(int i = 0; i < bisectionIts; i++) {

        midpoint = (left + right)*0.5;
        mid_val = bennettInequality(v, m, midpoint)- PsGd;

        if(left_val * mid_val > 0) {
            left = midpoint;
            left_val = mid_val;
        } else {
            right = midpoint;
        }
    }

    return (right + left) * 0.5;
}

double FabMapFBO::bennettInequality(double v, double m, double delta) {
    double DMonV = delta * m / v;
    double f_delta = log(DMonV + sqrt(pow(DMonV, 2.0) + 1));
    return exp((v / pow(m, 2.0))*(cosh(f_delta) - 1 - DMonV * f_delta));
}

bool FabMapFBO::compInfo(const WordStats& first, const WordStats& second) {
    return first.info < second.info;
}

FabMap2::FabMap2(const Mat& _clTree, double _PzGe, double _PzGNe,
        int _flags) :
FabMap(_clTree, _PzGe, _PzGNe, _flags) {
    CV_Assert(flags & SAMPLED);

    children.resize(clTree.cols);

    for (int q = 0; q < clTree.cols; q++) {
        d1.push_back(log((this->*PzGL)(q, false, false, true) /
                (this->*PzGL)(q, false, false, false)));
        d2.push_back(log((this->*PzGL)(q, false, true, true) /
                (this->*PzGL)(q, false, true, false)) - d1[q]);
        d3.push_back(log((this->*PzGL)(q, true, false, true) /
                (this->*PzGL)(q, true, false, false))- d1[q]);
        d4.push_back(log((this->*PzGL)(q, true, true, true) /
                (this->*PzGL)(q, true, true, false))- d1[q]);
        children[pq(q)].push_back(q);
    }

}

FabMap2::~FabMap2() {
}


void FabMap2::addTraining(const vector<Mat>& queryImgDescriptors) {
    for (size_t i = 0; i < queryImgDescriptors.size(); i++) {
        CV_Assert(!queryImgDescriptors[i].empty());
        CV_Assert(queryImgDescriptors[i].rows == 1);
        CV_Assert(queryImgDescriptors[i].cols == clTree.cols);
        CV_Assert(queryImgDescriptors[i].type() == CV_32F);
        trainingImgDescriptors.push_back(queryImgDescriptors[i]);
        addToIndex(queryImgDescriptors[i], trainingDefaults, trainingInvertedMap);
    }
}


void FabMap2::add(const vector<Mat>& queryImgDescriptors) {
    for (size_t i = 0; i < queryImgDescriptors.size(); i++) {
        CV_Assert(!queryImgDescriptors[i].empty());
        CV_Assert(queryImgDescriptors[i].rows == 1);
        CV_Assert(queryImgDescriptors[i].cols == clTree.cols);
        CV_Assert(queryImgDescriptors[i].type() == CV_32F);
        testImgDescriptors.push_back(queryImgDescriptors[i]);
        addToIndex(queryImgDescriptors[i], testDefaults, testInvertedMap);
    }
}

void FabMap2::getLikelihoods(const Mat& queryImgDescriptor,
        const vector<Mat>& testImageDescriptors, vector<IMatch>& matches) {

    if (&testImageDescriptors == &testImgDescriptors) {
        getIndexLikelihoods(queryImgDescriptor, testDefaults, testInvertedMap,
            matches);
    } else {
        CV_Assert(!(flags & MOTION_MODEL));
        vector<double> defaults;
        std::map<int, vector<int> > invertedMap;
        for (size_t i = 0; i < testImageDescriptors.size(); i++) {
            addToIndex(testImageDescriptors[i],defaults,invertedMap);
        }
        getIndexLikelihoods(queryImgDescriptor, defaults, invertedMap, matches);
    }
}

double FabMap2::getNewPlaceLikelihood(const Mat& queryImgDescriptor) {

    CV_Assert(!trainingImgDescriptors.empty());

    vector<IMatch> matches;
    getIndexLikelihoods(queryImgDescriptor, trainingDefaults,
            trainingInvertedMap, matches);

    double averageLogLikelihood = -DBL_MAX + matches.front().likelihood + 1;
    for (size_t i = 0; i < matches.size(); i++) {
        averageLogLikelihood =
            logsumexp(matches[i].likelihood, averageLogLikelihood);
    }

    return averageLogLikelihood - log((double)trainingDefaults.size());

}

void FabMap2::addToIndex(const Mat& queryImgDescriptor,
        vector<double>& defaults,
        std::map<int, vector<int> >& invertedMap) {
    defaults.push_back(0);
    for (int q = 0; q < clTree.cols; q++) {
        if (queryImgDescriptor.at<float>(0,q) > 0) {
            defaults.back() += d1[q];
            invertedMap[q].push_back((int)defaults.size()-1);
        }
    }
}

void FabMap2::getIndexLikelihoods(const Mat& queryImgDescriptor,
        std::vector<double>& defaults,
        std::map<int, vector<int> >& invertedMap,
        std::vector<IMatch>& matches) {

    vector<int>::iterator LwithI, child;

    std::vector<double> likelihoods = defaults;

    for (int q = 0; q < clTree.cols; q++) {
        if (queryImgDescriptor.at<float>(0,q) > 0) {
            for (LwithI = invertedMap[q].begin();
                LwithI != invertedMap[q].end(); LwithI++) {

                if (queryImgDescriptor.at<float>(0,pq(q)) > 0) {
                    likelihoods[*LwithI] += d4[q];
                } else {
                    likelihoods[*LwithI] += d3[q];
                }
            }
            for (child = children[q].begin(); child != children[q].end();
                child++) {

                if (queryImgDescriptor.at<float>(0,*child) == 0) {
                    for (LwithI = invertedMap[*child].begin();
                        LwithI != invertedMap[*child].end(); LwithI++) {

                        likelihoods[*LwithI] += d2[*child];
                    }
                }
            }
        }
    }

    for (size_t i = 0; i < likelihoods.size(); i++) {
        matches.push_back(IMatch(0,(int)i,likelihoods[i],0));
    }
}

}

}
