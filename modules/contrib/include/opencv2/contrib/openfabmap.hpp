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

#ifndef __OPENCV_OPENFABMAP_H_
#define __OPENCV_OPENFABMAP_H_

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

#include <vector>
#include <list>
#include <map>
#include <set>
#include <valarray>

namespace cv {

namespace of2 {

using std::list;
using std::map;
using std::multiset;

/*
    Return data format of a FABMAP compare call
*/
struct CV_EXPORTS IMatch {

    IMatch() :
        queryIdx(-1), imgIdx(-1), likelihood(-DBL_MAX), match(-DBL_MAX) {
    }
    IMatch(int _queryIdx, int _imgIdx, double _likelihood, double _match) :
        queryIdx(_queryIdx), imgIdx(_imgIdx), likelihood(_likelihood), match(
                _match) {
    }

    int queryIdx;    //query index
    int imgIdx;      //test index

    double likelihood;  //raw loglikelihood
    double match;      //normalised probability

    bool operator<(const IMatch& m) const {
        return match < m.match;
    }

};

/*
    Base FabMap class. Each FabMap method inherits from this class.
*/
class CV_EXPORTS FabMap {
public:

    //FabMap options
    enum {
        MEAN_FIELD = 1,
        SAMPLED = 2,
        NAIVE_BAYES = 4,
        CHOW_LIU = 8,
        MOTION_MODEL = 16
    };

    FabMap(const Mat& clTree, double PzGe, double PzGNe, int flags,
            int numSamples = 0);
    virtual ~FabMap();

    //methods to add training data for sampling method
    virtual void addTraining(const Mat& queryImgDescriptor);
    virtual void addTraining(const vector<Mat>& queryImgDescriptors);

    //methods to add to the test data
    virtual void add(const Mat& queryImgDescriptor);
    virtual void add(const vector<Mat>& queryImgDescriptors);

    //accessors
    const vector<Mat>& getTrainingImgDescriptors() const;
    const vector<Mat>& getTestImgDescriptors() const;

    //Main FabMap image comparison
    void compare(const Mat& queryImgDescriptor,
            vector<IMatch>& matches, bool addQuery = false,
            const Mat& mask = Mat());
    void compare(const Mat& queryImgDescriptor,
            const Mat& testImgDescriptors, vector<IMatch>& matches,
            const Mat& mask = Mat());
    void compare(const Mat& queryImgDescriptor,
            const vector<Mat>& testImgDescriptors,
            vector<IMatch>& matches, const Mat& mask = Mat());
    void compare(const vector<Mat>& queryImgDescriptors, vector<
            IMatch>& matches, bool addQuery = false, const Mat& mask =
            Mat());
    void compare(const vector<Mat>& queryImgDescriptors,
            const vector<Mat>& testImgDescriptors,
            vector<IMatch>& matches, const Mat& mask = Mat());

protected:

    void compareImgDescriptor(const Mat& queryImgDescriptor,
            int queryIndex, const vector<Mat>& testImgDescriptors,
            vector<IMatch>& matches);

    void addImgDescriptor(const Mat& queryImgDescriptor);

    //the getLikelihoods method is overwritten for each different FabMap
    //method.
    virtual void getLikelihoods(const Mat& queryImgDescriptor,
            const vector<Mat>& testImgDescriptors,
            vector<IMatch>& matches);
    virtual double getNewPlaceLikelihood(const Mat& queryImgDescriptor);

    //turn likelihoods into probabilities (also add in motion model if used)
    void normaliseDistribution(vector<IMatch>& matches);

    //Chow-Liu Tree
    int pq(int q);
    double Pzq(int q, bool zq);
    double PzqGzpq(int q, bool zq, bool zpq);

    //FAB-MAP Core
    double PzqGeq(bool zq, bool eq);
    double PeqGL(int q, bool Lzq, bool eq);
    double PzqGL(int q, bool zq, bool zpq, bool Lzq);
    double PzqGzpqL(int q, bool zq, bool zpq, bool Lzq);
    double (FabMap::*PzGL)(int q, bool zq, bool zpq, bool Lzq);

    //data
    Mat clTree;
    vector<Mat> trainingImgDescriptors;
    vector<Mat> testImgDescriptors;
    vector<IMatch> priorMatches;

    //parameters
    double PzGe;
    double PzGNe;
    double Pnew;

    double mBias;
    double sFactor;

    int flags;
    int numSamples;

};

/*
    The original FAB-MAP algorithm, developed based on:
    http://ijr.sagepub.com/content/27/6/647.short
*/
class CV_EXPORTS FabMap1: public FabMap {
public:
    FabMap1(const Mat& clTree, double PzGe, double PzGNe, int flags,
            int numSamples = 0);
    virtual ~FabMap1();
protected:

    //FabMap1 implementation of likelihood comparison
    void getLikelihoods(const Mat& queryImgDescriptor, const vector<
            Mat>& testImgDescriptors, vector<IMatch>& matches);
};

/*
    A computationally faster version of the original FAB-MAP algorithm. A look-
    up-table is used to precompute many of the reoccuring calculations
*/
class CV_EXPORTS FabMapLUT: public FabMap {
public:
    FabMapLUT(const Mat& clTree, double PzGe, double PzGNe,
            int flags, int numSamples = 0, int precision = 6);
    virtual ~FabMapLUT();
protected:

    //FabMap look-up-table implementation of the likelihood comparison
    void getLikelihoods(const Mat& queryImgDescriptor, const vector<
            Mat>& testImgDescriptors, vector<IMatch>& matches);

    //precomputed data
    int (*table)[8];

    //data precision
    int precision;
};

/*
    The Accelerated FAB-MAP algorithm, developed based on:
    http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=5613942
*/
class CV_EXPORTS FabMapFBO: public FabMap {
public:
    FabMapFBO(const Mat& clTree, double PzGe, double PzGNe, int flags,
            int numSamples = 0, double rejectionThreshold = 1e-8, double PsGd =
                    1e-8, int bisectionStart = 512, int bisectionIts = 9);
    virtual ~FabMapFBO();

protected:

    //FabMap Fast Bail-out implementation of the likelihood comparison
    void getLikelihoods(const Mat& queryImgDescriptor, const vector<
            Mat>& testImgDescriptors, vector<IMatch>& matches);

    //stucture used to determine word comparison order
    struct WordStats {
        WordStats() :
            q(0), info(0), V(0), M(0) {
        }

        WordStats(int _q, double _info) :
            q(_q), info(_info), V(0), M(0) {
        }

        int q;
        double info;
        mutable double V;
        mutable double M;

        bool operator<(const WordStats& w) const {
            return info < w.info;
        }

    };

    //private fast bail-out necessary functions
    void setWordStatistics(const Mat& queryImgDescriptor, multiset<WordStats>& wordData);
    double limitbisection(double v, double m);
    double bennettInequality(double v, double m, double delta);
    static bool compInfo(const WordStats& first, const WordStats& second);

    //parameters
    double PsGd;
    double rejectionThreshold;
    int bisectionStart;
    int bisectionIts;
};

/*
    The FAB-MAP2.0 algorithm, developed based on:
    http://ijr.sagepub.com/content/30/9/1100.abstract
*/
class CV_EXPORTS FabMap2: public FabMap {
public:

    FabMap2(const Mat& clTree, double PzGe, double PzGNe, int flags);
    virtual ~FabMap2();

    //FabMap2 builds the inverted index and requires an additional training/test
    //add function
    void addTraining(const Mat& queryImgDescriptors) {
        FabMap::addTraining(queryImgDescriptors);
    }
    void addTraining(const vector<Mat>& queryImgDescriptors);

    void add(const Mat& queryImgDescriptors) {
        FabMap::add(queryImgDescriptors);
    }
    void add(const vector<Mat>& queryImgDescriptors);

protected:

    //FabMap2 implementation of the likelihood comparison
    void getLikelihoods(const Mat& queryImgDescriptor, const vector<
            Mat>& testImgDescriptors, vector<IMatch>& matches);
    double getNewPlaceLikelihood(const Mat& queryImgDescriptor);

    //the likelihood function using the inverted index
    void getIndexLikelihoods(const Mat& queryImgDescriptor, vector<
                             double>& defaults, map<int, vector<int> >& invertedMap,
            vector<IMatch>& matches);
    void addToIndex(const Mat& queryImgDescriptor,
            vector<double>& defaults,
            map<int, vector<int> >& invertedMap);

    //data
    vector<double> d1, d2, d3, d4;
    vector<vector<int> > children;

    // TODO: inverted map a vector?

    vector<double> trainingDefaults;
    map<int, vector<int> > trainingInvertedMap;

    vector<double> testDefaults;
    map<int, vector<int> > testInvertedMap;

};
/*
    A Chow-Liu tree is required by FAB-MAP. The Chow-Liu tree provides an
    estimate of the full distribution of visual words using a minimum spanning
    tree. The tree is generated through training data.
*/
class CV_EXPORTS ChowLiuTree {
public:
    ChowLiuTree();
    virtual ~ChowLiuTree();

    //add data to the chow-liu tree before calling make
    void add(const Mat& imgDescriptor);
    void add(const vector<Mat>& imgDescriptors);

    const vector<Mat>& getImgDescriptors() const;

    Mat make(double infoThreshold = 0.0);

private:
    vector<Mat> imgDescriptors;
    Mat mergedImgDescriptors;

    typedef struct info {
        float score;
        short word1;
        short word2;
    } info;

    //probabilities extracted from mergedImgDescriptors
    double P(int a, bool za);
    double JP(int a, bool za, int b, bool zb); //a & b
    double CP(int a, bool za, int b, bool zb); // a | b

    //calculating mutual information of all edges
    void createBaseEdges(list<info>& edges, double infoThreshold);
    double calcMutInfo(int word1, int word2);
    static bool sortInfoScores(const info& first, const info& second);

    //selecting minimum spanning egdges with maximum information
    bool reduceEdgesToMinSpan(list<info>& edges);

    //building the tree sctructure
    Mat buildTree(int root_word, list<info> &edges);
    void recAddToTree(Mat &cltree, int q, int pq,
        list<info> &remaining_edges);
    vector<int> extractChildren(list<info> &remaining_edges, int q);

};

/*
    A custom vocabulary training method based on:
    http://www.springerlink.com/content/d1h6j8x552532003/
*/
class CV_EXPORTS BOWMSCTrainer: public BOWTrainer {
public:
    BOWMSCTrainer(double clusterSize = 0.4);
    virtual ~BOWMSCTrainer();

    // Returns trained vocabulary (i.e. cluster centers).
    virtual Mat cluster() const;
    virtual Mat cluster(const Mat& descriptors) const;

protected:

    double clusterSize;

};

}

}

#endif /* OPENFABMAP_H_ */
