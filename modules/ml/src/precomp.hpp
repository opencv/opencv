/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#ifndef __OPENCV_ML_PRECOMP_HPP__
#define __OPENCV_ML_PRECOMP_HPP__

#include "opencv2/core.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/core/utility.hpp"

#include "opencv2/core/private.hpp"

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <vector>

/****************************************************************************************\
 *                               Main struct definitions                                  *
 \****************************************************************************************/

/* log(2*PI) */
#define CV_LOG2PI (1.8378770664093454835606594728112)

namespace cv
{
namespace ml
{
    using std::vector;

    #define CV_DTREE_CAT_DIR(idx,subset) \
        (2*((subset[(idx)>>5]&(1 << ((idx) & 31)))==0)-1)

    template<typename _Tp> struct cmp_lt_idx
    {
        cmp_lt_idx(const _Tp* _arr) : arr(_arr) {}
        bool operator ()(int a, int b) const { return arr[a] < arr[b]; }
        const _Tp* arr;
    };

    template<typename _Tp> struct cmp_lt_ptr
    {
        cmp_lt_ptr() {}
        bool operator ()(const _Tp* a, const _Tp* b) const { return *a < *b; }
    };

    static inline void setRangeVector(std::vector<int>& vec, int n)
    {
        vec.resize(n);
        for( int i = 0; i < n; i++ )
            vec[i] = i;
    }

    static inline void writeTermCrit(FileStorage& fs, const TermCriteria& termCrit)
    {
        if( (termCrit.type & TermCriteria::EPS) != 0 )
            fs << "epsilon" << termCrit.epsilon;
        if( (termCrit.type & TermCriteria::COUNT) != 0 )
            fs << "iterations" << termCrit.maxCount;
    }

    static inline TermCriteria readTermCrit(const FileNode& fn)
    {
        TermCriteria termCrit;
        double epsilon = (double)fn["epsilon"];
        if( epsilon > 0 )
        {
            termCrit.type |= TermCriteria::EPS;
            termCrit.epsilon = epsilon;
        }
        int iters = (int)fn["iterations"];
        if( iters > 0 )
        {
            termCrit.type |= TermCriteria::COUNT;
            termCrit.maxCount = iters;
        }
        return termCrit;
    }

    struct TreeParams
    {
        TreeParams();
        TreeParams( int maxDepth, int minSampleCount,
                    double regressionAccuracy, bool useSurrogates,
                    int maxCategories, int CVFolds,
                    bool use1SERule, bool truncatePrunedTree,
                    const Mat& priors );

        inline void setMaxCategories(int val)
        {
            if( val < 2 )
                CV_Error( CV_StsOutOfRange, "max_categories should be >= 2" );
            maxCategories = std::min(val, 15 );
        }
        inline void setMaxDepth(int val)
        {
            if( val < 0 )
                CV_Error( CV_StsOutOfRange, "max_depth should be >= 0" );
            maxDepth = std::min( val, 25 );
        }
        inline void setMinSampleCount(int val)
        {
            minSampleCount = std::max(val, 1);
        }
        inline void setCVFolds(int val)
        {
            if( val < 0 )
                CV_Error( CV_StsOutOfRange,
                          "params.CVFolds should be =0 (the tree is not pruned) "
                          "or n>0 (tree is pruned using n-fold cross-validation)" );
            if(val > 1)
                CV_Error( CV_StsNotImplemented,
                          "tree pruning using cross-validation is not implemented."
                          "Set CVFolds to 1");

            if( val == 1 )
                val = 0;
            CVFolds = val;
        }
        inline void setRegressionAccuracy(float val)
        {
            if( val < 0 )
                CV_Error( CV_StsOutOfRange, "params.regression_accuracy should be >= 0" );
            regressionAccuracy = val;
        }

        inline int getMaxCategories() const { return maxCategories; }
        inline int getMaxDepth() const { return maxDepth; }
        inline int getMinSampleCount() const { return minSampleCount; }
        inline int getCVFolds() const { return CVFolds; }
        inline float getRegressionAccuracy() const { return regressionAccuracy; }

        inline bool getUseSurrogates() const { return useSurrogates; }
        inline void setUseSurrogates(bool val) { useSurrogates = val; }
        inline bool getUse1SERule() const { return use1SERule; }
        inline void setUse1SERule(bool val) { use1SERule = val; }
        inline bool getTruncatePrunedTree() const { return truncatePrunedTree; }
        inline void setTruncatePrunedTree(bool val) { truncatePrunedTree = val; }
        inline cv::Mat getPriors() const { return priors; }
        inline void setPriors(const cv::Mat& val) { priors = val; }

        public:
        bool  useSurrogates;
        bool  use1SERule;
        bool  truncatePrunedTree;
        Mat priors;

    protected:
        int   maxCategories;
        int   maxDepth;
        int   minSampleCount;
        int   CVFolds;
        float regressionAccuracy;
    };

    struct RTreeParams
    {
        RTreeParams();
        RTreeParams(bool calcVarImportance, int nactiveVars, TermCriteria termCrit );
        bool calcVarImportance;
        int nactiveVars;
        TermCriteria termCrit;
    };

    struct BoostTreeParams
    {
        BoostTreeParams();
        BoostTreeParams(int boostType, int weakCount, double weightTrimRate);
        int boostType;
        int weakCount;
        double weightTrimRate;
    };

    class DTreesImpl : public DTrees
    {
    public:
        struct WNode
        {
            WNode()
            {
                class_idx = sample_count = depth = complexity = 0;
                parent = left = right = split = defaultDir = -1;
                Tn = INT_MAX;
                value = maxlr = alpha = node_risk = tree_risk = tree_error = 0.;
            }

            int class_idx;
            double Tn;
            double value;

            int parent;
            int left;
            int right;
            int defaultDir;

            int split;

            int sample_count;
            int depth;
            double maxlr;

            // global pruning data
            int complexity;
            double alpha;
            double node_risk, tree_risk, tree_error;
        };

        struct WSplit
        {
            WSplit()
            {
                varIdx = next = 0;
                inversed = false;
                quality = c = 0.f;
                subsetOfs = -1;
            }

            int varIdx;
            bool inversed;
            float quality;
            int next;
            float c;
            int subsetOfs;
        };

        struct WorkData
        {
            WorkData(const Ptr<TrainData>& _data);

            Ptr<TrainData> data;
            vector<WNode> wnodes;
            vector<WSplit> wsplits;
            vector<int> wsubsets;
            vector<double> cv_Tn;
            vector<double> cv_node_risk;
            vector<double> cv_node_error;
            vector<int> cv_labels;
            vector<double> sample_weights;
            vector<int> cat_responses;
            vector<double> ord_responses;
            vector<int> sidx;
            int maxSubsetSize;
        };

        inline int getMaxCategories() const CV_OVERRIDE { return params.getMaxCategories(); }
        inline void setMaxCategories(int val) CV_OVERRIDE { params.setMaxCategories(val); }
        inline int getMaxDepth() const CV_OVERRIDE { return params.getMaxDepth(); }
        inline void setMaxDepth(int val) CV_OVERRIDE { params.setMaxDepth(val); }
        inline int getMinSampleCount() const CV_OVERRIDE { return params.getMinSampleCount(); }
        inline void setMinSampleCount(int val) CV_OVERRIDE { params.setMinSampleCount(val); }
        inline int getCVFolds() const CV_OVERRIDE { return params.getCVFolds(); }
        inline void setCVFolds(int val) CV_OVERRIDE { params.setCVFolds(val); }
        inline bool getUseSurrogates() const CV_OVERRIDE { return params.getUseSurrogates(); }
        inline void setUseSurrogates(bool val) CV_OVERRIDE { params.setUseSurrogates(val); }
        inline bool getUse1SERule() const CV_OVERRIDE { return params.getUse1SERule(); }
        inline void setUse1SERule(bool val) CV_OVERRIDE { params.setUse1SERule(val); }
        inline bool getTruncatePrunedTree() const CV_OVERRIDE { return params.getTruncatePrunedTree(); }
        inline void setTruncatePrunedTree(bool val) CV_OVERRIDE { params.setTruncatePrunedTree(val); }
        inline float getRegressionAccuracy() const CV_OVERRIDE { return params.getRegressionAccuracy(); }
        inline void setRegressionAccuracy(float val) CV_OVERRIDE { params.setRegressionAccuracy(val); }
        inline cv::Mat getPriors() const CV_OVERRIDE { return params.getPriors(); }
        inline void setPriors(const cv::Mat& val) CV_OVERRIDE { params.setPriors(val); }

        DTreesImpl();
        virtual ~DTreesImpl() CV_OVERRIDE;
        virtual void clear() CV_OVERRIDE;

        String getDefaultName() const CV_OVERRIDE { return "opencv_ml_dtree"; }
        bool isTrained() const CV_OVERRIDE { return !roots.empty(); }
        bool isClassifier() const CV_OVERRIDE { return _isClassifier; }
        int getVarCount() const CV_OVERRIDE { return varType.empty() ? 0 : (int)(varType.size() - 1); }
        int getCatCount(int vi) const { return catOfs[vi][1] - catOfs[vi][0]; }
        int getSubsetSize(int vi) const { return (getCatCount(vi) + 31)/32; }

        virtual void setDParams(const TreeParams& _params);
        virtual void startTraining( const Ptr<TrainData>& trainData, int flags );
        virtual void endTraining();
        virtual void initCompVarIdx();
        virtual bool train( const Ptr<TrainData>& trainData, int flags ) CV_OVERRIDE;

        virtual int addTree( const vector<int>& sidx );
        virtual int addNodeAndTrySplit( int parent, const vector<int>& sidx );
        virtual const vector<int>& getActiveVars();
        virtual int findBestSplit( const vector<int>& _sidx );
        virtual void calcValue( int nidx, const vector<int>& _sidx );

        virtual WSplit findSplitOrdClass( int vi, const vector<int>& _sidx, double initQuality );

        // simple k-means, slightly modified to take into account the "weight" (L1-norm) of each vector.
        virtual void clusterCategories( const double* vectors, int n, int m, double* csums, int k, int* labels );
        virtual WSplit findSplitCatClass( int vi, const vector<int>& _sidx, double initQuality, int* subset );

        virtual WSplit findSplitOrdReg( int vi, const vector<int>& _sidx, double initQuality );
        virtual WSplit findSplitCatReg( int vi, const vector<int>& _sidx, double initQuality, int* subset );

        virtual int calcDir( int splitidx, const vector<int>& _sidx, vector<int>& _sleft, vector<int>& _sright );
        virtual int pruneCV( int root );

        virtual double updateTreeRNC( int root, double T, int fold );
        virtual bool cutTree( int root, double T, int fold, double min_alpha );
        virtual float predictTrees( const Range& range, const Mat& sample, int flags ) const;
        virtual float predict( InputArray inputs, OutputArray outputs, int flags ) const CV_OVERRIDE;

        virtual void writeTrainingParams( FileStorage& fs ) const;
        virtual void writeParams( FileStorage& fs ) const;
        virtual void writeSplit( FileStorage& fs, int splitidx ) const;
        virtual void writeNode( FileStorage& fs, int nidx, int depth ) const;
        virtual void writeTree( FileStorage& fs, int root ) const;
        virtual void write( FileStorage& fs ) const CV_OVERRIDE;

        virtual void readParams( const FileNode& fn );
        virtual int readSplit( const FileNode& fn );
        virtual int readNode( const FileNode& fn );
        virtual int readTree( const FileNode& fn );
        virtual void read( const FileNode& fn ) CV_OVERRIDE;

        virtual const std::vector<int>& getRoots() const CV_OVERRIDE { return roots; }
        virtual const std::vector<Node>& getNodes() const CV_OVERRIDE { return nodes; }
        virtual const std::vector<Split>& getSplits() const CV_OVERRIDE { return splits; }
        virtual const std::vector<int>& getSubsets() const CV_OVERRIDE { return subsets; }

        TreeParams params;

        vector<int> varIdx;
        vector<int> compVarIdx;
        vector<uchar> varType;
        vector<Vec2i> catOfs;
        vector<int> catMap;
        vector<int> roots;
        vector<Node> nodes;
        vector<Split> splits;
        vector<int> subsets;
        vector<int> classLabels;
        vector<float> missingSubst;
        vector<int> varMapping;
        bool _isClassifier;

        Ptr<WorkData> w;
    };

    template <typename T>
    static inline void readVectorOrMat(const FileNode & node, std::vector<T> & v)
    {
        if (node.type() == FileNode::MAP)
        {
            Mat m;
            node >> m;
            m.copyTo(v);
        }
        else if (node.type() == FileNode::SEQ)
        {
            node >> v;
        }
    }

}}

#endif /* __OPENCV_ML_PRECOMP_HPP__ */
