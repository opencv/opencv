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

        DTreesImpl();
        virtual ~DTreesImpl();
        virtual void clear();

        String getDefaultModelName() const { return "opencv_ml_dtree"; }
        bool isTrained() const { return !roots.empty(); }
        bool isClassifier() const { return _isClassifier; }
        int getVarCount() const { return varType.empty() ? 0 : (int)(varType.size() - 1); }
        int getCatCount(int vi) const { return catOfs[vi][1] - catOfs[vi][0]; }
        int getSubsetSize(int vi) const { return (getCatCount(vi) + 31)/32; }

        virtual void setDParams(const Params& _params);
        virtual Params getDParams() const;
        virtual void startTraining( const Ptr<TrainData>& trainData, int flags );
        virtual void endTraining();
        virtual void initCompVarIdx();
        virtual bool train( const Ptr<TrainData>& trainData, int flags );

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
        virtual float predict( InputArray inputs, OutputArray outputs, int flags ) const;

        virtual void writeTrainingParams( FileStorage& fs ) const;
        virtual void writeParams( FileStorage& fs ) const;
        virtual void writeSplit( FileStorage& fs, int splitidx ) const;
        virtual void writeNode( FileStorage& fs, int nidx, int depth ) const;
        virtual void writeTree( FileStorage& fs, int root ) const;
        virtual void write( FileStorage& fs ) const;

        virtual void readParams( const FileNode& fn );
        virtual int readSplit( const FileNode& fn );
        virtual int readNode( const FileNode& fn );
        virtual int readTree( const FileNode& fn );
        virtual void read( const FileNode& fn );

        virtual const std::vector<int>& getRoots() const { return roots; }
        virtual const std::vector<Node>& getNodes() const { return nodes; }
        virtual const std::vector<Split>& getSplits() const { return splits; }
        virtual const std::vector<int>& getSubsets() const { return subsets; }

        Params params0, params;

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
        bool _isClassifier;

        Ptr<WorkData> w;
    };

}}

#endif /* __OPENCV_ML_PRECOMP_HPP__ */
