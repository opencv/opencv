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
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2014, Itseez Inc, all rights reserved.
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
#include <ctype.h>

namespace cv {
namespace ml {

using std::vector;

TreeParams::TreeParams()
{
    maxDepth = INT_MAX;
    minSampleCount = 10;
    regressionAccuracy = 0.01f;
    useSurrogates = false;
    maxCategories = 10;
    CVFolds = 10;
    use1SERule = true;
    truncatePrunedTree = true;
    priors = Mat();
}

TreeParams::TreeParams(int _maxDepth, int _minSampleCount,
                       double _regressionAccuracy, bool _useSurrogates,
                       int _maxCategories, int _CVFolds,
                       bool _use1SERule, bool _truncatePrunedTree,
                       const Mat& _priors)
{
    maxDepth = _maxDepth;
    minSampleCount = _minSampleCount;
    regressionAccuracy = (float)_regressionAccuracy;
    useSurrogates = _useSurrogates;
    maxCategories = _maxCategories;
    CVFolds = _CVFolds;
    use1SERule = _use1SERule;
    truncatePrunedTree = _truncatePrunedTree;
    priors = _priors;
}

DTrees::Node::Node()
{
    classIdx = 0;
    value = 0;
    parent = left = right = split = defaultDir = -1;
}

DTrees::Split::Split()
{
    varIdx = 0;
    inversed = false;
    quality = 0.f;
    next = -1;
    c = 0.f;
    subsetOfs = 0;
}


DTreesImpl::WorkData::WorkData(const Ptr<TrainData>& _data)
{
    data = _data;
    vector<int> subsampleIdx;
    Mat sidx0 = _data->getTrainSampleIdx();
    if( !sidx0.empty() )
    {
        sidx0.copyTo(sidx);
        std::sort(sidx.begin(), sidx.end());
    }
    else
    {
        int n = _data->getNSamples();
        setRangeVector(sidx, n);
    }

    maxSubsetSize = 0;
}

DTreesImpl::DTreesImpl() {}
DTreesImpl::~DTreesImpl() {}
void DTreesImpl::clear()
{
    varIdx.clear();
    compVarIdx.clear();
    varType.clear();
    catOfs.clear();
    catMap.clear();
    roots.clear();
    nodes.clear();
    splits.clear();
    subsets.clear();
    classLabels.clear();

    w.release();
    _isClassifier = false;
}

void DTreesImpl::startTraining( const Ptr<TrainData>& data, int )
{
    clear();
    w = makePtr<WorkData>(data);

    Mat vtype = data->getVarType();
    vtype.copyTo(varType);

    data->getCatOfs().copyTo(catOfs);
    data->getCatMap().copyTo(catMap);
    data->getDefaultSubstValues().copyTo(missingSubst);

    int nallvars = data->getNAllVars();

    Mat vidx0 = data->getVarIdx();
    if( !vidx0.empty() )
        vidx0.copyTo(varIdx);
    else
        setRangeVector(varIdx, nallvars);

    initCompVarIdx();

    w->maxSubsetSize = 0;

    int i, nvars = (int)varIdx.size();
    for( i = 0; i < nvars; i++ )
        w->maxSubsetSize = std::max(w->maxSubsetSize, getCatCount(varIdx[i]));

    w->maxSubsetSize = std::max((w->maxSubsetSize + 31)/32, 1);

    data->getSampleWeights().copyTo(w->sample_weights);

    _isClassifier = data->getResponseType() == VAR_CATEGORICAL;

    if( _isClassifier )
    {
        data->getNormCatResponses().copyTo(w->cat_responses);
        data->getClassLabels().copyTo(classLabels);
        int nclasses = (int)classLabels.size();

        Mat class_weights = params.priors;
        if( !class_weights.empty() )
        {
            if( class_weights.type() != CV_64F || !class_weights.isContinuous() )
            {
                Mat temp;
                class_weights.convertTo(temp, CV_64F);
                class_weights = temp;
            }
            CV_Assert( class_weights.checkVector(1, CV_64F) == nclasses );

            int nsamples = (int)w->cat_responses.size();
            const double* cw = class_weights.ptr<double>();
            CV_Assert( (int)w->sample_weights.size() == nsamples );

            for( i = 0; i < nsamples; i++ )
            {
                int ci = w->cat_responses[i];
                CV_Assert( 0 <= ci && ci < nclasses );
                w->sample_weights[i] *= cw[ci];
            }
        }
    }
    else
        data->getResponses().copyTo(w->ord_responses);
}


void DTreesImpl::initCompVarIdx()
{
    int nallvars = (int)varType.size();
    compVarIdx.assign(nallvars, -1);
    int i, nvars = (int)varIdx.size(), prevIdx = -1;
    for( i = 0; i < nvars; i++ )
    {
        int vi = varIdx[i];
        CV_Assert( 0 <= vi && vi < nallvars && vi > prevIdx );
        prevIdx = vi;
        compVarIdx[vi] = i;
    }
}

void DTreesImpl::endTraining()
{
    w.release();
}

bool DTreesImpl::train( const Ptr<TrainData>& trainData, int flags )
{
    startTraining(trainData, flags);
    bool ok = addTree( w->sidx ) >= 0;
    w.release();
    endTraining();
    return ok;
}

const vector<int>& DTreesImpl::getActiveVars()
{
    return varIdx;
}

int DTreesImpl::addTree(const vector<int>& sidx )
{
    size_t n = (params.getMaxDepth() > 0 ? (1 << params.getMaxDepth()) : 1024) + w->wnodes.size();

    w->wnodes.reserve(n);
    w->wsplits.reserve(n);
    w->wsubsets.reserve(n*w->maxSubsetSize);
    w->wnodes.clear();
    w->wsplits.clear();
    w->wsubsets.clear();

    int cv_n = params.getCVFolds();

    if( cv_n > 0 )
    {
        w->cv_Tn.resize(n*cv_n);
        w->cv_node_error.resize(n*cv_n);
        w->cv_node_risk.resize(n*cv_n);
    }

    // build the tree recursively
    int w_root = addNodeAndTrySplit(-1, sidx);
    int maxdepth = INT_MAX;//pruneCV(root);

    int w_nidx = w_root, pidx = -1, depth = 0;
    int root = (int)nodes.size();

    for(;;)
    {
        const WNode& wnode = w->wnodes[w_nidx];
        Node node;
        node.parent = pidx;
        node.classIdx = wnode.class_idx;
        node.value = wnode.value;
        node.defaultDir = wnode.defaultDir;

        int wsplit_idx = wnode.split;
        if( wsplit_idx >= 0 )
        {
            const WSplit& wsplit = w->wsplits[wsplit_idx];
            Split split;
            split.c = wsplit.c;
            split.quality = wsplit.quality;
            split.inversed = wsplit.inversed;
            split.varIdx = wsplit.varIdx;
            split.subsetOfs = -1;
            if( wsplit.subsetOfs >= 0 )
            {
                int ssize = getSubsetSize(split.varIdx);
                split.subsetOfs = (int)subsets.size();
                subsets.resize(split.subsetOfs + ssize);
                // This check verifies that subsets index is in the correct range
                // as in case ssize == 0 no real resize performed.
                // Thus memory kept safe.
                // Also this skips useless memcpy call when size parameter is zero
                if(ssize > 0)
                {
                    memcpy(&subsets[split.subsetOfs], &w->wsubsets[wsplit.subsetOfs], ssize*sizeof(int));
                }
            }
            node.split = (int)splits.size();
            splits.push_back(split);
        }
        int nidx = (int)nodes.size();
        nodes.push_back(node);
        if( pidx >= 0 )
        {
            int w_pidx = w->wnodes[w_nidx].parent;
            if( w->wnodes[w_pidx].left == w_nidx )
            {
                nodes[pidx].left = nidx;
            }
            else
            {
                CV_Assert(w->wnodes[w_pidx].right == w_nidx);
                nodes[pidx].right = nidx;
            }
        }

        if( wnode.left >= 0 && depth+1 < maxdepth )
        {
            w_nidx = wnode.left;
            pidx = nidx;
            depth++;
        }
        else
        {
            int w_pidx = wnode.parent;
            while( w_pidx >= 0 && w->wnodes[w_pidx].right == w_nidx )
            {
                w_nidx = w_pidx;
                w_pidx = w->wnodes[w_pidx].parent;
                nidx = pidx;
                pidx = nodes[pidx].parent;
                depth--;
            }

            if( w_pidx < 0 )
                break;

            w_nidx = w->wnodes[w_pidx].right;
            CV_Assert( w_nidx >= 0 );
        }
    }
    roots.push_back(root);
    return root;
}

void DTreesImpl::setDParams(const TreeParams& _params)
{
    params = _params;
}

int DTreesImpl::addNodeAndTrySplit( int parent, const vector<int>& sidx )
{
    w->wnodes.push_back(WNode());
    int nidx = (int)(w->wnodes.size() - 1);
    WNode& node = w->wnodes.back();

    node.parent = parent;
    node.depth = parent >= 0 ? w->wnodes[parent].depth + 1 : 0;
    int nfolds = params.getCVFolds();

    if( nfolds > 0 )
    {
        w->cv_Tn.resize((nidx+1)*nfolds);
        w->cv_node_error.resize((nidx+1)*nfolds);
        w->cv_node_risk.resize((nidx+1)*nfolds);
    }

    int i, n = node.sample_count = (int)sidx.size();
    bool can_split = true;
    vector<int> sleft, sright;

    calcValue( nidx, sidx );

    if( n <= params.getMinSampleCount() || node.depth >= params.getMaxDepth() )
        can_split = false;
    else if( _isClassifier )
    {
        const int* responses = &w->cat_responses[0];
        const int* s = &sidx[0];
        int first = responses[s[0]];
        for( i = 1; i < n; i++ )
            if( responses[s[i]] != first )
                break;
        if( i == n )
            can_split = false;
    }
    else
    {
        if( sqrt(node.node_risk) < params.getRegressionAccuracy() )
            can_split = false;
    }

    if( can_split )
        node.split = findBestSplit( sidx );

    //printf("depth=%d, nidx=%d, parent=%d, n=%d, %s, value=%.1f, risk=%.1f\n", node.depth, nidx, node.parent, n, (node.split < 0 ? "leaf" : varType[w->wsplits[node.split].varIdx] == VAR_CATEGORICAL ? "cat" : "ord"), node.value, node.node_risk);

    if( node.split >= 0 )
    {
        node.defaultDir = calcDir( node.split, sidx, sleft, sright );
        if( params.useSurrogates )
            CV_Error( CV_StsNotImplemented, "surrogate splits are not implemented yet");

        int left = addNodeAndTrySplit( nidx, sleft );
        int right = addNodeAndTrySplit( nidx, sright );
        w->wnodes[nidx].left = left;
        w->wnodes[nidx].right = right;
        CV_Assert( w->wnodes[nidx].left > 0 && w->wnodes[nidx].right > 0 );
    }

    return nidx;
}

int DTreesImpl::findBestSplit( const vector<int>& _sidx )
{
    const vector<int>& activeVars = getActiveVars();
    int splitidx = -1;
    int vi_, nv = (int)activeVars.size();
    AutoBuffer<int> buf(w->maxSubsetSize*2);
    int *subset = buf, *best_subset = subset + w->maxSubsetSize;
    WSplit split, best_split;
    best_split.quality = 0.;

    for( vi_ = 0; vi_ < nv; vi_++ )
    {
        int vi = activeVars[vi_];
        if( varType[vi] == VAR_CATEGORICAL )
        {
            if( _isClassifier )
                split = findSplitCatClass(vi, _sidx, 0, subset);
            else
                split = findSplitCatReg(vi, _sidx, 0, subset);
        }
        else
        {
            if( _isClassifier )
                split = findSplitOrdClass(vi, _sidx, 0);
            else
                split = findSplitOrdReg(vi, _sidx, 0);
        }
        if( split.quality > best_split.quality )
        {
            best_split = split;
            std::swap(subset, best_subset);
        }
    }

    if( best_split.quality > 0 )
    {
        int best_vi = best_split.varIdx;
        CV_Assert( compVarIdx[best_split.varIdx] >= 0 && best_vi >= 0 );
        int i, prevsz = (int)w->wsubsets.size(), ssize = getSubsetSize(best_vi);
        w->wsubsets.resize(prevsz + ssize);
        for( i = 0; i < ssize; i++ )
            w->wsubsets[prevsz + i] = best_subset[i];
        best_split.subsetOfs = prevsz;
        w->wsplits.push_back(best_split);
        splitidx = (int)(w->wsplits.size()-1);
    }

    return splitidx;
}

void DTreesImpl::calcValue( int nidx, const vector<int>& _sidx )
{
    WNode* node = &w->wnodes[nidx];
    int i, j, k, n = (int)_sidx.size(), cv_n = params.getCVFolds();
    int m = (int)classLabels.size();

    cv::AutoBuffer<double> buf(std::max(m, 3)*(cv_n+1));

    if( cv_n > 0 )
    {
        size_t sz = w->cv_Tn.size();
        w->cv_Tn.resize(sz + cv_n);
        w->cv_node_risk.resize(sz + cv_n);
        w->cv_node_error.resize(sz + cv_n);
    }

    if( _isClassifier )
    {
        // in case of classification tree:
        //  * node value is the label of the class that has the largest weight in the node.
        //  * node risk is the weighted number of misclassified samples,
        //  * j-th cross-validation fold value and risk are calculated as above,
        //    but using the samples with cv_labels(*)!=j.
        //  * j-th cross-validation fold error is calculated as the weighted number of
        //    misclassified samples with cv_labels(*)==j.

        // compute the number of instances of each class
        double* cls_count = buf;
        double* cv_cls_count = cls_count + m;

        double max_val = -1, total_weight = 0;
        int max_k = -1;

        for( k = 0; k < m; k++ )
            cls_count[k] = 0;

        if( cv_n == 0 )
        {
            for( i = 0; i < n; i++ )
            {
                int si = _sidx[i];
                cls_count[w->cat_responses[si]] += w->sample_weights[si];
            }
        }
        else
        {
            for( j = 0; j < cv_n; j++ )
                for( k = 0; k < m; k++ )
                    cv_cls_count[j*m + k] = 0;

            for( i = 0; i < n; i++ )
            {
                int si = _sidx[i];
                j = w->cv_labels[si]; k = w->cat_responses[si];
                cv_cls_count[j*m + k] += w->sample_weights[si];
            }

            for( j = 0; j < cv_n; j++ )
                for( k = 0; k < m; k++ )
                    cls_count[k] += cv_cls_count[j*m + k];
        }

        for( k = 0; k < m; k++ )
        {
            double val = cls_count[k];
            total_weight += val;
            if( max_val < val )
            {
                max_val = val;
                max_k = k;
            }
        }

        node->class_idx = max_k;
        node->value = classLabels[max_k];
        node->node_risk = total_weight - max_val;

        for( j = 0; j < cv_n; j++ )
        {
            double sum_k = 0, sum = 0, max_val_k = 0;
            max_val = -1; max_k = -1;

            for( k = 0; k < m; k++ )
            {
                double val_k = cv_cls_count[j*m + k];
                double val = cls_count[k] - val_k;
                sum_k += val_k;
                sum += val;
                if( max_val < val )
                {
                    max_val = val;
                    max_val_k = val_k;
                    max_k = k;
                }
            }

            w->cv_Tn[nidx*cv_n + j] = INT_MAX;
            w->cv_node_risk[nidx*cv_n + j] = sum - max_val;
            w->cv_node_error[nidx*cv_n + j] = sum_k - max_val_k;
        }
    }
    else
    {
        // in case of regression tree:
        //  * node value is 1/n*sum_i(Y_i), where Y_i is i-th response,
        //    n is the number of samples in the node.
        //  * node risk is the sum of squared errors: sum_i((Y_i - <node_value>)^2)
        //  * j-th cross-validation fold value and risk are calculated as above,
        //    but using the samples with cv_labels(*)!=j.
        //  * j-th cross-validation fold error is calculated
        //    using samples with cv_labels(*)==j as the test subset:
        //    error_j = sum_(i,cv_labels(i)==j)((Y_i - <node_value_j>)^2),
        //    where node_value_j is the node value calculated
        //    as described in the previous bullet, and summation is done
        //    over the samples with cv_labels(*)==j.
        double sum = 0, sum2 = 0, sumw = 0;

        if( cv_n == 0 )
        {
            for( i = 0; i < n; i++ )
            {
                int si = _sidx[i];
                double wval = w->sample_weights[si];
                double t = w->ord_responses[si];
                sum += t*wval;
                sum2 += t*t*wval;
                sumw += wval;
            }
        }
        else
        {
            double *cv_sum = buf, *cv_sum2 = cv_sum + cv_n;
            double* cv_count = (double*)(cv_sum2 + cv_n);

            for( j = 0; j < cv_n; j++ )
            {
                cv_sum[j] = cv_sum2[j] = 0.;
                cv_count[j] = 0;
            }

            for( i = 0; i < n; i++ )
            {
                int si = _sidx[i];
                j = w->cv_labels[si];
                double wval = w->sample_weights[si];
                double t = w->ord_responses[si];
                cv_sum[j] += t*wval;
                cv_sum2[j] += t*t*wval;
                cv_count[j] += wval;
            }

            for( j = 0; j < cv_n; j++ )
            {
                sum += cv_sum[j];
                sum2 += cv_sum2[j];
                sumw += cv_count[j];
            }

            for( j = 0; j < cv_n; j++ )
            {
                double s = sum - cv_sum[j], si = sum - s;
                double s2 = sum2 - cv_sum2[j], s2i = sum2 - s2;
                double c = cv_count[j], ci = sumw - c;
                double r = si/std::max(ci, DBL_EPSILON);
                w->cv_node_risk[nidx*cv_n + j] = s2i - r*r*ci;
                w->cv_node_error[nidx*cv_n + j] = s2 - 2*r*s + c*r*r;
                w->cv_Tn[nidx*cv_n + j] = INT_MAX;
            }
        }

        node->node_risk = sum2 - (sum/sumw)*sum;
        node->value = sum/sumw;
    }
}

DTreesImpl::WSplit DTreesImpl::findSplitOrdClass( int vi, const vector<int>& _sidx, double initQuality )
{
    int n = (int)_sidx.size();
    int m = (int)classLabels.size();

    cv::AutoBuffer<uchar> buf(n*(sizeof(float) + sizeof(int)) + m*2*sizeof(double));
    const int* sidx = &_sidx[0];
    const int* responses = &w->cat_responses[0];
    const double* weights = &w->sample_weights[0];
    double* lcw = (double*)(uchar*)buf;
    double* rcw = lcw + m;
    float* values = (float*)(rcw + m);
    int* sorted_idx = (int*)(values + n);
    int i, best_i = -1;
    double best_val = initQuality;

    for( i = 0; i < m; i++ )
        lcw[i] = rcw[i] = 0.;

    w->data->getValues( vi, _sidx, values );

    for( i = 0; i < n; i++ )
    {
        sorted_idx[i] = i;
        int si = sidx[i];
        rcw[responses[si]] += weights[si];
    }

    std::sort(sorted_idx, sorted_idx + n, cmp_lt_idx<float>(values));

    double L = 0, R = 0, lsum2 = 0, rsum2 = 0;
    for( i = 0; i < m; i++ )
    {
        double wval = rcw[i];
        R += wval;
        rsum2 += wval*wval;
    }

    for( i = 0; i < n - 1; i++ )
    {
        int curr = sorted_idx[i];
        int next = sorted_idx[i+1];
        int si = sidx[curr];
        double wval = weights[si], w2 = wval*wval;
        L += wval; R -= wval;
        int idx = responses[si];
        double lv = lcw[idx], rv = rcw[idx];
        lsum2 += 2*lv*wval + w2;
        rsum2 -= 2*rv*wval - w2;
        lcw[idx] = lv + wval; rcw[idx] = rv - wval;

        float value_between = (values[next] + values[curr]) * 0.5f;
        if( value_between > values[curr] && value_between < values[next] )
        {
            double val = (lsum2*R + rsum2*L)/(L*R);
            if( best_val < val )
            {
                best_val = val;
                best_i = i;
            }
        }
    }

    WSplit split;
    if( best_i >= 0 )
    {
        split.varIdx = vi;
        split.c = (values[sorted_idx[best_i]] + values[sorted_idx[best_i+1]])*0.5f;
        split.inversed = false;
        split.quality = (float)best_val;
    }
    return split;
}

// simple k-means, slightly modified to take into account the "weight" (L1-norm) of each vector.
void DTreesImpl::clusterCategories( const double* vectors, int n, int m, double* csums, int k, int* labels )
{
    int iters = 0, max_iters = 100;
    int i, j, idx;
    cv::AutoBuffer<double> buf(n + k);
    double *v_weights = buf, *c_weights = buf + n;
    bool modified = true;
    RNG r((uint64)-1);

    // assign labels randomly
    for( i = 0; i < n; i++ )
    {
        double sum = 0;
        const double* v = vectors + i*m;
        labels[i] = i < k ? i : r.uniform(0, k);

        // compute weight of each vector
        for( j = 0; j < m; j++ )
            sum += v[j];
        v_weights[i] = sum ? 1./sum : 0.;
    }

    for( i = 0; i < n; i++ )
    {
        int i1 = r.uniform(0, n);
        int i2 = r.uniform(0, n);
        std::swap( labels[i1], labels[i2] );
    }

    for( iters = 0; iters <= max_iters; iters++ )
    {
        // calculate csums
        for( i = 0; i < k; i++ )
        {
            for( j = 0; j < m; j++ )
                csums[i*m + j] = 0;
        }

        for( i = 0; i < n; i++ )
        {
            const double* v = vectors + i*m;
            double* s = csums + labels[i]*m;
            for( j = 0; j < m; j++ )
                s[j] += v[j];
        }

        // exit the loop here, when we have up-to-date csums
        if( iters == max_iters || !modified )
            break;

        modified = false;

        // calculate weight of each cluster
        for( i = 0; i < k; i++ )
        {
            const double* s = csums + i*m;
            double sum = 0;
            for( j = 0; j < m; j++ )
                sum += s[j];
            c_weights[i] = sum ? 1./sum : 0;
        }

        // now for each vector determine the closest cluster
        for( i = 0; i < n; i++ )
        {
            const double* v = vectors + i*m;
            double alpha = v_weights[i];
            double min_dist2 = DBL_MAX;
            int min_idx = -1;

            for( idx = 0; idx < k; idx++ )
            {
                const double* s = csums + idx*m;
                double dist2 = 0., beta = c_weights[idx];
                for( j = 0; j < m; j++ )
                {
                    double t = v[j]*alpha - s[j]*beta;
                    dist2 += t*t;
                }
                if( min_dist2 > dist2 )
                {
                    min_dist2 = dist2;
                    min_idx = idx;
                }
            }

            if( min_idx != labels[i] )
                modified = true;
            labels[i] = min_idx;
        }
    }
}

DTreesImpl::WSplit DTreesImpl::findSplitCatClass( int vi, const vector<int>& _sidx,
                                                  double initQuality, int* subset )
{
    int _mi = getCatCount(vi), mi = _mi;
    int n = (int)_sidx.size();
    int m = (int)classLabels.size();

    int base_size = m*(3 + mi) + mi + 1;
    if( m > 2 && mi > params.getMaxCategories() )
        base_size += m*std::min(params.getMaxCategories(), n) + mi;
    else
        base_size += mi;
    AutoBuffer<double> buf(base_size + n);

    double* lc = (double*)buf;
    double* rc = lc + m;
    double* _cjk = rc + m*2, *cjk = _cjk;
    double* c_weights = cjk + m*mi;

    int* labels = (int*)(buf + base_size);
    w->data->getNormCatValues(vi, _sidx, labels);
    const int* responses = &w->cat_responses[0];
    const double* weights = &w->sample_weights[0];

    int* cluster_labels = 0;
    double** dbl_ptr = 0;
    int i, j, k, si, idx;
    double L = 0, R = 0;
    double best_val = initQuality;
    int prevcode = 0, best_subset = -1, subset_i, subset_n, subtract = 0;

    // init array of counters:
    // c_{jk} - number of samples that have vi-th input variable = j and response = k.
    for( j = -1; j < mi; j++ )
        for( k = 0; k < m; k++ )
            cjk[j*m + k] = 0;

    for( i = 0; i < n; i++ )
    {
        si = _sidx[i];
        j = labels[i];
        k = responses[si];
        cjk[j*m + k] += weights[si];
    }

    if( m > 2 )
    {
        if( mi > params.getMaxCategories() )
        {
            mi = std::min(params.getMaxCategories(), n);
            cjk = c_weights + _mi;
            cluster_labels = (int*)(cjk + m*mi);
            clusterCategories( _cjk, _mi, m, cjk, mi, cluster_labels );
        }
        subset_i = 1;
        subset_n = 1 << mi;
    }
    else
    {
        assert( m == 2 );
        dbl_ptr = (double**)(c_weights + _mi);
        for( j = 0; j < mi; j++ )
            dbl_ptr[j] = cjk + j*2 + 1;
        std::sort(dbl_ptr, dbl_ptr + mi, cmp_lt_ptr<double>());
        subset_i = 0;
        subset_n = mi;
    }

    for( k = 0; k < m; k++ )
    {
        double sum = 0;
        for( j = 0; j < mi; j++ )
            sum += cjk[j*m + k];
        CV_Assert(sum > 0);
        rc[k] = sum;
        lc[k] = 0;
    }

    for( j = 0; j < mi; j++ )
    {
        double sum = 0;
        for( k = 0; k < m; k++ )
            sum += cjk[j*m + k];
        c_weights[j] = sum;
        R += c_weights[j];
    }

    for( ; subset_i < subset_n; subset_i++ )
    {
        double lsum2 = 0, rsum2 = 0;

        if( m == 2 )
            idx = (int)(dbl_ptr[subset_i] - cjk)/2;
        else
        {
            int graycode = (subset_i>>1)^subset_i;
            int diff = graycode ^ prevcode;

            // determine index of the changed bit.
            Cv32suf u;
            idx = diff >= (1 << 16) ? 16 : 0;
            u.f = (float)(((diff >> 16) | diff) & 65535);
            idx += (u.i >> 23) - 127;
            subtract = graycode < prevcode;
            prevcode = graycode;
        }

        double* crow = cjk + idx*m;
        double weight = c_weights[idx];
        if( weight < FLT_EPSILON )
            continue;

        if( !subtract )
        {
            for( k = 0; k < m; k++ )
            {
                double t = crow[k];
                double lval = lc[k] + t;
                double rval = rc[k] - t;
                lsum2 += lval*lval;
                rsum2 += rval*rval;
                lc[k] = lval; rc[k] = rval;
            }
            L += weight;
            R -= weight;
        }
        else
        {
            for( k = 0; k < m; k++ )
            {
                double t = crow[k];
                double lval = lc[k] - t;
                double rval = rc[k] + t;
                lsum2 += lval*lval;
                rsum2 += rval*rval;
                lc[k] = lval; rc[k] = rval;
            }
            L -= weight;
            R += weight;
        }

        if( L > FLT_EPSILON && R > FLT_EPSILON )
        {
            double val = (lsum2*R + rsum2*L)/(L*R);
            if( best_val < val )
            {
                best_val = val;
                best_subset = subset_i;
            }
        }
    }

    WSplit split;
    if( best_subset >= 0 )
    {
        split.varIdx = vi;
        split.quality = (float)best_val;
        memset( subset, 0, getSubsetSize(vi) * sizeof(int) );
        if( m == 2 )
        {
            for( i = 0; i <= best_subset; i++ )
            {
                idx = (int)(dbl_ptr[i] - cjk) >> 1;
                subset[idx >> 5] |= 1 << (idx & 31);
            }
        }
        else
        {
            for( i = 0; i < _mi; i++ )
            {
                idx = cluster_labels ? cluster_labels[i] : i;
                if( best_subset & (1 << idx) )
                    subset[i >> 5] |= 1 << (i & 31);
            }
        }
    }
    return split;
}

DTreesImpl::WSplit DTreesImpl::findSplitOrdReg( int vi, const vector<int>& _sidx, double initQuality )
{
    const double* weights = &w->sample_weights[0];
    int n = (int)_sidx.size();

    AutoBuffer<uchar> buf(n*(sizeof(int) + sizeof(float)));

    float* values = (float*)(uchar*)buf;
    int* sorted_idx = (int*)(values + n);
    w->data->getValues(vi, _sidx, values);
    const double* responses = &w->ord_responses[0];

    int i, si, best_i = -1;
    double L = 0, R = 0;
    double best_val = initQuality, lsum = 0, rsum = 0;

    for( i = 0; i < n; i++ )
    {
        sorted_idx[i] = i;
        si = _sidx[i];
        R += weights[si];
        rsum += weights[si]*responses[si];
    }

    std::sort(sorted_idx, sorted_idx + n, cmp_lt_idx<float>(values));

    // find the optimal split
    for( i = 0; i < n - 1; i++ )
    {
        int curr = sorted_idx[i];
        int next = sorted_idx[i+1];
        si = _sidx[curr];
        double wval = weights[si];
        double t = responses[si]*wval;
        L += wval; R -= wval;
        lsum += t; rsum -= t;

        float value_between = (values[next] + values[curr]) * 0.5f;
        if( value_between > values[curr] && value_between < values[next] )
        {
            double val = (lsum*lsum*R + rsum*rsum*L)/(L*R);
            if( best_val < val )
            {
                best_val = val;
                best_i = i;
            }
        }
    }

    WSplit split;
    if( best_i >= 0 )
    {
        split.varIdx = vi;
        split.c = (values[sorted_idx[best_i]] + values[sorted_idx[best_i+1]])*0.5f;
        split.inversed = false;
        split.quality = (float)best_val;
    }
    return split;
}

DTreesImpl::WSplit DTreesImpl::findSplitCatReg( int vi, const vector<int>& _sidx,
                                                double initQuality, int* subset )
{
    const double* weights = &w->sample_weights[0];
    const double* responses = &w->ord_responses[0];
    int n = (int)_sidx.size();
    int mi = getCatCount(vi);

    AutoBuffer<double> buf(3*mi + 3 + n);
    double* sum = (double*)buf + 1;
    double* counts = sum + mi + 1;
    double** sum_ptr = (double**)(counts + mi);
    int* cat_labels = (int*)(sum_ptr + mi);

    w->data->getNormCatValues(vi, _sidx, cat_labels);

    double L = 0, R = 0, best_val = initQuality, lsum = 0, rsum = 0;
    int i, si, best_subset = -1, subset_i;

    for( i = -1; i < mi; i++ )
        sum[i] = counts[i] = 0;

    // calculate sum response and weight of each category of the input var
    for( i = 0; i < n; i++ )
    {
        int idx = cat_labels[i];
        si = _sidx[i];
        double wval = weights[si];
        sum[idx] += responses[si]*wval;
        counts[idx] += wval;
    }

    // calculate average response in each category
    for( i = 0; i < mi; i++ )
    {
        R += counts[i];
        rsum += sum[i];
        sum[i] = fabs(counts[i]) > DBL_EPSILON ? sum[i]/counts[i] : 0;
        sum_ptr[i] = sum + i;
    }

    std::sort(sum_ptr, sum_ptr + mi, cmp_lt_ptr<double>());

    // revert back to unnormalized sums
    // (there should be a very little loss in accuracy)
    for( i = 0; i < mi; i++ )
        sum[i] *= counts[i];

    for( subset_i = 0; subset_i < mi-1; subset_i++ )
    {
        int idx = (int)(sum_ptr[subset_i] - sum);
        double ni = counts[idx];

        if( ni > FLT_EPSILON )
        {
            double s = sum[idx];
            lsum += s; L += ni;
            rsum -= s; R -= ni;

            if( L > FLT_EPSILON && R > FLT_EPSILON )
            {
                double val = (lsum*lsum*R + rsum*rsum*L)/(L*R);
                if( best_val < val )
                {
                    best_val = val;
                    best_subset = subset_i;
                }
            }
        }
    }

    WSplit split;
    if( best_subset >= 0 )
    {
        split.varIdx = vi;
        split.quality = (float)best_val;
        memset( subset, 0, getSubsetSize(vi) * sizeof(int));
        for( i = 0; i <= best_subset; i++ )
        {
            int idx = (int)(sum_ptr[i] - sum);
            subset[idx >> 5] |= 1 << (idx & 31);
        }
    }
    return split;
}

int DTreesImpl::calcDir( int splitidx, const vector<int>& _sidx,
                         vector<int>& _sleft, vector<int>& _sright )
{
    WSplit split = w->wsplits[splitidx];
    int i, si, n = (int)_sidx.size(), vi = split.varIdx;
    _sleft.reserve(n);
    _sright.reserve(n);
    _sleft.clear();
    _sright.clear();

    AutoBuffer<float> buf(n);
    int mi = getCatCount(vi);
    double wleft = 0, wright = 0;
    const double* weights = &w->sample_weights[0];

    if( mi <= 0 ) // split on an ordered variable
    {
        float c = split.c;
        float* values = buf;
        w->data->getValues(vi, _sidx, values);

        for( i = 0; i < n; i++ )
        {
            si = _sidx[i];
            if( values[i] <= c )
            {
                _sleft.push_back(si);
                wleft += weights[si];
            }
            else
            {
                _sright.push_back(si);
                wright += weights[si];
            }
        }
    }
    else
    {
        const int* subset = &w->wsubsets[split.subsetOfs];
        int* cat_labels = (int*)(float*)buf;
        w->data->getNormCatValues(vi, _sidx, cat_labels);

        for( i = 0; i < n; i++ )
        {
            si = _sidx[i];
            unsigned u = cat_labels[i];
            if( CV_DTREE_CAT_DIR(u, subset) < 0 )
            {
                _sleft.push_back(si);
                wleft += weights[si];
            }
            else
            {
                _sright.push_back(si);
                wright += weights[si];
            }
        }
    }
    CV_Assert( (int)_sleft.size() < n && (int)_sright.size() < n );
    return wleft > wright ? -1 : 1;
}

int DTreesImpl::pruneCV( int root )
{
    vector<double> ab;

    // 1. build tree sequence for each cv fold, calculate error_{Tj,beta_k}.
    // 2. choose the best tree index (if need, apply 1SE rule).
    // 3. store the best index and cut the branches.

    int ti, tree_count = 0, j, cv_n = params.getCVFolds(), n = w->wnodes[root].sample_count;
    // currently, 1SE for regression is not implemented
    bool use_1se = params.use1SERule != 0 && _isClassifier;
    double min_err = 0, min_err_se = 0;
    int min_idx = -1;

    // build the main tree sequence, calculate alpha's
    for(;;tree_count++)
    {
        double min_alpha = updateTreeRNC(root, tree_count, -1);
        if( cutTree(root, tree_count, -1, min_alpha) )
            break;

        ab.push_back(min_alpha);
    }

    if( tree_count > 0 )
    {
        ab[0] = 0.;

        for( ti = 1; ti < tree_count-1; ti++ )
            ab[ti] = std::sqrt(ab[ti]*ab[ti+1]);
        ab[tree_count-1] = DBL_MAX*0.5;

        Mat err_jk(cv_n, tree_count, CV_64F);

        for( j = 0; j < cv_n; j++ )
        {
            int tj = 0, tk = 0;
            for( ; tj < tree_count; tj++ )
            {
                double min_alpha = updateTreeRNC(root, tj, j);
                if( cutTree(root, tj, j, min_alpha) )
                    min_alpha = DBL_MAX;

                for( ; tk < tree_count; tk++ )
                {
                    if( ab[tk] > min_alpha )
                        break;
                    err_jk.at<double>(j, tk) = w->wnodes[root].tree_error;
                }
            }
        }

        for( ti = 0; ti < tree_count; ti++ )
        {
            double sum_err = 0;
            for( j = 0; j < cv_n; j++ )
                sum_err += err_jk.at<double>(j, ti);
            if( ti == 0 || sum_err < min_err )
            {
                min_err = sum_err;
                min_idx = ti;
                if( use_1se )
                    min_err_se = sqrt( sum_err*(n - sum_err) );
            }
            else if( sum_err < min_err + min_err_se )
                min_idx = ti;
        }
    }

    return min_idx;
}

double DTreesImpl::updateTreeRNC( int root, double T, int fold )
{
    int nidx = root, pidx = -1, cv_n = params.getCVFolds();
    double min_alpha = DBL_MAX;

    for(;;)
    {
        WNode *node = 0, *parent = 0;

        for(;;)
        {
            node = &w->wnodes[nidx];
            double t = fold >= 0 ? w->cv_Tn[nidx*cv_n + fold] : node->Tn;
            if( t <= T || node->left < 0 )
            {
                node->complexity = 1;
                node->tree_risk = node->node_risk;
                node->tree_error = 0.;
                if( fold >= 0 )
                {
                    node->tree_risk = w->cv_node_risk[nidx*cv_n + fold];
                    node->tree_error = w->cv_node_error[nidx*cv_n + fold];
                }
                break;
            }
            nidx = node->left;
        }

        for( pidx = node->parent; pidx >= 0 && w->wnodes[pidx].right == nidx;
             nidx = pidx, pidx = w->wnodes[pidx].parent )
        {
            node = &w->wnodes[nidx];
            parent = &w->wnodes[pidx];
            parent->complexity += node->complexity;
            parent->tree_risk += node->tree_risk;
            parent->tree_error += node->tree_error;

            parent->alpha = ((fold >= 0 ? w->cv_node_risk[pidx*cv_n + fold] : parent->node_risk)
                             - parent->tree_risk)/(parent->complexity - 1);
            min_alpha = std::min( min_alpha, parent->alpha );
        }

        if( pidx < 0 )
            break;

        node = &w->wnodes[nidx];
        parent = &w->wnodes[pidx];
        parent->complexity = node->complexity;
        parent->tree_risk = node->tree_risk;
        parent->tree_error = node->tree_error;
        nidx = parent->right;
    }

    return min_alpha;
}

bool DTreesImpl::cutTree( int root, double T, int fold, double min_alpha )
{
    int cv_n = params.getCVFolds(), nidx = root, pidx = -1;
    WNode* node = &w->wnodes[root];
    if( node->left < 0 )
        return true;

    for(;;)
    {
        for(;;)
        {
            node = &w->wnodes[nidx];
            double t = fold >= 0 ? w->cv_Tn[nidx*cv_n + fold] : node->Tn;
            if( t <= T || node->left < 0 )
                break;
            if( node->alpha <= min_alpha + FLT_EPSILON )
            {
                if( fold >= 0 )
                    w->cv_Tn[nidx*cv_n + fold] = T;
                else
                    node->Tn = T;
                if( nidx == root )
                    return true;
                break;
            }
            nidx = node->left;
        }

        for( pidx = node->parent; pidx >= 0 && w->wnodes[pidx].right == nidx;
             nidx = pidx, pidx = w->wnodes[pidx].parent )
            ;

        if( pidx < 0 )
            break;

        nidx = w->wnodes[pidx].right;
    }

    return false;
}

float DTreesImpl::predictTrees( const Range& range, const Mat& sample, int flags ) const
{
    CV_Assert( sample.type() == CV_32F );

    int predictType = flags & PREDICT_MASK;
    int nvars = (int)varIdx.size();
    if( nvars == 0 )
        nvars = (int)varType.size();
    int i, ncats = (int)catOfs.size(), nclasses = (int)classLabels.size();
    int catbufsize = ncats > 0 ? nvars : 0;
    AutoBuffer<int> buf(nclasses + catbufsize + 1);
    int* votes = buf;
    int* catbuf = votes + nclasses;
    const int* cvidx = (flags & (COMPRESSED_INPUT|PREPROCESSED_INPUT)) == 0 && !varIdx.empty() ? &compVarIdx[0] : 0;
    const uchar* vtype = &varType[0];
    const Vec2i* cofs = !catOfs.empty() ? &catOfs[0] : 0;
    const int* cmap = !catMap.empty() ? &catMap[0] : 0;
    const float* psample = sample.ptr<float>();
    const float* missingSubstPtr = !missingSubst.empty() ? &missingSubst[0] : 0;
    size_t sstep = sample.isContinuous() ? 1 : sample.step/sizeof(float);
    double sum = 0.;
    int lastClassIdx = -1;
    const float MISSED_VAL = TrainData::missingValue();

    for( i = 0; i < catbufsize; i++ )
        catbuf[i] = -1;

    if( predictType == PREDICT_AUTO )
    {
        predictType = !_isClassifier || (classLabels.size() == 2 && (flags & RAW_OUTPUT) != 0) ?
            PREDICT_SUM : PREDICT_MAX_VOTE;
    }

    if( predictType == PREDICT_MAX_VOTE )
    {
        for( i = 0; i < nclasses; i++ )
            votes[i] = 0;
    }

    for( int ridx = range.start; ridx < range.end; ridx++ )
    {
        int nidx = roots[ridx], prev = nidx, c = 0;

        for(;;)
        {
            prev = nidx;
            const Node& node = nodes[nidx];
            if( node.split < 0 )
                break;
            const Split& split = splits[node.split];
            int vi = split.varIdx;
            int ci = cvidx ? cvidx[vi] : vi;
            float val = psample[ci*sstep];
            if( val == MISSED_VAL )
            {
                if( !missingSubstPtr )
                {
                    nidx = node.defaultDir < 0 ? node.left : node.right;
                    continue;
                }
                val = missingSubstPtr[vi];
            }

            if( vtype[vi] == VAR_ORDERED )
                nidx = val <= split.c ? node.left : node.right;
            else
            {
                if( flags & PREPROCESSED_INPUT )
                    c = cvRound(val);
                else
                {
                    c = catbuf[ci];
                    if( c < 0 )
                    {
                        int a = c = cofs[vi][0];
                        int b = cofs[vi][1];

                        int ival = cvRound(val);
                        if( ival != val )
                            CV_Error( CV_StsBadArg,
                                     "one of input categorical variable is not an integer" );

                        while( a < b )
                        {
                            c = (a + b) >> 1;
                            if( ival < cmap[c] )
                                b = c;
                            else if( ival > cmap[c] )
                                a = c+1;
                            else
                                break;
                        }

                        CV_Assert( c >= 0 && ival == cmap[c] );

                        c -= cofs[vi][0];
                        catbuf[ci] = c;
                    }
                    const int* subset = &subsets[split.subsetOfs];
                    unsigned u = c;
                    nidx = CV_DTREE_CAT_DIR(u, subset) < 0 ? node.left : node.right;
                }
            }
        }

        if( predictType == PREDICT_SUM )
            sum += nodes[prev].value;
        else
        {
            lastClassIdx = nodes[prev].classIdx;
            votes[lastClassIdx]++;
        }
    }

    if( predictType == PREDICT_MAX_VOTE )
    {
        int best_idx = lastClassIdx;
        if( range.end - range.start > 1 )
        {
            best_idx = 0;
            for( i = 1; i < nclasses; i++ )
                if( votes[best_idx] < votes[i] )
                    best_idx = i;
        }
        sum = (flags & RAW_OUTPUT) ? (float)best_idx : classLabels[best_idx];
    }

    return (float)sum;
}


float DTreesImpl::predict( InputArray _samples, OutputArray _results, int flags ) const
{
    CV_Assert( !roots.empty() );
    Mat samples = _samples.getMat(), results;
    int i, nsamples = samples.rows;
    int rtype = CV_32F;
    bool needresults = _results.needed();
    float retval = 0.f;
    bool iscls = isClassifier();
    float scale = !iscls ? 1.f/(int)roots.size() : 1.f;

    if( iscls && (flags & PREDICT_MASK) == PREDICT_MAX_VOTE )
        rtype = CV_32S;

    if( needresults )
    {
        _results.create(nsamples, 1, rtype);
        results = _results.getMat();
    }
    else
        nsamples = std::min(nsamples, 1);

    for( i = 0; i < nsamples; i++ )
    {
        float val = predictTrees( Range(0, (int)roots.size()), samples.row(i), flags )*scale;
        if( needresults )
        {
            if( rtype == CV_32F )
                results.at<float>(i) = val;
            else
                results.at<int>(i) = cvRound(val);
        }
        if( i == 0 )
            retval = val;
    }
    return retval;
}

void DTreesImpl::writeTrainingParams(FileStorage& fs) const
{
    fs << "use_surrogates" << (params.useSurrogates ? 1 : 0);
    fs << "max_categories" << params.getMaxCategories();
    fs << "regression_accuracy" << params.getRegressionAccuracy();

    fs << "max_depth" << params.getMaxDepth();
    fs << "min_sample_count" << params.getMinSampleCount();
    fs << "cross_validation_folds" << params.getCVFolds();

    if( params.getCVFolds() > 1 )
        fs << "use_1se_rule" << (params.use1SERule ? 1 : 0);

    if( !params.priors.empty() )
        fs << "priors" << params.priors;
}

void DTreesImpl::writeParams(FileStorage& fs) const
{
    fs << "is_classifier" << isClassifier();
    fs << "var_all" << (int)varType.size();
    fs << "var_count" << getVarCount();

    int ord_var_count = 0, cat_var_count = 0;
    int i, n = (int)varType.size();
    for( i = 0; i < n; i++ )
        if( varType[i] == VAR_ORDERED )
            ord_var_count++;
        else
            cat_var_count++;
    fs << "ord_var_count" << ord_var_count;
    fs << "cat_var_count" << cat_var_count;

    fs << "training_params" << "{";
    writeTrainingParams(fs);

    fs << "}";

    if( !varIdx.empty() )
    {
        fs << "global_var_idx" << 1;
        fs << "var_idx" << varIdx;
    }

    fs << "var_type" << varType;

    if( !catOfs.empty() )
        fs << "cat_ofs" << catOfs;
    if( !catMap.empty() )
        fs << "cat_map" << catMap;
    if( !classLabels.empty() )
        fs << "class_labels" << classLabels;
    if( !missingSubst.empty() )
        fs << "missing_subst" << missingSubst;
}

void DTreesImpl::writeSplit( FileStorage& fs, int splitidx ) const
{
    const Split& split = splits[splitidx];

    fs << "{:";

    int vi = split.varIdx;
    fs << "var" << vi;
    fs << "quality" << split.quality;

    if( varType[vi] == VAR_CATEGORICAL ) // split on a categorical var
    {
        int i, n = getCatCount(vi), to_right = 0;
        const int* subset = &subsets[split.subsetOfs];
        for( i = 0; i < n; i++ )
            to_right += CV_DTREE_CAT_DIR(i, subset) > 0;

        // ad-hoc rule when to use inverse categorical split notation
        // to achieve more compact and clear representation
        int default_dir = to_right <= 1 || to_right <= std::min(3, n/2) || to_right <= n/3 ? -1 : 1;

        fs << (default_dir*(split.inversed ? -1 : 1) > 0 ? "in" : "not_in") << "[:";

        for( i = 0; i < n; i++ )
        {
            int dir = CV_DTREE_CAT_DIR(i, subset);
            if( dir*default_dir < 0 )
                fs << i;
        }

        fs << "]";
    }
    else
        fs << (!split.inversed ? "le" : "gt") << split.c;

    fs << "}";
}

void DTreesImpl::writeNode( FileStorage& fs, int nidx, int depth ) const
{
    const Node& node = nodes[nidx];
    fs << "{";
    fs << "depth" << depth;
    fs << "value" << node.value;

    if( _isClassifier )
        fs << "norm_class_idx" << node.classIdx;

    if( node.split >= 0 )
    {
        fs << "splits" << "[";

        for( int splitidx = node.split; splitidx >= 0; splitidx = splits[splitidx].next )
            writeSplit( fs, splitidx );

        fs << "]";
    }

    fs << "}";
}

void DTreesImpl::writeTree( FileStorage& fs, int root ) const
{
    fs << "nodes" << "[";

    int nidx = root, pidx = 0, depth = 0;
    const Node *node = 0;

    // traverse the tree and save all the nodes in depth-first order
    for(;;)
    {
        for(;;)
        {
            writeNode( fs, nidx, depth );
            node = &nodes[nidx];
            if( node->left < 0 )
                break;
            nidx = node->left;
            depth++;
        }

        for( pidx = node->parent; pidx >= 0 && nodes[pidx].right == nidx;
             nidx = pidx, pidx = nodes[pidx].parent )
            depth--;

        if( pidx < 0 )
            break;

        nidx = nodes[pidx].right;
    }

    fs << "]";
}

void DTreesImpl::write( FileStorage& fs ) const
{
    writeFormat(fs);
    writeParams(fs);
    writeTree(fs, roots[0]);
}

void DTreesImpl::readParams( const FileNode& fn )
{
    _isClassifier = (int)fn["is_classifier"] != 0;
    /*int var_all = (int)fn["var_all"];
    int var_count = (int)fn["var_count"];
    int cat_var_count = (int)fn["cat_var_count"];
    int ord_var_count = (int)fn["ord_var_count"];*/

    FileNode tparams_node = fn["training_params"];

    TreeParams params0 = TreeParams();

    if( !tparams_node.empty() ) // training parameters are not necessary
    {
        params0.useSurrogates = (int)tparams_node["use_surrogates"] != 0;
        params0.setMaxCategories((int)(tparams_node["max_categories"].empty() ? 16 : tparams_node["max_categories"]));
        params0.setRegressionAccuracy((float)tparams_node["regression_accuracy"]);
        params0.setMaxDepth((int)tparams_node["max_depth"]);
        params0.setMinSampleCount((int)tparams_node["min_sample_count"]);
        params0.setCVFolds((int)tparams_node["cross_validation_folds"]);

        if( params0.getCVFolds() > 1 )
        {
            params.use1SERule = (int)tparams_node["use_1se_rule"] != 0;
        }

        tparams_node["priors"] >> params0.priors;
    }

    readVectorOrMat(fn["var_idx"], varIdx);
    fn["var_type"] >> varType;

    int format = 0;
    fn["format"] >> format;
    bool isLegacy = format < 3;

    int varAll = (int)fn["var_all"];
    if (isLegacy && (int)varType.size() <= varAll)
    {
        std::vector<uchar> extendedTypes(varAll + 1, 0);

        int i = 0, n;
        if (!varIdx.empty())
        {
            n = (int)varIdx.size();
            for (; i < n; ++i)
            {
                int var = varIdx[i];
                extendedTypes[var] = varType[i];
            }
        }
        else
        {
            n = (int)varType.size();
            for (; i < n; ++i)
            {
                extendedTypes[i] = varType[i];
            }
        }
        extendedTypes[varAll] = (uchar)(_isClassifier ? VAR_CATEGORICAL : VAR_ORDERED);
        extendedTypes.swap(varType);
    }

    readVectorOrMat(fn["cat_map"], catMap);

    if (isLegacy)
    {
        // generating "catOfs" from "cat_count"
        catOfs.clear();
        classLabels.clear();
        std::vector<int> counts;
        readVectorOrMat(fn["cat_count"], counts);
        unsigned int i = 0, j = 0, curShift = 0, size = (int)varType.size() - 1;
        for (; i < size; ++i)
        {
            Vec2i newOffsets(0, 0);
            if (varType[i] == VAR_CATEGORICAL) // only categorical vars are represented in catMap
            {
                newOffsets[0] = curShift;
                curShift += counts[j];
                newOffsets[1] = curShift;
                ++j;
            }
            catOfs.push_back(newOffsets);
        }
        // other elements in "catMap" are "classLabels"
        if (curShift < catMap.size())
        {
            classLabels.insert(classLabels.end(), catMap.begin() + curShift, catMap.end());
            catMap.erase(catMap.begin() + curShift, catMap.end());
        }
    }
    else
    {
        fn["cat_ofs"] >> catOfs;
        fn["missing_subst"] >> missingSubst;
        fn["class_labels"] >> classLabels;
    }

    // init var mapping for node reading (var indexes or varIdx indexes)
    bool globalVarIdx = false;
    fn["global_var_idx"] >> globalVarIdx;
    if (globalVarIdx || varIdx.empty())
        setRangeVector(varMapping, (int)varType.size());
    else
        varMapping = varIdx;

    initCompVarIdx();
    setDParams(params0);
}

int DTreesImpl::readSplit( const FileNode& fn )
{
    Split split;

    int vi = (int)fn["var"];
    CV_Assert( 0 <= vi && vi <= (int)varType.size() );
    vi = varMapping[vi]; // convert to varIdx if needed
    split.varIdx = vi;

    if( varType[vi] == VAR_CATEGORICAL ) // split on categorical var
    {
        int i, val, ssize = getSubsetSize(vi);
        split.subsetOfs = (int)subsets.size();
        for( i = 0; i < ssize; i++ )
            subsets.push_back(0);
        int* subset = &subsets[split.subsetOfs];
        FileNode fns = fn["in"];
        if( fns.empty() )
        {
            fns = fn["not_in"];
            split.inversed = true;
        }

        if( fns.isInt() )
        {
            val = (int)fns;
            subset[val >> 5] |= 1 << (val & 31);
        }
        else
        {
            FileNodeIterator it = fns.begin();
            int n = (int)fns.size();
            for( i = 0; i < n; i++, ++it )
            {
                val = (int)*it;
                subset[val >> 5] |= 1 << (val & 31);
            }
        }

        // for categorical splits we do not use inversed splits,
        // instead we inverse the variable set in the split
        if( split.inversed )
        {
            for( i = 0; i < ssize; i++ )
                subset[i] ^= -1;
            split.inversed = false;
        }
    }
    else
    {
        FileNode cmpNode = fn["le"];
        if( cmpNode.empty() )
        {
            cmpNode = fn["gt"];
            split.inversed = true;
        }
        split.c = (float)cmpNode;
    }

    split.quality = (float)fn["quality"];
    splits.push_back(split);

    return (int)(splits.size() - 1);
}

int DTreesImpl::readNode( const FileNode& fn )
{
    Node node;
    node.value = (double)fn["value"];

    if( _isClassifier )
        node.classIdx = (int)fn["norm_class_idx"];

    FileNode sfn = fn["splits"];
    if( !sfn.empty() )
    {
        int i, n = (int)sfn.size(), prevsplit = -1;
        FileNodeIterator it = sfn.begin();

        for( i = 0; i < n; i++, ++it )
        {
            int splitidx = readSplit(*it);
            if( splitidx < 0 )
                break;
            if( prevsplit < 0 )
                node.split = splitidx;
            else
                splits[prevsplit].next = splitidx;
            prevsplit = splitidx;
        }
    }
    nodes.push_back(node);
    return (int)(nodes.size() - 1);
}

int DTreesImpl::readTree( const FileNode& fn )
{
    int i, n = (int)fn.size(), root = -1, pidx = -1;
    FileNodeIterator it = fn.begin();

    for( i = 0; i < n; i++, ++it )
    {
        int nidx = readNode(*it);
        if( nidx < 0 )
            break;
        Node& node = nodes[nidx];
        node.parent = pidx;
        if( pidx < 0 )
            root = nidx;
        else
        {
            Node& parent = nodes[pidx];
            if( parent.left < 0 )
                parent.left = nidx;
            else
                parent.right = nidx;
        }
        if( node.split >= 0 )
            pidx = nidx;
        else
        {
            while( pidx >= 0 && nodes[pidx].right >= 0 )
                pidx = nodes[pidx].parent;
        }
    }
    roots.push_back(root);
    return root;
}

void DTreesImpl::read( const FileNode& fn )
{
    clear();
    readParams(fn);

    FileNode fnodes = fn["nodes"];
    CV_Assert( !fnodes.empty() );
    readTree(fnodes);
}

Ptr<DTrees> DTrees::create()
{
    return makePtr<DTreesImpl>();
}

}
}

/* End of file. */
