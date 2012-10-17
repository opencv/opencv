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

#include "precomp.hpp"

static inline double
log_ratio( double val )
{
    const double eps = 1e-5;

    val = MAX( val, eps );
    val = MIN( val, 1. - eps );
    return log( val/(1. - val) );
}


CvBoostParams::CvBoostParams()
{
    boost_type = CvBoost::REAL;
    weak_count = 100;
    weight_trim_rate = 0.95;
    cv_folds = 0;
    max_depth = 1;
}


CvBoostParams::CvBoostParams( int _boost_type, int _weak_count,
                                        double _weight_trim_rate, int _max_depth,
                                        bool _use_surrogates, const float* _priors )
{
    boost_type = _boost_type;
    weak_count = _weak_count;
    weight_trim_rate = _weight_trim_rate;
    split_criteria = CvBoost::DEFAULT;
    cv_folds = 0;
    max_depth = _max_depth;
    use_surrogates = _use_surrogates;
    priors = _priors;
}



///////////////////////////////// CvBoostTree ///////////////////////////////////

CvBoostTree::CvBoostTree()
{
    ensemble = 0;
}


CvBoostTree::~CvBoostTree()
{
    clear();
}


void
CvBoostTree::clear()
{
    CvDTree::clear();
    ensemble = 0;
}


bool
CvBoostTree::train( CvDTreeTrainData* _train_data,
                    const CvMat* _subsample_idx, CvBoost* _ensemble )
{
    clear();
    ensemble = _ensemble;
    data = _train_data;
    data->shared = true;
    return do_train( _subsample_idx );
}


bool
CvBoostTree::train( const CvMat*, int, const CvMat*, const CvMat*,
                    const CvMat*, const CvMat*, const CvMat*, CvDTreeParams )
{
    assert(0);
    return false;
}


bool
CvBoostTree::train( CvDTreeTrainData*, const CvMat* )
{
    assert(0);
    return false;
}


void
CvBoostTree::scale( double _scale )
{
    CvDTreeNode* node = root;

    // traverse the tree and scale all the node values
    for(;;)
    {
        CvDTreeNode* parent;
        for(;;)
        {
            node->value *= _scale;
            if( !node->left )
                break;
            node = node->left;
        }

        for( parent = node->parent; parent && parent->right == node;
            node = parent, parent = parent->parent )
            ;

        if( !parent )
            break;

        node = parent->right;
    }
}


void
CvBoostTree::try_split_node( CvDTreeNode* node )
{
    CvDTree::try_split_node( node );

    if( !node->left )
    {
        // if the node has not been split,
        // store the responses for the corresponding training samples
        double* weak_eval = ensemble->get_weak_response()->data.db;
        cv::AutoBuffer<int> inn_buf(node->sample_count);
        const int* labels = data->get_cv_labels( node, (int*)inn_buf );
        int i, count = node->sample_count;
        double value = node->value;

        for( i = 0; i < count; i++ )
            weak_eval[labels[i]] = value;
    }
}


double
CvBoostTree::calc_node_dir( CvDTreeNode* node )
{
    char* dir = (char*)data->direction->data.ptr;
    const double* weights = ensemble->get_subtree_weights()->data.db;
    int i, n = node->sample_count, vi = node->split->var_idx;
    double L, R;

    assert( !node->split->inversed );

    if( data->get_var_type(vi) >= 0 ) // split on categorical var
    {
        cv::AutoBuffer<int> inn_buf(n);
        const int* cat_labels = data->get_cat_var_data( node, vi, (int*)inn_buf );
        const int* subset = node->split->subset;
        double sum = 0, sum_abs = 0;

        for( i = 0; i < n; i++ )
        {
            int idx = ((cat_labels[i] == 65535) && data->is_buf_16u) ? -1 : cat_labels[i];
            double w = weights[i];
            int d = idx >= 0 ? CV_DTREE_CAT_DIR(idx,subset) : 0;
            sum += d*w; sum_abs += (d & 1)*w;
            dir[i] = (char)d;
        }

        R = (sum_abs + sum) * 0.5;
        L = (sum_abs - sum) * 0.5;
    }
    else // split on ordered var
    {
        cv::AutoBuffer<uchar> inn_buf(2*n*sizeof(int)+n*sizeof(float));
        float* values_buf = (float*)(uchar*)inn_buf;
        int* sorted_indices_buf = (int*)(values_buf + n);
        int* sample_indices_buf = sorted_indices_buf + n;
        const float* values = 0;
        const int* sorted_indices = 0;
        data->get_ord_var_data( node, vi, values_buf, sorted_indices_buf, &values, &sorted_indices, sample_indices_buf );
        int split_point = node->split->ord.split_point;
        int n1 = node->get_num_valid(vi);

        assert( 0 <= split_point && split_point < n1-1 );
        L = R = 0;

        for( i = 0; i <= split_point; i++ )
        {
            int idx = sorted_indices[i];
            double w = weights[idx];
            dir[idx] = (char)-1;
            L += w;
        }

        for( ; i < n1; i++ )
        {
            int idx = sorted_indices[i];
            double w = weights[idx];
            dir[idx] = (char)1;
            R += w;
        }

        for( ; i < n; i++ )
            dir[sorted_indices[i]] = (char)0;
    }

    node->maxlr = MAX( L, R );
    return node->split->quality/(L + R);
}


CvDTreeSplit*
CvBoostTree::find_split_ord_class( CvDTreeNode* node, int vi, float init_quality,
                                    CvDTreeSplit* _split, uchar* _ext_buf )
{
    const float epsilon = FLT_EPSILON*2;

    const double* weights = ensemble->get_subtree_weights()->data.db;
    int n = node->sample_count;
    int n1 = node->get_num_valid(vi);

    cv::AutoBuffer<uchar> inn_buf;
    if( !_ext_buf )
        inn_buf.allocate(n*(3*sizeof(int)+sizeof(float)));
    uchar* ext_buf = _ext_buf ? _ext_buf : (uchar*)inn_buf;
    float* values_buf = (float*)ext_buf;
    int* sorted_indices_buf = (int*)(values_buf + n);
    int* sample_indices_buf = sorted_indices_buf + n;
    const float* values = 0;
    const int* sorted_indices = 0;
    data->get_ord_var_data( node, vi, values_buf, sorted_indices_buf, &values, &sorted_indices, sample_indices_buf );
    int* responses_buf = sorted_indices_buf + n;
    const int* responses = data->get_class_labels( node, responses_buf );
    const double* rcw0 = weights + n;
    double lcw[2] = {0,0}, rcw[2];
    int i, best_i = -1;
    double best_val = init_quality;
    int boost_type = ensemble->get_params().boost_type;
    int split_criteria = ensemble->get_params().split_criteria;

    rcw[0] = rcw0[0]; rcw[1] = rcw0[1];
    for( i = n1; i < n; i++ )
    {
        int idx = sorted_indices[i];
        double w = weights[idx];
        rcw[responses[idx]] -= w;
    }

    if( split_criteria != CvBoost::GINI && split_criteria != CvBoost::MISCLASS )
        split_criteria = boost_type == CvBoost::DISCRETE ? CvBoost::MISCLASS : CvBoost::GINI;

    if( split_criteria == CvBoost::GINI )
    {
        double L = 0, R = rcw[0] + rcw[1];
        double lsum2 = 0, rsum2 = rcw[0]*rcw[0] + rcw[1]*rcw[1];

        for( i = 0; i < n1 - 1; i++ )
        {
            int idx = sorted_indices[i];
            double w = weights[idx], w2 = w*w;
            double lv, rv;
            idx = responses[idx];
            L += w; R -= w;
            lv = lcw[idx]; rv = rcw[idx];
            lsum2 += 2*lv*w + w2;
            rsum2 -= 2*rv*w - w2;
            lcw[idx] = lv + w; rcw[idx] = rv - w;

            if( values[i] + epsilon < values[i+1] )
            {
                double val = (lsum2*R + rsum2*L)/(L*R);
                if( best_val < val )
                {
                    best_val = val;
                    best_i = i;
                }
            }
        }
    }
    else
    {
        for( i = 0; i < n1 - 1; i++ )
        {
            int idx = sorted_indices[i];
            double w = weights[idx];
            idx = responses[idx];
            lcw[idx] += w;
            rcw[idx] -= w;

            if( values[i] + epsilon < values[i+1] )
            {
                double val = lcw[0] + rcw[1], val2 = lcw[1] + rcw[0];
                val = MAX(val, val2);
                if( best_val < val )
                {
                    best_val = val;
                    best_i = i;
                }
            }
        }
    }

    CvDTreeSplit* split = 0;
    if( best_i >= 0 )
    {
        split = _split ? _split : data->new_split_ord( 0, 0.0f, 0, 0, 0.0f );
        split->var_idx = vi;
        split->ord.c = (values[best_i] + values[best_i+1])*0.5f;
        split->ord.split_point = best_i;
        split->inversed = 0;
        split->quality = (float)best_val;
    }
    return split;
}


#define CV_CMP_NUM_PTR(a,b) (*(a) < *(b))
static CV_IMPLEMENT_QSORT_EX( icvSortDblPtr, double*, CV_CMP_NUM_PTR, int )

CvDTreeSplit*
CvBoostTree::find_split_cat_class( CvDTreeNode* node, int vi, float init_quality, CvDTreeSplit* _split, uchar* _ext_buf )
{
    int ci = data->get_var_type(vi);
    int n = node->sample_count;
    int mi = data->cat_count->data.i[ci];

    int base_size = (2*mi+3)*sizeof(double) + mi*sizeof(double*);
    cv::AutoBuffer<uchar> inn_buf((2*mi+3)*sizeof(double) + mi*sizeof(double*));
    if( !_ext_buf)
        inn_buf.allocate( base_size + 2*n*sizeof(int) );
    uchar* base_buf = (uchar*)inn_buf;
    uchar* ext_buf = _ext_buf ? _ext_buf : base_buf + base_size;

    int* cat_labels_buf = (int*)ext_buf;
    const int* cat_labels = data->get_cat_var_data(node, vi, cat_labels_buf);
    int* responses_buf = cat_labels_buf + n;
    const int* responses = data->get_class_labels(node, responses_buf);
    double lcw[2]={0,0}, rcw[2]={0,0};

    double* cjk = (double*)cv::alignPtr(base_buf,sizeof(double))+2;
    const double* weights = ensemble->get_subtree_weights()->data.db;
    double** dbl_ptr = (double**)(cjk + 2*mi);
    int i, j, k, idx;
    double L = 0, R;
    double best_val = init_quality;
    int best_subset = -1, subset_i;
    int boost_type = ensemble->get_params().boost_type;
    int split_criteria = ensemble->get_params().split_criteria;

    // init array of counters:
    // c_{jk} - number of samples that have vi-th input variable = j and response = k.
    for( j = -1; j < mi; j++ )
        cjk[j*2] = cjk[j*2+1] = 0;

    for( i = 0; i < n; i++ )
    {
        double w = weights[i];
        j = ((cat_labels[i] == 65535) && data->is_buf_16u) ? -1 : cat_labels[i];
        k = responses[i];
        cjk[j*2 + k] += w;
    }

    for( j = 0; j < mi; j++ )
    {
        rcw[0] += cjk[j*2];
        rcw[1] += cjk[j*2+1];
        dbl_ptr[j] = cjk + j*2 + 1;
    }

    R = rcw[0] + rcw[1];

    if( split_criteria != CvBoost::GINI && split_criteria != CvBoost::MISCLASS )
        split_criteria = boost_type == CvBoost::DISCRETE ? CvBoost::MISCLASS : CvBoost::GINI;

    // sort rows of c_jk by increasing c_j,1
    // (i.e. by the weight of samples in j-th category that belong to class 1)
    icvSortDblPtr( dbl_ptr, mi, 0 );

    for( subset_i = 0; subset_i < mi-1; subset_i++ )
    {
        idx = (int)(dbl_ptr[subset_i] - cjk)/2;
        const double* crow = cjk + idx*2;
        double w0 = crow[0], w1 = crow[1];
        double weight = w0 + w1;

        if( weight < FLT_EPSILON )
            continue;

        lcw[0] += w0; rcw[0] -= w0;
        lcw[1] += w1; rcw[1] -= w1;

        if( split_criteria == CvBoost::GINI )
        {
            double lsum2 = lcw[0]*lcw[0] + lcw[1]*lcw[1];
            double rsum2 = rcw[0]*rcw[0] + rcw[1]*rcw[1];

            L += weight;
            R -= weight;

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
        else
        {
            double val = lcw[0] + rcw[1];
            double val2 = lcw[1] + rcw[0];

            val = MAX(val, val2);
            if( best_val < val )
            {
                best_val = val;
                best_subset = subset_i;
            }
        }
    }

    CvDTreeSplit* split = 0;
    if( best_subset >= 0 )
    {
        split = _split ? _split : data->new_split_cat( 0, -1.0f);
        split->var_idx = vi;
        split->quality = (float)best_val;
        memset( split->subset, 0, (data->max_c_count + 31)/32 * sizeof(int));
        for( i = 0; i <= best_subset; i++ )
        {
            idx = (int)(dbl_ptr[i] - cjk) >> 1;
            split->subset[idx >> 5] |= 1 << (idx & 31);
        }
    }
    return split;
}


CvDTreeSplit*
CvBoostTree::find_split_ord_reg( CvDTreeNode* node, int vi, float init_quality, CvDTreeSplit* _split, uchar* _ext_buf )
{
    const float epsilon = FLT_EPSILON*2;
    const double* weights = ensemble->get_subtree_weights()->data.db;
    int n = node->sample_count;
    int n1 = node->get_num_valid(vi);

    cv::AutoBuffer<uchar> inn_buf;
    if( !_ext_buf )
        inn_buf.allocate(2*n*(sizeof(int)+sizeof(float)));
    uchar* ext_buf = _ext_buf ? _ext_buf : (uchar*)inn_buf;

    float* values_buf = (float*)ext_buf;
    int* indices_buf = (int*)(values_buf + n);
    int* sample_indices_buf = indices_buf + n;
    const float* values = 0;
    const int* indices = 0;
    data->get_ord_var_data( node, vi, values_buf, indices_buf, &values, &indices, sample_indices_buf );
    float* responses_buf = (float*)(indices_buf + n);
    const float* responses = data->get_ord_responses( node, responses_buf, sample_indices_buf );

    int i, best_i = -1;
    double L = 0, R = weights[n];
    double best_val = init_quality, lsum = 0, rsum = node->value*R;

    // compensate for missing values
    for( i = n1; i < n; i++ )
    {
        int idx = indices[i];
        double w = weights[idx];
        rsum -= responses[idx]*w;
        R -= w;
    }

    // find the optimal split
    for( i = 0; i < n1 - 1; i++ )
    {
        int idx = indices[i];
        double w = weights[idx];
        double t = responses[idx]*w;
        L += w; R -= w;
        lsum += t; rsum -= t;

        if( values[i] + epsilon < values[i+1] )
        {
            double val = (lsum*lsum*R + rsum*rsum*L)/(L*R);
            if( best_val < val )
            {
                best_val = val;
                best_i = i;
            }
        }
    }

    CvDTreeSplit* split = 0;
    if( best_i >= 0 )
    {
        split = _split ? _split : data->new_split_ord( 0, 0.0f, 0, 0, 0.0f );
        split->var_idx = vi;
        split->ord.c = (values[best_i] + values[best_i+1])*0.5f;
        split->ord.split_point = best_i;
        split->inversed = 0;
        split->quality = (float)best_val;
    }
    return split;
}


CvDTreeSplit*
CvBoostTree::find_split_cat_reg( CvDTreeNode* node, int vi, float init_quality, CvDTreeSplit* _split, uchar* _ext_buf )
{
    const double* weights = ensemble->get_subtree_weights()->data.db;
    int ci = data->get_var_type(vi);
    int n = node->sample_count;
    int mi = data->cat_count->data.i[ci];
    int base_size = (2*mi+3)*sizeof(double) + mi*sizeof(double*);
    cv::AutoBuffer<uchar> inn_buf(base_size);
    if( !_ext_buf )
        inn_buf.allocate(base_size + n*(2*sizeof(int) + sizeof(float)));
    uchar* base_buf = (uchar*)inn_buf;
    uchar* ext_buf = _ext_buf ? _ext_buf : base_buf + base_size;

    int* cat_labels_buf = (int*)ext_buf;
    const int* cat_labels = data->get_cat_var_data(node, vi, cat_labels_buf);
    float* responses_buf = (float*)(cat_labels_buf + n);
    int* sample_indices_buf = (int*)(responses_buf + n);
    const float* responses = data->get_ord_responses(node, responses_buf, sample_indices_buf);

    double* sum = (double*)cv::alignPtr(base_buf,sizeof(double)) + 1;
    double* counts = sum + mi + 1;
    double** sum_ptr = (double**)(counts + mi);
    double L = 0, R = 0, best_val = init_quality, lsum = 0, rsum = 0;
    int i, best_subset = -1, subset_i;

    for( i = -1; i < mi; i++ )
        sum[i] = counts[i] = 0;

    // calculate sum response and weight of each category of the input var
    for( i = 0; i < n; i++ )
    {
        int idx = ((cat_labels[i] == 65535) && data->is_buf_16u) ? -1 : cat_labels[i];
        double w = weights[i];
        double s = sum[idx] + responses[i]*w;
        double nc = counts[idx] + w;
        sum[idx] = s;
        counts[idx] = nc;
    }

    // calculate average response in each category
    for( i = 0; i < mi; i++ )
    {
        R += counts[i];
        rsum += sum[i];
        sum[i] = fabs(counts[i]) > DBL_EPSILON ? sum[i]/counts[i] : 0;
        sum_ptr[i] = sum + i;
    }

    icvSortDblPtr( sum_ptr, mi, 0 );

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

    CvDTreeSplit* split = 0;
    if( best_subset >= 0 )
    {
        split = _split ? _split : data->new_split_cat( 0, -1.0f);
        split->var_idx = vi;
        split->quality = (float)best_val;
        memset( split->subset, 0, (data->max_c_count + 31)/32 * sizeof(int));
        for( i = 0; i <= best_subset; i++ )
        {
            int idx = (int)(sum_ptr[i] - sum);
            split->subset[idx >> 5] |= 1 << (idx & 31);
        }
    }
    return split;
}


CvDTreeSplit*
CvBoostTree::find_surrogate_split_ord( CvDTreeNode* node, int vi, uchar* _ext_buf )
{
    const float epsilon = FLT_EPSILON*2;
    int n = node->sample_count;
    cv::AutoBuffer<uchar> inn_buf;
    if( !_ext_buf )
        inn_buf.allocate(n*(2*sizeof(int)+sizeof(float)));
    uchar* ext_buf = _ext_buf ? _ext_buf : (uchar*)inn_buf;
    float* values_buf = (float*)ext_buf;
    int* indices_buf = (int*)(values_buf + n);
    int* sample_indices_buf = indices_buf + n;
    const float* values = 0;
    const int* indices = 0;
    data->get_ord_var_data( node, vi, values_buf, indices_buf, &values, &indices, sample_indices_buf );

    const double* weights = ensemble->get_subtree_weights()->data.db;
    const char* dir = (char*)data->direction->data.ptr;
    int n1 = node->get_num_valid(vi);
    // LL - number of samples that both the primary and the surrogate splits send to the left
    // LR - ... primary split sends to the left and the surrogate split sends to the right
    // RL - ... primary split sends to the right and the surrogate split sends to the left
    // RR - ... both send to the right
    int i, best_i = -1, best_inversed = 0;
    double best_val;
    double LL = 0, RL = 0, LR, RR;
    double worst_val = node->maxlr;
    double sum = 0, sum_abs = 0;
    best_val = worst_val;

    for( i = 0; i < n1; i++ )
    {
        int idx = indices[i];
        double w = weights[idx];
        int d = dir[idx];
        sum += d*w; sum_abs += (d & 1)*w;
    }

    // sum_abs = R + L; sum = R - L
    RR = (sum_abs + sum)*0.5;
    LR = (sum_abs - sum)*0.5;

    // initially all the samples are sent to the right by the surrogate split,
    // LR of them are sent to the left by primary split, and RR - to the right.
    // now iteratively compute LL, LR, RL and RR for every possible surrogate split value.
    for( i = 0; i < n1 - 1; i++ )
    {
        int idx = indices[i];
        double w = weights[idx];
        int d = dir[idx];

        if( d < 0 )
        {
            LL += w; LR -= w;
            if( LL + RR > best_val && values[i] + epsilon < values[i+1] )
            {
                best_val = LL + RR;
                best_i = i; best_inversed = 0;
            }
        }
        else if( d > 0 )
        {
            RL += w; RR -= w;
            if( RL + LR > best_val && values[i] + epsilon < values[i+1] )
            {
                best_val = RL + LR;
                best_i = i; best_inversed = 1;
            }
        }
    }

    return best_i >= 0 && best_val > node->maxlr ? data->new_split_ord( vi,
        (values[best_i] + values[best_i+1])*0.5f, best_i,
        best_inversed, (float)best_val ) : 0;
}


CvDTreeSplit*
CvBoostTree::find_surrogate_split_cat( CvDTreeNode* node, int vi, uchar* _ext_buf )
{
    const char* dir = (char*)data->direction->data.ptr;
    const double* weights = ensemble->get_subtree_weights()->data.db;
    int n = node->sample_count;
    int i, mi = data->cat_count->data.i[data->get_var_type(vi)];

    int base_size = (2*mi+3)*sizeof(double);
    cv::AutoBuffer<uchar> inn_buf(base_size);
    if( !_ext_buf )
        inn_buf.allocate(base_size + n*sizeof(int));
    uchar* ext_buf = _ext_buf ? _ext_buf : (uchar*)inn_buf;
    int* cat_labels_buf = (int*)ext_buf;
    const int* cat_labels = data->get_cat_var_data(node, vi, cat_labels_buf);

    // LL - number of samples that both the primary and the surrogate splits send to the left
    // LR - ... primary split sends to the left and the surrogate split sends to the right
    // RL - ... primary split sends to the right and the surrogate split sends to the left
    // RR - ... both send to the right
    CvDTreeSplit* split = data->new_split_cat( vi, 0 );
    double best_val = 0;
    double* lc = (double*)cv::alignPtr(cat_labels_buf + n, sizeof(double)) + 1;
    double* rc = lc + mi + 1;

    for( i = -1; i < mi; i++ )
        lc[i] = rc[i] = 0;

    // 1. for each category calculate the weight of samples
    // sent to the left (lc) and to the right (rc) by the primary split
    for( i = 0; i < n; i++ )
    {
        int idx = ((cat_labels[i] == 65535) && data->is_buf_16u) ? -1 : cat_labels[i];
        double w = weights[i];
        int d = dir[i];
        double sum = lc[idx] + d*w;
        double sum_abs = rc[idx] + (d & 1)*w;
        lc[idx] = sum; rc[idx] = sum_abs;
    }

    for( i = 0; i < mi; i++ )
    {
        double sum = lc[i];
        double sum_abs = rc[i];
        lc[i] = (sum_abs - sum) * 0.5;
        rc[i] = (sum_abs + sum) * 0.5;
    }

    // 2. now form the split.
    // in each category send all the samples to the same direction as majority
    for( i = 0; i < mi; i++ )
    {
        double lval = lc[i], rval = rc[i];
        if( lval > rval )
        {
            split->subset[i >> 5] |= 1 << (i & 31);
            best_val += lval;
        }
        else
            best_val += rval;
    }

    split->quality = (float)best_val;
    if( split->quality <= node->maxlr )
        cvSetRemoveByPtr( data->split_heap, split ), split = 0;

    return split;
}


void
CvBoostTree::calc_node_value( CvDTreeNode* node )
{
    int i, n = node->sample_count;
    const double* weights = ensemble->get_weights()->data.db;
    cv::AutoBuffer<uchar> inn_buf(n*(sizeof(int) + ( data->is_classifier ? sizeof(int) : sizeof(int) + sizeof(float))));
    int* labels_buf = (int*)(uchar*)inn_buf;
    const int* labels = data->get_cv_labels(node, labels_buf);
    double* subtree_weights = ensemble->get_subtree_weights()->data.db;
    double rcw[2] = {0,0};
    int boost_type = ensemble->get_params().boost_type;

    if( data->is_classifier )
    {
        int* _responses_buf = labels_buf + n;
        const int* _responses = data->get_class_labels(node, _responses_buf);
        int m = data->get_num_classes();
        int* cls_count = data->counts->data.i;
        for( int k = 0; k < m; k++ )
            cls_count[k] = 0;

        for( i = 0; i < n; i++ )
        {
            int idx = labels[i];
            double w = weights[idx];
            int r = _responses[i];
            rcw[r] += w;
            cls_count[r]++;
            subtree_weights[i] = w;
        }

        node->class_idx = rcw[1] > rcw[0];

        if( boost_type == CvBoost::DISCRETE )
        {
            // ignore cat_map for responses, and use {-1,1},
            // as the whole ensemble response is computes as sign(sum_i(weak_response_i)
            node->value = node->class_idx*2 - 1;
        }
        else
        {
            double p = rcw[1]/(rcw[0] + rcw[1]);
            assert( boost_type == CvBoost::REAL );

            // store log-ratio of the probability
            node->value = 0.5*log_ratio(p);
        }
    }
    else
    {
        // in case of regression tree:
        //  * node value is 1/n*sum_i(Y_i), where Y_i is i-th response,
        //    n is the number of samples in the node.
        //  * node risk is the sum of squared errors: sum_i((Y_i - <node_value>)^2)
        double sum = 0, sum2 = 0, iw;
        float* values_buf = (float*)(labels_buf + n);
        int* sample_indices_buf = (int*)(values_buf + n);
        const float* values = data->get_ord_responses(node, values_buf, sample_indices_buf);

        for( i = 0; i < n; i++ )
        {
            int idx = labels[i];
            double w = weights[idx]/*priors[values[i] > 0]*/;
            double t = values[i];
            rcw[0] += w;
            subtree_weights[i] = w;
            sum += t*w;
            sum2 += t*t*w;
        }

        iw = 1./rcw[0];
        node->value = sum*iw;
        node->node_risk = sum2 - (sum*iw)*sum;

        // renormalize the risk, as in try_split_node the unweighted formula
        // sqrt(risk)/n is used, rather than sqrt(risk)/sum(weights_i)
        node->node_risk *= n*iw*n*iw;
    }

    // store summary weights
    subtree_weights[n] = rcw[0];
    subtree_weights[n+1] = rcw[1];
}


void CvBoostTree::read( CvFileStorage* fs, CvFileNode* fnode, CvBoost* _ensemble, CvDTreeTrainData* _data )
{
    CvDTree::read( fs, fnode, _data );
    ensemble = _ensemble;
}


void CvBoostTree::read( CvFileStorage*, CvFileNode* )
{
    assert(0);
}

void CvBoostTree::read( CvFileStorage* _fs, CvFileNode* _node,
                        CvDTreeTrainData* _data )
{
    CvDTree::read( _fs, _node, _data );
}


/////////////////////////////////// CvBoost /////////////////////////////////////

CvBoost::CvBoost()
{
    data = 0;
    weak = 0;
    default_model_name = "my_boost_tree";

    active_vars = active_vars_abs = orig_response = sum_response = weak_eval =
        subsample_mask = weights = subtree_weights = 0;
    have_active_cat_vars = have_subsample = false;

    clear();
}


void CvBoost::prune( CvSlice slice )
{
    if( weak && weak->total > 0 )
    {
        CvSeqReader reader;
        int i, count = cvSliceLength( slice, weak );

        cvStartReadSeq( weak, &reader );
        cvSetSeqReaderPos( &reader, slice.start_index );

        for( i = 0; i < count; i++ )
        {
            CvBoostTree* w;
            CV_READ_SEQ_ELEM( w, reader );
            delete w;
        }

        cvSeqRemoveSlice( weak, slice );
    }
}


void CvBoost::clear()
{
    if( weak )
    {
        prune( CV_WHOLE_SEQ );
        cvReleaseMemStorage( &weak->storage );
    }
    if( data )
        delete data;
    weak = 0;
    data = 0;
    cvReleaseMat( &active_vars );
    cvReleaseMat( &active_vars_abs );
    cvReleaseMat( &orig_response );
    cvReleaseMat( &sum_response );
    cvReleaseMat( &weak_eval );
    cvReleaseMat( &subsample_mask );
    cvReleaseMat( &weights );
    cvReleaseMat( &subtree_weights );

    have_subsample = false;
}


CvBoost::~CvBoost()
{
    clear();
}


CvBoost::CvBoost( const CvMat* _train_data, int _tflag,
                  const CvMat* _responses, const CvMat* _var_idx,
                  const CvMat* _sample_idx, const CvMat* _var_type,
                  const CvMat* _missing_mask, CvBoostParams _params )
{
    weak = 0;
    data = 0;
    default_model_name = "my_boost_tree";

    active_vars = active_vars_abs = orig_response = sum_response = weak_eval =
        subsample_mask = weights = subtree_weights = 0;

    train( _train_data, _tflag, _responses, _var_idx, _sample_idx,
           _var_type, _missing_mask, _params );
}


bool
CvBoost::set_params( const CvBoostParams& _params )
{
    bool ok = false;

    CV_FUNCNAME( "CvBoost::set_params" );

    __BEGIN__;

    params = _params;
    if( params.boost_type != DISCRETE && params.boost_type != REAL &&
        params.boost_type != LOGIT && params.boost_type != GENTLE )
        CV_ERROR( CV_StsBadArg, "Unknown/unsupported boosting type" );

    params.weak_count = MAX( params.weak_count, 1 );
    params.weight_trim_rate = MAX( params.weight_trim_rate, 0. );
    params.weight_trim_rate = MIN( params.weight_trim_rate, 1. );
    if( params.weight_trim_rate < FLT_EPSILON )
        params.weight_trim_rate = 1.f;

    if( params.boost_type == DISCRETE &&
        params.split_criteria != GINI && params.split_criteria != MISCLASS )
        params.split_criteria = MISCLASS;
    if( params.boost_type == REAL &&
        params.split_criteria != GINI && params.split_criteria != MISCLASS )
        params.split_criteria = GINI;
    if( (params.boost_type == LOGIT || params.boost_type == GENTLE) &&
        params.split_criteria != SQERR )
        params.split_criteria = SQERR;

    ok = true;

    __END__;

    return ok;
}


bool
CvBoost::train( const CvMat* _train_data, int _tflag,
              const CvMat* _responses, const CvMat* _var_idx,
              const CvMat* _sample_idx, const CvMat* _var_type,
              const CvMat* _missing_mask,
              CvBoostParams _params, bool _update )
{
    bool ok = false;
    CvMemStorage* storage = 0;

    CV_FUNCNAME( "CvBoost::train" );

    __BEGIN__;

    int i;

    set_params( _params );

    cvReleaseMat( &active_vars );
    cvReleaseMat( &active_vars_abs );

    if( !_update || !data )
    {
        clear();
        data = new CvDTreeTrainData( _train_data, _tflag, _responses, _var_idx,
            _sample_idx, _var_type, _missing_mask, _params, true, true );

        if( data->get_num_classes() != 2 )
            CV_ERROR( CV_StsNotImplemented,
            "Boosted trees can only be used for 2-class classification." );
        CV_CALL( storage = cvCreateMemStorage() );
        weak = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvBoostTree*), storage );
        storage = 0;
    }
    else
    {
        data->set_data( _train_data, _tflag, _responses, _var_idx,
            _sample_idx, _var_type, _missing_mask, _params, true, true, true );
    }

    if ( (_params.boost_type == LOGIT) || (_params.boost_type == GENTLE) )
        data->do_responses_copy();

    update_weights( 0 );

    for( i = 0; i < params.weak_count; i++ )
    {
        CvBoostTree* tree = new CvBoostTree;
        if( !tree->train( data, subsample_mask, this ) )
        {
            delete tree;
            break;
        }
        //cvCheckArr( get_weak_response());
        cvSeqPush( weak, &tree );
        update_weights( tree );
        trim_weights();
        if( cvCountNonZero(subsample_mask) == 0 )
            break;
    }

    if(weak->total > 0)
    {
        get_active_vars(); // recompute active_vars* maps and condensed_idx's in the splits.
        data->is_classifier = true;
        data->free_train_data();
        ok = true;
    }
    else
        clear();

    __END__;

    return ok;
}

bool CvBoost::train( CvMLData* _data,
             CvBoostParams _params,
             bool update )
{
    bool result = false;

    CV_FUNCNAME( "CvBoost::train" );

    __BEGIN__;

    const CvMat* values = _data->get_values();
    const CvMat* response = _data->get_responses();
    const CvMat* missing = _data->get_missing();
    const CvMat* var_types = _data->get_var_types();
    const CvMat* train_sidx = _data->get_train_sample_idx();
    const CvMat* var_idx = _data->get_var_idx();

    CV_CALL( result = train( values, CV_ROW_SAMPLE, response, var_idx,
        train_sidx, var_types, missing, _params, update ) );

    __END__;

    return result;
}

void
CvBoost::update_weights( CvBoostTree* tree )
{
    CV_FUNCNAME( "CvBoost::update_weights" );

    __BEGIN__;

    int i, n = data->sample_count;
    double sumw = 0.;
    int step = 0;
    float* fdata = 0;
    int *sample_idx_buf;
    const int* sample_idx = 0;
    cv::AutoBuffer<uchar> inn_buf;
    size_t _buf_size = (params.boost_type == LOGIT) || (params.boost_type == GENTLE) ? data->sample_count*sizeof(int) : 0;
    if( !tree )
        _buf_size += n*sizeof(int);
    else
    {
        if( have_subsample )
            _buf_size += data->buf->cols*(sizeof(float)+sizeof(uchar));
    }
    inn_buf.allocate(_buf_size);
    uchar* cur_buf_pos = (uchar*)inn_buf;

    if ( (params.boost_type == LOGIT) || (params.boost_type == GENTLE) )
    {
        step = CV_IS_MAT_CONT(data->responses_copy->type) ?
            1 : data->responses_copy->step / CV_ELEM_SIZE(data->responses_copy->type);
        fdata = data->responses_copy->data.fl;
        sample_idx_buf = (int*)cur_buf_pos;
        cur_buf_pos = (uchar*)(sample_idx_buf + data->sample_count);
        sample_idx = data->get_sample_indices( data->data_root, sample_idx_buf );
    }
    CvMat* dtree_data_buf = data->buf;
    if( !tree ) // before training the first tree, initialize weights and other parameters
    {
        int* class_labels_buf = (int*)cur_buf_pos;
        cur_buf_pos = (uchar*)(class_labels_buf + n);
        const int* class_labels = data->get_class_labels(data->data_root, class_labels_buf);
        // in case of logitboost and gentle adaboost each weak tree is a regression tree,
        // so we need to convert class labels to floating-point values

        double w0 = 1./n;
        double p[2] = { 1, 1 };

        cvReleaseMat( &orig_response );
        cvReleaseMat( &sum_response );
        cvReleaseMat( &weak_eval );
        cvReleaseMat( &subsample_mask );
        cvReleaseMat( &weights );
        cvReleaseMat( &subtree_weights );

        CV_CALL( orig_response = cvCreateMat( 1, n, CV_32S ));
        CV_CALL( weak_eval = cvCreateMat( 1, n, CV_64F ));
        CV_CALL( subsample_mask = cvCreateMat( 1, n, CV_8U ));
        CV_CALL( weights = cvCreateMat( 1, n, CV_64F ));
        CV_CALL( subtree_weights = cvCreateMat( 1, n + 2, CV_64F ));

        if( data->have_priors )
        {
            // compute weight scale for each class from their prior probabilities
            int c1 = 0;
            for( i = 0; i < n; i++ )
                c1 += class_labels[i];
            p[0] = data->priors->data.db[0]*(c1 < n ? 1./(n - c1) : 0.);
            p[1] = data->priors->data.db[1]*(c1 > 0 ? 1./c1 : 0.);
            p[0] /= p[0] + p[1];
            p[1] = 1. - p[0];
        }

        if (data->is_buf_16u)
        {
            unsigned short* labels = (unsigned short*)(dtree_data_buf->data.s + data->data_root->buf_idx*dtree_data_buf->cols +
                data->data_root->offset + (data->work_var_count-1)*data->sample_count);
            for( i = 0; i < n; i++ )
            {
                // save original categorical responses {0,1}, convert them to {-1,1}
                orig_response->data.i[i] = class_labels[i]*2 - 1;
                // make all the samples active at start.
                // later, in trim_weights() deactivate/reactive again some, if need
                subsample_mask->data.ptr[i] = (uchar)1;
                // make all the initial weights the same.
                weights->data.db[i] = w0*p[class_labels[i]];
                // set the labels to find (from within weak tree learning proc)
                // the particular sample weight, and where to store the response.
                labels[i] = (unsigned short)i;
            }
        }
        else
        {
            int* labels = dtree_data_buf->data.i + data->data_root->buf_idx*dtree_data_buf->cols +
                data->data_root->offset + (data->work_var_count-1)*data->sample_count;

            for( i = 0; i < n; i++ )
            {
                // save original categorical responses {0,1}, convert them to {-1,1}
                orig_response->data.i[i] = class_labels[i]*2 - 1;
                // make all the samples active at start.
                // later, in trim_weights() deactivate/reactive again some, if need
                subsample_mask->data.ptr[i] = (uchar)1;
                // make all the initial weights the same.
                weights->data.db[i] = w0*p[class_labels[i]];
                // set the labels to find (from within weak tree learning proc)
                // the particular sample weight, and where to store the response.
                labels[i] = i;
            }
        }

        if( params.boost_type == LOGIT )
        {
            CV_CALL( sum_response = cvCreateMat( 1, n, CV_64F ));

            for( i = 0; i < n; i++ )
            {
                sum_response->data.db[i] = 0;
                fdata[sample_idx[i]*step] = orig_response->data.i[i] > 0 ? 2.f : -2.f;
            }

            // in case of logitboost each weak tree is a regression tree.
            // the target function values are recalculated for each of the trees
            data->is_classifier = false;
        }
        else if( params.boost_type == GENTLE )
        {
            for( i = 0; i < n; i++ )
                fdata[sample_idx[i]*step] = (float)orig_response->data.i[i];

            data->is_classifier = false;
        }
    }
    else
    {
        // at this moment, for all the samples that participated in the training of the most
        // recent weak classifier we know the responses. For other samples we need to compute them
        if( have_subsample )
        {
            float* values = (float*)cur_buf_pos;
            cur_buf_pos = (uchar*)(values + data->buf->cols);
            uchar* missing = cur_buf_pos;
            cur_buf_pos = missing + data->buf->step;
            CvMat _sample, _mask;

            // invert the subsample mask
            cvXorS( subsample_mask, cvScalar(1.), subsample_mask );
            data->get_vectors( subsample_mask, values, missing, 0 );

            _sample = cvMat( 1, data->var_count, CV_32F );
            _mask = cvMat( 1, data->var_count, CV_8U );

            // run tree through all the non-processed samples
            for( i = 0; i < n; i++ )
                if( subsample_mask->data.ptr[i] )
                {
                    _sample.data.fl = values;
                    _mask.data.ptr = missing;
                    values += _sample.cols;
                    missing += _mask.cols;
                    weak_eval->data.db[i] = tree->predict( &_sample, &_mask, true )->value;
                }
        }

        // now update weights and other parameters for each type of boosting
        if( params.boost_type == DISCRETE )
        {
            // Discrete AdaBoost:
            //   weak_eval[i] (=f(x_i)) is in {-1,1}
            //   err = sum(w_i*(f(x_i) != y_i))/sum(w_i)
            //   C = log((1-err)/err)
            //   w_i *= exp(C*(f(x_i) != y_i))

            double C, err = 0.;
            double scale[] = { 1., 0. };

            for( i = 0; i < n; i++ )
            {
                double w = weights->data.db[i];
                sumw += w;
                err += w*(weak_eval->data.db[i] != orig_response->data.i[i]);
            }

            if( sumw != 0 )
                err /= sumw;
            C = err = -log_ratio( err );
            scale[1] = exp(err);

            sumw = 0;
            for( i = 0; i < n; i++ )
            {
                double w = weights->data.db[i]*
                    scale[weak_eval->data.db[i] != orig_response->data.i[i]];
                sumw += w;
                weights->data.db[i] = w;
            }

            tree->scale( C );
        }
        else if( params.boost_type == REAL )
        {
            // Real AdaBoost:
            //   weak_eval[i] = f(x_i) = 0.5*log(p(x_i)/(1-p(x_i))), p(x_i)=P(y=1|x_i)
            //   w_i *= exp(-y_i*f(x_i))

            for( i = 0; i < n; i++ )
                weak_eval->data.db[i] *= -orig_response->data.i[i];

            cvExp( weak_eval, weak_eval );

            for( i = 0; i < n; i++ )
            {
                double w = weights->data.db[i]*weak_eval->data.db[i];
                sumw += w;
                weights->data.db[i] = w;
            }
        }
        else if( params.boost_type == LOGIT )
        {
            // LogitBoost:
            //   weak_eval[i] = f(x_i) in [-z_max,z_max]
            //   sum_response = F(x_i).
            //   F(x_i) += 0.5*f(x_i)
            //   p(x_i) = exp(F(x_i))/(exp(F(x_i)) + exp(-F(x_i))=1/(1+exp(-2*F(x_i)))
            //   reuse weak_eval: weak_eval[i] <- p(x_i)
            //   w_i = p(x_i)*1(1 - p(x_i))
            //   z_i = ((y_i+1)/2 - p(x_i))/(p(x_i)*(1 - p(x_i)))
            //   store z_i to the data->data_root as the new target responses

            const double lb_weight_thresh = FLT_EPSILON;
            const double lb_z_max = 10.;
            /*float* responses_buf = data->get_resp_float_buf();
            const float* responses = 0;
            data->get_ord_responses(data->data_root, responses_buf, &responses);*/

            /*if( weak->total == 7 )
                putchar('*');*/

            for( i = 0; i < n; i++ )
            {
                double s = sum_response->data.db[i] + 0.5*weak_eval->data.db[i];
                sum_response->data.db[i] = s;
                weak_eval->data.db[i] = -2*s;
            }

            cvExp( weak_eval, weak_eval );

            for( i = 0; i < n; i++ )
            {
                double p = 1./(1. + weak_eval->data.db[i]);
                double w = p*(1 - p), z;
                w = MAX( w, lb_weight_thresh );
                weights->data.db[i] = w;
                sumw += w;
                if( orig_response->data.i[i] > 0 )
                {
                    z = 1./p;
                    fdata[sample_idx[i]*step] = (float)MIN(z, lb_z_max);
                }
                else
                {
                    z = 1./(1-p);
                    fdata[sample_idx[i]*step] = (float)-MIN(z, lb_z_max);
                }
            }
        }
        else
        {
            // Gentle AdaBoost:
            //   weak_eval[i] = f(x_i) in [-1,1]
            //   w_i *= exp(-y_i*f(x_i))
            assert( params.boost_type == GENTLE );

            for( i = 0; i < n; i++ )
                weak_eval->data.db[i] *= -orig_response->data.i[i];

            cvExp( weak_eval, weak_eval );

            for( i = 0; i < n; i++ )
            {
                double w = weights->data.db[i] * weak_eval->data.db[i];
                weights->data.db[i] = w;
                sumw += w;
            }
        }
    }

    // renormalize weights
    if( sumw > FLT_EPSILON )
    {
        sumw = 1./sumw;
        for( i = 0; i < n; ++i )
            weights->data.db[i] *= sumw;
    }

    __END__;
}


static CV_IMPLEMENT_QSORT_EX( icvSort_64f, double, CV_LT, int )


void
CvBoost::trim_weights()
{
    //CV_FUNCNAME( "CvBoost::trim_weights" );

    __BEGIN__;

    int i, count = data->sample_count, nz_count = 0;
    double sum, threshold;

    if( params.weight_trim_rate <= 0. || params.weight_trim_rate >= 1. )
        EXIT;

    // use weak_eval as temporary buffer for sorted weights
    cvCopy( weights, weak_eval );

    icvSort_64f( weak_eval->data.db, count, 0 );

    // as weight trimming occurs immediately after updating the weights,
    // where they are renormalized, we assume that the weight sum = 1.
    sum = 1. - params.weight_trim_rate;

    for( i = 0; i < count; i++ )
    {
        double w = weak_eval->data.db[i];
        if( sum <= 0 )
            break;
        sum -= w;
    }

    threshold = i < count ? weak_eval->data.db[i] : DBL_MAX;

    for( i = 0; i < count; i++ )
    {
        double w = weights->data.db[i];
        int f = w >= threshold;
        subsample_mask->data.ptr[i] = (uchar)f;
        nz_count += f;
    }

    have_subsample = nz_count < count;

    __END__;
}


const CvMat*
CvBoost::get_active_vars( bool absolute_idx )
{
    CvMat* mask = 0;
    CvMat* inv_map = 0;
    CvMat* result = 0;

    CV_FUNCNAME( "CvBoost::get_active_vars" );

    __BEGIN__;

    if( !weak )
        CV_ERROR( CV_StsError, "The boosted tree ensemble has not been trained yet" );

    if( !active_vars || !active_vars_abs )
    {
        CvSeqReader reader;
        int i, j, nactive_vars;
        CvBoostTree* wtree;
        const CvDTreeNode* node;

        assert(!active_vars && !active_vars_abs);
        mask = cvCreateMat( 1, data->var_count, CV_8U );
        inv_map = cvCreateMat( 1, data->var_count, CV_32S );
        cvZero( mask );
        cvSet( inv_map, cvScalar(-1) );

        // first pass: compute the mask of used variables
        cvStartReadSeq( weak, &reader );
        for( i = 0; i < weak->total; i++ )
        {
            CV_READ_SEQ_ELEM(wtree, reader);

            node = wtree->get_root();
            assert( node != 0 );
            for(;;)
            {
                const CvDTreeNode* parent;
                for(;;)
                {
                    CvDTreeSplit* split = node->split;
                    for( ; split != 0; split = split->next )
                        mask->data.ptr[split->var_idx] = 1;
                    if( !node->left )
                        break;
                    node = node->left;
                }

                for( parent = node->parent; parent && parent->right == node;
                    node = parent, parent = parent->parent )
                    ;

                if( !parent )
                    break;

                node = parent->right;
            }
        }

        nactive_vars = cvCountNonZero(mask);

        //if ( nactive_vars > 0 )
        {
            active_vars = cvCreateMat( 1, nactive_vars, CV_32S );
            active_vars_abs = cvCreateMat( 1, nactive_vars, CV_32S );

            have_active_cat_vars = false;

            for( i = j = 0; i < data->var_count; i++ )
            {
                if( mask->data.ptr[i] )
                {
                    active_vars->data.i[j] = i;
                    active_vars_abs->data.i[j] = data->var_idx ? data->var_idx->data.i[i] : i;
                    inv_map->data.i[i] = j;
                    if( data->var_type->data.i[i] >= 0 )
                        have_active_cat_vars = true;
                    j++;
                }
            }


            // second pass: now compute the condensed indices
            cvStartReadSeq( weak, &reader );
            for( i = 0; i < weak->total; i++ )
            {
                CV_READ_SEQ_ELEM(wtree, reader);
                node = wtree->get_root();
                for(;;)
                {
                    const CvDTreeNode* parent;
                    for(;;)
                    {
                        CvDTreeSplit* split = node->split;
                        for( ; split != 0; split = split->next )
                        {
                            split->condensed_idx = inv_map->data.i[split->var_idx];
                            assert( split->condensed_idx >= 0 );
                        }

                        if( !node->left )
                            break;
                        node = node->left;
                    }

                    for( parent = node->parent; parent && parent->right == node;
                        node = parent, parent = parent->parent )
                        ;

                    if( !parent )
                        break;

                    node = parent->right;
                }
            }
        }
    }

    result = absolute_idx ? active_vars_abs : active_vars;

    __END__;

    cvReleaseMat( &mask );
    cvReleaseMat( &inv_map );

    return result;
}


float
CvBoost::predict( const CvMat* _sample, const CvMat* _missing,
                  CvMat* weak_responses, CvSlice slice,
                  bool raw_mode, bool return_sum ) const
{
    float value = -FLT_MAX;

    CvMat sample, missing;
    CvSeqReader reader;
    double sum = 0;
    int wstep = 0;
    const float* sample_data;

    if( !weak )
        CV_Error( CV_StsError, "The boosted tree ensemble has not been trained yet" );

    if( !CV_IS_MAT(_sample) || CV_MAT_TYPE(_sample->type) != CV_32FC1 ||
        (_sample->cols != 1 && _sample->rows != 1) ||
        (_sample->cols + _sample->rows - 1 != data->var_all && !raw_mode) ||
        (active_vars && _sample->cols + _sample->rows - 1 != active_vars->cols && raw_mode) )
            CV_Error( CV_StsBadArg,
        "the input sample must be 1d floating-point vector with the same "
        "number of elements as the total number of variables or "
        "as the number of variables used for training" );

    if( _missing )
    {
        if( !CV_IS_MAT(_missing) || !CV_IS_MASK_ARR(_missing) ||
            !CV_ARE_SIZES_EQ(_missing, _sample) )
            CV_Error( CV_StsBadArg,
            "the missing data mask must be 8-bit vector of the same size as input sample" );
    }

    int i, weak_count = cvSliceLength( slice, weak );
    if( weak_count >= weak->total )
    {
        weak_count = weak->total;
        slice.start_index = 0;
    }

    if( weak_responses )
    {
        if( !CV_IS_MAT(weak_responses) ||
            CV_MAT_TYPE(weak_responses->type) != CV_32FC1 ||
            (weak_responses->cols != 1 && weak_responses->rows != 1) ||
            weak_responses->cols + weak_responses->rows - 1 != weak_count )
            CV_Error( CV_StsBadArg,
            "The output matrix of weak classifier responses must be valid "
            "floating-point vector of the same number of components as the length of input slice" );
        wstep = CV_IS_MAT_CONT(weak_responses->type) ? 1 : weak_responses->step/sizeof(float);
    }

    int var_count = active_vars->cols;
    const int* vtype = data->var_type->data.i;
    const int* cmap = data->cat_map->data.i;
    const int* cofs = data->cat_ofs->data.i;

    // if need, preprocess the input vector
    if( !raw_mode )
    {
        int step, mstep = 0;
        const float* src_sample;
        const uchar* src_mask = 0;
        float* dst_sample;
        uchar* dst_mask;
        const int* vidx = active_vars->data.i;
        const int* vidx_abs = active_vars_abs->data.i;
        bool have_mask = _missing != 0;

        cv::AutoBuffer<float> buf(var_count + (var_count+3)/4);
        dst_sample = &buf[0];
        dst_mask = (uchar*)&buf[var_count];

        src_sample = _sample->data.fl;
        step = CV_IS_MAT_CONT(_sample->type) ? 1 : _sample->step/sizeof(src_sample[0]);

        if( _missing )
        {
            src_mask = _missing->data.ptr;
            mstep = CV_IS_MAT_CONT(_missing->type) ? 1 : _missing->step;
        }

        for( i = 0; i < var_count; i++ )
        {
            int idx = vidx[i], idx_abs = vidx_abs[i];
            float val = src_sample[idx_abs*step];
            int ci = vtype[idx];
            uchar m = src_mask ? src_mask[idx_abs*mstep] : (uchar)0;

            if( ci >= 0 )
            {
                int a = cofs[ci], b = (ci+1 >= data->cat_ofs->cols) ? data->cat_map->cols : cofs[ci+1],
                    c = a;
                int ival = cvRound(val);
                if ( (ival != val) && (!m) )
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

                if( c < 0 || ival != cmap[c] )
                {
                    m = 1;
                    have_mask = true;
                }
                else
                {
                    val = (float)(c - cofs[ci]);
                }
            }

            dst_sample[i] = val;
            dst_mask[i] = m;
        }

        sample = cvMat( 1, var_count, CV_32F, dst_sample );
        _sample = &sample;

        if( have_mask )
        {
            missing = cvMat( 1, var_count, CV_8UC1, dst_mask );
            _missing = &missing;
        }
    }
    else
    {
        if( !CV_IS_MAT_CONT(_sample->type & (_missing ? _missing->type : -1)) )
            CV_Error( CV_StsBadArg, "In raw mode the input vectors must be continuous" );
    }

    cvStartReadSeq( weak, &reader );
    cvSetSeqReaderPos( &reader, slice.start_index );

    sample_data = _sample->data.fl;

    if( !have_active_cat_vars && !_missing && !weak_responses )
    {
        for( i = 0; i < weak_count; i++ )
        {
            CvBoostTree* wtree;
            const CvDTreeNode* node;
            CV_READ_SEQ_ELEM( wtree, reader );

            node = wtree->get_root();
            while( node->left )
            {
                CvDTreeSplit* split = node->split;
                int vi = split->condensed_idx;
                float val = sample_data[vi];
                int dir = val <= split->ord.c ? -1 : 1;
                if( split->inversed )
                    dir = -dir;
                node = dir < 0 ? node->left : node->right;
            }
            sum += node->value;
        }
    }
    else
    {
        const int* avars = active_vars->data.i;
        const uchar* m = _missing ? _missing->data.ptr : 0;

        // full-featured version
        for( i = 0; i < weak_count; i++ )
        {
            CvBoostTree* wtree;
            const CvDTreeNode* node;
            CV_READ_SEQ_ELEM( wtree, reader );

            node = wtree->get_root();
            while( node->left )
            {
                const CvDTreeSplit* split = node->split;
                int dir = 0;
                for( ; !dir && split != 0; split = split->next )
                {
                    int vi = split->condensed_idx;
                    int ci = vtype[avars[vi]];
                    float val = sample_data[vi];
                    if( m && m[vi] )
                        continue;
                    if( ci < 0 ) // ordered
                        dir = val <= split->ord.c ? -1 : 1;
                    else // categorical
                    {
                        int c = cvRound(val);
                        dir = CV_DTREE_CAT_DIR(c, split->subset);
                    }
                    if( split->inversed )
                        dir = -dir;
                }

                if( !dir )
                {
                    int diff = node->right->sample_count - node->left->sample_count;
                    dir = diff < 0 ? -1 : 1;
                }
                node = dir < 0 ? node->left : node->right;
            }
            if( weak_responses )
                weak_responses->data.fl[i*wstep] = (float)node->value;
            sum += node->value;
        }
    }

    if( return_sum )
        value = (float)sum;
    else
    {
        int cls_idx = sum >= 0;
        if( raw_mode )
            value = (float)cls_idx;
        else
            value = (float)cmap[cofs[vtype[data->var_count]] + cls_idx];
    }

    return value;
}

float CvBoost::calc_error( CvMLData* _data, int type, std::vector<float> *resp )
{
    float err = 0;
    const CvMat* values = _data->get_values();
    const CvMat* response = _data->get_responses();
    const CvMat* missing = _data->get_missing();
    const CvMat* sample_idx = (type == CV_TEST_ERROR) ? _data->get_test_sample_idx() : _data->get_train_sample_idx();
    const CvMat* var_types = _data->get_var_types();
    int* sidx = sample_idx ? sample_idx->data.i : 0;
    int r_step = CV_IS_MAT_CONT(response->type) ?
                1 : response->step / CV_ELEM_SIZE(response->type);
    bool is_classifier = var_types->data.ptr[var_types->cols-1] == CV_VAR_CATEGORICAL;
    int sample_count = sample_idx ? sample_idx->cols : 0;
    sample_count = (type == CV_TRAIN_ERROR && sample_count == 0) ? values->rows : sample_count;
    float* pred_resp = 0;
    if( resp && (sample_count > 0) )
    {
        resp->resize( sample_count );
        pred_resp = &((*resp)[0]);
    }
    if ( is_classifier )
    {
        for( int i = 0; i < sample_count; i++ )
        {
            CvMat sample, miss;
            int si = sidx ? sidx[i] : i;
            cvGetRow( values, &sample, si );
            if( missing )
                cvGetRow( missing, &miss, si );
            float r = (float)predict( &sample, missing ? &miss : 0 );
            if( pred_resp )
                pred_resp[i] = r;
            int d = fabs((double)r - response->data.fl[si*r_step]) <= FLT_EPSILON ? 0 : 1;
            err += d;
        }
        err = sample_count ? err / (float)sample_count * 100 : -FLT_MAX;
    }
    else
    {
        for( int i = 0; i < sample_count; i++ )
        {
            CvMat sample, miss;
            int si = sidx ? sidx[i] : i;
            cvGetRow( values, &sample, si );
            if( missing )
                cvGetRow( missing, &miss, si );
            float r = (float)predict( &sample, missing ? &miss : 0 );
            if( pred_resp )
                pred_resp[i] = r;
            float d = r - response->data.fl[si*r_step];
            err += d*d;
        }
        err = sample_count ? err / (float)sample_count : -FLT_MAX;
    }
    return err;
}

void CvBoost::write_params( CvFileStorage* fs ) const
{
    const char* boost_type_str =
        params.boost_type == DISCRETE ? "DiscreteAdaboost" :
        params.boost_type == REAL ? "RealAdaboost" :
        params.boost_type == LOGIT ? "LogitBoost" :
        params.boost_type == GENTLE ? "GentleAdaboost" : 0;

    const char* split_crit_str =
        params.split_criteria == DEFAULT ? "Default" :
        params.split_criteria == GINI ? "Gini" :
        params.boost_type == MISCLASS ? "Misclassification" :
        params.boost_type == SQERR ? "SquaredErr" : 0;

    if( boost_type_str )
        cvWriteString( fs, "boosting_type", boost_type_str );
    else
        cvWriteInt( fs, "boosting_type", params.boost_type );

    if( split_crit_str )
        cvWriteString( fs, "splitting_criteria", split_crit_str );
    else
        cvWriteInt( fs, "splitting_criteria", params.split_criteria );

    cvWriteInt( fs, "ntrees", weak->total );
    cvWriteReal( fs, "weight_trimming_rate", params.weight_trim_rate );

    data->write_params( fs );
}


void CvBoost::read_params( CvFileStorage* fs, CvFileNode* fnode )
{
    CV_FUNCNAME( "CvBoost::read_params" );

    __BEGIN__;

    CvFileNode* temp;

    if( !fnode || !CV_NODE_IS_MAP(fnode->tag) )
        return;

    data = new CvDTreeTrainData();
    CV_CALL( data->read_params(fs, fnode));
    data->shared = true;

    params.max_depth = data->params.max_depth;
    params.min_sample_count = data->params.min_sample_count;
    params.max_categories = data->params.max_categories;
    params.priors = data->params.priors;
    params.regression_accuracy = data->params.regression_accuracy;
    params.use_surrogates = data->params.use_surrogates;

    temp = cvGetFileNodeByName( fs, fnode, "boosting_type" );
    if( !temp )
        return;

    if( temp && CV_NODE_IS_STRING(temp->tag) )
    {
        const char* boost_type_str = cvReadString( temp, "" );
        params.boost_type = strcmp( boost_type_str, "DiscreteAdaboost" ) == 0 ? DISCRETE :
                            strcmp( boost_type_str, "RealAdaboost" ) == 0 ? REAL :
                            strcmp( boost_type_str, "LogitBoost" ) == 0 ? LOGIT :
                            strcmp( boost_type_str, "GentleAdaboost" ) == 0 ? GENTLE : -1;
    }
    else
        params.boost_type = cvReadInt( temp, -1 );

    if( params.boost_type < DISCRETE || params.boost_type > GENTLE )
        CV_ERROR( CV_StsBadArg, "Unknown boosting type" );

    temp = cvGetFileNodeByName( fs, fnode, "splitting_criteria" );
    if( temp && CV_NODE_IS_STRING(temp->tag) )
    {
        const char* split_crit_str = cvReadString( temp, "" );
        params.split_criteria = strcmp( split_crit_str, "Default" ) == 0 ? DEFAULT :
                                strcmp( split_crit_str, "Gini" ) == 0 ? GINI :
                                strcmp( split_crit_str, "Misclassification" ) == 0 ? MISCLASS :
                                strcmp( split_crit_str, "SquaredErr" ) == 0 ? SQERR : -1;
    }
    else
        params.split_criteria = cvReadInt( temp, -1 );

    if( params.split_criteria < DEFAULT || params.boost_type > SQERR )
        CV_ERROR( CV_StsBadArg, "Unknown boosting type" );

    params.weak_count = cvReadIntByName( fs, fnode, "ntrees" );
    params.weight_trim_rate = cvReadRealByName( fs, fnode, "weight_trimming_rate", 0. );

    __END__;
}



void
CvBoost::read( CvFileStorage* fs, CvFileNode* node )
{
    CV_FUNCNAME( "CvBoost::read" );

    __BEGIN__;

    CvSeqReader reader;
    CvFileNode* trees_fnode;
    CvMemStorage* storage;
    int i, ntrees;

    clear();
    read_params( fs, node );

    if( !data )
        EXIT;

    trees_fnode = cvGetFileNodeByName( fs, node, "trees" );
    if( !trees_fnode || !CV_NODE_IS_SEQ(trees_fnode->tag) )
        CV_ERROR( CV_StsParseError, "<trees> tag is missing" );

    cvStartReadSeq( trees_fnode->data.seq, &reader );
    ntrees = trees_fnode->data.seq->total;

    if( ntrees != params.weak_count )
        CV_ERROR( CV_StsUnmatchedSizes,
        "The number of trees stored does not match <ntrees> tag value" );

    CV_CALL( storage = cvCreateMemStorage() );
    weak = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvBoostTree*), storage );

    for( i = 0; i < ntrees; i++ )
    {
        CvBoostTree* tree = new CvBoostTree();
        CV_CALL(tree->read( fs, (CvFileNode*)reader.ptr, this, data ));
        CV_NEXT_SEQ_ELEM( reader.seq->elem_size, reader );
        cvSeqPush( weak, &tree );
    }
    get_active_vars();

    __END__;
}


void
CvBoost::write( CvFileStorage* fs, const char* name ) const
{
    CV_FUNCNAME( "CvBoost::write" );

    __BEGIN__;

    CvSeqReader reader;
    int i;

    cvStartWriteStruct( fs, name, CV_NODE_MAP, CV_TYPE_NAME_ML_BOOSTING );

    if( !weak )
        CV_ERROR( CV_StsBadArg, "The classifier has not been trained yet" );

    write_params( fs );
    cvStartWriteStruct( fs, "trees", CV_NODE_SEQ );

    cvStartReadSeq( weak, &reader );

    for( i = 0; i < weak->total; i++ )
    {
        CvBoostTree* tree;
        CV_READ_SEQ_ELEM( tree, reader );
        cvStartWriteStruct( fs, 0, CV_NODE_MAP );
        tree->write( fs );
        cvEndWriteStruct( fs );
    }

    cvEndWriteStruct( fs );
    cvEndWriteStruct( fs );

    __END__;
}


CvMat*
CvBoost::get_weights()
{
    return weights;
}


CvMat*
CvBoost::get_subtree_weights()
{
    return subtree_weights;
}


CvMat*
CvBoost::get_weak_response()
{
    return weak_eval;
}


const CvBoostParams&
CvBoost::get_params() const
{
    return params;
}

CvSeq* CvBoost::get_weak_predictors()
{
    return weak;
}

const CvDTreeTrainData* CvBoost::get_data() const
{
    return data;
}

using namespace cv;

CvBoost::CvBoost( const Mat& _train_data, int _tflag,
               const Mat& _responses, const Mat& _var_idx,
               const Mat& _sample_idx, const Mat& _var_type,
               const Mat& _missing_mask,
               CvBoostParams _params )
{
    weak = 0;
    data = 0;
    default_model_name = "my_boost_tree";
    active_vars = active_vars_abs = orig_response = sum_response = weak_eval =
        subsample_mask = weights = subtree_weights = 0;

    train( _train_data, _tflag, _responses, _var_idx, _sample_idx,
          _var_type, _missing_mask, _params );
}


bool
CvBoost::train( const Mat& _train_data, int _tflag,
               const Mat& _responses, const Mat& _var_idx,
               const Mat& _sample_idx, const Mat& _var_type,
               const Mat& _missing_mask,
               CvBoostParams _params, bool _update )
{
    CvMat tdata = _train_data, responses = _responses, vidx = _var_idx,
        sidx = _sample_idx, vtype = _var_type, mmask = _missing_mask;
    return train(&tdata, _tflag, &responses, vidx.data.ptr ? &vidx : 0,
          sidx.data.ptr ? &sidx : 0, vtype.data.ptr ? &vtype : 0,
          mmask.data.ptr ? &mmask : 0, _params, _update);
}

float
CvBoost::predict( const Mat& _sample, const Mat& _missing,
                  const Range& slice, bool raw_mode, bool return_sum ) const
{
    CvMat sample = _sample, mmask = _missing;
    /*if( weak_responses )
    {
        int weak_count = cvSliceLength( slice, weak );
        if( weak_count >= weak->total )
        {
            weak_count = weak->total;
            slice.start_index = 0;
        }

        if( !(weak_responses->data && weak_responses->type() == CV_32FC1 &&
              (weak_responses->cols == 1 || weak_responses->rows == 1) &&
              weak_responses->cols + weak_responses->rows - 1 == weak_count) )
            weak_responses->create(weak_count, 1, CV_32FC1);
        pwr = &(wr = *weak_responses);
    }*/
    return predict(&sample, _missing.empty() ? 0 : &mmask, 0,
                   slice == Range::all() ? CV_WHOLE_SEQ : cvSlice(slice.start, slice.end),
                   raw_mode, return_sum);
}

/* End of file. */


