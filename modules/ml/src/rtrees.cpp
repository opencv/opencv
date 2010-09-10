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

CvForestTree::CvForestTree()
{
    forest = NULL;
}


CvForestTree::~CvForestTree()
{
    clear();
}


bool CvForestTree::train( CvDTreeTrainData* _data,
                          const CvMat* _subsample_idx,
                          CvRTrees* _forest )
{
    clear();
    forest = _forest;

    data = _data;
    data->shared = true;
    return do_train(_subsample_idx);
}


bool
CvForestTree::train( const CvMat*, int, const CvMat*, const CvMat*,
                    const CvMat*, const CvMat*, const CvMat*, CvDTreeParams )
{
    assert(0);
    return false;
}


bool
CvForestTree::train( CvDTreeTrainData*, const CvMat* )
{
    assert(0);
    return false;
}



namespace cv
{

ForestTreeBestSplitFinder::ForestTreeBestSplitFinder( CvForestTree* _tree, CvDTreeNode* _node ) :
    DTreeBestSplitFinder(_tree, _node) {}

ForestTreeBestSplitFinder::ForestTreeBestSplitFinder( const ForestTreeBestSplitFinder& finder, Split spl ) :
    DTreeBestSplitFinder( finder, spl ) {}

void ForestTreeBestSplitFinder::operator()(const BlockedRange& range)
{
    int vi, vi1 = range.begin(), vi2 = range.end();
    int n = node->sample_count;
    CvDTreeTrainData* data = tree->get_data();
    AutoBuffer<uchar> inn_buf(2*n*(sizeof(int) + sizeof(float)));

    CvForestTree* ftree = (CvForestTree*)tree;
    const CvMat* active_var_mask = ftree->forest->get_active_var_mask();

    for( vi = vi1; vi < vi2; vi++ )
    {
        CvDTreeSplit *res;
        int ci = data->var_type->data.i[vi];
        if( node->num_valid[vi] <= 1
            || (active_var_mask && !active_var_mask->data.ptr[vi]) )
            continue;

        if( data->is_classifier )
        {
            if( ci >= 0 )
                res = ftree->find_split_cat_class( node, vi, bestSplit->quality, split, (uchar*)inn_buf );
            else
                res = ftree->find_split_ord_class( node, vi, bestSplit->quality, split, (uchar*)inn_buf );
        }
        else
        {
            if( ci >= 0 )
                res = ftree->find_split_cat_reg( node, vi, bestSplit->quality, split, (uchar*)inn_buf );
            else
                res = ftree->find_split_ord_reg( node, vi, bestSplit->quality, split, (uchar*)inn_buf );
        }

        if( res && bestSplit->quality < split->quality )
                memcpy( (CvDTreeSplit*)bestSplit, (CvDTreeSplit*)split, splitSize );
    }
}
}

CvDTreeSplit* CvForestTree::find_best_split( CvDTreeNode* node )
{
    CvMat* active_var_mask = 0;
    if( forest )
    {
        int var_count;
        CvRNG* rng = forest->get_rng();

        active_var_mask = forest->get_active_var_mask();
        var_count = active_var_mask->cols;

        CV_Assert( var_count == data->var_count );

        for( int vi = 0; vi < var_count; vi++ )
        {
            uchar temp;
            int i1 = cvRandInt(rng) % var_count;
            int i2 = cvRandInt(rng) % var_count;
            CV_SWAP( active_var_mask->data.ptr[i1],
                active_var_mask->data.ptr[i2], temp );
        }
    }

    cv::ForestTreeBestSplitFinder finder( this, node );

    cv::parallel_reduce(cv::BlockedRange(0, data->var_count), finder);

    CvDTreeSplit *bestSplit = 0;
    if( finder.bestSplit->quality > 0 )
    {
        bestSplit = data->new_split_cat( 0, -1.0f );
        memcpy( bestSplit, finder.bestSplit, finder.splitSize );
    }

    return bestSplit;
}

void CvForestTree::read( CvFileStorage* fs, CvFileNode* fnode, CvRTrees* _forest, CvDTreeTrainData* _data )
{
    CvDTree::read( fs, fnode, _data );
    forest = _forest;
}


void CvForestTree::read( CvFileStorage*, CvFileNode* )
{
    assert(0);
}

void CvForestTree::read( CvFileStorage* _fs, CvFileNode* _node,
                         CvDTreeTrainData* _data )
{
    CvDTree::read( _fs, _node, _data );
}


//////////////////////////////////////////////////////////////////////////////////////////
//                                  Random trees                                        //
//////////////////////////////////////////////////////////////////////////////////////////

CvRTrees::CvRTrees()
{
    nclasses         = 0;
    oob_error        = 0;
    ntrees           = 0;
    trees            = NULL;
    data             = NULL;
    active_var_mask  = NULL;
    var_importance   = NULL;
    rng = cvRNG(0xffffffff);
    default_model_name = "my_random_trees";
}


void CvRTrees::clear()
{
    int k;
    for( k = 0; k < ntrees; k++ )
        delete trees[k];
    cvFree( &trees );

    delete data;
    data = 0;

    cvReleaseMat( &active_var_mask );
    cvReleaseMat( &var_importance );
    ntrees = 0;
}


CvRTrees::~CvRTrees()
{
    clear();
}


CvMat* CvRTrees::get_active_var_mask()
{
    return active_var_mask;
}


CvRNG* CvRTrees::get_rng()
{
    return &rng;
}

bool CvRTrees::train( const CvMat* _train_data, int _tflag,
                        const CvMat* _responses, const CvMat* _var_idx,
                        const CvMat* _sample_idx, const CvMat* _var_type,
                        const CvMat* _missing_mask, CvRTParams params )
{
    clear();

    CvDTreeParams tree_params( params.max_depth, params.min_sample_count,
        params.regression_accuracy, params.use_surrogates, params.max_categories,
        params.cv_folds, params.use_1se_rule, false, params.priors );

    data = new CvDTreeTrainData();
    data->set_data( _train_data, _tflag, _responses, _var_idx,
        _sample_idx, _var_type, _missing_mask, tree_params, true);

    int var_count = data->var_count;
    if( params.nactive_vars > var_count )
        params.nactive_vars = var_count;
    else if( params.nactive_vars == 0 )
        params.nactive_vars = (int)sqrt((double)var_count);
    else if( params.nactive_vars < 0 )
        CV_Error( CV_StsBadArg, "<nactive_vars> must be non-negative" );

    // Create mask of active variables at the tree nodes
    active_var_mask = cvCreateMat( 1, var_count, CV_8UC1 );
    if( params.calc_var_importance )
    {
        var_importance  = cvCreateMat( 1, var_count, CV_32FC1 );
        cvZero(var_importance);
    }
    { // initialize active variables mask
        CvMat submask1, submask2;
        CV_Assert( (active_var_mask->cols >= 1) && (params.nactive_vars > 0) && (params.nactive_vars <= active_var_mask->cols) );
        cvGetCols( active_var_mask, &submask1, 0, params.nactive_vars );
        cvSet( &submask1, cvScalar(1) );
        if( params.nactive_vars < active_var_mask->cols )
        {
            cvGetCols( active_var_mask, &submask2, params.nactive_vars, var_count );
            cvZero( &submask2 );
        }
    }

    return grow_forest( params.term_crit );
}

bool CvRTrees::train( CvMLData* data, CvRTParams params )
{
    const CvMat* values = data->get_values();
    const CvMat* response = data->get_responses();
    const CvMat* missing = data->get_missing();
    const CvMat* var_types = data->get_var_types();
    const CvMat* train_sidx = data->get_train_sample_idx();
    const CvMat* var_idx = data->get_var_idx();

    return train( values, CV_ROW_SAMPLE, response, var_idx,
                  train_sidx, var_types, missing, params );
}

bool CvRTrees::grow_forest( const CvTermCriteria term_crit )
{
    CvMat* sample_idx_mask_for_tree = 0;
    CvMat* sample_idx_for_tree      = 0;

    const int max_ntrees = term_crit.max_iter;
    const double max_oob_err = term_crit.epsilon;

    const int dims = data->var_count;
    float maximal_response = 0;

    CvMat* oob_sample_votes	   = 0;
    CvMat* oob_responses       = 0;

    float* oob_samples_perm_ptr= 0;

    float* samples_ptr     = 0;
    uchar* missing_ptr     = 0;
    float* true_resp_ptr   = 0;
    bool is_oob_or_vimportance = (max_oob_err > 0 && term_crit.type != CV_TERMCRIT_ITER) || var_importance;

    // oob_predictions_sum[i] = sum of predicted values for the i-th sample
    // oob_num_of_predictions[i] = number of summands
    //                            (number of predictions for the i-th sample)
    // initialize these variable to avoid warning C4701
    CvMat oob_predictions_sum = cvMat( 1, 1, CV_32FC1 );
    CvMat oob_num_of_predictions = cvMat( 1, 1, CV_32FC1 );
     
    nsamples = data->sample_count;
    nclasses = data->get_num_classes();

    if ( is_oob_or_vimportance )
    {
        if( data->is_classifier )
        {
            oob_sample_votes = cvCreateMat( nsamples, nclasses, CV_32SC1 );
            cvZero(oob_sample_votes);
        }
        else
        {
            // oob_responses[0,i] = oob_predictions_sum[i]
            //    = sum of predicted values for the i-th sample
            // oob_responses[1,i] = oob_num_of_predictions[i]
            //    = number of summands (number of predictions for the i-th sample)
            oob_responses = cvCreateMat( 2, nsamples, CV_32FC1 );
            cvZero(oob_responses);
            cvGetRow( oob_responses, &oob_predictions_sum, 0 );
            cvGetRow( oob_responses, &oob_num_of_predictions, 1 );
        }
        
        oob_samples_perm_ptr     = (float*)cvAlloc( sizeof(float)*nsamples*dims );
        samples_ptr              = (float*)cvAlloc( sizeof(float)*nsamples*dims );
        missing_ptr              = (uchar*)cvAlloc( sizeof(uchar)*nsamples*dims );
        true_resp_ptr            = (float*)cvAlloc( sizeof(float)*nsamples );            

        data->get_vectors( 0, samples_ptr, missing_ptr, true_resp_ptr );
        
        double minval, maxval;
        CvMat responses = cvMat(1, nsamples, CV_32FC1, true_resp_ptr);
        cvMinMaxLoc( &responses, &minval, &maxval );
        maximal_response = (float)MAX( MAX( fabs(minval), fabs(maxval) ), 0 );
    }

    trees = (CvForestTree**)cvAlloc( sizeof(trees[0])*max_ntrees );
    memset( trees, 0, sizeof(trees[0])*max_ntrees );

    sample_idx_mask_for_tree = cvCreateMat( 1, nsamples, CV_8UC1 );
    sample_idx_for_tree      = cvCreateMat( 1, nsamples, CV_32SC1 );

    ntrees = 0;
    while( ntrees < max_ntrees )
    {
        int i, oob_samples_count = 0;
        double ncorrect_responses = 0; // used for estimation of variable importance
        CvForestTree* tree = 0;

        cvZero( sample_idx_mask_for_tree );
        for(i = 0; i < nsamples; i++ ) //form sample for creation one tree
        {
            int idx = cvRandInt( &rng ) % nsamples;
            sample_idx_for_tree->data.i[i] = idx;
            sample_idx_mask_for_tree->data.ptr[idx] = 0xFF;
        }

        trees[ntrees] = new CvForestTree();
        tree = trees[ntrees];
        tree->train( data, sample_idx_for_tree, this );

        if ( is_oob_or_vimportance )
        {
            CvMat sample, missing;
            // form array of OOB samples indices and get these samples
            sample   = cvMat( 1, dims, CV_32FC1, samples_ptr );
            missing  = cvMat( 1, dims, CV_8UC1,  missing_ptr );

            oob_error = 0;
            for( i = 0; i < nsamples; i++,
                sample.data.fl += dims, missing.data.ptr += dims )
            {
                CvDTreeNode* predicted_node = 0;
                // check if the sample is OOB
                if( sample_idx_mask_for_tree->data.ptr[i] )
                    continue;

                // predict oob samples
                if( !predicted_node )
                    predicted_node = tree->predict(&sample, &missing, true);

                if( !data->is_classifier ) //regression
                {
                    double avg_resp, resp = predicted_node->value;
                    oob_predictions_sum.data.fl[i] += (float)resp;
                    oob_num_of_predictions.data.fl[i] += 1;

                    // compute oob error
                    avg_resp = oob_predictions_sum.data.fl[i]/oob_num_of_predictions.data.fl[i];
                    avg_resp -= true_resp_ptr[i];
                    oob_error += avg_resp*avg_resp;
                    resp = (resp - true_resp_ptr[i])/maximal_response;
                    ncorrect_responses += exp( -resp*resp );
                }
                else //classification
                {
                    double prdct_resp;
                    CvPoint max_loc;
                    CvMat votes;

                    cvGetRow(oob_sample_votes, &votes, i);
                    votes.data.i[predicted_node->class_idx]++;

                    // compute oob error
                    cvMinMaxLoc( &votes, 0, 0, 0, &max_loc );

                    prdct_resp = data->cat_map->data.i[max_loc.x];
                    oob_error += (fabs(prdct_resp - true_resp_ptr[i]) < FLT_EPSILON) ? 0 : 1;

                    ncorrect_responses += cvRound(predicted_node->value - true_resp_ptr[i]) == 0;
                }
                oob_samples_count++;
            }
            if( oob_samples_count > 0 )
                oob_error /= (double)oob_samples_count;

            // estimate variable importance
            if( var_importance && oob_samples_count > 0 )
            {
                int m;

                memcpy( oob_samples_perm_ptr, samples_ptr, dims*nsamples*sizeof(float));
                for( m = 0; m < dims; m++ )
                {
                    double ncorrect_responses_permuted = 0;
                    // randomly permute values of the m-th variable in the oob samples
                    float* mth_var_ptr = oob_samples_perm_ptr + m;

                    for( i = 0; i < nsamples; i++ )
                    {
                        int i1, i2;
                        float temp;

                        if( sample_idx_mask_for_tree->data.ptr[i] ) //the sample is not OOB
                            continue;
                        i1 = cvRandInt( &rng ) % nsamples;
                        i2 = cvRandInt( &rng ) % nsamples;
                        CV_SWAP( mth_var_ptr[i1*dims], mth_var_ptr[i2*dims], temp );

                        // turn values of (m-1)-th variable, that were permuted
                        // at the previous iteration, untouched
                        if( m > 1 )
                            oob_samples_perm_ptr[i*dims+m-1] = samples_ptr[i*dims+m-1];
                    }

                    // predict "permuted" cases and calculate the number of votes for the
                    // correct class in the variable-m-permuted oob data
                    sample  = cvMat( 1, dims, CV_32FC1, oob_samples_perm_ptr );
                    missing = cvMat( 1, dims, CV_8UC1, missing_ptr );
                    for( i = 0; i < nsamples; i++,
                        sample.data.fl += dims, missing.data.ptr += dims )
                    {
                        double predct_resp, true_resp;

                        if( sample_idx_mask_for_tree->data.ptr[i] ) //the sample is not OOB
                            continue;

                        predct_resp = tree->predict(&sample, &missing, true)->value;
                        true_resp   = true_resp_ptr[i];
                        if( data->is_classifier )
                            ncorrect_responses_permuted += cvRound(true_resp - predct_resp) == 0;
                        else
                        {
                            true_resp = (true_resp - predct_resp)/maximal_response;
                            ncorrect_responses_permuted += exp( -true_resp*true_resp );
                        }
                    }
                    var_importance->data.fl[m] += (float)(ncorrect_responses
                        - ncorrect_responses_permuted);
                }
            }
        }
        ntrees++;
        if( term_crit.type != CV_TERMCRIT_ITER && oob_error < max_oob_err )
            break;
    }

    if( var_importance )
    {
        for ( int vi = 0; vi < var_importance->cols; vi++ )
                var_importance->data.fl[vi] = ( var_importance->data.fl[vi] > 0 ) ?
                    var_importance->data.fl[vi] : 0;
        cvNormalize( var_importance, var_importance, 1., 0, CV_L1 );
    }

    cvFree( &oob_samples_perm_ptr );
    cvFree( &samples_ptr );
    cvFree( &missing_ptr );
    cvFree( &true_resp_ptr );
    
    cvReleaseMat( &sample_idx_mask_for_tree );
    cvReleaseMat( &sample_idx_for_tree );

    cvReleaseMat( &oob_sample_votes );
    cvReleaseMat( &oob_responses );

    return true;
}


const CvMat* CvRTrees::get_var_importance()
{
    return var_importance;
}


float CvRTrees::get_proximity( const CvMat* sample1, const CvMat* sample2,
                              const CvMat* missing1, const CvMat* missing2 ) const
{
    float result = 0;

    for( int i = 0; i < ntrees; i++ )
        result += trees[i]->predict( sample1, missing1 ) ==
        trees[i]->predict( sample2, missing2 ) ?  1 : 0;
    result = result/(float)ntrees;

    return result;
}

float CvRTrees::calc_error( CvMLData* _data, int type , std::vector<float> *resp )
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

float CvRTrees::get_train_error()
{
    float err = -1;

    int sample_count = data->sample_count;
    int var_count = data->var_count;

    float *values_ptr = (float*)cvAlloc( sizeof(float)*sample_count*var_count );
    uchar *missing_ptr = (uchar*)cvAlloc( sizeof(uchar)*sample_count*var_count );
    float *responses_ptr = (float*)cvAlloc( sizeof(float)*sample_count );

    data->get_vectors( 0, values_ptr, missing_ptr, responses_ptr);
    
    if (data->is_classifier)
    {
        int err_count = 0;
        float *vp = values_ptr;
        uchar *mp = missing_ptr;    
        for (int si = 0; si < sample_count; si++, vp += var_count, mp += var_count)
        {
            CvMat sample = cvMat( 1, var_count, CV_32FC1, vp );
            CvMat missing = cvMat( 1, var_count, CV_8UC1,  mp );
            float r = predict( &sample, &missing );
            if (fabs(r - responses_ptr[si]) >= FLT_EPSILON)
                err_count++;
        }
        err = (float)err_count / (float)sample_count;
    }
    else
        CV_Error( CV_StsBadArg, "This method is not supported for regression problems" );
    
    cvFree( &values_ptr );
    cvFree( &missing_ptr );
    cvFree( &responses_ptr ); 

    return err;
}


float CvRTrees::predict( const CvMat* sample, const CvMat* missing ) const
{
    double result = -1;
    int k;

    if( nclasses > 0 ) //classification
    {
        int max_nvotes = 0;
        int* votes = (int*)alloca( sizeof(int)*nclasses );
        memset( votes, 0, sizeof(*votes)*nclasses );
        for( k = 0; k < ntrees; k++ )
        {
            CvDTreeNode* predicted_node = trees[k]->predict( sample, missing );
            int nvotes;
            int class_idx = predicted_node->class_idx;
            CV_Assert( 0 <= class_idx && class_idx < nclasses );

            nvotes = ++votes[class_idx];
            if( nvotes > max_nvotes )
            {
                max_nvotes = nvotes;
                result = predicted_node->value;
            }
        }
    }
    else // regression
    {
        result = 0;
        for( k = 0; k < ntrees; k++ )
            result += trees[k]->predict( sample, missing )->value;
        result /= (double)ntrees;
    }

    return (float)result;
}

float CvRTrees::predict_prob( const CvMat* sample, const CvMat* missing) const
{
    double result = -1;
    int k;
	
	if( nclasses == 2 ) //classification
    {
        int max_nvotes = 0;
        int* votes = (int*)alloca( sizeof(int)*nclasses );
        memset( votes, 0, sizeof(*votes)*nclasses );
        for( k = 0; k < ntrees; k++ )
        {
            CvDTreeNode* predicted_node = trees[k]->predict( sample, missing );
            int nvotes;
            int class_idx = predicted_node->class_idx;
            CV_Assert( 0 <= class_idx && class_idx < nclasses );
			
            nvotes = ++votes[class_idx];
            if( nvotes > max_nvotes )
            {
                max_nvotes = nvotes;
                result = predicted_node->value;
            }
        }
		
		return float(votes[1])/ntrees;
    }
    else // regression
		CV_Error(CV_StsBadArg, "This function works for binary classification problems only...");
	
    return -1;
}

void CvRTrees::write( CvFileStorage* fs, const char* name ) const
{
    int k;

    if( ntrees < 1 || !trees || nsamples < 1 )
        CV_Error( CV_StsBadArg, "Invalid CvRTrees object" );

    cvStartWriteStruct( fs, name, CV_NODE_MAP, CV_TYPE_NAME_ML_RTREES );

    cvWriteInt( fs, "nclasses", nclasses );
    cvWriteInt( fs, "nsamples", nsamples );
    cvWriteInt( fs, "nactive_vars", (int)cvSum(active_var_mask).val[0] );
    cvWriteReal( fs, "oob_error", oob_error );

    if( var_importance )
        cvWrite( fs, "var_importance", var_importance );

    cvWriteInt( fs, "ntrees", ntrees );

    data->write_params( fs );

    cvStartWriteStruct( fs, "trees", CV_NODE_SEQ );

    for( k = 0; k < ntrees; k++ )
    {
        cvStartWriteStruct( fs, 0, CV_NODE_MAP );
        trees[k]->write( fs );
        cvEndWriteStruct( fs );
    }

    cvEndWriteStruct( fs ); //trees
    cvEndWriteStruct( fs ); //CV_TYPE_NAME_ML_RTREES
}


void CvRTrees::read( CvFileStorage* fs, CvFileNode* fnode )
{
    int nactive_vars, var_count, k;
    CvSeqReader reader;
    CvFileNode* trees_fnode = 0;

    clear();

    nclasses     = cvReadIntByName( fs, fnode, "nclasses", -1 );
    nsamples     = cvReadIntByName( fs, fnode, "nsamples" );
    nactive_vars = cvReadIntByName( fs, fnode, "nactive_vars", -1 );
    oob_error    = cvReadRealByName(fs, fnode, "oob_error", -1 );
    ntrees       = cvReadIntByName( fs, fnode, "ntrees", -1 );

    var_importance = (CvMat*)cvReadByName( fs, fnode, "var_importance" );

    if( nclasses < 0 || nsamples <= 0 || nactive_vars < 0 || oob_error < 0 || ntrees <= 0)
        CV_Error( CV_StsParseError, "Some <nclasses>, <nsamples>, <var_count>, "
        "<nactive_vars>, <oob_error>, <ntrees> of tags are missing" );

    rng = CvRNG( -1 );

    trees = (CvForestTree**)cvAlloc( sizeof(trees[0])*ntrees );
    memset( trees, 0, sizeof(trees[0])*ntrees );

    data = new CvDTreeTrainData();
    data->read_params( fs, fnode );
    data->shared = true;

    trees_fnode = cvGetFileNodeByName( fs, fnode, "trees" );
    if( !trees_fnode || !CV_NODE_IS_SEQ(trees_fnode->tag) )
        CV_Error( CV_StsParseError, "<trees> tag is missing" );

    cvStartReadSeq( trees_fnode->data.seq, &reader );
    if( reader.seq->total != ntrees )
        CV_Error( CV_StsParseError,
        "<ntrees> is not equal to the number of trees saved in file" );

    for( k = 0; k < ntrees; k++ )
    {
        trees[k] = new CvForestTree();
        trees[k]->read( fs, (CvFileNode*)reader.ptr, this, data );
        CV_NEXT_SEQ_ELEM( reader.seq->elem_size, reader );
    }

    var_count = data->var_count;
    active_var_mask = cvCreateMat( 1, var_count, CV_8UC1 );
    {
        // initialize active variables mask
        CvMat submask1, submask2;
        cvGetCols( active_var_mask, &submask1, 0, nactive_vars );
        cvGetCols( active_var_mask, &submask2, nactive_vars, var_count );
        cvSet( &submask1, cvScalar(1) );
        cvZero( &submask2 );
    }
}


int CvRTrees::get_tree_count() const
{
    return ntrees;
}

CvForestTree* CvRTrees::get_tree(int i) const
{
    return (unsigned)i < (unsigned)ntrees ? trees[i] : 0;
}

using namespace cv;

bool CvRTrees::train( const Mat& _train_data, int _tflag,
                     const Mat& _responses, const Mat& _var_idx,
                     const Mat& _sample_idx, const Mat& _var_type,
                     const Mat& _missing_mask, CvRTParams _params )
{
    CvMat tdata = _train_data, responses = _responses, vidx = _var_idx,
    sidx = _sample_idx, vtype = _var_type, mmask = _missing_mask;
    return train(&tdata, _tflag, &responses, vidx.data.ptr ? &vidx : 0,
                 sidx.data.ptr ? &sidx : 0, vtype.data.ptr ? &vtype : 0,
                 mmask.data.ptr ? &mmask : 0, _params);
}


float CvRTrees::predict( const Mat& _sample, const Mat& _missing ) const
{
    CvMat sample = _sample, mmask = _missing;
    return predict(&sample, mmask.data.ptr ? &mmask : 0);
}

float CvRTrees::predict_prob( const Mat& _sample, const Mat& _missing) const
{
    CvMat sample = _sample, mmask = _missing;
    return predict_prob(&sample, mmask.data.ptr ? &mmask : 0);
}


// End of file.
