/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//            Intel License Agreement
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

CvNormalBayesClassifier::CvNormalBayesClassifier()
{
    var_count = var_all = 0;
    var_idx = 0;
    cls_labels = 0;
    count = 0;
    sum = 0;
    productsum = 0;
    avg = 0;
    inv_eigen_values = 0;
    cov_rotate_mats = 0;
    c = 0;
    default_model_name = "my_nb";
}


void CvNormalBayesClassifier::clear()
{
    if( cls_labels )
    {
        for( int cls = 0; cls < cls_labels->cols; cls++ )
        {
            cvReleaseMat( &count[cls] );
            cvReleaseMat( &sum[cls] );
            cvReleaseMat( &productsum[cls] );
            cvReleaseMat( &avg[cls] );
            cvReleaseMat( &inv_eigen_values[cls] );
            cvReleaseMat( &cov_rotate_mats[cls] );
        }
    }

    cvReleaseMat( &cls_labels );
    cvReleaseMat( &var_idx );
    cvReleaseMat( &c );
    cvFree( &count );
}


CvNormalBayesClassifier::~CvNormalBayesClassifier()
{
    clear();
}


CvNormalBayesClassifier::CvNormalBayesClassifier(
    const CvMat* _train_data, const CvMat* _responses,
    const CvMat* _var_idx, const CvMat* _sample_idx )
{
    var_count = var_all = 0;
    var_idx = 0;
    cls_labels = 0;
    count = 0;
    sum = 0;
    productsum = 0;
    avg = 0;
    inv_eigen_values = 0;
    cov_rotate_mats = 0;
    c = 0;
    default_model_name = "my_nb";

    train( _train_data, _responses, _var_idx, _sample_idx );
}


bool CvNormalBayesClassifier::train( const CvMat* _train_data, const CvMat* _responses,
                            const CvMat* _var_idx, const CvMat* _sample_idx, bool update )
{
    const float min_variation = FLT_EPSILON;
    bool result = false;
    CvMat* responses   = 0;
    const float** train_data = 0;
    CvMat* __cls_labels = 0;
    CvMat* __var_idx = 0;
    CvMat* cov = 0;

    CV_FUNCNAME( "CvNormalBayesClassifier::train" );

    __BEGIN__;

    int cls, nsamples = 0, _var_count = 0, _var_all = 0, nclasses = 0;
    int s, c1, c2;
    const int* responses_data;

    CV_CALL( cvPrepareTrainData( 0,
        _train_data, CV_ROW_SAMPLE, _responses, CV_VAR_CATEGORICAL,
        _var_idx, _sample_idx, false, &train_data,
        &nsamples, &_var_count, &_var_all, &responses,
        &__cls_labels, &__var_idx ));

    if( !update )
    {
        const size_t mat_size = sizeof(CvMat*);
        size_t data_size;

        clear();

        var_idx = __var_idx;
        cls_labels = __cls_labels;
        __var_idx = __cls_labels = 0;
        var_count = _var_count;
        var_all = _var_all;

        nclasses = cls_labels->cols;
        data_size = nclasses*6*mat_size;

        CV_CALL( count = (CvMat**)cvAlloc( data_size ));
        memset( count, 0, data_size );

        sum             = count      + nclasses;
        productsum      = sum        + nclasses;
        avg             = productsum + nclasses;
        inv_eigen_values= avg        + nclasses;
        cov_rotate_mats = inv_eigen_values         + nclasses;

        CV_CALL( c = cvCreateMat( 1, nclasses, CV_64FC1 ));

        for( cls = 0; cls < nclasses; cls++ )
        {
            CV_CALL(count[cls]            = cvCreateMat( 1, var_count, CV_32SC1 ));
            CV_CALL(sum[cls]              = cvCreateMat( 1, var_count, CV_64FC1 ));
            CV_CALL(productsum[cls]       = cvCreateMat( var_count, var_count, CV_64FC1 ));
            CV_CALL(avg[cls]              = cvCreateMat( 1, var_count, CV_64FC1 ));
            CV_CALL(inv_eigen_values[cls] = cvCreateMat( 1, var_count, CV_64FC1 ));
            CV_CALL(cov_rotate_mats[cls]  = cvCreateMat( var_count, var_count, CV_64FC1 ));
            CV_CALL(cvZero( count[cls] ));
            CV_CALL(cvZero( sum[cls] ));
            CV_CALL(cvZero( productsum[cls] ));
            CV_CALL(cvZero( avg[cls] ));
            CV_CALL(cvZero( inv_eigen_values[cls] ));
            CV_CALL(cvZero( cov_rotate_mats[cls] ));
        }
    }
    else
    {
        // check that the new training data has the same dimensionality etc.
        if( _var_count != var_count || _var_all != var_all || !((!_var_idx && !var_idx) ||
            (_var_idx && var_idx && cvNorm(_var_idx,var_idx,CV_C) < DBL_EPSILON)) )
            CV_ERROR( CV_StsBadArg,
            "The new training data is inconsistent with the original training data" );

        if( cls_labels->cols != __cls_labels->cols ||
            cvNorm(cls_labels, __cls_labels, CV_C) > DBL_EPSILON )
            CV_ERROR( CV_StsNotImplemented,
            "In the current implementation the new training data must have absolutely "
            "the same set of class labels as used in the original training data" );

        nclasses = cls_labels->cols;
    }

    responses_data = responses->data.i;
    CV_CALL( cov = cvCreateMat( _var_count, _var_count, CV_64FC1 ));

    /* process train data (count, sum , productsum) */
    for( s = 0; s < nsamples; s++ )
    {
        cls = responses_data[s];
        int* count_data = count[cls]->data.i;
        double* sum_data = sum[cls]->data.db;
        double* prod_data = productsum[cls]->data.db;
        const float* train_vec = train_data[s];

        for( c1 = 0; c1 < _var_count; c1++, prod_data += _var_count )
        {
            double val1 = train_vec[c1];
            sum_data[c1] += val1;
            count_data[c1]++;
            for( c2 = c1; c2 < _var_count; c2++ )
                prod_data[c2] += train_vec[c2]*val1;
        }
    }
    cvReleaseMat( &responses );
    responses = 0;

    /* calculate avg, covariance matrix, c */
    for( cls = 0; cls < nclasses; cls++ )
    {
        double det = 1;
        int i, j;
        CvMat* w = inv_eigen_values[cls];
        int* count_data = count[cls]->data.i;
        double* avg_data = avg[cls]->data.db;
        double* sum1 = sum[cls]->data.db;

        cvCompleteSymm( productsum[cls], 0 );

        for( j = 0; j < _var_count; j++ )
        {
            int n = count_data[j];
            avg_data[j] = n ? sum1[j] / n : 0.;
        }

        count_data = count[cls]->data.i;
        avg_data = avg[cls]->data.db;
        sum1 = sum[cls]->data.db;

        for( i = 0; i < _var_count; i++ )
        {
            double* avg2_data = avg[cls]->data.db;
            double* sum2 = sum[cls]->data.db;
            double* prod_data = productsum[cls]->data.db + i*_var_count;
            double* cov_data = cov->data.db + i*_var_count;
            double s1val = sum1[i];
            double avg1 = avg_data[i];
            int _count = count_data[i];

            for( j = 0; j <= i; j++ )
            {
                double avg2 = avg2_data[j];
                double cov_val = prod_data[j] - avg1 * sum2[j] - avg2 * s1val + avg1 * avg2 * _count;
                cov_val = (_count > 1) ? cov_val / (_count - 1) : cov_val;
                cov_data[j] = cov_val;
            }
        }

        CV_CALL( cvCompleteSymm( cov, 1 ));
        CV_CALL( cvSVD( cov, w, cov_rotate_mats[cls], 0, CV_SVD_U_T ));
        CV_CALL( cvMaxS( w, min_variation, w ));
        for( j = 0; j < _var_count; j++ )
            det *= w->data.db[j];

        CV_CALL( cvDiv( NULL, w, w ));
        c->data.db[cls] = det > 0 ? log(det) : -700;
    }

    result = true;

    __END__;

    if( !result || cvGetErrStatus() < 0 )
        clear();

    cvReleaseMat( &cov );
    cvReleaseMat( &__cls_labels );
    cvReleaseMat( &__var_idx );
    cvFree( &train_data );

    return result;
}

struct predict_body : cv::ParallelLoopBody {
  predict_body(CvMat* _c, CvMat** _cov_rotate_mats, CvMat** _inv_eigen_values, CvMat** _avg,
     const CvMat* _samples, const int* _vidx, CvMat* _cls_labels,
     CvMat* _results, float* _value, int _var_count1, CvMat* _results_prob
  )
  {
    c = _c;
    cov_rotate_mats = _cov_rotate_mats;
    inv_eigen_values = _inv_eigen_values;
    avg = _avg;
    samples = _samples;
    vidx = _vidx;
    cls_labels = _cls_labels;
    results = _results;
    value = _value;
    var_count1 = _var_count1;
    results_prob = _results_prob;
  }

  CvMat* c;
  CvMat** cov_rotate_mats;
  CvMat** inv_eigen_values;
  CvMat** avg;
  const CvMat* samples;
  const int* vidx;
  CvMat* cls_labels;

  CvMat* results_prob;
  CvMat* results;
  float* value;
  int var_count1;

  void operator()( const cv::Range& range ) const
  {

    int cls = -1;
    int rtype = 0, rstep = 0, rptype = 0, rpstep = 0;
    int nclasses = cls_labels->cols;
    int _var_count = avg[0]->cols;
    double probability = 0;

    if (results)
    {
        rtype = CV_MAT_TYPE(results->type);
        rstep = CV_IS_MAT_CONT(results->type) ? 1 : results->step/CV_ELEM_SIZE(rtype);
    }
    if (results_prob)
    {
        rptype = CV_MAT_TYPE(results_prob->type);
        rpstep = CV_IS_MAT_CONT(results_prob->type) ? 1 : results_prob->step/CV_ELEM_SIZE(rptype);
    }
    // allocate memory and initializing headers for calculating
    cv::AutoBuffer<double> buffer(nclasses + var_count1);
    CvMat diff = cvMat( 1, var_count1, CV_64FC1, &buffer[0] );

    for(int k = range.start; k < range.end; k += 1 )
    {
        int ival;
        double opt = FLT_MAX;

        for(int i = 0; i < nclasses; i++ )
        {
            double cur = c->data.db[i];
            CvMat* u = cov_rotate_mats[i];
            CvMat* w = inv_eigen_values[i];

            const double* avg_data = avg[i]->data.db;
            const float* x = (const float*)(samples->data.ptr + samples->step*k);

            // cov = u w u'  -->  cov^(-1) = u w^(-1) u'
            for(int j = 0; j < _var_count; j++ )
                diff.data.db[j] = avg_data[j] - x[vidx ? vidx[j] : j];

            cvGEMM( &diff, u, 1, 0, 0, &diff, CV_GEMM_B_T );
            for(int j = 0; j < _var_count; j++ )
            {
                double d = diff.data.db[j];
                cur += d*d*w->data.db[j];
            }

            if( cur < opt )
            {
                cls = i;
                opt = cur;
            }
            /* probability = exp( -0.5 * cur ) */
            probability = exp( -0.5 * cur );
        }

        ival = cls_labels->data.i[cls];
        if( results )
        {
            if( rtype == CV_32SC1 )
                results->data.i[k*rstep] = ival;
            else
                results->data.fl[k*rstep] = (float)ival;
        }
        if ( results_prob )
        {
            if ( rptype == CV_32FC1 )
                results_prob->data.fl[k*rpstep] = (float)probability;
            else
                results_prob->data.db[k*rpstep] = probability;
        }
        if( k == 0 )
            *value = (float)ival;
    }
  }
};


float CvNormalBayesClassifier::predict( const CvMat* samples, CvMat* results, CvMat* results_prob ) const
{
    float value = 0;

    if( !CV_IS_MAT(samples) || CV_MAT_TYPE(samples->type) != CV_32FC1 || samples->cols != var_all )
        CV_Error( CV_StsBadArg,
        "The input samples must be 32f matrix with the number of columns = var_all" );

    if( samples->rows > 1 && !results )
        CV_Error( CV_StsNullPtr,
        "When the number of input samples is >1, the output vector of results must be passed" );

    if( results )
    {
        if( !CV_IS_MAT(results) || (CV_MAT_TYPE(results->type) != CV_32FC1 &&
                                    CV_MAT_TYPE(results->type) != CV_32SC1) ||
          (results->cols != 1 && results->rows != 1) ||
           results->cols + results->rows - 1 != samples->rows )
        CV_Error( CV_StsBadArg, "The output array must be integer or floating-point vector "
                 "with the number of elements = number of rows in the input matrix" );
    }

    if( results_prob )
    {
        if( !CV_IS_MAT(results_prob) || (CV_MAT_TYPE(results_prob->type) != CV_32FC1 &&
                                         CV_MAT_TYPE(results_prob->type) != CV_64FC1) ||
          (results_prob->cols != 1 && results_prob->rows != 1) ||
           results_prob->cols + results_prob->rows - 1 != samples->rows )
        CV_Error( CV_StsBadArg, "The output array must be double or float vector "
                 "with the number of elements = number of rows in the input matrix" );
    }

    const int* vidx = var_idx ? var_idx->data.i : 0;

    cv::parallel_for_(cv::Range(0, samples->rows),
                      predict_body(c, cov_rotate_mats, inv_eigen_values, avg, samples,
                                   vidx, cls_labels, results, &value, var_count, results_prob));

    return value;
}


void CvNormalBayesClassifier::write( CvFileStorage* fs, const char* name ) const
{
    CV_FUNCNAME( "CvNormalBayesClassifier::write" );

    __BEGIN__;

    int nclasses, i;

    nclasses = cls_labels->cols;

    cvStartWriteStruct( fs, name, CV_NODE_MAP, CV_TYPE_NAME_ML_NBAYES );

    CV_CALL( cvWriteInt( fs, "var_count", var_count ));
    CV_CALL( cvWriteInt( fs, "var_all", var_all ));

    if( var_idx )
        CV_CALL( cvWrite( fs, "var_idx", var_idx ));
    CV_CALL( cvWrite( fs, "cls_labels", cls_labels ));

    CV_CALL( cvStartWriteStruct( fs, "count", CV_NODE_SEQ ));
    for( i = 0; i < nclasses; i++ )
        CV_CALL( cvWrite( fs, NULL, count[i] ));
    CV_CALL( cvEndWriteStruct( fs ));

    CV_CALL( cvStartWriteStruct( fs, "sum", CV_NODE_SEQ ));
    for( i = 0; i < nclasses; i++ )
        CV_CALL( cvWrite( fs, NULL, sum[i] ));
    CV_CALL( cvEndWriteStruct( fs ));

    CV_CALL( cvStartWriteStruct( fs, "productsum", CV_NODE_SEQ ));
    for( i = 0; i < nclasses; i++ )
        CV_CALL( cvWrite( fs, NULL, productsum[i] ));
    CV_CALL( cvEndWriteStruct( fs ));

    CV_CALL( cvStartWriteStruct( fs, "avg", CV_NODE_SEQ ));
    for( i = 0; i < nclasses; i++ )
        CV_CALL( cvWrite( fs, NULL, avg[i] ));
    CV_CALL( cvEndWriteStruct( fs ));

    CV_CALL( cvStartWriteStruct( fs, "inv_eigen_values", CV_NODE_SEQ ));
    for( i = 0; i < nclasses; i++ )
        CV_CALL( cvWrite( fs, NULL, inv_eigen_values[i] ));
    CV_CALL( cvEndWriteStruct( fs ));

    CV_CALL( cvStartWriteStruct( fs, "cov_rotate_mats", CV_NODE_SEQ ));
    for( i = 0; i < nclasses; i++ )
        CV_CALL( cvWrite( fs, NULL, cov_rotate_mats[i] ));
    CV_CALL( cvEndWriteStruct( fs ));

    CV_CALL( cvWrite( fs, "c", c ));

    cvEndWriteStruct( fs );

    __END__;
}


void CvNormalBayesClassifier::read( CvFileStorage* fs, CvFileNode* root_node )
{
    bool ok = false;
    CV_FUNCNAME( "CvNormalBayesClassifier::read" );

    __BEGIN__;

    int nclasses, i;
    size_t data_size;
    CvFileNode* node;
    CvSeq* seq;
    CvSeqReader reader;

    clear();

    CV_CALL( var_count = cvReadIntByName( fs, root_node, "var_count", -1 ));
    CV_CALL( var_all = cvReadIntByName( fs, root_node, "var_all", -1 ));
    CV_CALL( var_idx = (CvMat*)cvReadByName( fs, root_node, "var_idx" ));
    CV_CALL( cls_labels = (CvMat*)cvReadByName( fs, root_node, "cls_labels" ));
    if( !cls_labels )
        CV_ERROR( CV_StsParseError, "No \"cls_labels\" in NBayes classifier" );
    if( cls_labels->cols < 1 )
        CV_ERROR( CV_StsBadArg, "Number of classes is less 1" );
    if( var_count <= 0 )
        CV_ERROR( CV_StsParseError,
        "The field \"var_count\" of NBayes classifier is missing" );
    nclasses = cls_labels->cols;

    data_size = nclasses*6*sizeof(CvMat*);
    CV_CALL( count = (CvMat**)cvAlloc( data_size ));
    memset( count, 0, data_size );

    sum = count + nclasses;
    productsum  = sum  + nclasses;
    avg = productsum + nclasses;
    inv_eigen_values = avg + nclasses;
    cov_rotate_mats = inv_eigen_values + nclasses;

    CV_CALL( node = cvGetFileNodeByName( fs, root_node, "count" ));
    seq = node->data.seq;
    if( !CV_NODE_IS_SEQ(node->tag) || seq->total != nclasses)
        CV_ERROR( CV_StsBadArg, "" );
    CV_CALL( cvStartReadSeq( seq, &reader, 0 ));
    for( i = 0; i < nclasses; i++ )
    {
        CV_CALL( count[i] = (CvMat*)cvRead( fs, (CvFileNode*)reader.ptr ));
        CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
    }

    CV_CALL( node = cvGetFileNodeByName( fs, root_node, "sum" ));
    seq = node->data.seq;
    if( !CV_NODE_IS_SEQ(node->tag) || seq->total != nclasses)
        CV_ERROR( CV_StsBadArg, "" );
    CV_CALL( cvStartReadSeq( seq, &reader, 0 ));
    for( i = 0; i < nclasses; i++ )
    {
        CV_CALL( sum[i] = (CvMat*)cvRead( fs, (CvFileNode*)reader.ptr ));
        CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
    }

    CV_CALL( node = cvGetFileNodeByName( fs, root_node, "productsum" ));
    seq = node->data.seq;
    if( !CV_NODE_IS_SEQ(node->tag) || seq->total != nclasses)
        CV_ERROR( CV_StsBadArg, "" );
    CV_CALL( cvStartReadSeq( seq, &reader, 0 ));
    for( i = 0; i < nclasses; i++ )
    {
        CV_CALL( productsum[i] = (CvMat*)cvRead( fs, (CvFileNode*)reader.ptr ));
        CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
    }

    CV_CALL( node = cvGetFileNodeByName( fs, root_node, "avg" ));
    seq = node->data.seq;
    if( !CV_NODE_IS_SEQ(node->tag) || seq->total != nclasses)
        CV_ERROR( CV_StsBadArg, "" );
    CV_CALL( cvStartReadSeq( seq, &reader, 0 ));
    for( i = 0; i < nclasses; i++ )
    {
        CV_CALL( avg[i] = (CvMat*)cvRead( fs, (CvFileNode*)reader.ptr ));
        CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
    }

    CV_CALL( node = cvGetFileNodeByName( fs, root_node, "inv_eigen_values" ));
    seq = node->data.seq;
    if( !CV_NODE_IS_SEQ(node->tag) || seq->total != nclasses)
        CV_ERROR( CV_StsBadArg, "" );
    CV_CALL( cvStartReadSeq( seq, &reader, 0 ));
    for( i = 0; i < nclasses; i++ )
    {
        CV_CALL( inv_eigen_values[i] = (CvMat*)cvRead( fs, (CvFileNode*)reader.ptr ));
        CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
    }

    CV_CALL( node = cvGetFileNodeByName( fs, root_node, "cov_rotate_mats" ));
    seq = node->data.seq;
    if( !CV_NODE_IS_SEQ(node->tag) || seq->total != nclasses)
        CV_ERROR( CV_StsBadArg, "" );
    CV_CALL( cvStartReadSeq( seq, &reader, 0 ));
    for( i = 0; i < nclasses; i++ )
    {
        CV_CALL( cov_rotate_mats[i] = (CvMat*)cvRead( fs, (CvFileNode*)reader.ptr ));
        CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
    }

    CV_CALL( c = (CvMat*)cvReadByName( fs, root_node, "c" ));

    ok = true;

    __END__;

    if( !ok )
        clear();
}

using namespace cv;

CvNormalBayesClassifier::CvNormalBayesClassifier( const Mat& _train_data, const Mat& _responses,
                                    const Mat& _var_idx, const Mat& _sample_idx )
{
    var_count = var_all = 0;
    var_idx = 0;
    cls_labels = 0;
    count = 0;
    sum = 0;
    productsum = 0;
    avg = 0;
    inv_eigen_values = 0;
    cov_rotate_mats = 0;
    c = 0;
    default_model_name = "my_nb";

    CvMat tdata = _train_data, responses = _responses, vidx = _var_idx, sidx = _sample_idx;
    train(&tdata, &responses, vidx.data.ptr ? &vidx : 0,
                 sidx.data.ptr ? &sidx : 0);
}

bool CvNormalBayesClassifier::train( const Mat& _train_data, const Mat& _responses,
                                    const Mat& _var_idx, const Mat& _sample_idx, bool update )
{
    CvMat tdata = _train_data, responses = _responses, vidx = _var_idx, sidx = _sample_idx;
    return train(&tdata, &responses, vidx.data.ptr ? &vidx : 0,
                 sidx.data.ptr ? &sidx : 0, update);
}

float CvNormalBayesClassifier::predict( const Mat& _samples, Mat* _results, Mat* _results_prob ) const
{
    CvMat samples = _samples, results, *presults = 0, results_prob, *presults_prob = 0;

    if( _results )
    {
        if( !(_results->data && _results->type() == CV_32F &&
              (_results->cols == 1 || _results->rows == 1) &&
              _results->cols + _results->rows - 1 == _samples.rows) )
            _results->create(_samples.rows, 1, CV_32F);
        presults = &(results = *_results);
    }

    if( _results_prob )
    {
        if( !(_results_prob->data && _results_prob->type() == CV_64F &&
              (_results_prob->cols == 1 || _results_prob->rows == 1) &&
              _results_prob->cols + _results_prob->rows - 1 == _samples.rows) )
            _results_prob->create(_samples.rows, 1, CV_64F);
        presults_prob = &(results_prob = *_results_prob);
    }

    return predict(&samples, presults, presults_prob);
}

/* End of file. */
