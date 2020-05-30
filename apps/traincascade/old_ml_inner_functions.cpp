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

#include "old_ml_precomp.hpp"


CvStatModel::CvStatModel()
{
    default_model_name = "my_stat_model";
}


CvStatModel::~CvStatModel()
{
    clear();
}


void CvStatModel::clear()
{
}


void CvStatModel::save( const char* filename, const char* name ) const
{
    CvFileStorage* fs = 0;

    CV_FUNCNAME( "CvStatModel::save" );

    __BEGIN__;

    CV_CALL( fs = cvOpenFileStorage( filename, 0, CV_STORAGE_WRITE ));
    if( !fs )
        CV_ERROR( CV_StsError, "Could not open the file storage. Check the path and permissions" );

    write( fs, name ? name : default_model_name );

    __END__;

    cvReleaseFileStorage( &fs );
}


void CvStatModel::load( const char* filename, const char* name )
{
    CvFileStorage* fs = 0;

    CV_FUNCNAME( "CvAlgorithm::load" );

    __BEGIN__;

    CvFileNode* model_node = 0;

    CV_CALL( fs = cvOpenFileStorage( filename, 0, CV_STORAGE_READ ));
    if( !fs )
        EXIT;

    if( name )
        model_node = cvGetFileNodeByName( fs, 0, name );
    else
    {
        CvFileNode* root = cvGetRootFileNode( fs );
        if( root->data.seq->total > 0 )
            model_node = (CvFileNode*)cvGetSeqElem( root->data.seq, 0 );
    }

    read( fs, model_node );

    __END__;

    cvReleaseFileStorage( &fs );
}


void CvStatModel::write( CvFileStorage*, const char* ) const
{
    OPENCV_ERROR( CV_StsNotImplemented, "CvStatModel::write", "" );
}

void CvStatModel::read( CvFileStorage*, CvFileNode* )
{
    OPENCV_ERROR( CV_StsNotImplemented, "CvStatModel::read", "" );
}

CvMat* icvGenerateRandomClusterCenters ( int seed, const CvMat* data,
                                         int num_of_clusters, CvMat* _centers )
{
    CvMat* centers = _centers;

    CV_FUNCNAME("icvGenerateRandomClusterCenters");
    __BEGIN__;

    CvRNG rng;
    CvMat data_comp, centers_comp;
    CvPoint minLoc, maxLoc; // Not used, just for function "cvMinMaxLoc"
    double minVal, maxVal;
    int i;
    int dim = data ? data->cols : 0;

    if( ICV_IS_MAT_OF_TYPE(data, CV_32FC1) )
    {
        if( _centers && !ICV_IS_MAT_OF_TYPE (_centers, CV_32FC1) )
        {
            CV_ERROR(CV_StsBadArg,"");
        }
        else if( !_centers )
            CV_CALL(centers = cvCreateMat (num_of_clusters, dim, CV_32FC1));
    }
    else if( ICV_IS_MAT_OF_TYPE(data, CV_64FC1) )
    {
        if( _centers && !ICV_IS_MAT_OF_TYPE (_centers, CV_64FC1) )
        {
            CV_ERROR(CV_StsBadArg,"");
        }
        else if( !_centers )
            CV_CALL(centers = cvCreateMat (num_of_clusters, dim, CV_64FC1));
    }
    else
        CV_ERROR (CV_StsBadArg,"");

    if( num_of_clusters < 1 )
        CV_ERROR (CV_StsBadArg,"");

    rng = cvRNG(seed);
    for (i = 0; i < dim; i++)
    {
        CV_CALL(cvGetCol (data, &data_comp, i));
        CV_CALL(cvMinMaxLoc (&data_comp, &minVal, &maxVal, &minLoc, &maxLoc));
        CV_CALL(cvGetCol (centers, &centers_comp, i));
        CV_CALL(cvRandArr (&rng, &centers_comp, CV_RAND_UNI, cvScalarAll(minVal), cvScalarAll(maxVal)));
    }

    __END__;

    if( (cvGetErrStatus () < 0) || (centers != _centers) )
        cvReleaseMat (&centers);

    return _centers ? _centers : centers;
} // end of icvGenerateRandomClusterCenters

static int CV_CDECL
icvCmpIntegers( const void* a, const void* b )
{
    return *(const int*)a - *(const int*)b;
}


static int CV_CDECL
icvCmpIntegersPtr( const void* _a, const void* _b )
{
    int a = **(const int**)_a;
    int b = **(const int**)_b;
    return (a < b ? -1 : 0)|(a > b);
}


static int icvCmpSparseVecElems( const void* a, const void* b )
{
    return ((CvSparseVecElem32f*)a)->idx - ((CvSparseVecElem32f*)b)->idx;
}


CvMat*
cvPreprocessIndexArray( const CvMat* idx_arr, int data_arr_size, bool check_for_duplicates )
{
    CvMat* idx = 0;

    CV_FUNCNAME( "cvPreprocessIndexArray" );

    __BEGIN__;

    int i, idx_total, idx_selected = 0, step, type, prev = INT_MIN, is_sorted = 1;
    uchar* srcb = 0;
    int* srci = 0;
    int* dsti;

    if( !CV_IS_MAT(idx_arr) )
        CV_ERROR( CV_StsBadArg, "Invalid index array" );

    if( idx_arr->rows != 1 && idx_arr->cols != 1 )
        CV_ERROR( CV_StsBadSize, "the index array must be 1-dimensional" );

    idx_total = idx_arr->rows + idx_arr->cols - 1;
    srcb = idx_arr->data.ptr;
    srci = idx_arr->data.i;

    type = CV_MAT_TYPE(idx_arr->type);
    step = CV_IS_MAT_CONT(idx_arr->type) ? 1 : idx_arr->step/CV_ELEM_SIZE(type);

    switch( type )
    {
    case CV_8UC1:
    case CV_8SC1:
        // idx_arr is array of 1's and 0's -
        // i.e. it is a mask of the selected components
        if( idx_total != data_arr_size )
            CV_ERROR( CV_StsUnmatchedSizes,
            "Component mask should contain as many elements as the total number of input variables" );

        for( i = 0; i < idx_total; i++ )
            idx_selected += srcb[i*step] != 0;

        if( idx_selected == 0 )
            CV_ERROR( CV_StsOutOfRange, "No components/input_variables is selected!" );

        break;
    case CV_32SC1:
        // idx_arr is array of integer indices of selected components
        if( idx_total > data_arr_size )
            CV_ERROR( CV_StsOutOfRange,
            "index array may not contain more elements than the total number of input variables" );
        idx_selected = idx_total;
        // check if sorted already
        for( i = 0; i < idx_total; i++ )
        {
            int val = srci[i*step];
            if( val >= prev )
            {
                is_sorted = 0;
                break;
            }
            prev = val;
        }
        break;
    default:
        CV_ERROR( CV_StsUnsupportedFormat, "Unsupported index array data type "
                                           "(it should be 8uC1, 8sC1 or 32sC1)" );
    }

    CV_CALL( idx = cvCreateMat( 1, idx_selected, CV_32SC1 ));
    dsti = idx->data.i;

    if( type < CV_32SC1 )
    {
        for( i = 0; i < idx_total; i++ )
            if( srcb[i*step] )
                *dsti++ = i;
    }
    else
    {
        for( i = 0; i < idx_total; i++ )
            dsti[i] = srci[i*step];

        if( !is_sorted )
            qsort( dsti, idx_total, sizeof(dsti[0]), icvCmpIntegers );

        if( dsti[0] < 0 || dsti[idx_total-1] >= data_arr_size )
            CV_ERROR( CV_StsOutOfRange, "the index array elements are out of range" );

        if( check_for_duplicates )
        {
            for( i = 1; i < idx_total; i++ )
                if( dsti[i] <= dsti[i-1] )
                    CV_ERROR( CV_StsBadArg, "There are duplicated index array elements" );
        }
    }

    __END__;

    if( cvGetErrStatus() < 0 )
        cvReleaseMat( &idx );

    return idx;
}


CvMat*
cvPreprocessVarType( const CvMat* var_type, const CvMat* var_idx,
                     int var_count, int* response_type )
{
    CvMat* out_var_type = 0;
    CV_FUNCNAME( "cvPreprocessVarType" );

    if( response_type )
        *response_type = -1;

    __BEGIN__;

    int i, tm_size, tm_step;
    //int* map = 0;
    const uchar* src;
    uchar* dst;

    if( !CV_IS_MAT(var_type) )
        CV_ERROR( var_type ? CV_StsBadArg : CV_StsNullPtr, "Invalid or absent var_type array" );

    if( var_type->rows != 1 && var_type->cols != 1 )
        CV_ERROR( CV_StsBadSize, "var_type array must be 1-dimensional" );

    if( !CV_IS_MASK_ARR(var_type))
        CV_ERROR( CV_StsUnsupportedFormat, "type mask must be 8uC1 or 8sC1 array" );

    tm_size = var_type->rows + var_type->cols - 1;
    tm_step = var_type->rows == 1 ? 1 : var_type->step/CV_ELEM_SIZE(var_type->type);

    if( /*tm_size != var_count &&*/ tm_size != var_count + 1 )
        CV_ERROR( CV_StsBadArg,
        "type mask must be of <input var count> + 1 size" );

    if( response_type && tm_size > var_count )
        *response_type = var_type->data.ptr[var_count*tm_step] != 0;

    if( var_idx )
    {
        if( !CV_IS_MAT(var_idx) || CV_MAT_TYPE(var_idx->type) != CV_32SC1 ||
            (var_idx->rows != 1 && var_idx->cols != 1) || !CV_IS_MAT_CONT(var_idx->type) )
            CV_ERROR( CV_StsBadArg, "var index array should be continuous 1-dimensional integer vector" );
        if( var_idx->rows + var_idx->cols - 1 > var_count )
            CV_ERROR( CV_StsBadSize, "var index array is too large" );
        //map = var_idx->data.i;
        var_count = var_idx->rows + var_idx->cols - 1;
    }

    CV_CALL( out_var_type = cvCreateMat( 1, var_count, CV_8UC1 ));
    src = var_type->data.ptr;
    dst = out_var_type->data.ptr;

    for( i = 0; i < var_count; i++ )
    {
        //int idx = map ? map[i] : i;
        assert( (unsigned)/*idx*/i < (unsigned)tm_size );
        dst[i] = (uchar)(src[/*idx*/i*tm_step] != 0);
    }

    __END__;

    return out_var_type;
}


CvMat*
cvPreprocessOrderedResponses( const CvMat* responses, const CvMat* sample_idx, int sample_all )
{
    CvMat* out_responses = 0;

    CV_FUNCNAME( "cvPreprocessOrderedResponses" );

    __BEGIN__;

    int i, r_type, r_step;
    const int* map = 0;
    float* dst;
    int sample_count = sample_all;

    if( !CV_IS_MAT(responses) )
        CV_ERROR( CV_StsBadArg, "Invalid response array" );

    if( responses->rows != 1 && responses->cols != 1 )
        CV_ERROR( CV_StsBadSize, "Response array must be 1-dimensional" );

    if( responses->rows + responses->cols - 1 != sample_count )
        CV_ERROR( CV_StsUnmatchedSizes,
        "Response array must contain as many elements as the total number of samples" );

    r_type = CV_MAT_TYPE(responses->type);
    if( r_type != CV_32FC1 && r_type != CV_32SC1 )
        CV_ERROR( CV_StsUnsupportedFormat, "Unsupported response type" );

    r_step = responses->step ? responses->step / CV_ELEM_SIZE(responses->type) : 1;

    if( r_type == CV_32FC1 && CV_IS_MAT_CONT(responses->type) && !sample_idx )
    {
        out_responses = cvCloneMat( responses );
        EXIT;
    }

    if( sample_idx )
    {
        if( !CV_IS_MAT(sample_idx) || CV_MAT_TYPE(sample_idx->type) != CV_32SC1 ||
            (sample_idx->rows != 1 && sample_idx->cols != 1) || !CV_IS_MAT_CONT(sample_idx->type) )
            CV_ERROR( CV_StsBadArg, "sample index array should be continuous 1-dimensional integer vector" );
        if( sample_idx->rows + sample_idx->cols - 1 > sample_count )
            CV_ERROR( CV_StsBadSize, "sample index array is too large" );
        map = sample_idx->data.i;
        sample_count = sample_idx->rows + sample_idx->cols - 1;
    }

    CV_CALL( out_responses = cvCreateMat( 1, sample_count, CV_32FC1 ));

    dst = out_responses->data.fl;
    if( r_type == CV_32FC1 )
    {
        const float* src = responses->data.fl;
        for( i = 0; i < sample_count; i++ )
        {
            int idx = map ? map[i] : i;
            assert( (unsigned)idx < (unsigned)sample_all );
            dst[i] = src[idx*r_step];
        }
    }
    else
    {
        const int* src = responses->data.i;
        for( i = 0; i < sample_count; i++ )
        {
            int idx = map ? map[i] : i;
            assert( (unsigned)idx < (unsigned)sample_all );
            dst[i] = (float)src[idx*r_step];
        }
    }

    __END__;

    return out_responses;
}

CvMat*
cvPreprocessCategoricalResponses( const CvMat* responses,
    const CvMat* sample_idx, int sample_all,
    CvMat** out_response_map, CvMat** class_counts )
{
    CvMat* out_responses = 0;
    int** response_ptr = 0;

    CV_FUNCNAME( "cvPreprocessCategoricalResponses" );

    if( out_response_map )
        *out_response_map = 0;

    if( class_counts )
        *class_counts = 0;

    __BEGIN__;

    int i, r_type, r_step;
    int cls_count = 1, prev_cls, prev_i;
    const int* map = 0;
    const int* srci;
    const float* srcfl;
    int* dst;
    int* cls_map;
    int* cls_counts = 0;
    int sample_count = sample_all;

    if( !CV_IS_MAT(responses) )
        CV_ERROR( CV_StsBadArg, "Invalid response array" );

    if( responses->rows != 1 && responses->cols != 1 )
        CV_ERROR( CV_StsBadSize, "Response array must be 1-dimensional" );

    if( responses->rows + responses->cols - 1 != sample_count )
        CV_ERROR( CV_StsUnmatchedSizes,
        "Response array must contain as many elements as the total number of samples" );

    r_type = CV_MAT_TYPE(responses->type);
    if( r_type != CV_32FC1 && r_type != CV_32SC1 )
        CV_ERROR( CV_StsUnsupportedFormat, "Unsupported response type" );

    r_step = responses->rows == 1 ? 1 : responses->step / CV_ELEM_SIZE(responses->type);

    if( sample_idx )
    {
        if( !CV_IS_MAT(sample_idx) || CV_MAT_TYPE(sample_idx->type) != CV_32SC1 ||
            (sample_idx->rows != 1 && sample_idx->cols != 1) || !CV_IS_MAT_CONT(sample_idx->type) )
            CV_ERROR( CV_StsBadArg, "sample index array should be continuous 1-dimensional integer vector" );
        if( sample_idx->rows + sample_idx->cols - 1 > sample_count )
            CV_ERROR( CV_StsBadSize, "sample index array is too large" );
        map = sample_idx->data.i;
        sample_count = sample_idx->rows + sample_idx->cols - 1;
    }

    CV_CALL( out_responses = cvCreateMat( 1, sample_count, CV_32SC1 ));

    if( !out_response_map )
        CV_ERROR( CV_StsNullPtr, "out_response_map pointer is NULL" );

    CV_CALL( response_ptr = (int**)cvAlloc( sample_count*sizeof(response_ptr[0])));

    srci = responses->data.i;
    srcfl = responses->data.fl;
    dst = out_responses->data.i;

    for( i = 0; i < sample_count; i++ )
    {
        int idx = map ? map[i] : i;
        assert( (unsigned)idx < (unsigned)sample_all );
        if( r_type == CV_32SC1 )
            dst[i] = srci[idx*r_step];
        else
        {
            float rf = srcfl[idx*r_step];
            int ri = cvRound(rf);
            if( ri != rf )
            {
                char buf[100];
                sprintf( buf, "response #%d is not integral", idx );
                CV_ERROR( CV_StsBadArg, buf );
            }
            dst[i] = ri;
        }
        response_ptr[i] = dst + i;
    }

    qsort( response_ptr, sample_count, sizeof(int*), icvCmpIntegersPtr );

    // count the classes
    for( i = 1; i < sample_count; i++ )
        cls_count += *response_ptr[i] != *response_ptr[i-1];

    if( cls_count < 2 )
        CV_ERROR( CV_StsBadArg, "There is only a single class" );

    CV_CALL( *out_response_map = cvCreateMat( 1, cls_count, CV_32SC1 ));

    if( class_counts )
    {
        CV_CALL( *class_counts = cvCreateMat( 1, cls_count, CV_32SC1 ));
        cls_counts = (*class_counts)->data.i;
    }

    // compact the class indices and build the map
    prev_cls = ~*response_ptr[0];
    cls_count = -1;
    cls_map = (*out_response_map)->data.i;

    for( i = 0, prev_i = -1; i < sample_count; i++ )
    {
        int cur_cls = *response_ptr[i];
        if( cur_cls != prev_cls )
        {
            if( cls_counts && cls_count >= 0 )
                cls_counts[cls_count] = i - prev_i;
            cls_map[++cls_count] = prev_cls = cur_cls;
            prev_i = i;
        }
        *response_ptr[i] = cls_count;
    }

    if( cls_counts )
        cls_counts[cls_count] = i - prev_i;

    __END__;

    cvFree( &response_ptr );

    return out_responses;
}


const float**
cvGetTrainSamples( const CvMat* train_data, int tflag,
                   const CvMat* var_idx, const CvMat* sample_idx,
                   int* _var_count, int* _sample_count,
                   bool always_copy_data )
{
    float** samples = 0;

    CV_FUNCNAME( "cvGetTrainSamples" );

    __BEGIN__;

    int i, j, var_count, sample_count, s_step, v_step;
    bool copy_data;
    const float* data;
    const int *s_idx, *v_idx;

    if( !CV_IS_MAT(train_data) )
        CV_ERROR( CV_StsBadArg, "Invalid or NULL training data matrix" );

    var_count = var_idx ? var_idx->cols + var_idx->rows - 1 :
                tflag == CV_ROW_SAMPLE ? train_data->cols : train_data->rows;
    sample_count = sample_idx ? sample_idx->cols + sample_idx->rows - 1 :
                   tflag == CV_ROW_SAMPLE ? train_data->rows : train_data->cols;

    if( _var_count )
        *_var_count = var_count;

    if( _sample_count )
        *_sample_count = sample_count;

    copy_data = tflag != CV_ROW_SAMPLE || var_idx || always_copy_data;

    CV_CALL( samples = (float**)cvAlloc(sample_count*sizeof(samples[0]) +
                (copy_data ? 1 : 0)*var_count*sample_count*sizeof(samples[0][0])) );
    data = train_data->data.fl;
    s_step = train_data->step / sizeof(samples[0][0]);
    v_step = 1;
    s_idx = sample_idx ? sample_idx->data.i : 0;
    v_idx = var_idx ? var_idx->data.i : 0;

    if( !copy_data )
    {
        for( i = 0; i < sample_count; i++ )
            samples[i] = (float*)(data + (s_idx ? s_idx[i] : i)*s_step);
    }
    else
    {
        samples[0] = (float*)(samples + sample_count);
        if( tflag != CV_ROW_SAMPLE )
            CV_SWAP( s_step, v_step, i );

        for( i = 0; i < sample_count; i++ )
        {
            float* dst = samples[i] = samples[0] + i*var_count;
            const float* src = data + (s_idx ? s_idx[i] : i)*s_step;

            if( !v_idx )
                for( j = 0; j < var_count; j++ )
                    dst[j] = src[j*v_step];
            else
                for( j = 0; j < var_count; j++ )
                    dst[j] = src[v_idx[j]*v_step];
        }
    }

    __END__;

    return (const float**)samples;
}


void
cvCheckTrainData( const CvMat* train_data, int tflag,
                  const CvMat* missing_mask,
                  int* var_all, int* sample_all )
{
    CV_FUNCNAME( "cvCheckTrainData" );

    if( var_all )
        *var_all = 0;

    if( sample_all )
        *sample_all = 0;

    __BEGIN__;

    // check parameter types and sizes
    if( !CV_IS_MAT(train_data) || CV_MAT_TYPE(train_data->type) != CV_32FC1 )
        CV_ERROR( CV_StsBadArg, "train data must be floating-point matrix" );

    if( missing_mask )
    {
        if( !CV_IS_MAT(missing_mask) || !CV_IS_MASK_ARR(missing_mask) ||
            !CV_ARE_SIZES_EQ(train_data, missing_mask) )
            CV_ERROR( CV_StsBadArg,
            "missing value mask must be 8-bit matrix of the same size as training data" );
    }

    if( tflag != CV_ROW_SAMPLE && tflag != CV_COL_SAMPLE )
        CV_ERROR( CV_StsBadArg,
        "Unknown training data layout (must be CV_ROW_SAMPLE or CV_COL_SAMPLE)" );

    if( var_all )
        *var_all = tflag == CV_ROW_SAMPLE ? train_data->cols : train_data->rows;

    if( sample_all )
        *sample_all = tflag == CV_ROW_SAMPLE ? train_data->rows : train_data->cols;

    __END__;
}


int
cvPrepareTrainData( const char* /*funcname*/,
                    const CvMat* train_data, int tflag,
                    const CvMat* responses, int response_type,
                    const CvMat* var_idx,
                    const CvMat* sample_idx,
                    bool always_copy_data,
                    const float*** out_train_samples,
                    int* _sample_count,
                    int* _var_count,
                    int* _var_all,
                    CvMat** out_responses,
                    CvMat** out_response_map,
                    CvMat** out_var_idx,
                    CvMat** out_sample_idx )
{
    int ok = 0;
    CvMat* _var_idx = 0;
    CvMat* _sample_idx = 0;
    CvMat* _responses = 0;
    int sample_all = 0, sample_count = 0, var_all = 0, var_count = 0;

    CV_FUNCNAME( "cvPrepareTrainData" );

    // step 0. clear all the output pointers to ensure we do not try
    // to call free() with uninitialized pointers
    if( out_responses )
        *out_responses = 0;

    if( out_response_map )
        *out_response_map = 0;

    if( out_var_idx )
        *out_var_idx = 0;

    if( out_sample_idx )
        *out_sample_idx = 0;

    if( out_train_samples )
        *out_train_samples = 0;

    if( _sample_count )
        *_sample_count = 0;

    if( _var_count )
        *_var_count = 0;

    if( _var_all )
        *_var_all = 0;

    __BEGIN__;

    if( !out_train_samples )
        CV_ERROR( CV_StsBadArg, "output pointer to train samples is NULL" );

    CV_CALL( cvCheckTrainData( train_data, tflag, 0, &var_all, &sample_all ));

    if( sample_idx )
        CV_CALL( _sample_idx = cvPreprocessIndexArray( sample_idx, sample_all ));
    if( var_idx )
        CV_CALL( _var_idx = cvPreprocessIndexArray( var_idx, var_all ));

    if( responses )
    {
        if( !out_responses )
            CV_ERROR( CV_StsNullPtr, "output response pointer is NULL" );

        if( response_type == CV_VAR_NUMERICAL )
        {
            CV_CALL( _responses = cvPreprocessOrderedResponses( responses,
                                                _sample_idx, sample_all ));
        }
        else
        {
            CV_CALL( _responses = cvPreprocessCategoricalResponses( responses,
                                _sample_idx, sample_all, out_response_map, 0 ));
        }
    }

    CV_CALL( *out_train_samples =
                cvGetTrainSamples( train_data, tflag, _var_idx, _sample_idx,
                                   &var_count, &sample_count, always_copy_data ));

    ok = 1;

    __END__;

    if( ok )
    {
        if( out_responses )
            *out_responses = _responses, _responses = 0;

        if( out_var_idx )
            *out_var_idx = _var_idx, _var_idx = 0;

        if( out_sample_idx )
            *out_sample_idx = _sample_idx, _sample_idx = 0;

        if( _sample_count )
            *_sample_count = sample_count;

        if( _var_count )
            *_var_count = var_count;

        if( _var_all )
            *_var_all = var_all;
    }
    else
    {
        if( out_response_map )
            cvReleaseMat( out_response_map );
        cvFree( out_train_samples );
    }

    if( _responses != responses )
        cvReleaseMat( &_responses );
    cvReleaseMat( &_var_idx );
    cvReleaseMat( &_sample_idx );

    return ok;
}


typedef struct CvSampleResponsePair
{
    const float* sample;
    const uchar* mask;
    int response;
    int index;
}
CvSampleResponsePair;


static int
CV_CDECL icvCmpSampleResponsePairs( const void* a, const void* b )
{
    int ra = ((const CvSampleResponsePair*)a)->response;
    int rb = ((const CvSampleResponsePair*)b)->response;
    int ia = ((const CvSampleResponsePair*)a)->index;
    int ib = ((const CvSampleResponsePair*)b)->index;

    return ra < rb ? -1 : ra > rb ? 1 : ia - ib;
    //return (ra > rb ? -1 : 0)|(ra < rb);
}


void
cvSortSamplesByClasses( const float** samples, const CvMat* classes,
                        int* class_ranges, const uchar** mask )
{
    CvSampleResponsePair* pairs = 0;
    CV_FUNCNAME( "cvSortSamplesByClasses" );

    __BEGIN__;

    int i, k = 0, sample_count;

    if( !samples || !classes || !class_ranges )
        CV_ERROR( CV_StsNullPtr, "INTERNAL ERROR: some of the args are NULL pointers" );

    if( classes->rows != 1 || CV_MAT_TYPE(classes->type) != CV_32SC1 )
        CV_ERROR( CV_StsBadArg, "classes array must be a single row of integers" );

    sample_count = classes->cols;
    CV_CALL( pairs = (CvSampleResponsePair*)cvAlloc( (sample_count+1)*sizeof(pairs[0])));

    for( i = 0; i < sample_count; i++ )
    {
        pairs[i].sample = samples[i];
        pairs[i].mask = (mask) ? (mask[i]) : 0;
        pairs[i].response = classes->data.i[i];
        pairs[i].index = i;
        assert( classes->data.i[i] >= 0 );
    }

    qsort( pairs, sample_count, sizeof(pairs[0]), icvCmpSampleResponsePairs );
    pairs[sample_count].response = -1;
    class_ranges[0] = 0;

    for( i = 0; i < sample_count; i++ )
    {
        samples[i] = pairs[i].sample;
        if (mask)
            mask[i] = pairs[i].mask;
        classes->data.i[i] = pairs[i].response;

        if( pairs[i].response != pairs[i+1].response )
            class_ranges[++k] = i+1;
    }

    __END__;

    cvFree( &pairs );
}


void
cvPreparePredictData( const CvArr* _sample, int dims_all,
                      const CvMat* comp_idx, int class_count,
                      const CvMat* prob, float** _row_sample,
                      int as_sparse )
{
    float* row_sample = 0;
    int* inverse_comp_idx = 0;

    CV_FUNCNAME( "cvPreparePredictData" );

    __BEGIN__;

    const CvMat* sample = (const CvMat*)_sample;
    float* sample_data;
    int sample_step;
    int is_sparse = CV_IS_SPARSE_MAT(sample);
    int d, sizes[CV_MAX_DIM];
    int i, dims_selected;
    int vec_size;

    if( !is_sparse && !CV_IS_MAT(sample) )
        CV_ERROR( !sample ? CV_StsNullPtr : CV_StsBadArg, "The sample is not a valid vector" );

    if( cvGetElemType( sample ) != CV_32FC1 )
        CV_ERROR( CV_StsUnsupportedFormat, "Input sample must have 32fC1 type" );

    CV_CALL( d = cvGetDims( sample, sizes ));

    if( !((is_sparse && d == 1) || (!is_sparse && d == 2 && (sample->rows == 1 || sample->cols == 1))) )
        CV_ERROR( CV_StsBadSize, "Input sample must be 1-dimensional vector" );

    if( d == 1 )
        sizes[1] = 1;

    if( sizes[0] + sizes[1] - 1 != dims_all )
        CV_ERROR( CV_StsUnmatchedSizes,
        "The sample size is different from what has been used for training" );

    if( !_row_sample )
        CV_ERROR( CV_StsNullPtr, "INTERNAL ERROR: The row_sample pointer is NULL" );

    if( comp_idx && (!CV_IS_MAT(comp_idx) || comp_idx->rows != 1 ||
        CV_MAT_TYPE(comp_idx->type) != CV_32SC1) )
        CV_ERROR( CV_StsBadArg, "INTERNAL ERROR: invalid comp_idx" );

    dims_selected = comp_idx ? comp_idx->cols : dims_all;

    if( prob )
    {
        if( !CV_IS_MAT(prob) )
            CV_ERROR( CV_StsBadArg, "The output matrix of probabilities is invalid" );

        if( (prob->rows != 1 && prob->cols != 1) ||
            (CV_MAT_TYPE(prob->type) != CV_32FC1 &&
            CV_MAT_TYPE(prob->type) != CV_64FC1) )
            CV_ERROR( CV_StsBadSize,
            "The matrix of probabilities must be 1-dimensional vector of 32fC1 type" );

        if( prob->rows + prob->cols - 1 != class_count )
            CV_ERROR( CV_StsUnmatchedSizes,
            "The vector of probabilities must contain as many elements as "
            "the number of classes in the training set" );
    }

    vec_size = !as_sparse ? dims_selected*sizeof(row_sample[0]) :
                (dims_selected + 1)*sizeof(CvSparseVecElem32f);

    if( CV_IS_MAT(sample) )
    {
        sample_data = sample->data.fl;
        sample_step = CV_IS_MAT_CONT(sample->type) ? 1 : sample->step/sizeof(row_sample[0]);

        if( !comp_idx && CV_IS_MAT_CONT(sample->type) && !as_sparse )
            *_row_sample = sample_data;
        else
        {
            CV_CALL( row_sample = (float*)cvAlloc( vec_size ));

            if( !comp_idx )
                for( i = 0; i < dims_selected; i++ )
                    row_sample[i] = sample_data[sample_step*i];
            else
            {
                int* comp = comp_idx->data.i;
                for( i = 0; i < dims_selected; i++ )
                    row_sample[i] = sample_data[sample_step*comp[i]];
            }

            *_row_sample = row_sample;
        }

        if( as_sparse )
        {
            const float* src = (const float*)row_sample;
            CvSparseVecElem32f* dst = (CvSparseVecElem32f*)row_sample;

            dst[dims_selected].idx = -1;
            for( i = dims_selected - 1; i >= 0; i-- )
            {
                dst[i].idx = i;
                dst[i].val = src[i];
            }
        }
    }
    else
    {
        CvSparseNode* node;
        CvSparseMatIterator mat_iterator;
        const CvSparseMat* sparse = (const CvSparseMat*)sample;
        assert( is_sparse );

        node = cvInitSparseMatIterator( sparse, &mat_iterator );
        CV_CALL( row_sample = (float*)cvAlloc( vec_size ));

        if( comp_idx )
        {
            CV_CALL( inverse_comp_idx = (int*)cvAlloc( dims_all*sizeof(int) ));
            memset( inverse_comp_idx, -1, dims_all*sizeof(int) );
            for( i = 0; i < dims_selected; i++ )
                inverse_comp_idx[comp_idx->data.i[i]] = i;
        }

        if( !as_sparse )
        {
            memset( row_sample, 0, vec_size );

            for( ; node != 0; node = cvGetNextSparseNode(&mat_iterator) )
            {
                int idx = *CV_NODE_IDX( sparse, node );
                if( inverse_comp_idx )
                {
                    idx = inverse_comp_idx[idx];
                    if( idx < 0 )
                        continue;
                }
                row_sample[idx] = *(float*)CV_NODE_VAL( sparse, node );
            }
        }
        else
        {
            CvSparseVecElem32f* ptr = (CvSparseVecElem32f*)row_sample;

            for( ; node != 0; node = cvGetNextSparseNode(&mat_iterator) )
            {
                int idx = *CV_NODE_IDX( sparse, node );
                if( inverse_comp_idx )
                {
                    idx = inverse_comp_idx[idx];
                    if( idx < 0 )
                        continue;
                }
                ptr->idx = idx;
                ptr->val = *(float*)CV_NODE_VAL( sparse, node );
                ptr++;
            }

            qsort( row_sample, ptr - (CvSparseVecElem32f*)row_sample,
                   sizeof(ptr[0]), icvCmpSparseVecElems );
            ptr->idx = -1;
        }

        *_row_sample = row_sample;
    }

    __END__;

    if( inverse_comp_idx )
        cvFree( &inverse_comp_idx );

    if( cvGetErrStatus() < 0 && _row_sample )
    {
        cvFree( &row_sample );
        *_row_sample = 0;
    }
}


static void
icvConvertDataToSparse( const uchar* src, int src_step, int src_type,
                        uchar* dst, int dst_step, int dst_type,
                        CvSize size, int* idx )
{
    CV_FUNCNAME( "icvConvertDataToSparse" );

    __BEGIN__;

    int i, j;
    src_type = CV_MAT_TYPE(src_type);
    dst_type = CV_MAT_TYPE(dst_type);

    if( CV_MAT_CN(src_type) != 1 || CV_MAT_CN(dst_type) != 1 )
        CV_ERROR( CV_StsUnsupportedFormat, "The function supports only single-channel arrays" );

    if( src_step == 0 )
        src_step = CV_ELEM_SIZE(src_type);

    if( dst_step == 0 )
        dst_step = CV_ELEM_SIZE(dst_type);

    // if there is no "idx" and if both arrays are continuous,
    // do the whole processing (copying or conversion) in a single loop
    if( !idx && CV_ELEM_SIZE(src_type)*size.width == src_step &&
        CV_ELEM_SIZE(dst_type)*size.width == dst_step )
    {
        size.width *= size.height;
        size.height = 1;
    }

    if( src_type == dst_type )
    {
        int full_width = CV_ELEM_SIZE(dst_type)*size.width;

        if( full_width == sizeof(int) ) // another common case: copy int's or float's
            for( i = 0; i < size.height; i++, src += src_step )
                *(int*)(dst + dst_step*(idx ? idx[i] : i)) = *(int*)src;
        else
            for( i = 0; i < size.height; i++, src += src_step )
                memcpy( dst + dst_step*(idx ? idx[i] : i), src, full_width );
    }
    else if( src_type == CV_32SC1 && (dst_type == CV_32FC1 || dst_type == CV_64FC1) )
        for( i = 0; i < size.height; i++, src += src_step )
        {
            uchar* _dst = dst + dst_step*(idx ? idx[i] : i);
            if( dst_type == CV_32FC1 )
                for( j = 0; j < size.width; j++ )
                    ((float*)_dst)[j] = (float)((int*)src)[j];
            else
                for( j = 0; j < size.width; j++ )
                    ((double*)_dst)[j] = ((int*)src)[j];
        }
    else if( (src_type == CV_32FC1 || src_type == CV_64FC1) && dst_type == CV_32SC1 )
        for( i = 0; i < size.height; i++, src += src_step )
        {
            uchar* _dst = dst + dst_step*(idx ? idx[i] : i);
            if( src_type == CV_32FC1 )
                for( j = 0; j < size.width; j++ )
                    ((int*)_dst)[j] = cvRound(((float*)src)[j]);
            else
                for( j = 0; j < size.width; j++ )
                    ((int*)_dst)[j] = cvRound(((double*)src)[j]);
        }
    else if( (src_type == CV_32FC1 && dst_type == CV_64FC1) ||
             (src_type == CV_64FC1 && dst_type == CV_32FC1) )
        for( i = 0; i < size.height; i++, src += src_step )
        {
            uchar* _dst = dst + dst_step*(idx ? idx[i] : i);
            if( src_type == CV_32FC1 )
                for( j = 0; j < size.width; j++ )
                    ((double*)_dst)[j] = ((float*)src)[j];
            else
                for( j = 0; j < size.width; j++ )
                    ((float*)_dst)[j] = (float)((double*)src)[j];
        }
    else
        CV_ERROR( CV_StsUnsupportedFormat, "Unsupported combination of input and output vectors" );

    __END__;
}


void
cvWritebackLabels( const CvMat* labels, CvMat* dst_labels,
                   const CvMat* centers, CvMat* dst_centers,
                   const CvMat* probs, CvMat* dst_probs,
                   const CvMat* sample_idx, int samples_all,
                   const CvMat* comp_idx, int dims_all )
{
    CV_FUNCNAME( "cvWritebackLabels" );

    __BEGIN__;

    int samples_selected = samples_all, dims_selected = dims_all;

    if( dst_labels && !CV_IS_MAT(dst_labels) )
        CV_ERROR( CV_StsBadArg, "Array of output labels is not a valid matrix" );

    if( dst_centers )
        if( !ICV_IS_MAT_OF_TYPE(dst_centers, CV_32FC1) &&
            !ICV_IS_MAT_OF_TYPE(dst_centers, CV_64FC1) )
            CV_ERROR( CV_StsBadArg, "Array of cluster centers is not a valid matrix" );

    if( dst_probs && !CV_IS_MAT(dst_probs) )
        CV_ERROR( CV_StsBadArg, "Probability matrix is not valid" );

    if( sample_idx )
    {
        CV_ASSERT( sample_idx->rows == 1 && CV_MAT_TYPE(sample_idx->type) == CV_32SC1 );
        samples_selected = sample_idx->cols;
    }

    if( comp_idx )
    {
        CV_ASSERT( comp_idx->rows == 1 && CV_MAT_TYPE(comp_idx->type) == CV_32SC1 );
        dims_selected = comp_idx->cols;
    }

    if( dst_labels && (!labels || labels->data.ptr != dst_labels->data.ptr) )
    {
        if( !labels )
            CV_ERROR( CV_StsNullPtr, "NULL labels" );

        CV_ASSERT( labels->rows == 1 );

        if( dst_labels->rows != 1 && dst_labels->cols != 1 )
            CV_ERROR( CV_StsBadSize, "Array of output labels should be 1d vector" );

        if( dst_labels->rows + dst_labels->cols - 1 != samples_all )
            CV_ERROR( CV_StsUnmatchedSizes,
            "Size of vector of output labels is not equal to the total number of input samples" );

        CV_ASSERT( labels->cols == samples_selected );

        CV_CALL( icvConvertDataToSparse( labels->data.ptr, labels->step, labels->type,
                        dst_labels->data.ptr, dst_labels->step, dst_labels->type,
                        cvSize( 1, samples_selected ), sample_idx ? sample_idx->data.i : 0 ));
    }

    if( dst_centers && (!centers || centers->data.ptr != dst_centers->data.ptr) )
    {
        int i;

        if( !centers )
            CV_ERROR( CV_StsNullPtr, "NULL centers" );

        if( centers->rows != dst_centers->rows )
            CV_ERROR( CV_StsUnmatchedSizes, "Invalid number of rows in matrix of output centers" );

        if( dst_centers->cols != dims_all )
            CV_ERROR( CV_StsUnmatchedSizes,
            "Number of columns in matrix of output centers is "
            "not equal to the total number of components in the input samples" );

        CV_ASSERT( centers->cols == dims_selected );

        for( i = 0; i < centers->rows; i++ )
            CV_CALL( icvConvertDataToSparse( centers->data.ptr + i*centers->step, 0, centers->type,
                        dst_centers->data.ptr + i*dst_centers->step, 0, dst_centers->type,
                        cvSize( 1, dims_selected ), comp_idx ? comp_idx->data.i : 0 ));
    }

    if( dst_probs && (!probs || probs->data.ptr != dst_probs->data.ptr) )
    {
        if( !probs )
            CV_ERROR( CV_StsNullPtr, "NULL probs" );

        if( probs->cols != dst_probs->cols )
            CV_ERROR( CV_StsUnmatchedSizes, "Invalid number of columns in output probability matrix" );

        if( dst_probs->rows != samples_all )
            CV_ERROR( CV_StsUnmatchedSizes,
            "Number of rows in output probability matrix is "
            "not equal to the total number of input samples" );

        CV_ASSERT( probs->rows == samples_selected );

        CV_CALL( icvConvertDataToSparse( probs->data.ptr, probs->step, probs->type,
                        dst_probs->data.ptr, dst_probs->step, dst_probs->type,
                        cvSize( probs->cols, samples_selected ),
                        sample_idx ? sample_idx->data.i : 0 ));
    }

    __END__;
}

#if 0
CV_IMPL void
cvStatModelMultiPredict( const CvStatModel* stat_model,
                         const CvArr* predict_input,
                         int flags, CvMat* predict_output,
                         CvMat* probs, const CvMat* sample_idx )
{
    CvMemStorage* storage = 0;
    CvMat* sample_idx_buffer = 0;
    CvSparseMat** sparse_rows = 0;
    int samples_selected = 0;

    CV_FUNCNAME( "cvStatModelMultiPredict" );

    __BEGIN__;

    int i;
    int predict_output_step = 1, sample_idx_step = 1;
    int type;
    int d, sizes[CV_MAX_DIM];
    int tflag = flags == CV_COL_SAMPLE;
    int samples_all, dims_all;
    int is_sparse = CV_IS_SPARSE_MAT(predict_input);
    CvMat predict_input_part;
    CvArr* sample = &predict_input_part;
    CvMat probs_part;
    CvMat* probs1 = probs ? &probs_part : 0;

    if( !CV_IS_STAT_MODEL(stat_model) )
        CV_ERROR( !stat_model ? CV_StsNullPtr : CV_StsBadArg, "Invalid statistical model" );

    if( !stat_model->predict )
        CV_ERROR( CV_StsNotImplemented, "There is no \"predict\" method" );

    if( !predict_input || !predict_output )
        CV_ERROR( CV_StsNullPtr, "NULL input or output matrices" );

    if( !is_sparse && !CV_IS_MAT(predict_input) )
        CV_ERROR( CV_StsBadArg, "predict_input should be a matrix or a sparse matrix" );

    if( !CV_IS_MAT(predict_output) )
        CV_ERROR( CV_StsBadArg, "predict_output should be a matrix" );

    type = cvGetElemType( predict_input );
    if( type != CV_32FC1 ||
        (CV_MAT_TYPE(predict_output->type) != CV_32FC1 &&
         CV_MAT_TYPE(predict_output->type) != CV_32SC1 ))
         CV_ERROR( CV_StsUnsupportedFormat, "The input or output matrix has unsupported format" );

    CV_CALL( d = cvGetDims( predict_input, sizes ));
    if( d > 2 )
        CV_ERROR( CV_StsBadSize, "The input matrix should be 1- or 2-dimensional" );

    if( !tflag )
    {
        samples_all = samples_selected = sizes[0];
        dims_all = sizes[1];
    }
    else
    {
        samples_all = samples_selected = sizes[1];
        dims_all = sizes[0];
    }

    if( sample_idx )
    {
        if( !CV_IS_MAT(sample_idx) )
            CV_ERROR( CV_StsBadArg, "Invalid sample_idx matrix" );

        if( sample_idx->cols != 1 && sample_idx->rows != 1 )
            CV_ERROR( CV_StsBadSize, "sample_idx must be 1-dimensional matrix" );

        samples_selected = sample_idx->rows + sample_idx->cols - 1;

        if( CV_MAT_TYPE(sample_idx->type) == CV_32SC1 )
        {
            if( samples_selected > samples_all )
                CV_ERROR( CV_StsBadSize, "sample_idx is too large vector" );
        }
        else if( samples_selected != samples_all )
            CV_ERROR( CV_StsUnmatchedSizes, "sample_idx has incorrect size" );

        sample_idx_step = sample_idx->step ?
            sample_idx->step / CV_ELEM_SIZE(sample_idx->type) : 1;
    }

    if( predict_output->rows != 1 && predict_output->cols != 1 )
        CV_ERROR( CV_StsBadSize, "predict_output should be a 1-dimensional matrix" );

    if( predict_output->rows + predict_output->cols - 1 != samples_all )
        CV_ERROR( CV_StsUnmatchedSizes, "predict_output and predict_input have uncoordinated sizes" );

    predict_output_step = predict_output->step ?
        predict_output->step / CV_ELEM_SIZE(predict_output->type) : 1;

    if( probs )
    {
        if( !CV_IS_MAT(probs) )
            CV_ERROR( CV_StsBadArg, "Invalid matrix of probabilities" );

        if( probs->rows != samples_all )
            CV_ERROR( CV_StsUnmatchedSizes,
            "matrix of probabilities must have as many rows as the total number of samples" );

        if( CV_MAT_TYPE(probs->type) != CV_32FC1 )
            CV_ERROR( CV_StsUnsupportedFormat, "matrix of probabilities must have 32fC1 type" );
    }

    if( is_sparse )
    {
        CvSparseNode* node;
        CvSparseMatIterator mat_iterator;
        CvSparseMat* sparse = (CvSparseMat*)predict_input;

        if( sample_idx && CV_MAT_TYPE(sample_idx->type) == CV_32SC1 )
        {
            CV_CALL( sample_idx_buffer = cvCreateMat( 1, samples_all, CV_8UC1 ));
            cvZero( sample_idx_buffer );
            for( i = 0; i < samples_selected; i++ )
                sample_idx_buffer->data.ptr[sample_idx->data.i[i*sample_idx_step]] = 1;
            samples_selected = samples_all;
            sample_idx = sample_idx_buffer;
            sample_idx_step = 1;
        }

        CV_CALL( sparse_rows = (CvSparseMat**)cvAlloc( samples_selected*sizeof(sparse_rows[0])));
        for( i = 0; i < samples_selected; i++ )
        {
            if( sample_idx && sample_idx->data.ptr[i*sample_idx_step] == 0 )
                continue;
            CV_CALL( sparse_rows[i] = cvCreateSparseMat( 1, &dims_all, type ));
            if( !storage )
                storage = sparse_rows[i]->heap->storage;
            else
            {
                // hack: to decrease memory footprint, make all the sparse matrices
                // reside in the same storage
                int elem_size = sparse_rows[i]->heap->elem_size;
                cvReleaseMemStorage( &sparse_rows[i]->heap->storage );
                sparse_rows[i]->heap = cvCreateSet( 0, sizeof(CvSet), elem_size, storage );
            }
        }

        // put each row (or column) of predict_input into separate sparse matrix.
        node = cvInitSparseMatIterator( sparse, &mat_iterator );
        for( ; node != 0; node = cvGetNextSparseNode( &mat_iterator ))
        {
            int* idx = CV_NODE_IDX( sparse, node );
            int idx0 = idx[tflag ^ 1];
            int idx1 = idx[tflag];

            if( sample_idx && sample_idx->data.ptr[idx0*sample_idx_step] == 0 )
                continue;

            assert( sparse_rows[idx0] != 0 );
            *(float*)cvPtrND( sparse, &idx1, 0, 1, 0 ) = *(float*)CV_NODE_VAL( sparse, node );
        }
    }

    for( i = 0; i < samples_selected; i++ )
    {
        int idx = i;
        float response;

        if( sample_idx )
        {
            if( CV_MAT_TYPE(sample_idx->type) == CV_32SC1 )
            {
                idx = sample_idx->data.i[i*sample_idx_step];
                if( (unsigned)idx >= (unsigned)samples_all )
                    CV_ERROR( CV_StsOutOfRange, "Some of sample_idx elements are out of range" );
            }
            else if( CV_MAT_TYPE(sample_idx->type) == CV_8UC1 &&
                     sample_idx->data.ptr[i*sample_idx_step] == 0 )
                continue;
        }

        if( !is_sparse )
        {
            if( !tflag )
                cvGetRow( predict_input, &predict_input_part, idx );
            else
            {
                cvGetCol( predict_input, &predict_input_part, idx );
            }
        }
        else
            sample = sparse_rows[idx];

        if( probs )
            cvGetRow( probs, probs1, idx );

        CV_CALL( response = stat_model->predict( stat_model, (CvMat*)sample, probs1 ));

        if( CV_MAT_TYPE(predict_output->type) == CV_32FC1 )
            predict_output->data.fl[idx*predict_output_step] = response;
        else
        {
            CV_ASSERT( cvRound(response) == response );
            predict_output->data.i[idx*predict_output_step] = cvRound(response);
        }
    }

    __END__;

    if( sparse_rows )
    {
        int i;
        for( i = 0; i < samples_selected; i++ )
            if( sparse_rows[i] )
            {
                sparse_rows[i]->heap->storage = 0;
                cvReleaseSparseMat( &sparse_rows[i] );
            }
        cvFree( &sparse_rows );
    }

    cvReleaseMat( &sample_idx_buffer );
    cvReleaseMemStorage( &storage );
}
#endif

// By P. Yarykin - begin -

void cvCombineResponseMaps (CvMat*  _responses,
                      const CvMat*  old_response_map,
                            CvMat*  new_response_map,
                            CvMat** out_response_map)
{
    int** old_data = NULL;
    int** new_data = NULL;

        CV_FUNCNAME ("cvCombineResponseMaps");
        __BEGIN__

    int i,j;
    int old_n, new_n, out_n;
    int samples, free_response;
    int* first;
    int* responses;
    int* out_data;

    if( out_response_map )
        *out_response_map = 0;

// Check input data.
    if ((!ICV_IS_MAT_OF_TYPE (_responses, CV_32SC1)) ||
        (!ICV_IS_MAT_OF_TYPE (old_response_map, CV_32SC1)) ||
        (!ICV_IS_MAT_OF_TYPE (new_response_map, CV_32SC1)))
    {
        CV_ERROR (CV_StsBadArg, "Some of input arguments is not the CvMat")
    }

// Prepare sorted responses.
    first = new_response_map->data.i;
    new_n = new_response_map->cols;
    CV_CALL (new_data = (int**)cvAlloc (new_n * sizeof (new_data[0])));
    for (i = 0; i < new_n; i++)
        new_data[i] = first + i;
    qsort (new_data, new_n, sizeof(int*), icvCmpIntegersPtr);

    first = old_response_map->data.i;
    old_n = old_response_map->cols;
    CV_CALL (old_data = (int**)cvAlloc (old_n * sizeof (old_data[0])));
    for (i = 0; i < old_n; i++)
        old_data[i] = first + i;
    qsort (old_data, old_n, sizeof(int*), icvCmpIntegersPtr);

// Count the number of different responses.
    for (i = 0, j = 0, out_n = 0; i < old_n && j < new_n; out_n++)
    {
        if (*old_data[i] == *new_data[j])
        {
            i++;
            j++;
        }
        else if (*old_data[i] < *new_data[j])
            i++;
        else
            j++;
    }
    out_n += old_n - i + new_n - j;

// Create and fill the result response maps.
    CV_CALL (*out_response_map = cvCreateMat (1, out_n, CV_32SC1));
    out_data = (*out_response_map)->data.i;
    memcpy (out_data, first, old_n * sizeof (int));

    free_response = old_n;
    for (i = 0, j = 0; i < old_n && j < new_n; )
    {
        if (*old_data[i] == *new_data[j])
        {
            *new_data[j] = (int)(old_data[i] - first);
            i++;
            j++;
        }
        else if (*old_data[i] < *new_data[j])
            i++;
        else
        {
            out_data[free_response] = *new_data[j];
            *new_data[j] = free_response++;
            j++;
        }
    }
    for (; j < new_n; j++)
    {
        out_data[free_response] = *new_data[j];
        *new_data[j] = free_response++;
    }
    CV_ASSERT (free_response == out_n);

// Change <responses> according to out response map.
    samples = _responses->cols + _responses->rows - 1;
    responses = _responses->data.i;
    first = new_response_map->data.i;
    for (i = 0; i < samples; i++)
    {
        responses[i] = first[responses[i]];
    }

        __END__

    cvFree(&old_data);
    cvFree(&new_data);

}


static int icvGetNumberOfCluster( double* prob_vector, int num_of_clusters, float r,
                           float outlier_thresh, int normalize_probs )
{
    int max_prob_loc = 0;

    CV_FUNCNAME("icvGetNumberOfCluster");
    __BEGIN__;

    double prob, maxprob, sum;
    int i;

    CV_ASSERT(prob_vector);
    CV_ASSERT(num_of_clusters >= 0);

    maxprob = prob_vector[0];
    max_prob_loc = 0;
    sum = maxprob;
    for( i = 1; i < num_of_clusters; i++ )
    {
        prob = prob_vector[i];
        sum += prob;
        if( prob > maxprob )
        {
            max_prob_loc = i;
            maxprob = prob;
        }
    }
    if( normalize_probs && fabs(sum - 1.) > FLT_EPSILON )
    {
        for( i = 0; i < num_of_clusters; i++ )
            prob_vector[i] /= sum;
    }
    if( fabs(r - 1.) > FLT_EPSILON && fabs(sum - 1.) < outlier_thresh )
        max_prob_loc = -1;

    __END__;

    return max_prob_loc;

} // End of icvGetNumberOfCluster


void icvFindClusterLabels( const CvMat* probs, float outlier_thresh, float r,
                          const CvMat* labels )
{
    CvMat* counts = 0;

    CV_FUNCNAME("icvFindClusterLabels");
    __BEGIN__;

    int nclusters, nsamples;
    int i, j;
    double* probs_data;

    CV_ASSERT( ICV_IS_MAT_OF_TYPE(probs, CV_64FC1) );
    CV_ASSERT( ICV_IS_MAT_OF_TYPE(labels, CV_32SC1) );

    nclusters = probs->cols;
    nsamples  = probs->rows;
    CV_ASSERT( nsamples == labels->cols );

    CV_CALL( counts = cvCreateMat( 1, nclusters + 1, CV_32SC1 ) );
    CV_CALL( cvSetZero( counts ));
    for( i = 0; i < nsamples; i++ )
    {
        labels->data.i[i] = icvGetNumberOfCluster( probs->data.db + i*probs->cols,
            nclusters, r, outlier_thresh, 1 );
        counts->data.i[labels->data.i[i] + 1]++;
    }
    CV_ASSERT((int)cvSum(counts).val[0] == nsamples);
    // Filling empty clusters with the vector, that has the maximal probability
    for( j = 0; j < nclusters; j++ ) // outliers are ignored
    {
        int maxprob_loc = -1;
        double maxprob = 0;

        if( counts->data.i[j+1] ) // j-th class is not empty
            continue;
        // look for the presentative, which is not lonely in it's cluster
        // and that has a maximal probability among all these vectors
        probs_data = probs->data.db;
        for( i = 0; i < nsamples; i++, probs_data++ )
        {
            int label = labels->data.i[i];
            double prob;
            if( counts->data.i[label+1] == 0 ||
                (counts->data.i[label+1] <= 1 && label != -1) )
                continue;
            prob = *probs_data;
            if( prob >= maxprob )
            {
                maxprob = prob;
                maxprob_loc = i;
            }
        }
        // maxprob_loc == 0 <=> number of vectors less then number of clusters
        CV_ASSERT( maxprob_loc >= 0 );
        counts->data.i[labels->data.i[maxprob_loc] + 1]--;
        labels->data.i[maxprob_loc] = j;
        counts->data.i[j + 1]++;
    }

    __END__;

    cvReleaseMat( &counts );
} // End of icvFindClusterLabels

/* End of file */
