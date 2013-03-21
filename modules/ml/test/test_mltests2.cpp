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
//                For Open Source Computer Vision Library
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

#include "test_precomp.hpp"

using namespace cv;
using namespace std;

// auxiliary functions
// 1. nbayes
void nbayes_check_data( CvMLData* _data )
{
    if( _data->get_missing() )
        CV_Error( CV_StsBadArg, "missing values are not supported" );
    const CvMat* var_types = _data->get_var_types();
    bool is_classifier = var_types->data.ptr[var_types->cols-1] == CV_VAR_CATEGORICAL;
    if( ( fabs( cvNorm( var_types, 0, CV_L1 ) -
        (var_types->rows + var_types->cols - 2)*CV_VAR_ORDERED - CV_VAR_CATEGORICAL ) > FLT_EPSILON ) ||
        !is_classifier )
        CV_Error( CV_StsBadArg, "incorrect types of predictors or responses" );
}
bool nbayes_train( CvNormalBayesClassifier* nbayes, CvMLData* _data )
{
    nbayes_check_data( _data );
    const CvMat* values = _data->get_values();
    const CvMat* responses = _data->get_responses();
    const CvMat* train_sidx = _data->get_train_sample_idx();
    const CvMat* var_idx = _data->get_var_idx();
    return nbayes->train( values, responses, var_idx, train_sidx );
}
float nbayes_calc_error( CvNormalBayesClassifier* nbayes, CvMLData* _data, int type, vector<float> *resp )
{
    float err = 0;
    nbayes_check_data( _data );
    const CvMat* values = _data->get_values();
    const CvMat* response = _data->get_responses();
    const CvMat* sample_idx = (type == CV_TEST_ERROR) ? _data->get_test_sample_idx() : _data->get_train_sample_idx();
    int* sidx = sample_idx ? sample_idx->data.i : 0;
    int r_step = CV_IS_MAT_CONT(response->type) ?
        1 : response->step / CV_ELEM_SIZE(response->type);
    int sample_count = sample_idx ? sample_idx->cols : 0;
    sample_count = (type == CV_TRAIN_ERROR && sample_count == 0) ? values->rows : sample_count;
    float* pred_resp = 0;
    if( resp && (sample_count > 0) )
    {
        resp->resize( sample_count );
        pred_resp = &((*resp)[0]);
    }

    for( int i = 0; i < sample_count; i++ )
    {
        CvMat sample;
        int si = sidx ? sidx[i] : i;
        cvGetRow( values, &sample, si );
        float r = (float)nbayes->predict( &sample, 0 );
        if( pred_resp )
            pred_resp[i] = r;
        int d = fabs((double)r - response->data.fl[si*r_step]) <= FLT_EPSILON ? 0 : 1;
        err += d;
    }
    err = sample_count ? err / (float)sample_count * 100 : -FLT_MAX;
    return err;
}

// 2. knearest
void knearest_check_data_and_get_predictors( CvMLData* _data, CvMat* _predictors )
{
    const CvMat* values = _data->get_values();
    const CvMat* var_idx = _data->get_var_idx();
    if( var_idx->cols + var_idx->rows != values->cols )
        CV_Error( CV_StsBadArg, "var_idx is not supported" );
    if( _data->get_missing() )
        CV_Error( CV_StsBadArg, "missing values are not supported" );
    int resp_idx = _data->get_response_idx();
    if( resp_idx == 0)
        cvGetCols( values, _predictors, 1, values->cols );
    else if( resp_idx == values->cols - 1 )
        cvGetCols( values, _predictors, 0, values->cols - 1 );
    else
        CV_Error( CV_StsBadArg, "responses must be in the first or last column; other cases are not supported" );
}
bool knearest_train( CvKNearest* knearest, CvMLData* _data )
{
    const CvMat* responses = _data->get_responses();
    const CvMat* train_sidx = _data->get_train_sample_idx();
    bool is_regression = _data->get_var_type( _data->get_response_idx() ) == CV_VAR_ORDERED;
    CvMat predictors;
    knearest_check_data_and_get_predictors( _data, &predictors );
    return knearest->train( &predictors, responses, train_sidx, is_regression );
}
float knearest_calc_error( CvKNearest* knearest, CvMLData* _data, int k, int type, vector<float> *resp )
{
    float err = 0;
    const CvMat* response = _data->get_responses();
    const CvMat* sample_idx = (type == CV_TEST_ERROR) ? _data->get_test_sample_idx() : _data->get_train_sample_idx();
    int* sidx = sample_idx ? sample_idx->data.i : 0;
    int r_step = CV_IS_MAT_CONT(response->type) ?
        1 : response->step / CV_ELEM_SIZE(response->type);
    bool is_regression = _data->get_var_type( _data->get_response_idx() ) == CV_VAR_ORDERED;
    CvMat predictors;
    knearest_check_data_and_get_predictors( _data, &predictors );
    int sample_count = sample_idx ? sample_idx->cols : 0;
    sample_count = (type == CV_TRAIN_ERROR && sample_count == 0) ? predictors.rows : sample_count;
    float* pred_resp = 0;
    if( resp && (sample_count > 0) )
    {
        resp->resize( sample_count );
        pred_resp = &((*resp)[0]);
    }
    if ( !is_regression )
    {
        for( int i = 0; i < sample_count; i++ )
        {
            CvMat sample;
            int si = sidx ? sidx[i] : i;
            cvGetRow( &predictors, &sample, si );
            float r = knearest->find_nearest( &sample, k );
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
            CvMat sample;
            int si = sidx ? sidx[i] : i;
            cvGetRow( &predictors, &sample, si );
            float r = knearest->find_nearest( &sample, k );
            if( pred_resp )
                pred_resp[i] = r;
            float d = r - response->data.fl[si*r_step];
            err += d*d;
        }
        err = sample_count ? err / (float)sample_count : -FLT_MAX;
    }
    return err;
}

// 3. svm
int str_to_svm_type(string& str)
{
    if( !str.compare("C_SVC") )
        return CvSVM::C_SVC;
    if( !str.compare("NU_SVC") )
        return CvSVM::NU_SVC;
    if( !str.compare("ONE_CLASS") )
        return CvSVM::ONE_CLASS;
    if( !str.compare("EPS_SVR") )
        return CvSVM::EPS_SVR;
    if( !str.compare("NU_SVR") )
        return CvSVM::NU_SVR;
    CV_Error( CV_StsBadArg, "incorrect svm type string" );
    return -1;
}
int str_to_svm_kernel_type( string& str )
{
    if( !str.compare("LINEAR") )
        return CvSVM::LINEAR;
    if( !str.compare("POLY") )
        return CvSVM::POLY;
    if( !str.compare("RBF") )
        return CvSVM::RBF;
    if( !str.compare("SIGMOID") )
        return CvSVM::SIGMOID;
    CV_Error( CV_StsBadArg, "incorrect svm type string" );
    return -1;
}
void svm_check_data( CvMLData* _data )
{
    if( _data->get_missing() )
        CV_Error( CV_StsBadArg, "missing values are not supported" );
    const CvMat* var_types = _data->get_var_types();
    for( int i = 0; i < var_types->cols-1; i++ )
        if (var_types->data.ptr[i] == CV_VAR_CATEGORICAL)
        {
            char msg[50];
            sprintf( msg, "incorrect type of %d-predictor", i );
            CV_Error( CV_StsBadArg, msg );
        }
}
bool svm_train( CvSVM* svm, CvMLData* _data, CvSVMParams _params )
{
    svm_check_data(_data);
    const CvMat* _train_data = _data->get_values();
    const CvMat* _responses = _data->get_responses();
    const CvMat* _var_idx = _data->get_var_idx();
    const CvMat* _sample_idx = _data->get_train_sample_idx();
    return svm->train( _train_data, _responses, _var_idx, _sample_idx, _params );
}
bool svm_train_auto( CvSVM* svm, CvMLData* _data, CvSVMParams _params,
                    int k_fold, CvParamGrid C_grid, CvParamGrid gamma_grid,
                    CvParamGrid p_grid, CvParamGrid nu_grid, CvParamGrid coef_grid,
                    CvParamGrid degree_grid )
{
    svm_check_data(_data);
    const CvMat* _train_data = _data->get_values();
    const CvMat* _responses = _data->get_responses();
    const CvMat* _var_idx = _data->get_var_idx();
    const CvMat* _sample_idx = _data->get_train_sample_idx();
    return svm->train_auto( _train_data, _responses, _var_idx,
        _sample_idx, _params, k_fold, C_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid );
}
float svm_calc_error( CvSVM* svm, CvMLData* _data, int type, vector<float> *resp )
{
    svm_check_data(_data);
    float err = 0;
    const CvMat* values = _data->get_values();
    const CvMat* response = _data->get_responses();
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
            CvMat sample;
            int si = sidx ? sidx[i] : i;
            cvGetRow( values, &sample, si );
            float r = svm->predict( &sample );
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
            CvMat sample;
            int si = sidx ? sidx[i] : i;
            cvGetRow( values, &sample, si );
            float r = svm->predict( &sample );
            if( pred_resp )
                pred_resp[i] = r;
            float d = r - response->data.fl[si*r_step];
            err += d*d;
        }
        err = sample_count ? err / (float)sample_count : -FLT_MAX;
    }
    return err;
}

// 4. em
// 5. ann
int str_to_ann_train_method( string& str )
{
    if( !str.compare("BACKPROP") )
        return CvANN_MLP_TrainParams::BACKPROP;
    if( !str.compare("RPROP") )
        return CvANN_MLP_TrainParams::RPROP;
    CV_Error( CV_StsBadArg, "incorrect ann train method string" );
    return -1;
}
void ann_check_data_and_get_predictors( CvMLData* _data, CvMat* _inputs )
{
    const CvMat* values = _data->get_values();
    const CvMat* var_idx = _data->get_var_idx();
    if( var_idx->cols + var_idx->rows != values->cols )
        CV_Error( CV_StsBadArg, "var_idx is not supported" );
    if( _data->get_missing() )
        CV_Error( CV_StsBadArg, "missing values are not supported" );
    int resp_idx = _data->get_response_idx();
    if( resp_idx == 0)
        cvGetCols( values, _inputs, 1, values->cols );
    else if( resp_idx == values->cols - 1 )
        cvGetCols( values, _inputs, 0, values->cols - 1 );
    else
        CV_Error( CV_StsBadArg, "outputs must be in the first or last column; other cases are not supported" );
}
void ann_get_new_responses( CvMLData* _data, Mat& new_responses, map<int, int>& cls_map )
{
    const CvMat* train_sidx = _data->get_train_sample_idx();
    int* train_sidx_ptr = train_sidx->data.i;
    const CvMat* responses = _data->get_responses();
    float* responses_ptr = responses->data.fl;
    int r_step = CV_IS_MAT_CONT(responses->type) ?
        1 : responses->step / CV_ELEM_SIZE(responses->type);
    int cls_count = 0;
    // construct cls_map
    cls_map.clear();
    for( int si = 0; si < train_sidx->cols; si++ )
    {
        int sidx = train_sidx_ptr[si];
        int r = cvRound(responses_ptr[sidx*r_step]);
        CV_DbgAssert( fabs(responses_ptr[sidx*r_step]-r) < FLT_EPSILON );
        int cls_map_size = (int)cls_map.size();
        cls_map[r];
        if ( (int)cls_map.size() > cls_map_size )
            cls_map[r] = cls_count++;
    }
    new_responses.create( responses->rows, cls_count, CV_32F );
    new_responses.setTo( 0 );
    for( int si = 0; si < train_sidx->cols; si++ )
    {
        int sidx = train_sidx_ptr[si];
        int r = cvRound(responses_ptr[sidx*r_step]);
        int cidx = cls_map[r];
        new_responses.ptr<float>(sidx)[cidx] = 1;
    }
}
int ann_train( CvANN_MLP* ann, CvMLData* _data, Mat& new_responses, CvANN_MLP_TrainParams _params, int flags = 0 )
{
    const CvMat* train_sidx = _data->get_train_sample_idx();
    CvMat predictors;
    ann_check_data_and_get_predictors( _data, &predictors );
    CvMat _new_responses = CvMat( new_responses );
    return ann->train( &predictors, &_new_responses, 0, train_sidx, _params, flags );
}
float ann_calc_error( CvANN_MLP* ann, CvMLData* _data, map<int, int>& cls_map, int type , vector<float> *resp_labels )
{
    float err = 0;
    const CvMat* responses = _data->get_responses();
    const CvMat* sample_idx = (type == CV_TEST_ERROR) ? _data->get_test_sample_idx() : _data->get_train_sample_idx();
    int* sidx = sample_idx ? sample_idx->data.i : 0;
    int r_step = CV_IS_MAT_CONT(responses->type) ?
        1 : responses->step / CV_ELEM_SIZE(responses->type);
    CvMat predictors;
    ann_check_data_and_get_predictors( _data, &predictors );
    int sample_count = sample_idx ? sample_idx->cols : 0;
    sample_count = (type == CV_TRAIN_ERROR && sample_count == 0) ? predictors.rows : sample_count;
    float* pred_resp = 0;
    vector<float> innresp;
    if( sample_count > 0 )
    {
        if( resp_labels )
        {
            resp_labels->resize( sample_count );
            pred_resp = &((*resp_labels)[0]);
        }
        else
        {
            innresp.resize( sample_count );
            pred_resp = &(innresp[0]);
        }
    }
    int cls_count = (int)cls_map.size();
    Mat output( 1, cls_count, CV_32FC1 );
    CvMat _output = CvMat(output);
    for( int i = 0; i < sample_count; i++ )
    {
        CvMat sample;
        int si = sidx ? sidx[i] : i;
        cvGetRow( &predictors, &sample, si );
        ann->predict( &sample, &_output );
        CvPoint best_cls = {0,0};
        cvMinMaxLoc( &_output, 0, 0, 0, &best_cls, 0 );
        int r = cvRound(responses->data.fl[si*r_step]);
        CV_DbgAssert( fabs(responses->data.fl[si*r_step]-r) < FLT_EPSILON );
        r = cls_map[r];
        int d = best_cls.x == r ? 0 : 1;
        err += d;
        pred_resp[i] = (float)best_cls.x;
    }
    err = sample_count ? err / (float)sample_count * 100 : -FLT_MAX;
    return err;
}

// 6. dtree
// 7. boost
int str_to_boost_type( string& str )
{
    if ( !str.compare("DISCRETE") )
        return CvBoost::DISCRETE;
    if ( !str.compare("REAL") )
        return CvBoost::REAL;
    if ( !str.compare("LOGIT") )
        return CvBoost::LOGIT;
    if ( !str.compare("GENTLE") )
        return CvBoost::GENTLE;
    CV_Error( CV_StsBadArg, "incorrect boost type string" );
    return -1;
}

// 8. rtrees
// 9. ertrees

// ---------------------------------- MLBaseTest ---------------------------------------------------

CV_MLBaseTest::CV_MLBaseTest(const char* _modelName)
{
    int64 seeds[] = { CV_BIG_INT(0x00009fff4f9c8d52),
                      CV_BIG_INT(0x0000a17166072c7c),
                      CV_BIG_INT(0x0201b32115cd1f9a),
                      CV_BIG_INT(0x0513cb37abcd1234),
                      CV_BIG_INT(0x0001a2b3c4d5f678)
                    };

    int seedCount = sizeof(seeds)/sizeof(seeds[0]);
    RNG& rng = theRNG();

    initSeed = rng.state;

    rng.state = seeds[rng(seedCount)];

    modelName = _modelName;
    nbayes = 0;
    knearest = 0;
    svm = 0;
    ann = 0;
    dtree = 0;
    boost = 0;
    rtrees = 0;
    ertrees = 0;
    if( !modelName.compare(CV_NBAYES) )
        nbayes = new CvNormalBayesClassifier;
    else if( !modelName.compare(CV_KNEAREST) )
        knearest = new CvKNearest;
    else if( !modelName.compare(CV_SVM) )
        svm = new CvSVM;
    else if( !modelName.compare(CV_ANN) )
        ann = new CvANN_MLP;
    else if( !modelName.compare(CV_DTREE) )
        dtree = new CvDTree;
    else if( !modelName.compare(CV_BOOST) )
        boost = new CvBoost;
    else if( !modelName.compare(CV_RTREES) )
        rtrees = new CvRTrees;
    else if( !modelName.compare(CV_ERTREES) )
        ertrees = new CvERTrees;
}

CV_MLBaseTest::~CV_MLBaseTest()
{
    if( validationFS.isOpened() )
        validationFS.release();
    if( nbayes )
        delete nbayes;
    if( knearest )
        delete knearest;
    if( svm )
        delete svm;
    if( ann )
        delete ann;
    if( dtree )
        delete dtree;
    if( boost )
        delete boost;
    if( rtrees )
        delete rtrees;
    if( ertrees )
        delete ertrees;
    theRNG().state = initSeed;
}

int CV_MLBaseTest::read_params( CvFileStorage* _fs )
{
    if( !_fs )
        test_case_count = -1;
    else
    {
        CvFileNode* fn = cvGetRootFileNode( _fs, 0 );
        fn = (CvFileNode*)cvGetSeqElem( fn->data.seq, 0 );
        fn = cvGetFileNodeByName( _fs, fn, "run_params" );
        CvSeq* dataSetNamesSeq = cvGetFileNodeByName( _fs, fn, modelName.c_str() )->data.seq;
        test_case_count = dataSetNamesSeq ? dataSetNamesSeq->total : -1;
        if( test_case_count > 0 )
        {
            dataSetNames.resize( test_case_count );
            vector<string>::iterator it = dataSetNames.begin();
            for( int i = 0; i < test_case_count; i++, it++ )
                *it = ((CvFileNode*)cvGetSeqElem( dataSetNamesSeq, i ))->data.str.ptr;
        }
    }
    return cvtest::TS::OK;;
}

void CV_MLBaseTest::run( int )
{
    string filename = ts->get_data_path();
    filename += get_validation_filename();
    validationFS.open( filename, FileStorage::READ );
    read_params( *validationFS );

    int code = cvtest::TS::OK;
    for (int i = 0; i < test_case_count; i++)
    {
        int temp_code = run_test_case( i );
        if (temp_code == cvtest::TS::OK)
            temp_code = validate_test_results( i );
        if (temp_code != cvtest::TS::OK)
            code = temp_code;
    }
    if ( test_case_count <= 0)
    {
        ts->printf( cvtest::TS::LOG, "validation file is not determined or not correct" );
        code = cvtest::TS::FAIL_INVALID_TEST_DATA;
    }
    ts->set_failed_test_info( code );
}

int CV_MLBaseTest::prepare_test_case( int test_case_idx )
{
    int trainSampleCount, respIdx;
    string varTypes;
    clear();

    string dataPath = ts->get_data_path();
    if ( dataPath.empty() )
    {
        ts->printf( cvtest::TS::LOG, "data path is empty" );
        return cvtest::TS::FAIL_INVALID_TEST_DATA;
    }

    string dataName = dataSetNames[test_case_idx],
        filename = dataPath + dataName + ".data";
    if ( data.read_csv( filename.c_str() ) != 0)
    {
        char msg[100];
        sprintf( msg, "file %s can not be read", filename.c_str() );
        ts->printf( cvtest::TS::LOG, msg );
        return cvtest::TS::FAIL_INVALID_TEST_DATA;
    }

    FileNode dataParamsNode = validationFS.getFirstTopLevelNode()["validation"][modelName][dataName]["data_params"];
    CV_DbgAssert( !dataParamsNode.empty() );

    CV_DbgAssert( !dataParamsNode["LS"].empty() );
    dataParamsNode["LS"] >> trainSampleCount;
    CvTrainTestSplit spl( trainSampleCount );
    data.set_train_test_split( &spl );

    CV_DbgAssert( !dataParamsNode["resp_idx"].empty() );
    dataParamsNode["resp_idx"] >> respIdx;
    data.set_response_idx( respIdx );

    CV_DbgAssert( !dataParamsNode["types"].empty() );
    dataParamsNode["types"] >> varTypes;
    data.set_var_types( varTypes.c_str() );

    return cvtest::TS::OK;
}

string& CV_MLBaseTest::get_validation_filename()
{
    return validationFN;
}

int CV_MLBaseTest::train( int testCaseIdx )
{
    bool is_trained = false;
    FileNode modelParamsNode =
        validationFS.getFirstTopLevelNode()["validation"][modelName][dataSetNames[testCaseIdx]]["model_params"];

    if( !modelName.compare(CV_NBAYES) )
        is_trained = nbayes_train( nbayes, &data );
    else if( !modelName.compare(CV_KNEAREST) )
    {
        assert( 0 );
        //is_trained = knearest->train( &data );
    }
    else if( !modelName.compare(CV_SVM) )
    {
        string svm_type_str, kernel_type_str;
        modelParamsNode["svm_type"] >> svm_type_str;
        modelParamsNode["kernel_type"] >> kernel_type_str;
        CvSVMParams params;
        params.svm_type = str_to_svm_type( svm_type_str );
        params.kernel_type = str_to_svm_kernel_type( kernel_type_str );
        modelParamsNode["degree"] >> params.degree;
        modelParamsNode["gamma"] >> params.gamma;
        modelParamsNode["coef0"] >> params.coef0;
        modelParamsNode["C"] >> params.C;
        modelParamsNode["nu"] >> params.nu;
        modelParamsNode["p"] >> params.p;
        is_trained = svm_train( svm, &data, params );
    }
    else if( !modelName.compare(CV_EM) )
    {
        assert( 0 );
    }
    else if( !modelName.compare(CV_ANN) )
    {
        string train_method_str;
        double param1, param2;
        modelParamsNode["train_method"] >> train_method_str;
        modelParamsNode["param1"] >> param1;
        modelParamsNode["param2"] >> param2;
        Mat new_responses;
        ann_get_new_responses( &data, new_responses, cls_map );
        int layer_sz[] = { data.get_values()->cols - 1, 100, 100, (int)cls_map.size() };
        CvMat layer_sizes =
            cvMat( 1, (int)(sizeof(layer_sz)/sizeof(layer_sz[0])), CV_32S, layer_sz );
        ann->create( &layer_sizes );
        is_trained = ann_train( ann, &data, new_responses, CvANN_MLP_TrainParams(cvTermCriteria(CV_TERMCRIT_ITER,300,0.01),
            str_to_ann_train_method(train_method_str), param1, param2) ) >= 0;
    }
    else if( !modelName.compare(CV_DTREE) )
    {
        int MAX_DEPTH, MIN_SAMPLE_COUNT, MAX_CATEGORIES, CV_FOLDS;
        float REG_ACCURACY = 0;
        bool USE_SURROGATE, IS_PRUNED;
        modelParamsNode["max_depth"] >> MAX_DEPTH;
        modelParamsNode["min_sample_count"] >> MIN_SAMPLE_COUNT;
        modelParamsNode["use_surrogate"] >> USE_SURROGATE;
        modelParamsNode["max_categories"] >> MAX_CATEGORIES;
        modelParamsNode["cv_folds"] >> CV_FOLDS;
        modelParamsNode["is_pruned"] >> IS_PRUNED;
        is_trained = dtree->train( &data,
            CvDTreeParams(MAX_DEPTH, MIN_SAMPLE_COUNT, REG_ACCURACY, USE_SURROGATE,
            MAX_CATEGORIES, CV_FOLDS, false, IS_PRUNED, 0 )) != 0;
    }
    else if( !modelName.compare(CV_BOOST) )
    {
        int BOOST_TYPE, WEAK_COUNT, MAX_DEPTH;
        float WEIGHT_TRIM_RATE;
        bool USE_SURROGATE;
        string typeStr;
        modelParamsNode["type"] >> typeStr;
        BOOST_TYPE = str_to_boost_type( typeStr );
        modelParamsNode["weak_count"] >> WEAK_COUNT;
        modelParamsNode["weight_trim_rate"] >> WEIGHT_TRIM_RATE;
        modelParamsNode["max_depth"] >> MAX_DEPTH;
        modelParamsNode["use_surrogate"] >> USE_SURROGATE;
        is_trained = boost->train( &data,
            CvBoostParams(BOOST_TYPE, WEAK_COUNT, WEIGHT_TRIM_RATE, MAX_DEPTH, USE_SURROGATE, 0) ) != 0;
    }
    else if( !modelName.compare(CV_RTREES) )
    {
        int MAX_DEPTH, MIN_SAMPLE_COUNT, MAX_CATEGORIES, CV_FOLDS, NACTIVE_VARS, MAX_TREES_NUM;
        float REG_ACCURACY = 0, OOB_EPS = 0.0;
        bool USE_SURROGATE, IS_PRUNED;
        modelParamsNode["max_depth"] >> MAX_DEPTH;
        modelParamsNode["min_sample_count"] >> MIN_SAMPLE_COUNT;
        modelParamsNode["use_surrogate"] >> USE_SURROGATE;
        modelParamsNode["max_categories"] >> MAX_CATEGORIES;
        modelParamsNode["cv_folds"] >> CV_FOLDS;
        modelParamsNode["is_pruned"] >> IS_PRUNED;
        modelParamsNode["nactive_vars"] >> NACTIVE_VARS;
        modelParamsNode["max_trees_num"] >> MAX_TREES_NUM;
        is_trained = rtrees->train( &data, CvRTParams(  MAX_DEPTH, MIN_SAMPLE_COUNT, REG_ACCURACY,
            USE_SURROGATE, MAX_CATEGORIES, 0, true, // (calc_var_importance == true) <=> RF processes variable importance
            NACTIVE_VARS, MAX_TREES_NUM, OOB_EPS, CV_TERMCRIT_ITER)) != 0;
    }
    else if( !modelName.compare(CV_ERTREES) )
    {
        int MAX_DEPTH, MIN_SAMPLE_COUNT, MAX_CATEGORIES, CV_FOLDS, NACTIVE_VARS, MAX_TREES_NUM;
        float REG_ACCURACY = 0, OOB_EPS = 0.0;
        bool USE_SURROGATE, IS_PRUNED;
        modelParamsNode["max_depth"] >> MAX_DEPTH;
        modelParamsNode["min_sample_count"] >> MIN_SAMPLE_COUNT;
        modelParamsNode["use_surrogate"] >> USE_SURROGATE;
        modelParamsNode["max_categories"] >> MAX_CATEGORIES;
        modelParamsNode["cv_folds"] >> CV_FOLDS;
        modelParamsNode["is_pruned"] >> IS_PRUNED;
        modelParamsNode["nactive_vars"] >> NACTIVE_VARS;
        modelParamsNode["max_trees_num"] >> MAX_TREES_NUM;
        is_trained = ertrees->train( &data, CvRTParams( MAX_DEPTH, MIN_SAMPLE_COUNT, REG_ACCURACY,
            USE_SURROGATE, MAX_CATEGORIES, 0, false, // (calc_var_importance == true) <=> RF processes variable importance
            NACTIVE_VARS, MAX_TREES_NUM, OOB_EPS, CV_TERMCRIT_ITER)) != 0;
    }

    if( !is_trained )
    {
        ts->printf( cvtest::TS::LOG, "in test case %d model training was failed", testCaseIdx );
        return cvtest::TS::FAIL_INVALID_OUTPUT;
    }
    return cvtest::TS::OK;
}

float CV_MLBaseTest::get_error( int /*testCaseIdx*/, int type, vector<float> *resp )
{
    float err = 0;
    if( !modelName.compare(CV_NBAYES) )
        err = nbayes_calc_error( nbayes, &data, type, resp );
    else if( !modelName.compare(CV_KNEAREST) )
    {
        assert( 0 );
        /*testCaseIdx = 0;
        int k = 2;
        validationFS.getFirstTopLevelNode()["validation"][modelName][dataSetNames[testCaseIdx]]["model_params"]["k"] >> k;
        err = knearest->calc_error( &data, k, type, resp );*/
    }
    else if( !modelName.compare(CV_SVM) )
        err = svm_calc_error( svm, &data, type, resp );
    else if( !modelName.compare(CV_EM) )
        assert( 0 );
    else if( !modelName.compare(CV_ANN) )
        err = ann_calc_error( ann, &data, cls_map, type, resp );
    else if( !modelName.compare(CV_DTREE) )
        err = dtree->calc_error( &data, type, resp );
    else if( !modelName.compare(CV_BOOST) )
        err = boost->calc_error( &data, type, resp );
    else if( !modelName.compare(CV_RTREES) )
        err = rtrees->calc_error( &data, type, resp );
    else if( !modelName.compare(CV_ERTREES) )
        err = ertrees->calc_error( &data, type, resp );
    return err;
}

void CV_MLBaseTest::save( const char* filename )
{
    if( !modelName.compare(CV_NBAYES) )
        nbayes->save( filename );
    else if( !modelName.compare(CV_KNEAREST) )
        knearest->save( filename );
    else if( !modelName.compare(CV_SVM) )
        svm->save( filename );
    else if( !modelName.compare(CV_ANN) )
        ann->save( filename );
    else if( !modelName.compare(CV_DTREE) )
        dtree->save( filename );
    else if( !modelName.compare(CV_BOOST) )
        boost->save( filename );
    else if( !modelName.compare(CV_RTREES) )
        rtrees->save( filename );
    else if( !modelName.compare(CV_ERTREES) )
        ertrees->save( filename );
}

void CV_MLBaseTest::load( const char* filename )
{
    if( !modelName.compare(CV_NBAYES) )
        nbayes->load( filename );
    else if( !modelName.compare(CV_KNEAREST) )
        knearest->load( filename );
    else if( !modelName.compare(CV_SVM) )
    {
        delete svm;
        svm = new CvSVM;
        svm->load( filename );
    }
    else if( !modelName.compare(CV_ANN) )
        ann->load( filename );
    else if( !modelName.compare(CV_DTREE) )
        dtree->load( filename );
    else if( !modelName.compare(CV_BOOST) )
        boost->load( filename );
    else if( !modelName.compare(CV_RTREES) )
        rtrees->load( filename );
    else if( !modelName.compare(CV_ERTREES) )
        ertrees->load( filename );
}

/* End of file. */
