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

int str_to_svm_type(String& str)
{
    if( !str.compare("C_SVC") )
        return SVM::C_SVC;
    if( !str.compare("NU_SVC") )
        return SVM::NU_SVC;
    if( !str.compare("ONE_CLASS") )
        return SVM::ONE_CLASS;
    if( !str.compare("EPS_SVR") )
        return SVM::EPS_SVR;
    if( !str.compare("NU_SVR") )
        return SVM::NU_SVR;
    CV_Error( CV_StsBadArg, "incorrect svm type string" );
    return -1;
}
int str_to_svm_kernel_type( String& str )
{
    if( !str.compare("LINEAR") )
        return SVM::LINEAR;
    if( !str.compare("POLY") )
        return SVM::POLY;
    if( !str.compare("RBF") )
        return SVM::RBF;
    if( !str.compare("SIGMOID") )
        return SVM::SIGMOID;
    CV_Error( CV_StsBadArg, "incorrect svm type string" );
    return -1;
}

Ptr<SVM> svm_train_auto( Ptr<TrainData> _data, SVM::Params _params,
                    int k_fold, ParamGrid C_grid, ParamGrid gamma_grid,
                    ParamGrid p_grid, ParamGrid nu_grid, ParamGrid coef_grid,
                    ParamGrid degree_grid )
{
    Mat _train_data = _data->getSamples();
    Mat _responses = _data->getResponses();
    Mat _var_idx = _data->getVarIdx();
    Mat _sample_idx = _data->getTrainSampleIdx();

    Ptr<SVM> svm = SVM::create(_params);
    if( svm->trainAuto( _data, k_fold, C_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid ) )
        return svm;
    return Ptr<SVM>();
}

// 4. em
// 5. ann
int str_to_ann_train_method( String& str )
{
    if( !str.compare("BACKPROP") )
        return ANN_MLP::Params::BACKPROP;
    if( !str.compare("RPROP") )
        return ANN_MLP::Params::RPROP;
    CV_Error( CV_StsBadArg, "incorrect ann train method string" );
    return -1;
}

void ann_check_data( Ptr<TrainData> _data )
{
    Mat values = _data->getSamples();
    Mat var_idx = _data->getVarIdx();
    int nvars = (int)var_idx.total();
    if( nvars != 0 && nvars != values.cols )
        CV_Error( CV_StsBadArg, "var_idx is not supported" );
    if( !_data->getMissing().empty() )
        CV_Error( CV_StsBadArg, "missing values are not supported" );
}

// unroll the categorical responses to binary vectors
Mat ann_get_new_responses( Ptr<TrainData> _data, map<int, int>& cls_map )
{
    Mat train_sidx = _data->getTrainSampleIdx();
    int* train_sidx_ptr = train_sidx.ptr<int>();
    Mat responses = _data->getResponses();
    int cls_count = 0;
    // construct cls_map
    cls_map.clear();
    int nresponses = (int)responses.total();
    int si, n = !train_sidx.empty() ? (int)train_sidx.total() : nresponses;

    for( si = 0; si < n; si++ )
    {
        int sidx = train_sidx_ptr ? train_sidx_ptr[si] : si;
        int r = cvRound(responses.at<float>(sidx));
        CV_DbgAssert( fabs(responses.at<float>(sidx) - r) < FLT_EPSILON );
        map<int,int>::iterator it = cls_map.find(r);
        if( it == cls_map.end() )
            cls_map[r] = cls_count++;
    }
    Mat new_responses = Mat::zeros( nresponses, cls_count, CV_32F );
    for( si = 0; si < n; si++ )
    {
        int sidx = train_sidx_ptr ? train_sidx_ptr[si] : si;
        int r = cvRound(responses.at<float>(sidx));
        int cidx = cls_map[r];
        new_responses.at<float>(sidx, cidx) = 1.f;
    }
    return new_responses;
}

float ann_calc_error( Ptr<StatModel> ann, Ptr<TrainData> _data, map<int, int>& cls_map, int type, vector<float> *resp_labels )
{
    float err = 0;
    Mat samples = _data->getSamples();
    Mat responses = _data->getResponses();
    Mat sample_idx = (type == CV_TEST_ERROR) ? _data->getTestSampleIdx() : _data->getTrainSampleIdx();
    int* sidx = !sample_idx.empty() ? sample_idx.ptr<int>() : 0;
    ann_check_data( _data );
    int sample_count = (int)sample_idx.total();
    sample_count = (type == CV_TRAIN_ERROR && sample_count == 0) ? samples.rows : sample_count;
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

    for( int i = 0; i < sample_count; i++ )
    {
        int si = sidx ? sidx[i] : i;
        Mat sample = samples.row(si);
        ann->predict( sample, output );
        Point best_cls;
        minMaxLoc(output, 0, 0, 0, &best_cls, 0);
        int r = cvRound(responses.at<float>(si));
        CV_DbgAssert( fabs(responses.at<float>(si) - r) < FLT_EPSILON );
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
int str_to_boost_type( String& str )
{
    if ( !str.compare("DISCRETE") )
        return Boost::DISCRETE;
    if ( !str.compare("REAL") )
        return Boost::REAL;
    if ( !str.compare("LOGIT") )
        return Boost::LOGIT;
    if ( !str.compare("GENTLE") )
        return Boost::GENTLE;
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
}

CV_MLBaseTest::~CV_MLBaseTest()
{
    if( validationFS.isOpened() )
        validationFS.release();
    theRNG().state = initSeed;
}

int CV_MLBaseTest::read_params( CvFileStorage* __fs )
{
    FileStorage _fs(__fs, false);
    if( !_fs.isOpened() )
        test_case_count = -1;
    else
    {
        FileNode fn = _fs.getFirstTopLevelNode()["run_params"][modelName];
        test_case_count = (int)fn.size();
        if( test_case_count <= 0 )
            test_case_count = -1;
        if( test_case_count > 0 )
        {
            dataSetNames.resize( test_case_count );
            FileNodeIterator it = fn.begin();
            for( int i = 0; i < test_case_count; i++, ++it )
            {
                dataSetNames[i] = (string)*it;
            }
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
    clear();

    string dataPath = ts->get_data_path();
    if ( dataPath.empty() )
    {
        ts->printf( cvtest::TS::LOG, "data path is empty" );
        return cvtest::TS::FAIL_INVALID_TEST_DATA;
    }

    string dataName = dataSetNames[test_case_idx],
        filename = dataPath + dataName + ".data";

    FileNode dataParamsNode = validationFS.getFirstTopLevelNode()["validation"][modelName][dataName]["data_params"];
    CV_DbgAssert( !dataParamsNode.empty() );

    CV_DbgAssert( !dataParamsNode["LS"].empty() );
    int trainSampleCount = (int)dataParamsNode["LS"];

    CV_DbgAssert( !dataParamsNode["resp_idx"].empty() );
    int respIdx = (int)dataParamsNode["resp_idx"];

    CV_DbgAssert( !dataParamsNode["types"].empty() );
    String varTypes = (String)dataParamsNode["types"];

    data = TrainData::loadFromCSV(filename, 0, respIdx, respIdx+1, varTypes);
    if( data.empty() )
    {
        ts->printf( cvtest::TS::LOG, "file %s can not be read\n", filename.c_str() );
        return cvtest::TS::FAIL_INVALID_TEST_DATA;
    }

    data->setTrainTestSplit(trainSampleCount);
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

    if( modelName == CV_NBAYES )
        model = NormalBayesClassifier::create();
    else if( modelName == CV_KNEAREST )
    {
        model = KNearest::create();
    }
    else if( modelName == CV_SVM )
    {
        String svm_type_str, kernel_type_str;
        modelParamsNode["svm_type"] >> svm_type_str;
        modelParamsNode["kernel_type"] >> kernel_type_str;
        SVM::Params params;
        params.svmType = str_to_svm_type( svm_type_str );
        params.kernelType = str_to_svm_kernel_type( kernel_type_str );
        modelParamsNode["degree"] >> params.degree;
        modelParamsNode["gamma"] >> params.gamma;
        modelParamsNode["coef0"] >> params.coef0;
        modelParamsNode["C"] >> params.C;
        modelParamsNode["nu"] >> params.nu;
        modelParamsNode["p"] >> params.p;
        model = SVM::create(params);
    }
    else if( modelName == CV_EM )
    {
        assert( 0 );
    }
    else if( modelName == CV_ANN )
    {
        String train_method_str;
        double param1, param2;
        modelParamsNode["train_method"] >> train_method_str;
        modelParamsNode["param1"] >> param1;
        modelParamsNode["param2"] >> param2;
        Mat new_responses = ann_get_new_responses( data, cls_map );
        // binarize the responses
        data = TrainData::create(data->getSamples(), data->getLayout(), new_responses,
                                 data->getVarIdx(), data->getTrainSampleIdx());
        int layer_sz[] = { data->getNAllVars(), 100, 100, (int)cls_map.size() };
        Mat layer_sizes( 1, (int)(sizeof(layer_sz)/sizeof(layer_sz[0])), CV_32S, layer_sz );
        model = ANN_MLP::create(ANN_MLP::Params(layer_sizes, ANN_MLP::SIGMOID_SYM, 0, 0,
                                                TermCriteria(TermCriteria::COUNT,300,0.01),
                                                str_to_ann_train_method(train_method_str), param1, param2));
    }
    else if( modelName == CV_DTREE )
    {
        int MAX_DEPTH, MIN_SAMPLE_COUNT, MAX_CATEGORIES, CV_FOLDS;
        float REG_ACCURACY = 0;
        bool USE_SURROGATE = false, IS_PRUNED;
        modelParamsNode["max_depth"] >> MAX_DEPTH;
        modelParamsNode["min_sample_count"] >> MIN_SAMPLE_COUNT;
        //modelParamsNode["use_surrogate"] >> USE_SURROGATE;
        modelParamsNode["max_categories"] >> MAX_CATEGORIES;
        modelParamsNode["cv_folds"] >> CV_FOLDS;
        modelParamsNode["is_pruned"] >> IS_PRUNED;
        model = DTrees::create(DTrees::Params(MAX_DEPTH, MIN_SAMPLE_COUNT, REG_ACCURACY, USE_SURROGATE,
                                MAX_CATEGORIES, CV_FOLDS, false, IS_PRUNED, Mat() ));
    }
    else if( modelName == CV_BOOST )
    {
        int BOOST_TYPE, WEAK_COUNT, MAX_DEPTH;
        float WEIGHT_TRIM_RATE;
        bool USE_SURROGATE = false;
        String typeStr;
        modelParamsNode["type"] >> typeStr;
        BOOST_TYPE = str_to_boost_type( typeStr );
        modelParamsNode["weak_count"] >> WEAK_COUNT;
        modelParamsNode["weight_trim_rate"] >> WEIGHT_TRIM_RATE;
        modelParamsNode["max_depth"] >> MAX_DEPTH;
        //modelParamsNode["use_surrogate"] >> USE_SURROGATE;
        model = Boost::create( Boost::Params(BOOST_TYPE, WEAK_COUNT, WEIGHT_TRIM_RATE, MAX_DEPTH, USE_SURROGATE, Mat()) );
    }
    else if( modelName == CV_RTREES )
    {
        int MAX_DEPTH, MIN_SAMPLE_COUNT, MAX_CATEGORIES, CV_FOLDS, NACTIVE_VARS, MAX_TREES_NUM;
        float REG_ACCURACY = 0, OOB_EPS = 0.0;
        bool USE_SURROGATE = false, IS_PRUNED;
        modelParamsNode["max_depth"] >> MAX_DEPTH;
        modelParamsNode["min_sample_count"] >> MIN_SAMPLE_COUNT;
        //modelParamsNode["use_surrogate"] >> USE_SURROGATE;
        modelParamsNode["max_categories"] >> MAX_CATEGORIES;
        modelParamsNode["cv_folds"] >> CV_FOLDS;
        modelParamsNode["is_pruned"] >> IS_PRUNED;
        modelParamsNode["nactive_vars"] >> NACTIVE_VARS;
        modelParamsNode["max_trees_num"] >> MAX_TREES_NUM;
        model = RTrees::create(RTrees::Params( MAX_DEPTH, MIN_SAMPLE_COUNT, REG_ACCURACY,
            USE_SURROGATE, MAX_CATEGORIES, Mat(), true, // (calc_var_importance == true) <=> RF processes variable importance
            NACTIVE_VARS, TermCriteria(TermCriteria::COUNT, MAX_TREES_NUM, OOB_EPS)));
    }

    if( !model.empty() )
        is_trained = model->train(data, 0);

    if( !is_trained )
    {
        ts->printf( cvtest::TS::LOG, "in test case %d model training was failed", testCaseIdx );
        return cvtest::TS::FAIL_INVALID_OUTPUT;
    }
    return cvtest::TS::OK;
}

float CV_MLBaseTest::get_test_error( int /*testCaseIdx*/, vector<float> *resp )
{
    int type = CV_TEST_ERROR;
    float err = 0;
    Mat _resp;
    if( modelName == CV_EM )
        assert( 0 );
    else if( modelName == CV_ANN )
        err = ann_calc_error( model, data, cls_map, type, resp );
    else if( modelName == CV_DTREE || modelName == CV_BOOST || modelName == CV_RTREES ||
             modelName == CV_SVM || modelName == CV_NBAYES || modelName == CV_KNEAREST )
        err = model->calcError( data, true, _resp );
    if( !_resp.empty() && resp )
        _resp.convertTo(*resp, CV_32F);
    return err;
}

void CV_MLBaseTest::save( const char* filename )
{
    model->save( filename );
}

void CV_MLBaseTest::load( const char* filename )
{
    if( modelName == CV_NBAYES )
        model = StatModel::load<NormalBayesClassifier>( filename );
    else if( modelName == CV_KNEAREST )
        model = StatModel::load<KNearest>( filename );
    else if( modelName == CV_SVM )
        model = StatModel::load<SVM>( filename );
    else if( modelName == CV_ANN )
        model = StatModel::load<ANN_MLP>( filename );
    else if( modelName == CV_DTREE )
        model = StatModel::load<DTrees>( filename );
    else if( modelName == CV_BOOST )
        model = StatModel::load<Boost>( filename );
    else if( modelName == CV_RTREES )
        model = StatModel::load<RTrees>( filename );
    else
        CV_Error( CV_StsNotImplemented, "invalid stat model name");
}

/* End of file. */
