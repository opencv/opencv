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

//#define GENERATE_TESTDATA

namespace opencv_test { namespace {

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
}

// 4. em
// 5. ann
int str_to_ann_train_method( String& str )
{
    if( !str.compare("BACKPROP") )
        return ANN_MLP::BACKPROP;
    if (!str.compare("RPROP"))
        return ANN_MLP::RPROP;
    if (!str.compare("ANNEAL"))
        return ANN_MLP::ANNEAL;
    CV_Error( CV_StsBadArg, "incorrect ann train method string" );
}

#if 0
int str_to_ann_activation_function(String& str)
{
    if (!str.compare("IDENTITY"))
        return ANN_MLP::IDENTITY;
    if (!str.compare("SIGMOID_SYM"))
        return ANN_MLP::SIGMOID_SYM;
    if (!str.compare("GAUSSIAN"))
        return ANN_MLP::GAUSSIAN;
    if (!str.compare("RELU"))
        return ANN_MLP::RELU;
    if (!str.compare("LEAKYRELU"))
        return ANN_MLP::LEAKYRELU;
    CV_Error(CV_StsBadArg, "incorrect ann activation function string");
}
#endif

void ann_check_data( Ptr<TrainData> _data )
{
    CV_TRACE_FUNCTION();
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
    CV_TRACE_FUNCTION();
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
    CV_TRACE_FUNCTION();
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

TEST(ML_ANN, ActivationFunction)
{
    String folder = string(cvtest::TS::ptr()->get_data_path());
    String original_path = folder + "waveform.data";
    String dataname = folder + "waveform";

    Ptr<TrainData> tdata = TrainData::loadFromCSV(original_path, 0);

    ASSERT_FALSE(tdata.empty()) << "Could not find test data file : " << original_path;
    RNG& rng = theRNG();
    rng.state = 1027401484159173092;
    tdata->setTrainTestSplit(500);

    vector<int> activationType;
    activationType.push_back(ml::ANN_MLP::IDENTITY);
    activationType.push_back(ml::ANN_MLP::SIGMOID_SYM);
    activationType.push_back(ml::ANN_MLP::GAUSSIAN);
    activationType.push_back(ml::ANN_MLP::RELU);
    activationType.push_back(ml::ANN_MLP::LEAKYRELU);
    vector<String> activationName;
    activationName.push_back("_identity");
    activationName.push_back("_sigmoid_sym");
    activationName.push_back("_gaussian");
    activationName.push_back("_relu");
    activationName.push_back("_leakyrelu");
    for (size_t i = 0; i < activationType.size(); i++)
    {
        Ptr<ml::ANN_MLP> x = ml::ANN_MLP::create();
        Mat_<int> layerSizes(1, 4);
        layerSizes(0, 0) = tdata->getNVars();
        layerSizes(0, 1) = 100;
        layerSizes(0, 2) = 100;
        layerSizes(0, 3) = tdata->getResponses().cols;
        x->setLayerSizes(layerSizes);
        x->setActivationFunction(activationType[i]);
        x->setTrainMethod(ml::ANN_MLP::RPROP, 0.01, 0.1);
        x->setTermCriteria(TermCriteria(TermCriteria::COUNT, 300, 0.01));
        x->train(tdata, ml::ANN_MLP::NO_OUTPUT_SCALE);
        ASSERT_TRUE(x->isTrained()) << "Could not train networks with  " << activationName[i];
#ifdef GENERATE_TESTDATA
        x->save(dataname + activationName[i] + ".yml");
#else
        Ptr<ml::ANN_MLP> y = Algorithm::load<ANN_MLP>(dataname + activationName[i] + ".yml");
        ASSERT_TRUE(y) << "Could not load   " << dataname + activationName[i] + ".yml";
        Mat testSamples = tdata->getTestSamples();
        Mat rx, ry, dst;
        x->predict(testSamples, rx);
        y->predict(testSamples, ry);
        double n = cvtest::norm(rx, ry, NORM_INF);
        EXPECT_LT(n,FLT_EPSILON) << "Predict are not equal for " << dataname + activationName[i] + ".yml and " << activationName[i];
#endif
    }
}

CV_ENUM(ANN_MLP_METHOD, ANN_MLP::RPROP, ANN_MLP::ANNEAL)

typedef tuple<ANN_MLP_METHOD, string, int> ML_ANN_METHOD_Params;
typedef TestWithParam<ML_ANN_METHOD_Params> ML_ANN_METHOD;

TEST_P(ML_ANN_METHOD, Test)
{
    int methodType = get<0>(GetParam());
    string methodName = get<1>(GetParam());
    int N = get<2>(GetParam());

    String folder = string(cvtest::TS::ptr()->get_data_path());
    String original_path = folder + "waveform.data";
    String dataname = folder + "waveform" + '_' + methodName;

    Ptr<TrainData> tdata2 = TrainData::loadFromCSV(original_path, 0);
    Mat samples = tdata2->getSamples()(Range(0, N), Range::all());
    Mat responses(N, 3, CV_32FC1, Scalar(0));
    for (int i = 0; i < N; i++)
        responses.at<float>(i, static_cast<int>(tdata2->getResponses().at<float>(i, 0))) = 1;
    Ptr<TrainData> tdata = TrainData::create(samples, ml::ROW_SAMPLE, responses);

    ASSERT_FALSE(tdata.empty()) << "Could not find test data file : " << original_path;
    RNG& rng = theRNG();
    rng.state = 0;
    tdata->setTrainTestSplitRatio(0.8);

    Mat testSamples = tdata->getTestSamples();

#ifdef GENERATE_TESTDATA
    {
    Ptr<ml::ANN_MLP> xx = ml::ANN_MLP::create();
    Mat_<int> layerSizesXX(1, 4);
    layerSizesXX(0, 0) = tdata->getNVars();
    layerSizesXX(0, 1) = 30;
    layerSizesXX(0, 2) = 30;
    layerSizesXX(0, 3) = tdata->getResponses().cols;
    xx->setLayerSizes(layerSizesXX);
    xx->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM);
    xx->setTrainMethod(ml::ANN_MLP::RPROP);
    xx->setTermCriteria(TermCriteria(TermCriteria::COUNT, 1, 0.01));
    xx->train(tdata, ml::ANN_MLP::NO_OUTPUT_SCALE + ml::ANN_MLP::NO_INPUT_SCALE);
    FileStorage fs;
    fs.open(dataname + "_init_weight.yml.gz", FileStorage::WRITE + FileStorage::BASE64);
    xx->write(fs);
    fs.release();
    }
#endif
    {
        FileStorage fs;
        fs.open(dataname + "_init_weight.yml.gz", FileStorage::READ);
        Ptr<ml::ANN_MLP> x = ml::ANN_MLP::create();
        x->read(fs.root());
        x->setTrainMethod(methodType);
        if (methodType == ml::ANN_MLP::ANNEAL)
        {
            x->setAnnealEnergyRNG(RNG(CV_BIG_INT(0xffffffff)));
            x->setAnnealInitialT(12);
            x->setAnnealFinalT(0.15);
            x->setAnnealCoolingRatio(0.96);
            x->setAnnealItePerStep(11);
        }
        x->setTermCriteria(TermCriteria(TermCriteria::COUNT, 100, 0.01));
        x->train(tdata, ml::ANN_MLP::NO_OUTPUT_SCALE + ml::ANN_MLP::NO_INPUT_SCALE + ml::ANN_MLP::UPDATE_WEIGHTS);
        ASSERT_TRUE(x->isTrained()) << "Could not train networks with  " << methodName;
        string filename = dataname + ".yml.gz";
        Mat r_gold;
#ifdef  GENERATE_TESTDATA
        x->save(filename);
        x->predict(testSamples, r_gold);
        {
            FileStorage fs_response(dataname + "_response.yml.gz", FileStorage::WRITE + FileStorage::BASE64);
            fs_response << "response" << r_gold;
        }
#else
        {
            FileStorage fs_response(dataname + "_response.yml.gz", FileStorage::READ);
            fs_response["response"] >> r_gold;
        }
#endif
        ASSERT_FALSE(r_gold.empty());
        Ptr<ml::ANN_MLP> y = Algorithm::load<ANN_MLP>(filename);
        ASSERT_TRUE(y) << "Could not load   " << filename;
        Mat rx, ry;
        for (int j = 0; j < 4; j++)
        {
            rx = x->getWeights(j);
            ry = y->getWeights(j);
            double n = cvtest::norm(rx, ry, NORM_INF);
            EXPECT_LT(n, FLT_EPSILON) << "Weights are not equal for layer: " << j;
        }
        x->predict(testSamples, rx);
        y->predict(testSamples, ry);
        double n = cvtest::norm(ry, rx, NORM_INF);
        EXPECT_LT(n, FLT_EPSILON) << "Predict are not equal to result of the saved model";
        n = cvtest::norm(r_gold, rx, NORM_INF);
        EXPECT_LT(n, FLT_EPSILON) << "Predict are not equal to 'gold' response";
    }
}

INSTANTIATE_TEST_CASE_P(/*none*/, ML_ANN_METHOD,
    testing::Values(
        make_tuple<ANN_MLP_METHOD, string, int>(ml::ANN_MLP::RPROP, "rprop", 5000),
        make_tuple<ANN_MLP_METHOD, string, int>(ml::ANN_MLP::ANNEAL, "anneal", 1000)
        //make_pair<ANN_MLP_METHOD, string>(ml::ANN_MLP::BACKPROP, "backprop", 5000); -----> NO BACKPROP TEST
    )
);


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
}

// 8. rtrees
// 9. ertrees

int str_to_svmsgd_type( String& str )
{
    if ( !str.compare("SGD") )
        return SVMSGD::SGD;
    if ( !str.compare("ASGD") )
        return SVMSGD::ASGD;
    CV_Error( CV_StsBadArg, "incorrect svmsgd type string" );
}

int str_to_margin_type( String& str )
{
    if ( !str.compare("SOFT_MARGIN") )
        return SVMSGD::SOFT_MARGIN;
    if ( !str.compare("HARD_MARGIN") )
        return SVMSGD::HARD_MARGIN;
    CV_Error( CV_StsBadArg, "incorrect svmsgd margin type string" );
}

}
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

int CV_MLBaseTest::read_params( const cv::FileStorage& _fs )
{
    CV_TRACE_FUNCTION();
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
    CV_TRACE_FUNCTION();
    string filename = ts->get_data_path();
    filename += get_validation_filename();
    validationFS.open( filename, FileStorage::READ );
    read_params( validationFS );

    int code = cvtest::TS::OK;
    for (int i = 0; i < test_case_count; i++)
    {
        CV_TRACE_REGION("iteration");
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
    CV_TRACE_FUNCTION();
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
    CV_TRACE_FUNCTION();
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
        Ptr<SVM> m = SVM::create();
        m->setType(str_to_svm_type( svm_type_str ));
        m->setKernel(str_to_svm_kernel_type( kernel_type_str ));
        m->setDegree(modelParamsNode["degree"]);
        m->setGamma(modelParamsNode["gamma"]);
        m->setCoef0(modelParamsNode["coef0"]);
        m->setC(modelParamsNode["C"]);
        m->setNu(modelParamsNode["nu"]);
        m->setP(modelParamsNode["p"]);
        model = m;
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
        Ptr<ANN_MLP> m = ANN_MLP::create();
        m->setLayerSizes(layer_sizes);
        m->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0, 0);
        m->setTermCriteria(TermCriteria(TermCriteria::COUNT,300,0.01));
        m->setTrainMethod(str_to_ann_train_method(train_method_str), param1, param2);
        model = m;

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

        Ptr<DTrees> m = DTrees::create();
        m->setMaxDepth(MAX_DEPTH);
        m->setMinSampleCount(MIN_SAMPLE_COUNT);
        m->setRegressionAccuracy(REG_ACCURACY);
        m->setUseSurrogates(USE_SURROGATE);
        m->setMaxCategories(MAX_CATEGORIES);
        m->setCVFolds(CV_FOLDS);
        m->setUse1SERule(false);
        m->setTruncatePrunedTree(IS_PRUNED);
        m->setPriors(Mat());
        model = m;
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

        Ptr<Boost> m = Boost::create();
        m->setBoostType(BOOST_TYPE);
        m->setWeakCount(WEAK_COUNT);
        m->setWeightTrimRate(WEIGHT_TRIM_RATE);
        m->setMaxDepth(MAX_DEPTH);
        m->setUseSurrogates(USE_SURROGATE);
        m->setPriors(Mat());
        model = m;
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

        Ptr<RTrees> m = RTrees::create();
        m->setMaxDepth(MAX_DEPTH);
        m->setMinSampleCount(MIN_SAMPLE_COUNT);
        m->setRegressionAccuracy(REG_ACCURACY);
        m->setUseSurrogates(USE_SURROGATE);
        m->setMaxCategories(MAX_CATEGORIES);
        m->setPriors(Mat());
        m->setCalculateVarImportance(true);
        m->setActiveVarCount(NACTIVE_VARS);
        m->setTermCriteria(TermCriteria(TermCriteria::COUNT, MAX_TREES_NUM, OOB_EPS));
        model = m;
    }

    else if( modelName == CV_SVMSGD )
    {
        String svmsgdTypeStr;
        modelParamsNode["svmsgdType"] >> svmsgdTypeStr;

        Ptr<SVMSGD> m = SVMSGD::create();
        int svmsgdType = str_to_svmsgd_type( svmsgdTypeStr );
        m->setSvmsgdType(svmsgdType);

        String marginTypeStr;
        modelParamsNode["marginType"] >> marginTypeStr;
        int marginType = str_to_margin_type( marginTypeStr );
        m->setMarginType(marginType);

        m->setMarginRegularization(modelParamsNode["marginRegularization"]);
        m->setInitialStepSize(modelParamsNode["initialStepSize"]);
        m->setStepDecreasingPower(modelParamsNode["stepDecreasingPower"]);
        m->setTermCriteria(TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10000, 0.00001));
        model = m;
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
    CV_TRACE_FUNCTION();
    int type = CV_TEST_ERROR;
    float err = 0;
    Mat _resp;
    if( modelName == CV_EM )
        assert( 0 );
    else if( modelName == CV_ANN )
        err = ann_calc_error( model, data, cls_map, type, resp );
    else if( modelName == CV_DTREE || modelName == CV_BOOST || modelName == CV_RTREES ||
             modelName == CV_SVM || modelName == CV_NBAYES || modelName == CV_KNEAREST || modelName == CV_SVMSGD )
        err = model->calcError( data, true, _resp );
    if( !_resp.empty() && resp )
        _resp.convertTo(*resp, CV_32F);
    return err;
}

void CV_MLBaseTest::save( const char* filename )
{
    CV_TRACE_FUNCTION();
    model->save( filename );
}

void CV_MLBaseTest::load( const char* filename )
{
    CV_TRACE_FUNCTION();
    if( modelName == CV_NBAYES )
        model = Algorithm::load<NormalBayesClassifier>( filename );
    else if( modelName == CV_KNEAREST )
        model = Algorithm::load<KNearest>( filename );
    else if( modelName == CV_SVM )
        model = Algorithm::load<SVM>( filename );
    else if( modelName == CV_ANN )
        model = Algorithm::load<ANN_MLP>( filename );
    else if( modelName == CV_DTREE )
        model = Algorithm::load<DTrees>( filename );
    else if( modelName == CV_BOOST )
        model = Algorithm::load<Boost>( filename );
    else if( modelName == CV_RTREES )
        model = Algorithm::load<RTrees>( filename );
    else if( modelName == CV_SVMSGD )
        model = Algorithm::load<SVMSGD>( filename );
    else
        CV_Error( CV_StsNotImplemented, "invalid stat model name");
}



TEST(TrainDataGet, layout_ROW_SAMPLE)  // Details: #12236
{
    cv::Mat test = cv::Mat::ones(150, 30, CV_32FC1) * 2;
    test.col(3) += Scalar::all(3);
    cv::Mat labels = cv::Mat::ones(150, 3, CV_32SC1) * 5;
    labels.col(1) += 1;
    cv::Ptr<cv::ml::TrainData> train_data = cv::ml::TrainData::create(test, cv::ml::ROW_SAMPLE, labels);
    train_data->setTrainTestSplitRatio(0.9);

    Mat tidx = train_data->getTestSampleIdx();
    EXPECT_EQ((size_t)15, tidx.total());

    Mat tresp = train_data->getTestResponses();
    EXPECT_EQ(15, tresp.rows);
    EXPECT_EQ(labels.cols, tresp.cols);
    EXPECT_EQ(5, tresp.at<int>(0, 0)) << tresp;
    EXPECT_EQ(6, tresp.at<int>(0, 1)) << tresp;
    EXPECT_EQ(6, tresp.at<int>(14, 1)) << tresp;
    EXPECT_EQ(5, tresp.at<int>(14, 2)) << tresp;

    Mat tsamples = train_data->getTestSamples();
    EXPECT_EQ(15, tsamples.rows);
    EXPECT_EQ(test.cols, tsamples.cols);
    EXPECT_EQ(2, tsamples.at<float>(0, 0)) << tsamples;
    EXPECT_EQ(5, tsamples.at<float>(0, 3)) << tsamples;
    EXPECT_EQ(2, tsamples.at<float>(14, test.cols - 1)) << tsamples;
    EXPECT_EQ(5, tsamples.at<float>(14, 3)) << tsamples;
}

TEST(TrainDataGet, layout_COL_SAMPLE)  // Details: #12236
{
    cv::Mat test = cv::Mat::ones(30, 150, CV_32FC1) * 3;
    test.row(3) += Scalar::all(3);
    cv::Mat labels = cv::Mat::ones(3, 150, CV_32SC1) * 5;
    labels.row(1) += 1;
    cv::Ptr<cv::ml::TrainData> train_data = cv::ml::TrainData::create(test, cv::ml::COL_SAMPLE, labels);
    train_data->setTrainTestSplitRatio(0.9);

    Mat tidx = train_data->getTestSampleIdx();
    EXPECT_EQ((size_t)15, tidx.total());

    Mat tresp = train_data->getTestResponses();  // always row-based, transposed
    EXPECT_EQ(15, tresp.rows);
    EXPECT_EQ(labels.rows, tresp.cols);
    EXPECT_EQ(5, tresp.at<int>(0, 0)) << tresp;
    EXPECT_EQ(6, tresp.at<int>(0, 1)) << tresp;
    EXPECT_EQ(6, tresp.at<int>(14, 1)) << tresp;
    EXPECT_EQ(5, tresp.at<int>(14, 2)) << tresp;


    Mat tsamples = train_data->getTestSamples();
    EXPECT_EQ(15, tsamples.cols);
    EXPECT_EQ(test.rows, tsamples.rows);
    EXPECT_EQ(3, tsamples.at<float>(0, 0)) << tsamples;
    EXPECT_EQ(6, tsamples.at<float>(3, 0)) << tsamples;
    EXPECT_EQ(6, tsamples.at<float>(3, 14)) << tsamples;
    EXPECT_EQ(3, tsamples.at<float>(test.rows - 1, 14)) << tsamples;
}



} // namespace
/* End of file. */
