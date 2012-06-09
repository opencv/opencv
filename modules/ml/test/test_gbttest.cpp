
#include "test_precomp.hpp"

#include <string>
#include <fstream>
#include <iostream>

using namespace std;


class CV_GBTreesTest : public cvtest::BaseTest
{
public:
    CV_GBTreesTest();
    ~CV_GBTreesTest();

protected:
    void run(int);

    int TestTrainPredict(int test_num);
    int TestSaveLoad();

    int checkPredictError(int test_num);
    int checkLoadSave();

    string model_file_name1;
    string model_file_name2;

    string* datasets;
    string data_path;

    CvMLData* data;
    CvGBTrees* gtb;

    vector<float> test_resps1;
    vector<float> test_resps2;

    int64 initSeed;
};


int _get_len(const CvMat* mat)
{
    return (mat->cols > mat->rows) ? mat->cols : mat->rows;
}


CV_GBTreesTest::CV_GBTreesTest()
{
    int64 seeds[] = { CV_BIG_INT(0x00009fff4f9c8d52),
                      CV_BIG_INT(0x0000a17166072c7c),
                      CV_BIG_INT(0x0201b32115cd1f9a),
                      CV_BIG_INT(0x0513cb37abcd1234),
                      CV_BIG_INT(0x0001a2b3c4d5f678)
                    };

    int seedCount = sizeof(seeds)/sizeof(seeds[0]);
    cv::RNG& rng = cv::theRNG();
    initSeed = rng.state;
    rng.state = seeds[rng(seedCount)];

    datasets = 0;
    data = 0;
    gtb = 0;
}

CV_GBTreesTest::~CV_GBTreesTest()
{
    if (data)
        delete data;
    delete[] datasets;
    cv::theRNG().state = initSeed;
}


int CV_GBTreesTest::TestTrainPredict(int test_num)
{
    int code = cvtest::TS::OK;

    int weak_count = 200;
    float shrinkage = 0.1f;
    float subsample_portion = 0.5f;
    int max_depth = 5;
    bool use_surrogates = false;
    int loss_function_type = 0;
    switch (test_num)
    {
        case (1) : loss_function_type = CvGBTrees::SQUARED_LOSS; break;
        case (2) : loss_function_type = CvGBTrees::ABSOLUTE_LOSS; break;
        case (3) : loss_function_type = CvGBTrees::HUBER_LOSS; break;
        case (0) : loss_function_type = CvGBTrees::DEVIANCE_LOSS; break;
        default  :
            {
            ts->printf( cvtest::TS::LOG, "Bad test_num value in CV_GBTreesTest::TestTrainPredict(..) function." );
            return cvtest::TS::FAIL_BAD_ARG_CHECK;
            }
    }

    int dataset_num = test_num == 0 ? 0 : 1;
    if (!data)
    {
        data = new CvMLData();
        data->set_delimiter(',');

        if (data->read_csv(datasets[dataset_num].c_str()))
        {
            ts->printf( cvtest::TS::LOG, "File reading error." );
            return cvtest::TS::FAIL_INVALID_TEST_DATA;
        }

        if (test_num == 0)
        {
            data->set_response_idx(57);
            data->set_var_types("ord[0-56],cat[57]");
        }
        else
        {
            data->set_response_idx(13);
            data->set_var_types("ord[0-2,4-13],cat[3]");
            subsample_portion = 0.7f;
        }

        int train_sample_count = cvFloor(_get_len(data->get_responses())*0.5f);
        CvTrainTestSplit spl( train_sample_count );
        data->set_train_test_split( &spl );
    }

    data->mix_train_and_test_idx();


    if (gtb) delete gtb;
    gtb = new CvGBTrees();
    bool tmp_code = true;
    tmp_code = gtb->train(data, CvGBTreesParams(loss_function_type, weak_count,
                          shrinkage, subsample_portion,
                          max_depth, use_surrogates));

    if (!tmp_code)
    {
        ts->printf( cvtest::TS::LOG, "Model training was failed.");
        return cvtest::TS::FAIL_INVALID_OUTPUT;
    }

    code = checkPredictError(test_num);

    return code;

}


int CV_GBTreesTest::checkPredictError(int test_num)
{
    if (!gtb)
        return cvtest::TS::FAIL_GENERIC;

    //float mean[] = {5.430247f, 13.5654f, 12.6569f, 13.1661f};
    //float sigma[] = {0.4162694f, 3.21161f, 3.43297f, 3.00624f};
    float mean[] = {5.80226f, 12.68689f, 13.49095f, 13.19628f};
    float sigma[] = {0.4764534f, 3.166919f, 3.022405f, 2.868722f};

    float current_error = gtb->calc_error(data, CV_TEST_ERROR);

    if ( abs( current_error - mean[test_num]) > 6*sigma[test_num] )
    {
        ts->printf( cvtest::TS::LOG, "Test error is out of range:\n"
                    "abs(%f/*curEr*/ - %f/*mean*/ > %f/*6*sigma*/",
                    current_error, mean[test_num], 6*sigma[test_num] );
        return cvtest::TS::FAIL_BAD_ACCURACY;
    }

    return cvtest::TS::OK;

}


int CV_GBTreesTest::TestSaveLoad()
{
    if (!gtb)
        return cvtest::TS::FAIL_GENERIC;

    model_file_name1 = cv::tempfile();
    model_file_name2 = cv::tempfile();

    gtb->save(model_file_name1.c_str());
    gtb->calc_error(data, CV_TEST_ERROR, &test_resps1);
    gtb->load(model_file_name1.c_str());
    gtb->calc_error(data, CV_TEST_ERROR, &test_resps2);
    gtb->save(model_file_name2.c_str());

    return checkLoadSave();

}



int CV_GBTreesTest::checkLoadSave()
{
    int code = cvtest::TS::OK;

    // 1. compare files
    ifstream f1( model_file_name1.c_str() ), f2( model_file_name2.c_str() );
    string s1, s2;
    int lineIdx = 0;
    CV_Assert( f1.is_open() && f2.is_open() );
    for( ; !f1.eof() && !f2.eof(); lineIdx++ )
    {
        getline( f1, s1 );
        getline( f2, s2 );
        if( s1.compare(s2) )
        {
            ts->printf( cvtest::TS::LOG, "first and second saved files differ in %n-line; first %n line: %s; second %n-line: %s",
               lineIdx, lineIdx, s1.c_str(), lineIdx, s2.c_str() );
            code = cvtest::TS::FAIL_INVALID_OUTPUT;
        }
    }
    if( !f1.eof() || !f2.eof() )
    {
        ts->printf( cvtest::TS::LOG, "First and second saved files differ in %n-line; first %n line: %s; second %n-line: %s",
            lineIdx, lineIdx, s1.c_str(), lineIdx, s2.c_str() );
        code = cvtest::TS::FAIL_INVALID_OUTPUT;
    }
    f1.close();
    f2.close();
    // delete temporary files
    remove( model_file_name1.c_str() );
    remove( model_file_name2.c_str() );

    // 2. compare responses
    CV_Assert( test_resps1.size() == test_resps2.size() );
    vector<float>::const_iterator it1 = test_resps1.begin(), it2 = test_resps2.begin();
    for( ; it1 != test_resps1.end(); ++it1, ++it2 )
    {
        if( fabs(*it1 - *it2) > FLT_EPSILON )
        {
            ts->printf( cvtest::TS::LOG, "Responses predicted before saving and after loading are different" );
            code = cvtest::TS::FAIL_INVALID_OUTPUT;
        }
    }
    return code;
}



void CV_GBTreesTest::run(int)
{

    string dataPath = string(ts->get_data_path());
    datasets = new string[2];
    datasets[0] = dataPath + string("spambase.data"); /*string("dataset_classification.csv");*/
    datasets[1] = dataPath + string("housing_.data");  /*string("dataset_regression.csv");*/

    int code = cvtest::TS::OK;

    for (int i = 0; i < 4; i++)
    {

        int temp_code = TestTrainPredict(i);
        if (temp_code != cvtest::TS::OK)
        {
            code = temp_code;
            break;
        }

        else if (i==0)
        {
            temp_code = TestSaveLoad();
            if (temp_code != cvtest::TS::OK)
                code = temp_code;
            delete data;
            data = 0;
        }

        delete gtb;
        gtb = 0;
    }
    delete data;
    data = 0;

    ts->set_failed_test_info( code );
}

/////////////////////////////////////////////////////////////////////////////
//////////////////// test registration  /////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

TEST(ML_GBTrees, regression) { CV_GBTreesTest test; test.safe_run(); }
