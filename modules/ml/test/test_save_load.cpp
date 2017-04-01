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

#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

CV_SLMLTest::CV_SLMLTest( const char* _modelName ) : CV_MLBaseTest( _modelName )
{
    validationFN = "slvalidation.xml";
}

int CV_SLMLTest::run_test_case( int testCaseIdx )
{
    int code = cvtest::TS::OK;
    code = prepare_test_case( testCaseIdx );

    if( code == cvtest::TS::OK )
    {
        data->setTrainTestSplit(data->getNTrainSamples(), true);
        code = train( testCaseIdx );
        if( code == cvtest::TS::OK )
        {
            get_test_error( testCaseIdx, &test_resps1 );
            fname1 = tempfile(".json.gz");
            save( (fname1 + "?base64").c_str() );
            load( fname1.c_str() );
            get_test_error( testCaseIdx, &test_resps2 );
            fname2 = tempfile(".json.gz");
            save( (fname2 + "?base64").c_str() );
        }
        else
            ts->printf( cvtest::TS::LOG, "model can not be trained" );
    }
    return code;
}

int CV_SLMLTest::validate_test_results( int testCaseIdx )
{
    int code = cvtest::TS::OK;

    // 1. compare files
    FILE *fs1 = fopen(fname1.c_str(), "rb"), *fs2 = fopen(fname2.c_str(), "rb");
    size_t sz1 = 0, sz2 = 0;
    if( !fs1 || !fs2 )
        code = cvtest::TS::FAIL_MISSING_TEST_DATA;
    if( code >= 0 )
    {
        fseek(fs1, 0, SEEK_END); fseek(fs2, 0, SEEK_END);
        sz1 = ftell(fs1);
        sz2 = ftell(fs2);
        fseek(fs1, 0, SEEK_SET); fseek(fs2, 0, SEEK_SET);
    }

    if( sz1 != sz2 )
        code = cvtest::TS::FAIL_INVALID_OUTPUT;

    if( code >= 0 )
    {
        const int BUFSZ = 1024;
        uchar buf1[BUFSZ], buf2[BUFSZ];
        for( size_t pos = 0; pos < sz1;  )
        {
            size_t r1 = fread(buf1, 1, BUFSZ, fs1);
            size_t r2 = fread(buf2, 1, BUFSZ, fs2);
            if( r1 != r2 || memcmp(buf1, buf2, r1) != 0 )
            {
                ts->printf( cvtest::TS::LOG,
                           "in test case %d first (%s) and second (%s) saved files differ in %d-th kb\n",
                           testCaseIdx, fname1.c_str(), fname2.c_str(),
                           (int)pos );
                code = cvtest::TS::FAIL_INVALID_OUTPUT;
                break;
            }
            pos += r1;
        }
    }

    if(fs1)
        fclose(fs1);
    if(fs2)
        fclose(fs2);

    // delete temporary files
    if( code >= 0 )
    {
        remove( fname1.c_str() );
        remove( fname2.c_str() );
    }

    if( code >= 0 )
    {
        // 2. compare responses
        CV_Assert( test_resps1.size() == test_resps2.size() );
        vector<float>::const_iterator it1 = test_resps1.begin(), it2 = test_resps2.begin();
        for( ; it1 != test_resps1.end(); ++it1, ++it2 )
        {
            if( fabs(*it1 - *it2) > FLT_EPSILON )
            {
                ts->printf( cvtest::TS::LOG, "in test case %d responses predicted before saving and after loading is different", testCaseIdx );
                code = cvtest::TS::FAIL_INVALID_OUTPUT;
                break;
            }
        }
    }
    return code;
}

TEST(ML_NaiveBayes, save_load) { CV_SLMLTest test( CV_NBAYES ); test.safe_run(); }
TEST(ML_KNearest, save_load) { CV_SLMLTest test( CV_KNEAREST ); test.safe_run(); }
TEST(ML_SVM, save_load) { CV_SLMLTest test( CV_SVM ); test.safe_run(); }
TEST(ML_ANN, save_load) { CV_SLMLTest test( CV_ANN ); test.safe_run(); }
TEST(ML_DTree, save_load) { CV_SLMLTest test( CV_DTREE ); test.safe_run(); }
TEST(ML_Boost, save_load) { CV_SLMLTest test( CV_BOOST ); test.safe_run(); }
TEST(ML_RTrees, save_load) { CV_SLMLTest test( CV_RTREES ); test.safe_run(); }
TEST(DISABLED_ML_ERTrees, save_load) { CV_SLMLTest test( CV_ERTREES ); test.safe_run(); }
TEST(MV_SVMSGD, save_load){ CV_SLMLTest test( CV_SVMSGD ); test.safe_run(); }

class CV_LegacyTest : public cvtest::BaseTest
{
public:
    CV_LegacyTest(const std::string &_modelName, const std::string &_suffixes = std::string())
        : cvtest::BaseTest(), modelName(_modelName), suffixes(_suffixes)
    {
    }
    virtual ~CV_LegacyTest() {}
protected:
    void run(int)
    {
        unsigned int idx = 0;
        for (;;)
        {
            if (idx >= suffixes.size())
                break;
            int found = (int)suffixes.find(';', idx);
            string piece = suffixes.substr(idx, found - idx);
            if (piece.empty())
                break;
            oneTest(piece);
            idx += (unsigned int)piece.size() + 1;
        }
    }
    void oneTest(const string & suffix)
    {
        using namespace cv::ml;

        int code = cvtest::TS::OK;
        string filename = ts->get_data_path() + "legacy/" + modelName + suffix;
        bool isTree = modelName == CV_BOOST || modelName == CV_DTREE || modelName == CV_RTREES;
        Ptr<StatModel> model;
        if (modelName == CV_BOOST)
            model = Algorithm::load<Boost>(filename);
        else if (modelName == CV_ANN)
            model = Algorithm::load<ANN_MLP>(filename);
        else if (modelName == CV_DTREE)
            model = Algorithm::load<DTrees>(filename);
        else if (modelName == CV_NBAYES)
            model = Algorithm::load<NormalBayesClassifier>(filename);
        else if (modelName == CV_SVM)
            model = Algorithm::load<SVM>(filename);
        else if (modelName == CV_RTREES)
            model = Algorithm::load<RTrees>(filename);
        else if (modelName == CV_SVMSGD)
            model = Algorithm::load<SVMSGD>(filename);
        if (!model)
        {
            code = cvtest::TS::FAIL_INVALID_TEST_DATA;
        }
        else
        {
            Mat input = Mat(isTree ? 10 : 1, model->getVarCount(), CV_32F);
            ts->get_rng().fill(input, RNG::UNIFORM, 0, 40);

            if (isTree)
                randomFillCategories(filename, input);

            Mat output;
            model->predict(input, output, StatModel::RAW_OUTPUT | (isTree ? DTrees::PREDICT_SUM : 0));
            // just check if no internal assertions or errors thrown
        }
        ts->set_failed_test_info(code);
    }
    void randomFillCategories(const string & filename, Mat & input)
    {
        Mat catMap;
        Mat catCount;
        std::vector<uchar> varTypes;

        FileStorage fs(filename, FileStorage::READ);
        FileNode root = fs.getFirstTopLevelNode();
        root["cat_map"] >> catMap;
        root["cat_count"] >> catCount;
        root["var_type"] >> varTypes;

        int offset = 0;
        int countOffset = 0;
        uint var = 0, varCount = (uint)varTypes.size();
        for (; var < varCount; ++var)
        {
            if (varTypes[var] == ml::VAR_CATEGORICAL)
            {
                int size = catCount.at<int>(0, countOffset);
                for (int row = 0; row < input.rows; ++row)
                {
                    int randomChosenIndex = offset + ((uint)ts->get_rng()) % size;
                    int value = catMap.at<int>(0, randomChosenIndex);
                    input.at<float>(row, var) = (float)value;
                }
                offset += size;
                ++countOffset;
            }
        }
    }
    string modelName;
    string suffixes;
};

TEST(ML_ANN, legacy_load) { CV_LegacyTest test(CV_ANN, "_waveform.xml"); test.safe_run(); }
TEST(ML_Boost, legacy_load) { CV_LegacyTest test(CV_BOOST, "_adult.xml;_1.xml;_2.xml;_3.xml"); test.safe_run(); }
TEST(ML_DTree, legacy_load) { CV_LegacyTest test(CV_DTREE, "_abalone.xml;_mushroom.xml"); test.safe_run(); }
TEST(ML_NBayes, legacy_load) { CV_LegacyTest test(CV_NBAYES, "_waveform.xml"); test.safe_run(); }
TEST(ML_SVM, legacy_load) { CV_LegacyTest test(CV_SVM, "_poletelecomm.xml;_waveform.xml"); test.safe_run(); }
TEST(ML_RTrees, legacy_load) { CV_LegacyTest test(CV_RTREES, "_waveform.xml"); test.safe_run(); }
TEST(ML_SVMSGD, legacy_load) { CV_LegacyTest test(CV_SVMSGD, "_waveform.xml"); test.safe_run(); }

/*TEST(ML_SVM, throw_exception_when_save_untrained_model)
{
    Ptr<cv::ml::SVM> svm;
    string filename = tempfile("svm.xml");
    ASSERT_THROW(svm.save(filename.c_str()), Exception);
    remove(filename.c_str());
}*/

TEST(DISABLED_ML_SVM, linear_save_load)
{
    Ptr<cv::ml::SVM> svm1, svm2, svm3;

    svm1 = Algorithm::load<SVM>("SVM45_X_38-1.xml");
    svm2 = Algorithm::load<SVM>("SVM45_X_38-2.xml");
    string tname = tempfile("a.json");
    svm2->save(tname + "?base64");
    svm3 = Algorithm::load<SVM>(tname);

    ASSERT_EQ(svm1->getVarCount(), svm2->getVarCount());
    ASSERT_EQ(svm1->getVarCount(), svm3->getVarCount());

    int m = 10000, n = svm1->getVarCount();
    Mat samples(m, n, CV_32F), r1, r2, r3;
    randu(samples, 0., 1.);

    svm1->predict(samples, r1);
    svm2->predict(samples, r2);
    svm3->predict(samples, r3);

    double eps = 1e-4;
    EXPECT_LE(norm(r1, r2, NORM_INF), eps);
    EXPECT_LE(norm(r1, r3, NORM_INF), eps);

    remove(tname.c_str());
}

/* End of file. */
