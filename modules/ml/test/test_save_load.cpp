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
            data.mix_train_and_test_idx();
            code = train( testCaseIdx );
            if( code == cvtest::TS::OK )
            {
                get_error( testCaseIdx, CV_TEST_ERROR, &test_resps1 );
                fname1 = tempfile(".yml.gz");
                save( fname1.c_str() );
                load( fname1.c_str() );
                get_error( testCaseIdx, CV_TEST_ERROR, &test_resps2 );
                fname2 = tempfile(".yml.gz");
                save( fname2.c_str() );
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

    // 2. compare responses
    CV_Assert( test_resps1.size() == test_resps2.size() );
    vector<float>::const_iterator it1 = test_resps1.begin(), it2 = test_resps2.begin();
    for( ; it1 != test_resps1.end(); ++it1, ++it2 )
    {
        if( fabs(*it1 - *it2) > FLT_EPSILON )
        {
            ts->printf( cvtest::TS::LOG, "in test case %d responses predicted before saving and after loading is different", testCaseIdx );
            code = cvtest::TS::FAIL_INVALID_OUTPUT;
        }
    }
    return code;
}

TEST(ML_NaiveBayes, save_load) { CV_SLMLTest test( CV_NBAYES ); test.safe_run(); }
//CV_SLMLTest lsmlknearest( CV_KNEAREST, "slknearest" ); // does not support save!
TEST(ML_SVM, save_load) { CV_SLMLTest test( CV_SVM ); test.safe_run(); }
//CV_SLMLTest lsmlem( CV_EM, "slem" ); // does not support save!
TEST(ML_ANN, save_load) { CV_SLMLTest test( CV_ANN ); test.safe_run(); }
TEST(ML_DTree, save_load) { CV_SLMLTest test( CV_DTREE ); test.safe_run(); }
TEST(ML_Boost, save_load) { CV_SLMLTest test( CV_BOOST ); test.safe_run(); }
TEST(ML_RTrees, save_load) { CV_SLMLTest test( CV_RTREES ); test.safe_run(); }
TEST(ML_ERTrees, save_load) { CV_SLMLTest test( CV_ERTREES ); test.safe_run(); }


TEST(ML_SVM, throw_exception_when_save_untrained_model)
{
    SVM svm;
    string filename = tempfile("svm.xml");
    ASSERT_THROW(svm.save(filename.c_str()), Exception);
    remove(filename.c_str());
}

TEST(DISABLED_ML_SVM, linear_save_load)
{
    CvSVM svm1, svm2, svm3;
    svm1.load("SVM45_X_38-1.xml");
    svm2.load("SVM45_X_38-2.xml");
    string tname = tempfile("a.xml");
    svm2.save(tname.c_str());
    svm3.load(tname.c_str());

    ASSERT_EQ(svm1.get_var_count(), svm2.get_var_count());
    ASSERT_EQ(svm1.get_var_count(), svm3.get_var_count());

    int m = 10000, n = svm1.get_var_count();
    Mat samples(m, n, CV_32F), r1, r2, r3;
    randu(samples, 0., 1.);

    svm1.predict(samples, r1);
    svm2.predict(samples, r2);
    svm3.predict(samples, r3);

    double eps = 1e-4;
    EXPECT_LE(norm(r1, r2, NORM_INF), eps);
    EXPECT_LE(norm(r1, r3, NORM_INF), eps);

    remove(tname.c_str());
}

/* End of file. */
