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

#include "mltest.h"
#include <iostream>
#include <fstream>

CV_SLMLTest::CV_SLMLTest( const char* _modelName, const char* _testName ) :
    CV_MLBaseTest( _modelName, _testName, "load-save" )
{
    validationFN = "slvalidation.xml";
}

int CV_SLMLTest::run_test_case( int testCaseIdx )
{
    int code = CvTS::OK;
    code = prepare_test_case( testCaseIdx );

    if( code == CvTS::OK )
    {
            data.mix_train_and_test_idx();
            code = train( testCaseIdx );
            if( code == CvTS::OK )
            {
                get_error( testCaseIdx, CV_TEST_ERROR, &test_resps1 );
                save( tmpnam( fname1 ) );
                load( fname1);
                get_error( testCaseIdx, CV_TEST_ERROR, &test_resps2 );
                save( tmpnam( fname2 ) );
            }
            else
                ts->printf( CvTS::LOG, "model can not be trained" );
    }
    return code;
}

int CV_SLMLTest::validate_test_results( int testCaseIdx )
{
    int code = CvTS::OK;

    // 1. compare files
    ifstream f1( fname1 ), f2( fname2 );
    string s1, s2;
    int lineIdx = 0; 
    CV_Assert( f1.is_open() && f2.is_open() );
    for( ; !f1.eof() && !f2.eof(); lineIdx++ )
    {
        getline( f1, s1 );
        getline( f2, s2 );
        if( s1.compare(s2) )
        {
            ts->printf( CvTS::LOG, "first and second saved files differ in %n-line; first %n line: %s; second %n-line: %s",
               lineIdx, lineIdx, s1.c_str(), lineIdx, s2.c_str() );
            code = CvTS::FAIL_INVALID_OUTPUT;
        }
    }
    if( !f1.eof() || !f2.eof() )
    {
        ts->printf( CvTS::LOG, "in test case %d first and second saved files differ in %n-line; first %n line: %s; second %n-line: %s",
            testCaseIdx, lineIdx, lineIdx, s1.c_str(), lineIdx, s2.c_str() );
        code = CvTS::FAIL_INVALID_OUTPUT;
    }
    f1.close();
    f2.close();
    // delete temporary files
    unlink( fname1 );
    unlink( fname2 );

    // 2. compare responses
    CV_Assert( test_resps1.size() == test_resps2.size() );
    vector<float>::const_iterator it1 = test_resps1.begin(), it2 = test_resps2.begin();
    for( ; it1 != test_resps1.end(); ++it1, ++it2 )
    {
        if( fabs(*it1 - *it2) > FLT_EPSILON )
        {
            ts->printf( CvTS::LOG, "in test case %d responses predicted before saving and after loading is different", testCaseIdx );
            code = CvTS::FAIL_INVALID_OUTPUT;
        }
    }
    return code;
}

CV_SLMLTest lsmlnbayes( CV_NBAYES, "slnbayes" );
//CV_SLMLTest lsmlknearest( CV_KNEAREST, "slknearest" ); // does not support save!
CV_SLMLTest lsmlsvm( CV_SVM, "slsvm" );
//CV_SLMLTest lsmlem( CV_EM, "slem" ); // does not support save!
CV_SLMLTest lsmlann( CV_ANN, "slann" );
CV_SLMLTest slmldtree( CV_DTREE, "sldtree" );
CV_SLMLTest slmlboost( CV_BOOST, "slboost" );
CV_SLMLTest slmlrtrees( CV_RTREES, "slrtrees" );
CV_SLMLTest slmlertrees( CV_ERTREES, "slertrees" );

/* End of file. */
