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

CV_AMLTest::CV_AMLTest( const char* _modelName, const char* _testName ) :
    CV_MLBaseTest( _modelName, _testName, "train-predict" )
{
    validationFN = "avalidation.xml";
}

int CV_AMLTest::run_test_case( int testCaseIdx )
{
    int code = CvTS::OK;
    code = prepare_test_case( testCaseIdx );

    if (code == CvTS::OK)
    {
        //#define GET_STAT
#ifdef GET_STAT
        const char* data_name = ((CvFileNode*)cvGetSeqElem( dataSetNames, testCaseIdx ))->data.str.ptr;     
        printf("%s, %s      ", name, data_name);
        const int icount = 100;
        float res[icount];
        for (int k = 0; k < icount; k++)
        {
#endif
            data.mix_train_and_test_idx();
            code = train( testCaseIdx );            
#ifdef GET_STAT
            float case_result = get_error();

            res[k] = case_result;
        }
        float mean = 0, sigma = 0;
        for (int k = 0; k < icount; k++)
        {
            mean += res[k];
        }
        mean = mean /icount;
        for (int k = 0; k < icount; k++)
        {
            sigma += (res[k] - mean)*(res[k] - mean);
        }
        sigma = sqrt(sigma/icount);
        printf("%f, %f\n", mean, sigma);
#endif
    }
    return code;
}

int CV_AMLTest::validate_test_results( int testCaseIdx )
{
    int iters;
    float mean, sigma;
    // read validation params
    FileNode resultNode = 
        validationFS.getFirstTopLevelNode()["validation"][modelName][dataSetNames[testCaseIdx]]["result"];
    resultNode["iter_count"] >> iters; 
    if ( iters > 0)
    {
        resultNode["mean"] >> mean;
        resultNode["sigma"] >> sigma;
        float curErr = get_error( testCaseIdx, CV_TEST_ERROR );
        const int coeff = 4;
        ts->printf( CvTS::LOG, "Test case = %d; test error = %f; mean error = %f (diff=%f), %d*sigma = %f",
                                testCaseIdx, curErr, mean, abs( curErr - mean), coeff, coeff*sigma );
        if ( abs( curErr - mean) > coeff*sigma )
        {
            ts->printf( CvTS::LOG, "abs(%f - %f) > %f - OUT OF RANGE!\n", curErr, mean, coeff*sigma, coeff );
            return CvTS::FAIL_BAD_ACCURACY;
        }
        else
            ts->printf( CvTS::LOG, ".\n" );

    }
    else
    {
        ts->printf( CvTS::LOG, "validation info is not suitable" );
        return CvTS::FAIL_INVALID_TEST_DATA;
    }
    return CvTS::OK;
}

CV_AMLTest amldtree( CV_DTREE, "adtree" );
CV_AMLTest amlboost( CV_BOOST, "aboost" );
CV_AMLTest amlrtrees( CV_RTREES, "artrees" );
CV_AMLTest amlertrees( CV_ERTREES, "aertrees" );

/* End of file. */
