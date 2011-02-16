/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

#ifndef _OPENCV_MLTEST_H_
#define _OPENCV_MLTEST_H_

#if defined _MSC_VER && _MSC_VER >= 1200
#pragma warning( disable: 4710 4711 4514 4996 )
#endif

#include "cxcore.h"
#include "cxmisc.h"
#include "cxts.h"
#include "ml.h"
#include <map>
#include <string>
#include <iostream>

using namespace cv;

#define CV_NBAYES   "nbayes"
#define CV_KNEAREST "knearest"
#define CV_SVM      "svm"
#define CV_EM       "em"
#define CV_ANN      "ann"
#define CV_DTREE    "dtree"
#define CV_BOOST    "boost"
#define CV_RTREES   "rtrees"
#define CV_ERTREES  "ertrees"

class CV_MLBaseTest : public CvTest
{
public:
    CV_MLBaseTest( const char* _modelName, const char* _testName, const char* _testFuncs );
    virtual ~CV_MLBaseTest();
    virtual int init( CvTS* system );
protected:
    virtual int read_params( CvFileStorage* fs );
    virtual void run( int startFrom );
    virtual int prepare_test_case( int testCaseIdx );
    virtual string& get_validation_filename();
    virtual int run_test_case( int testCaseIdx ) = 0;
    virtual int validate_test_results( int testCaseIdx ) = 0;

    int train( int testCaseIdx );
    float get_error( int testCaseIdx, int type, std::vector<float> *resp = 0 );
    void save( const char* filename );
    void load( const char* filename );

    CvMLData data;
    string modelName, validationFN;
    std::vector<string> dataSetNames;
    FileStorage validationFS;

    // MLL models
    CvNormalBayesClassifier* nbayes;
    CvKNearest* knearest;
    CvSVM* svm;
    CvEM* em;
    CvANN_MLP* ann;
    CvDTree* dtree;
    CvBoost* boost;
    CvRTrees* rtrees;
    CvERTrees* ertrees;

    std::map<int, int> cls_map;

    int64 initSeed;
};

class CV_AMLTest : public CV_MLBaseTest
{
public:
    CV_AMLTest( const char* _modelName, const char* _testName ); 
protected:
    virtual int run_test_case( int testCaseIdx );
    virtual int validate_test_results( int testCaseIdx );
};

class CV_SLMLTest : public CV_MLBaseTest
{
public:
    CV_SLMLTest( const char* _modelName, const char* _testName ); 
protected:
    virtual int run_test_case( int testCaseIdx );
    virtual int validate_test_results( int testCaseIdx );

    std::vector<float> test_resps1, test_resps2; // predicted responses for test data
    char fname1[50], fname2[50];
};

/* End of file. */

#endif
