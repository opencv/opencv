#ifndef __OPENCV_TEST_PRECOMP_HPP__
#define __OPENCV_TEST_PRECOMP_HPP__

#include "opencv2/ts/ts.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/core/core_c.h"
#include <iostream>
#include <map>

#define CV_NBAYES   "nbayes"
#define CV_KNEAREST "knearest"
#define CV_SVM      "svm"
#define CV_EM       "em"
#define CV_ANN      "ann"
#define CV_DTREE    "dtree"
#define CV_BOOST    "boost"
#define CV_RTREES   "rtrees"
#define CV_ERTREES  "ertrees"

class CV_MLBaseTest : public cvtest::BaseTest
{
public:
    CV_MLBaseTest( const char* _modelName );
    virtual ~CV_MLBaseTest();
protected:
    virtual int read_params( CvFileStorage* fs );
    virtual void run( int startFrom );
    virtual int prepare_test_case( int testCaseIdx );
    virtual std::string& get_validation_filename();
    virtual int run_test_case( int testCaseIdx ) = 0;
    virtual int validate_test_results( int testCaseIdx ) = 0;

    int train( int testCaseIdx );
    float get_error( int testCaseIdx, int type, std::vector<float> *resp = 0 );
    void save( const char* filename );
    void load( const char* filename );

    CvMLData data;
    std::string modelName, validationFN;
    std::vector<std::string> dataSetNames;
    cv::FileStorage validationFS;

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
    CV_AMLTest( const char* _modelName ); 
protected:
    virtual int run_test_case( int testCaseIdx );
    virtual int validate_test_results( int testCaseIdx );
};

class CV_SLMLTest : public CV_MLBaseTest
{
public:
    CV_SLMLTest( const char* _modelName ); 
protected:
    virtual int run_test_case( int testCaseIdx );
    virtual int validate_test_results( int testCaseIdx );

    std::vector<float> test_resps1, test_resps2; // predicted responses for test data
    char fname1[50], fname2[50];
};

#endif
