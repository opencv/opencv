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

using namespace std;
using namespace cv;

static
void defaultDistribs( Mat& means, vector<Mat>& covs )
{
    float mp0[] = {0.0f, 0.0f}, cp0[] = {0.67f, 0.0f, 0.0f, 0.67f};
    float mp1[] = {5.0f, 0.0f}, cp1[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float mp2[] = {1.0f, 5.0f}, cp2[] = {1.0f, 0.0f, 0.0f, 1.0f};
    means.create(3, 2, CV_32FC1);
    Mat m0( 1, 2, CV_32FC1, mp0 ), c0( 2, 2, CV_32FC1, cp0 );
    Mat m1( 1, 2, CV_32FC1, mp1 ), c1( 2, 2, CV_32FC1, cp1 );
    Mat m2( 1, 2, CV_32FC1, mp2 ), c2( 2, 2, CV_32FC1, cp2 );
    means.resize(3), covs.resize(3);

    Mat mr0 = means.row(0);
    m0.copyTo(mr0);
    c0.copyTo(covs[0]);

    Mat mr1 = means.row(1);
    m1.copyTo(mr1);
    c1.copyTo(covs[1]);

    Mat mr2 = means.row(2);
    m2.copyTo(mr2);
    c2.copyTo(covs[2]);
}

// generate points sets by normal distributions
static
void generateData( Mat& data, Mat& labels, const vector<int>& sizes, const Mat& _means, const vector<Mat>& covs, int labelType )
{
    vector<int>::const_iterator sit = sizes.begin();
    int total = 0;
    for( ; sit != sizes.end(); ++sit )
        total += *sit;
    assert( _means.rows == (int)sizes.size() && covs.size() == sizes.size() );
    assert( !data.empty() && data.rows == total );
    assert( data.type() == CV_32FC1 );
    
    labels.create( data.rows, 1, labelType );

    randn( data, Scalar::all(0.0), Scalar::all(1.0) );
    vector<Mat> means(sizes.size());
    for(int i = 0; i < _means.rows; i++)
        means[i] = _means.row(i);
    vector<Mat>::const_iterator mit = means.begin(), cit = covs.begin();
    int bi, ei = 0;
    sit = sizes.begin();
    for( int p = 0, l = 0; sit != sizes.end(); ++sit, ++mit, ++cit, l++ )
    {
        bi = ei;
        ei = bi + *sit;
        assert( mit->rows == 1 && mit->cols == data.cols );
        assert( cit->rows == data.cols && cit->cols == data.cols );
        for( int i = bi; i < ei; i++, p++ )
        {
            Mat r(1, data.cols, CV_32FC1, data.ptr<float>(i));
            r =  r * (*cit) + *mit; 
            if( labelType == CV_32FC1 )
                labels.at<float>(p, 0) = (float)l;
            else if( labelType == CV_32SC1 )
                labels.at<int>(p, 0) = l;
            else
                CV_DbgAssert(0);
        }
    }
}

static
int maxIdx( const vector<int>& count )
{
    int idx = -1;
    int maxVal = -1;
    vector<int>::const_iterator it = count.begin();
    for( int i = 0; it != count.end(); ++it, i++ )
    {
        if( *it > maxVal)
        {
            maxVal = *it;
            idx = i;
        }
    }
    assert( idx >= 0);
    return idx;
}

static
bool getLabelsMap( const Mat& labels, const vector<int>& sizes, vector<int>& labelsMap )
{
    size_t total = 0, nclusters = sizes.size();
    for(size_t i = 0; i < sizes.size(); i++)
        total += sizes[i];

    assert( !labels.empty() );
    assert( labels.total() == total && (labels.cols == 1 || labels.rows == 1));
    assert( labels.type() == CV_32SC1 || labels.type() == CV_32FC1 );

    bool isFlt = labels.type() == CV_32FC1;

    labelsMap.resize(nclusters);

    vector<bool> buzy(nclusters, false);
    int startIndex = 0;
    for( size_t clusterIndex = 0; clusterIndex < sizes.size(); clusterIndex++ )
    {
        vector<int> count( nclusters, 0 );
        for( int i = startIndex; i < startIndex + sizes[clusterIndex]; i++)
        {
            int lbl = isFlt ? (int)labels.at<float>(i) : labels.at<int>(i);
            CV_Assert(lbl < (int)nclusters);
            count[lbl]++;
            CV_Assert(count[lbl] < (int)total);
        }
        startIndex += sizes[clusterIndex];

        int cls = maxIdx( count );
        CV_Assert( !buzy[cls] );

        labelsMap[clusterIndex] = cls;

        buzy[cls] = true;
    }
    for(size_t i = 0; i < buzy.size(); i++)
        if(!buzy[i])
            return false;

    return true;
}

static
bool calcErr( const Mat& labels, const Mat& origLabels, const vector<int>& sizes, float& err, bool labelsEquivalent = true )
{
    err = 0;
    CV_Assert( !labels.empty() && !origLabels.empty() );
    CV_Assert( labels.rows == 1 || labels.cols == 1 );
    CV_Assert( origLabels.rows == 1 || origLabels.cols == 1 );
    CV_Assert( labels.total() == origLabels.total() );
    CV_Assert( labels.type() == CV_32SC1 || labels.type() == CV_32FC1 );
    CV_Assert( origLabels.type() == labels.type() );

    vector<int> labelsMap;
    bool isFlt = labels.type() == CV_32FC1;
    if( !labelsEquivalent )
    {
        if( !getLabelsMap( labels, sizes, labelsMap ) )
            return false;

        for( int i = 0; i < labels.rows; i++ )
            if( isFlt )
                err += labels.at<float>(i) != labelsMap[(int)origLabels.at<float>(i)] ? 1.f : 0.f;
            else
                err += labels.at<int>(i) != labelsMap[origLabels.at<int>(i)] ? 1.f : 0.f;
    }
    else
    {
        for( int i = 0; i < labels.rows; i++ )
            if( isFlt )
                err += labels.at<float>(i) != origLabels.at<float>(i) ? 1.f : 0.f;
            else
                err += labels.at<int>(i) != origLabels.at<int>(i) ? 1.f : 0.f;
    }
    err /= (float)labels.rows;
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class CV_CvEMTest : public cvtest::BaseTest
{
public:
    CV_CvEMTest() {}
protected:
    virtual void run( int start_from );
    int runCase( int caseIndex, const CvEMParams& params,
                  const cv::Mat& trainData, const cv::Mat& trainLabels,
                  const cv::Mat& testData, const cv::Mat& testLabels,
                  const vector<int>& sizes);
};

int CV_CvEMTest::runCase( int caseIndex, const CvEMParams& params,
                        const cv::Mat& trainData, const cv::Mat& trainLabels,
                        const cv::Mat& testData, const cv::Mat& testLabels,
                        const vector<int>& sizes )
{
    int code = cvtest::TS::OK;

    cv::Mat labels;
    float err;

    CvEM em;
    em.train( trainData, Mat(), params, &labels );

    // check train error
    if( !calcErr( labels, trainLabels, sizes, err , false ) )
    {
        ts->printf( cvtest::TS::LOG, "Case index %i : Bad output labels.\n", caseIndex );
        code = cvtest::TS::FAIL_INVALID_OUTPUT;
    }
    else if( err > 0.006f )
    {
        ts->printf( cvtest::TS::LOG, "Case index %i : Bad accuracy (%f) on train data.\n", caseIndex, err );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
    }

    // check test error
    labels.create( testData.rows, 1, CV_32SC1 );
    for( int i = 0; i < testData.rows; i++ )
    {
        Mat sample = testData.row(i);
        labels.at<int>(i,0) = (int)em.predict( sample, 0 );
    }
    if( !calcErr( labels, testLabels, sizes, err, false ) )
    {
        ts->printf( cvtest::TS::LOG, "Case index %i : Bad output labels.\n", caseIndex );
        code = cvtest::TS::FAIL_INVALID_OUTPUT;
    }
    else if( err > 0.006f )
    {
        ts->printf( cvtest::TS::LOG, "Case index %i : Bad accuracy (%f) on test data.\n", caseIndex, err );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
    }

    return code;
}

void CV_CvEMTest::run( int /*start_from*/ )
{
    int sizesArr[] = { 500, 700, 800 };
    int pointsCount = sizesArr[0]+ sizesArr[1] + sizesArr[2];

    // Points distribution
    Mat means;
    vector<Mat> covs;
    defaultDistribs( means, covs );

    // train data
    Mat trainData( pointsCount, 2, CV_32FC1 ), trainLabels;
    vector<int> sizes( sizesArr, sizesArr + sizeof(sizesArr) / sizeof(sizesArr[0]) );
    generateData( trainData, trainLabels, sizes, means, covs, CV_32SC1 );

    // test data
    Mat testData( pointsCount, 2, CV_32FC1 ), testLabels;
    generateData( testData, testLabels, sizes, means, covs, CV_32SC1 );

    CvEMParams params;
    params.nclusters = 3;
    Mat probs(trainData.rows, params.nclusters, CV_32FC1, cv::Scalar(1));
    CvMat probsHdr = probs;
    params.probs = &probsHdr;
    Mat weights(1, params.nclusters, CV_32FC1, cv::Scalar(1));
    CvMat weightsHdr = weights;
    params.weights = &weightsHdr;
    CvMat meansHdr = means;
    params.means = &meansHdr;
    std::vector<CvMat> covsHdrs(params.nclusters);
    std::vector<const CvMat*> covsPtrs(params.nclusters);
    for(int i = 0; i < params.nclusters; i++)
    {
        covsHdrs[i] = covs[i];
        covsPtrs[i] = &covsHdrs[i];
    }
    params.covs = &covsPtrs[0];

    int code = cvtest::TS::OK;
    int caseIndex = 0;
    {
        params.start_step = cv::EM::START_AUTO_STEP;
        params.cov_mat_type = cv::EM::COV_MAT_GENERIC;
        int currCode = runCase(caseIndex++, params, trainData, trainLabels, testData, testLabels, sizes);
        code = currCode == cvtest::TS::OK ? code : currCode;
    }
    {
        params.start_step = cv::EM::START_AUTO_STEP;
        params.cov_mat_type = cv::EM::COV_MAT_DIAGONAL;
        int currCode = runCase(caseIndex++, params, trainData, trainLabels, testData, testLabels, sizes);
        code = currCode == cvtest::TS::OK ? code : currCode;
    }
    {
        params.start_step = cv::EM::START_AUTO_STEP;
        params.cov_mat_type = cv::EM::COV_MAT_SPHERICAL;
        int currCode = runCase(caseIndex++, params, trainData, trainLabels, testData, testLabels, sizes);
        code = currCode == cvtest::TS::OK ? code : currCode;
    }
    {
        params.start_step = cv::EM::START_M_STEP;
        params.cov_mat_type = cv::EM::COV_MAT_GENERIC;
        int currCode = runCase(caseIndex++, params, trainData, trainLabels, testData, testLabels, sizes);
        code = currCode == cvtest::TS::OK ? code : currCode;
    }
    {
        params.start_step = cv::EM::START_M_STEP;
        params.cov_mat_type = cv::EM::COV_MAT_DIAGONAL;
        int currCode = runCase(caseIndex++, params, trainData, trainLabels, testData, testLabels, sizes);
        code = currCode == cvtest::TS::OK ? code : currCode;
    }
    {
        params.start_step = cv::EM::START_M_STEP;
        params.cov_mat_type = cv::EM::COV_MAT_SPHERICAL;
        int currCode = runCase(caseIndex++, params, trainData, trainLabels, testData, testLabels, sizes);
        code = currCode == cvtest::TS::OK ? code : currCode;
    }
    {
        params.start_step = cv::EM::START_E_STEP;
        params.cov_mat_type = cv::EM::COV_MAT_GENERIC;
        int currCode = runCase(caseIndex++, params, trainData, trainLabels, testData, testLabels, sizes);
        code = currCode == cvtest::TS::OK ? code : currCode;
    }
    {
        params.start_step = cv::EM::START_E_STEP;
        params.cov_mat_type = cv::EM::COV_MAT_DIAGONAL;
        int currCode = runCase(caseIndex++, params, trainData, trainLabels, testData, testLabels, sizes);
        code = currCode == cvtest::TS::OK ? code : currCode;
    }
    {
        params.start_step = cv::EM::START_E_STEP;
        params.cov_mat_type = cv::EM::COV_MAT_SPHERICAL;
        int currCode = runCase(caseIndex++, params, trainData, trainLabels, testData, testLabels, sizes);
        code = currCode == cvtest::TS::OK ? code : currCode;
    }
    
    ts->set_failed_test_info( code );
}

class CV_CvEMTest_SaveLoad : public cvtest::BaseTest {
public:
    CV_CvEMTest_SaveLoad() {}
protected:
    virtual void run( int /*start_from*/ )
    {
        int code = cvtest::TS::OK;
        cv::EM em;

        Mat samples = Mat(3,1,CV_32F);
        samples.at<float>(0,0) = 1;
        samples.at<float>(1,0) = 2;
        samples.at<float>(2,0) = 3;

        cv::EM::Params params;
        params.nclusters = 2;

        Mat labels;

        em.train(samples, Mat(), params, &labels);

        Mat firstResult(samples.rows, 1, CV_32FC1);
        for( int i = 0; i < samples.rows; i++)
            firstResult.at<float>(i) = em.predict( samples.row(i) );

        // Write out

        string filename = tempfile() + ".xml";
        {
            FileStorage fs = FileStorage(filename, FileStorage::WRITE);
            try
            {
                fs << "em" << "{";
                em.write(fs);
                fs << "}";
            }
            catch(...)
            {
                ts->printf( cvtest::TS::LOG, "Crash in write method.\n" );
                ts->set_failed_test_info( cvtest::TS::FAIL_EXCEPTION );
            }
        }

        em.clear();

        // Read in
        {
            FileStorage fs = FileStorage(filename, FileStorage::READ);
            CV_Assert(fs.isOpened());
            FileNode fn = fs["em"];
            try
            {
                em.read(fn);
            }
            catch(...)
            {
                ts->printf( cvtest::TS::LOG, "Crash in read method.\n" );
                ts->set_failed_test_info( cvtest::TS::FAIL_EXCEPTION );
            }
        }

        remove( filename.c_str() );

        int errCaseCount = 0;
        for( int i = 0; i < samples.rows; i++)
            errCaseCount = std::abs(em.predict(samples.row(i)) - firstResult.at<float>(i)) < FLT_EPSILON ? 0 : 1;

        if( errCaseCount > 0 )
        {
            ts->printf( cvtest::TS::LOG, "Different prediction results before writeing and after reading (errCaseCount=%d).\n", errCaseCount );
            code = cvtest::TS::FAIL_BAD_ACCURACY;
        }

        ts->set_failed_test_info( code );
    }
};

TEST(ML_CvEM, accuracy) { CV_CvEMTest test; test.safe_run(); }
TEST(ML_CvEM, save_load) { CV_CvEMTest_SaveLoad test; test.safe_run(); }
