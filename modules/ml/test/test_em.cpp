// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

CV_ENUM(EM_START_STEP, EM::START_AUTO_STEP, EM::START_M_STEP, EM::START_E_STEP)
CV_ENUM(EM_COV_MAT, EM::COV_MAT_GENERIC, EM::COV_MAT_DIAGONAL, EM::COV_MAT_SPHERICAL)

typedef testing::TestWithParam< tuple<EM_START_STEP, EM_COV_MAT> > ML_EM_Params;

TEST_P(ML_EM_Params, accuracy)
{
    const int nclusters = 3;
    const int sizesArr[] = { 500, 700, 800 };
    const vector<int> sizes( sizesArr, sizesArr + sizeof(sizesArr) / sizeof(sizesArr[0]) );
    const int pointsCount = sizesArr[0] + sizesArr[1] + sizesArr[2];
    Mat means;
    vector<Mat> covs;
    defaultDistribs( means, covs, CV_64FC1 );
    Mat trainData(pointsCount, 2, CV_64FC1 );
    Mat trainLabels;
    generateData( trainData, trainLabels, sizes, means, covs, CV_64FC1, CV_32SC1 );
    Mat testData( pointsCount, 2, CV_64FC1 );
    Mat testLabels;
    generateData( testData, testLabels, sizes, means, covs, CV_64FC1, CV_32SC1 );
    Mat probs(trainData.rows, nclusters, CV_64FC1, cv::Scalar(1));
    Mat weights(1, nclusters, CV_64FC1, cv::Scalar(1));
    TermCriteria termCrit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, FLT_EPSILON);
    int startStep = get<0>(GetParam());
    int covMatType = get<1>(GetParam());
    cv::Mat labels;

    Ptr<EM> em = EM::create();
    em->setClustersNumber(nclusters);
    em->setCovarianceMatrixType(covMatType);
    em->setTermCriteria(termCrit);
    if( startStep == EM::START_AUTO_STEP )
        em->trainEM( trainData, noArray(), labels, noArray() );
    else if( startStep == EM::START_E_STEP )
        em->trainE( trainData, means, covs, weights, noArray(), labels, noArray() );
    else if( startStep == EM::START_M_STEP )
        em->trainM( trainData, probs, noArray(), labels, noArray() );

    {
        SCOPED_TRACE("Train");
        float err = 1000;
        EXPECT_TRUE(calcErr( labels, trainLabels, sizes, err , false, false ));
        EXPECT_LE(err, 0.008f);
    }

    {
        SCOPED_TRACE("Test");
        float err = 1000;
        labels.create( testData.rows, 1, CV_32SC1 );
        for( int i = 0; i < testData.rows; i++ )
        {
            Mat sample = testData.row(i);
            Mat out_probs;
            labels.at<int>(i) = static_cast<int>(em->predict2( sample, out_probs )[1]);
        }
        EXPECT_TRUE(calcErr( labels, testLabels, sizes, err, false, false ));
        EXPECT_LE(err, 0.008f);
    }
}

INSTANTIATE_TEST_CASE_P(/**/, ML_EM_Params,
    testing::Combine(
        testing::Values(EM::START_AUTO_STEP, EM::START_M_STEP, EM::START_E_STEP),
        testing::Values(EM::COV_MAT_GENERIC, EM::COV_MAT_DIAGONAL, EM::COV_MAT_SPHERICAL)
    ));

//==================================================================================================

TEST(ML_EM, save_load)
{
    const int nclusters = 2;
    Mat_<double> samples(3, 1);
    samples << 1., 2., 3.;

    std::vector<double> firstResult;
    string filename = cv::tempfile(".xml");
    {
        Mat labels;
        Ptr<EM> em = EM::create();
        em->setClustersNumber(nclusters);
        em->trainEM(samples, noArray(), labels, noArray());
        for( int i = 0; i < samples.rows; i++)
        {
            Vec2d res = em->predict2(samples.row(i), noArray());
            firstResult.push_back(res[1]);
        }
        {
            FileStorage fs = FileStorage(filename, FileStorage::WRITE);
            ASSERT_NO_THROW(fs << "em" << "{");
            ASSERT_NO_THROW(em->write(fs));
            ASSERT_NO_THROW(fs << "}");
        }
    }
    {
        Ptr<EM> em;
        ASSERT_NO_THROW(em = Algorithm::load<EM>(filename));
        for( int i = 0; i < samples.rows; i++)
        {
            SCOPED_TRACE(i);
            Vec2d res = em->predict2(samples.row(i), noArray());
            EXPECT_DOUBLE_EQ(firstResult[i], res[1]);
        }
    }
    remove(filename.c_str());
}

//==================================================================================================

TEST(ML_EM, classification)
{
    // This test classifies spam by the following way:
    // 1. estimates distributions of "spam" / "not spam"
    // 2. predict classID using Bayes classifier for estimated distributions.
    string dataFilename = findDataFile("spambase.data");
    Ptr<TrainData> data = TrainData::loadFromCSV(dataFilename, 0);
    ASSERT_FALSE(data.empty());

    Mat samples = data->getSamples();
    ASSERT_EQ(samples.cols, 57);
    Mat responses = data->getResponses();

    vector<int> trainSamplesMask(samples.rows, 0);
    const int trainSamplesCount = (int)(0.5f * samples.rows);
    const int testSamplesCount = samples.rows - trainSamplesCount;
    for(int i = 0; i < trainSamplesCount; i++)
        trainSamplesMask[i] = 1;
    RNG &rng = cv::theRNG();
    for(size_t i = 0; i < trainSamplesMask.size(); i++)
    {
        int i1 = rng(static_cast<unsigned>(trainSamplesMask.size()));
        int i2 = rng(static_cast<unsigned>(trainSamplesMask.size()));
        std::swap(trainSamplesMask[i1], trainSamplesMask[i2]);
    }

    Mat samples0, samples1;
    for(int i = 0; i < samples.rows; i++)
    {
        if(trainSamplesMask[i])
        {
            Mat sample = samples.row(i);
            int resp = (int)responses.at<float>(i);
            if(resp == 0)
                samples0.push_back(sample);
            else
                samples1.push_back(sample);
        }
    }

    Ptr<EM> model0 = EM::create();
    model0->setClustersNumber(3);
    model0->trainEM(samples0, noArray(), noArray(), noArray());

    Ptr<EM> model1 = EM::create();
    model1->setClustersNumber(3);
    model1->trainEM(samples1, noArray(), noArray(), noArray());

    // confusion matrices
    Mat_<int> trainCM(2, 2, 0);
    Mat_<int> testCM(2, 2, 0);
    const double lambda = 1.;
    for(int i = 0; i < samples.rows; i++)
    {
        Mat sample = samples.row(i);
        double sampleLogLikelihoods0 = model0->predict2(sample, noArray())[0];
        double sampleLogLikelihoods1 = model1->predict2(sample, noArray())[0];
        int classID = (sampleLogLikelihoods0 >= lambda * sampleLogLikelihoods1) ? 0 : 1;
        int resp = (int)responses.at<float>(i);
        EXPECT_TRUE(resp == 0 || resp == 1);
        if(trainSamplesMask[i])
            trainCM(resp, classID)++;
        else
            testCM(resp, classID)++;
    }
    EXPECT_LE((double)(trainCM(1,0) + trainCM(0,1)) / trainSamplesCount, 0.23);
    EXPECT_LE((double)(testCM(1,0) + testCM(0,1)) / testSamplesCount, 0.26);
}

}} // namespace
