// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// AUTHOR: Rahul Kavi rahulkavi[at]live[at]com

//
// Test data uses subset of data from the popular Iris Dataset (1936):
// - http://archive.ics.uci.edu/ml/datasets/Iris
// - https://en.wikipedia.org/wiki/Iris_flower_data_set
//

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(ML_LR, accuracy)
{
    std::string dataFileName = findDataFile("iris.data");
    Ptr<TrainData> tdata = TrainData::loadFromCSV(dataFileName, 0);
    ASSERT_FALSE(tdata.empty());

    Ptr<LogisticRegression> p = LogisticRegression::create();
    p->setLearningRate(1.0);
    p->setIterations(10001);
    p->setRegularization(LogisticRegression::REG_L2);
    p->setTrainMethod(LogisticRegression::BATCH);
    p->setMiniBatchSize(10);
    p->train(tdata);

    Mat responses;
    p->predict(tdata->getSamples(), responses);

    float error = 1000;
    EXPECT_TRUE(calculateError(responses, tdata->getResponses(), error));
    EXPECT_LE(error, 0.05f);
}

//==================================================================================================

TEST(ML_LR, save_load)
{
    string dataFileName = findDataFile("iris.data");
    Ptr<TrainData> tdata = TrainData::loadFromCSV(dataFileName, 0);
    ASSERT_FALSE(tdata.empty());
    Mat responses1, responses2;
    Mat learnt_mat1, learnt_mat2;
    String filename = tempfile(".xml");
    {
        Ptr<LogisticRegression> lr1 = LogisticRegression::create();
        lr1->setLearningRate(1.0);
        lr1->setIterations(10001);
        lr1->setRegularization(LogisticRegression::REG_L2);
        lr1->setTrainMethod(LogisticRegression::BATCH);
        lr1->setMiniBatchSize(10);
        ASSERT_NO_THROW(lr1->train(tdata));
        ASSERT_NO_THROW(lr1->predict(tdata->getSamples(), responses1));
        ASSERT_NO_THROW(lr1->save(filename));
        learnt_mat1 = lr1->get_learnt_thetas();
    }
    {
        Ptr<LogisticRegression> lr2;
        ASSERT_NO_THROW(lr2 = Algorithm::load<LogisticRegression>(filename));
        ASSERT_NO_THROW(lr2->predict(tdata->getSamples(), responses2));
        learnt_mat2 = lr2->get_learnt_thetas();
    }
    // compare difference in prediction outputs and stored inputs
    EXPECT_MAT_NEAR(responses1, responses2, 0.f);

    Mat comp_learnt_mats;
    comp_learnt_mats = (learnt_mat1 == learnt_mat2);
    comp_learnt_mats = comp_learnt_mats.reshape(1, comp_learnt_mats.rows*comp_learnt_mats.cols);
    comp_learnt_mats.convertTo(comp_learnt_mats, CV_32S);
    comp_learnt_mats = comp_learnt_mats/255;
    // check if there is any difference between computed learnt mat and retrieved mat
    EXPECT_EQ(comp_learnt_mats.rows, sum(comp_learnt_mats)[0]);

    remove( filename.c_str() );
}

}} // namespace
