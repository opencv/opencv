// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

// #define GENERATE_TESTDATA

namespace opencv_test { namespace {

struct Activation
{
    int id;
    const char * name;
};
void PrintTo(const Activation &a, std::ostream *os) { *os << a.name; }

Activation activation_list[] =
{
    { ml::ANN_MLP::IDENTITY, "identity" },
    { ml::ANN_MLP::SIGMOID_SYM, "sigmoid_sym" },
    { ml::ANN_MLP::GAUSSIAN, "gaussian" },
    { ml::ANN_MLP::RELU, "relu" },
    { ml::ANN_MLP::LEAKYRELU, "leakyrelu" },
};

typedef testing::TestWithParam< Activation > ML_ANN_Params;

TEST_P(ML_ANN_Params, ActivationFunction)
{
    const Activation &activation = GetParam();
    const string dataname = "waveform";
    const string data_path = findDataFile(dataname + ".data");
    const string model_name = dataname + "_" + activation.name + ".yml";

    Ptr<TrainData> tdata = TrainData::loadFromCSV(data_path, 0);
    ASSERT_FALSE(tdata.empty());

    // hack?
    const uint64 old_state = theRNG().state;
    theRNG().state = 1027401484159173092;
    tdata->setTrainTestSplit(500);
    theRNG().state = old_state;

    Mat_<int> layerSizes(1, 4);
    layerSizes(0, 0) = tdata->getNVars();
    layerSizes(0, 1) = 100;
    layerSizes(0, 2) = 100;
    layerSizes(0, 3) = tdata->getResponses().cols;

    Mat testSamples = tdata->getTestSamples();
    Mat rx, ry;

    {
        Ptr<ml::ANN_MLP> x = ml::ANN_MLP::create();
        x->setActivationFunction(activation.id);
        x->setLayerSizes(layerSizes);
        x->setTrainMethod(ml::ANN_MLP::RPROP, 0.01, 0.1);
        x->setTermCriteria(TermCriteria(TermCriteria::COUNT, 300, 0.01));
        x->train(tdata, ml::ANN_MLP::NO_OUTPUT_SCALE);
        ASSERT_TRUE(x->isTrained());
        x->predict(testSamples, rx);
#ifdef GENERATE_TESTDATA
        x->save(cvtest::TS::ptr()->get_data_path() + model_name);
#endif
    }

    {
        const string model_path = findDataFile(model_name);
        Ptr<ml::ANN_MLP> y = Algorithm::load<ANN_MLP>(model_path);
        ASSERT_TRUE(y);
        y->predict(testSamples, ry);
        EXPECT_MAT_NEAR(rx, ry, FLT_EPSILON);
    }
}

INSTANTIATE_TEST_CASE_P(/**/, ML_ANN_Params, testing::ValuesIn(activation_list));

//==================================================================================================

CV_ENUM(ANN_MLP_METHOD, ANN_MLP::RPROP, ANN_MLP::ANNEAL)

typedef tuple<ANN_MLP_METHOD, string, int> ML_ANN_METHOD_Params;
typedef TestWithParam<ML_ANN_METHOD_Params> ML_ANN_METHOD;

TEST_P(ML_ANN_METHOD, Test)
{
    int methodType = get<0>(GetParam());
    string methodName = get<1>(GetParam());
    int N = get<2>(GetParam());

    String folder = string(cvtest::TS::ptr()->get_data_path());
    String original_path = findDataFile("waveform.data");
    string dataname = "waveform_" + methodName;
    string weight_name = dataname + "_init_weight.yml.gz";
    string model_name = dataname + ".yml.gz";
    string response_name = dataname + "_response.yml.gz";

    Ptr<TrainData> tdata2 = TrainData::loadFromCSV(original_path, 0);
    ASSERT_FALSE(tdata2.empty());

    Mat samples = tdata2->getSamples()(Range(0, N), Range::all());
    Mat responses(N, 3, CV_32FC1, Scalar(0));
    for (int i = 0; i < N; i++)
        responses.at<float>(i, static_cast<int>(tdata2->getResponses().at<float>(i, 0))) = 1;

    Ptr<TrainData> tdata = TrainData::create(samples, ml::ROW_SAMPLE, responses);
    ASSERT_FALSE(tdata.empty());

    // hack?
    const uint64 old_state = theRNG().state;
    theRNG().state = 0;
    tdata->setTrainTestSplitRatio(0.8);
    theRNG().state = old_state;

    Mat testSamples = tdata->getTestSamples();

    // train 1st stage

    Ptr<ml::ANN_MLP> xx = ml::ANN_MLP_ANNEAL::create();
    Mat_<int> layerSizes(1, 4);
    layerSizes(0, 0) = tdata->getNVars();
    layerSizes(0, 1) = 30;
    layerSizes(0, 2) = 30;
    layerSizes(0, 3) = tdata->getResponses().cols;
    xx->setLayerSizes(layerSizes);
    xx->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM);
    xx->setTrainMethod(ml::ANN_MLP::RPROP);
    xx->setTermCriteria(TermCriteria(TermCriteria::COUNT, 1, 0.01));
    xx->train(tdata, ml::ANN_MLP::NO_OUTPUT_SCALE + ml::ANN_MLP::NO_INPUT_SCALE);
#ifdef GENERATE_TESTDATA
    {
        FileStorage fs;
        fs.open(cvtest::TS::ptr()->get_data_path() + weight_name, FileStorage::WRITE + FileStorage::BASE64);
        xx->write(fs);
    }
#endif

    // train 2nd stage
    Mat r_gold;
    Ptr<ml::ANN_MLP> x = ml::ANN_MLP_ANNEAL::create();
    {
        const string weight_file = findDataFile(weight_name);
        FileStorage fs;
        fs.open(weight_file, FileStorage::READ);
        x->read(fs.root());
    }
    x->setTrainMethod(methodType);
    if (methodType == ml::ANN_MLP::ANNEAL)
    {
        x->setAnnealEnergyRNG(RNG(CV_BIG_INT(0xffffffff)));
        x->setAnnealInitialT(12);
        x->setAnnealFinalT(0.15);
        x->setAnnealCoolingRatio(0.96);
        x->setAnnealItePerStep(11);
    }
    x->setTermCriteria(TermCriteria(TermCriteria::COUNT, 100, 0.01));
    x->train(tdata, ml::ANN_MLP::NO_OUTPUT_SCALE + ml::ANN_MLP::NO_INPUT_SCALE + ml::ANN_MLP::UPDATE_WEIGHTS);
    ASSERT_TRUE(x->isTrained());
#ifdef GENERATE_TESTDATA
    x->save(cvtest::TS::ptr()->get_data_path() + model_name);
    x->predict(testSamples, r_gold);
    {
        FileStorage fs_response(cvtest::TS::ptr()->get_data_path() + response_name, FileStorage::WRITE + FileStorage::BASE64);
        fs_response << "response" << r_gold;
    }
#endif
    {
        const string response_file = findDataFile(response_name);
        FileStorage fs_response(response_file, FileStorage::READ);
        fs_response["response"] >> r_gold;
    }
    ASSERT_FALSE(r_gold.empty());

    // verify
    const string model_file = findDataFile(model_name);
    Ptr<ml::ANN_MLP> y = Algorithm::load<ANN_MLP>(model_file);
    ASSERT_TRUE(y);
    Mat rx, ry;
    for (int j = 0; j < 4; j++)
    {
        rx = x->getWeights(j);
        ry = y->getWeights(j);
        EXPECT_MAT_NEAR(rx, ry, FLT_EPSILON) << "Weights are not equal for layer: " << j;
    }
    x->predict(testSamples, rx);
    y->predict(testSamples, ry);
    EXPECT_MAT_NEAR(ry, rx, FLT_EPSILON) << "Predict are not equal to result of the saved model";
    EXPECT_MAT_NEAR(r_gold, rx, FLT_EPSILON) << "Predict are not equal to 'gold' response";
}

INSTANTIATE_TEST_CASE_P(/*none*/, ML_ANN_METHOD,
    testing::Values(
        ML_ANN_METHOD_Params(ml::ANN_MLP::RPROP, "rprop", 5000),
        ML_ANN_METHOD_Params(ml::ANN_MLP::ANNEAL, "anneal", 1000)
        // ML_ANN_METHOD_Params(ml::ANN_MLP::BACKPROP, "backprop", 500) -----> NO BACKPROP TEST
    )
);

}} // namespace
