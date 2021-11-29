// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test {

struct LstmParams {
    // Batch size
    int nrSamples;

    // Size of the input vector
    int inputSize;

    // Size of the internal state vector
    int hiddenSize;

    // Number of timesteps for the LSTM
    int nrSteps;
};

static inline void PrintTo(const LstmParams& params, ::std::ostream* os) {
    (*os) << "BATCH=" << params.nrSamples
        << ", IN=" << params.inputSize
        << ", HIDDEN=" << params.hiddenSize
        << ", TS=" << params.nrSteps;
}

static const LstmParams testLstmConfigs[] = {
    {1, 192, 192, 100},
    {1, 1024, 192, 100},
    {1, 64, 192, 100},
    {1, 192, 512, 100},
    {64, 192, 192, 2},
    {64, 1024, 192, 2},
    {64, 64, 192, 2},
    {64, 192, 512, 2},
    {128, 192, 192, 2},
    {128, 1024, 192, 2},
    {128, 64, 192, 2},
    {128, 192, 512, 2}
};

class Layer_LSTM : public TestBaseWithParam<LstmParams> {};

PERF_TEST_P_(Layer_LSTM, lstm) {
    const LstmParams& params = GetParam();
    LayerParams lp;
    lp.type = "LSTM";
    lp.name = "testLstm";
    lp.set("produce_cell_output", false);
    lp.set("use_timestamp_dim", true);

    Mat weightH(params.hiddenSize * 4, params.hiddenSize, CV_32FC1, cv::Scalar(0));
    Mat weightX(params.hiddenSize * 4, params.inputSize, CV_32FC1, cv::Scalar(0));
    Mat bias(params.hiddenSize * 4, 1, CV_32FC1, cv::Scalar(0));
    Mat hInternal(params.nrSteps, params.hiddenSize, CV_32FC1, cv::Scalar(0));
    Mat cInternal(params.nrSteps, params.hiddenSize, CV_32FC1, cv::Scalar(0));
    lp.blobs.push_back(weightH);
    lp.blobs.push_back(weightX);
    lp.blobs.push_back(bias);
    lp.blobs.push_back(hInternal);
    lp.blobs.push_back(cInternal);

    std::vector<int> inputDims;
    inputDims.push_back(params.nrSamples);
    inputDims.push_back(params.nrSteps);
    inputDims.push_back(params.inputSize);
    Mat input(inputDims.size(), inputDims.data(), CV_32FC1);
    input = cv::Scalar(0);

    Net net;
    net.addLayerToPrev(lp.name, lp.type, lp);
    net.setInput(input);

    // Warm up
    std::vector<Mat> outputs(2);
    net.forward(outputs, "testLstm");

    TEST_CYCLE()
    {
        net.forward(outputs, "testLstm");
    }
    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Layer_LSTM, testing::ValuesIn(testLstmConfigs));

} // namespace
