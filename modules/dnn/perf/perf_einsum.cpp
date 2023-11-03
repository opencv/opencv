// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test {

struct EinsumParams {
    int inputSize;
    int outputSize;
    std::string equation;
    std::vector<MatShape> einsumInpShapes;
    EinsumParams(std::string equation_, int inputSize_, int outputSize_,  std::vector<MatShape> einsumInpShapes_ = std::vector<MatShape>())
    {
        inputSize = inputSize_;
        outputSize = outputSize_;
        equation = equation_;
        einsumInpShapes = einsumInpShapes_;
    }
};

static inline void PrintTo(const EinsumParams& params, ::std::ostream* os) {
     (*os) << "Eqiation=" << params.equation << ", "
        << "InputSize=" << params.inputSize << ", "
        << "OutputSize=" << params.outputSize << ", ";

        (*os) << "InputShape={";
        for(int i = 0; i < params.einsumInpShapes.size(); i++)
        {
            (*os) << "{";
            for(int j = 0; j < params.einsumInpShapes[i].size(); j++)
            {
                (*os) << params.einsumInpShapes[i][j] << ((j < params.einsumInpShapes[i].size() - 1) ?  ", " : "");
            }
            (*os) << ((i < params.einsumInpShapes.size() - 1) ? "}, " : "}");
        }
        (*os) << "}";
}

// test cases
static const EinsumParams testEinsumConfigs[] = {
    // TODO: Add tests with one input after ellips merge
    {"ij, jk -> ik", 2, 1,  {{2, 3}, {3, 2}}},
    {"ij, jk -> ik", 2, 1,  {{20, 30}, {30, 20}}},
    {"ij, jk -> ik", 2, 1,  {{113, 127}, {127, 113}}},

    {"imkj, injs -> imnks", 2, 1,  {{1, 4, 7, 9}, {1, 5, 9, 8}}},
    {"imkj, injs -> imnks", 2, 1,  {{1, 4, 70, 90}, {1, 5, 90, 80}}},
    {"imkj, injs -> imnks", 2, 1,  {{1, 4, 73, 91}, {1, 5, 91, 57}}},

    {"ij -> i", 1, 1, {{30, 40}}},
    {"ij -> i", 1, 1, {{113, 374}}},

    {"...ij -> ...i", 1, 1, {{30, 40}}},
    {"...ij -> ...i", 1, 1, {{113, 374}}},

    {"...ij, ...jk -> ...ik", 2, 1, {{40, 50}, {50, 80}}},
    {"...ij, ...jk -> ...ik", 2, 1, {{47, 51}, {51, 83}}},
};

class Layer_Einsum: public TestBaseWithParam<EinsumParams> {};

PERF_TEST_P_(Layer_Einsum, einsum) {
    const EinsumParams& params = GetParam();
    LayerParams lp;
    lp.type = "Einsum";
    lp.name = "testEinsum";
    lp.set("equation", params.equation);
    lp.set("inputSize", params.inputSize);
    lp.set("outputSize", params.outputSize);

    CV_CheckFalse(params.einsumInpShapes.empty(), "ERROR no inputs shapes provided");

    for (int i = 0; i < params.einsumInpShapes.size(); i++) {
        lp.set("inputShapes" + cv::format("%d", i), DictValue::arrayInt(params.einsumInpShapes[i].begin(), params.einsumInpShapes[i].size()));
    }

    Net net;
    std::vector<Mat> inputs;
    std::vector<std::string> input_names;
    if (params.inputSize == 1){

        // create inputs
        inputs.emplace_back(Mat(params.einsumInpShapes[0].size(), params.einsumInpShapes[0].data(), CV_32FC1));

        int id = net.addLayerToPrev(lp.name, lp.type, lp);
        net.connect(0, 0, id, 0);

        input_names.emplace_back("input1");

    } else {

        // create inputs
        inputs.emplace_back(Mat(params.einsumInpShapes[0].size(), params.einsumInpShapes[0].data(), CV_32FC1));
        inputs.emplace_back(Mat(params.einsumInpShapes[1].size(), params.einsumInpShapes[1].data(), CV_32FC1));

        int id = net.addLayerToPrev(lp.name, lp.type, lp);
        net.connect(0, 0, id, 0);
        net.connect(0, 1, id, 1);

        input_names.emplace_back("input1");
        input_names.emplace_back("input2");
    }

    //warm up
    net.setInputsNames(input_names);
    for (int i = 0; i < input_names.size(); i++){
        net.setInput(inputs[i], input_names[i]);
    }
    Mat out = net.forward();

    std::vector<Mat> outputs;
    TEST_CYCLE()
    {
        net.forward(outputs, "testEinsum");
    }
    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Layer_Einsum, testing::ValuesIn(testEinsumConfigs));

}; //namespace
