// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test {

struct EinsumParams {
    int inputSize;
    int outputSize;
    std::string equation;
    std::vector<std::vector<int> > einsumInpShapes;
    EinsumParams(std::string equation_, std::vector<std::vector<int> > einsumInpShapes_ = std::vector<std::vector<int> >())
    {
        inputSize = einsumInpShapes_.size();
        equation = equation_;
        einsumInpShapes = einsumInpShapes_;
    }
};

static inline void PrintTo(const EinsumParams& params, ::std::ostream* os) {
     (*os) << "Equation=" << params.equation << " ";

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
    {"ij, jk -> ik", {{2, 3}, {3, 2}}},
    {"ij, jk -> ik", {{20, 30}, {30, 20}}},
    {"ij, jk -> ik", {{113, 127}, {127, 113}}},

    {"imkj, injs -> imnks", {{1, 4, 7, 9}, {1, 5, 9, 8}}},
    {"imkj, injs -> imnks", {{1, 4, 70, 90}, {1, 5, 90, 80}}},
    {"imkj, injs -> imnks", {{1, 4, 73, 91}, {1, 5, 91, 57}}},

    {"ij -> i",  {{30, 40}}},
    {"ij -> i",  {{113, 374}}},

    {"...ij -> ...i", {{30, 40}}},
    {"...ij -> ...i", {{113, 374}}},

    {"...ij, ...jk -> ...ik",  {{40, 50}, {50, 80}}},
    {"...ij, ...jk -> ...ik",  {{47, 51}, {51, 83}}},
};

class Layer_Einsum: public TestBaseWithParam<EinsumParams> {};

PERF_TEST_P_(Layer_Einsum, einsum) {
    const EinsumParams& params = GetParam();
    LayerParams lp;
    lp.type = "Einsum";
    lp.name = "testEinsum";
    lp.set("equation", params.equation);
    lp.set("inputSize", params.inputSize);
    lp.set("outputSize", 1);

    CV_CheckFalse(params.einsumInpShapes.empty(), "ERROR no inputs shapes provided");

    for (int i = 0; i < params.einsumInpShapes.size(); i++) {
        lp.set("inputShapes" + cv::format("%d", i), DictValue::arrayInt(params.einsumInpShapes[i].begin(), params.einsumInpShapes[i].size()));
    }

    Net net;
    std::vector<Mat> inputs;
    std::vector<std::string> input_names;
    int id = net.addLayer(lp.name, lp.type, lp);

    for (int i = 0; i < params.inputSize; ++i) {
        // create inputs
        inputs.emplace_back(Mat(params.einsumInpShapes[i], CV_32FC1));

        // connect each input to the layer
        net.connect(0, i, id, i);

        // create input names dynamically, assuming input naming follows a consistent pattern
        input_names.emplace_back("input" + std::to_string(i + 1));
    }

    //warm up
    std::vector<Mat> outputs;
    net.setInputsNames(input_names);
    for (int i = 0; i < input_names.size(); i++){
        net.setInput(inputs[i], input_names[i]);
    }
    net.forward(outputs, "testEinsum");

    TEST_CYCLE()
    {
        net.forward(outputs, "testEinsum");
    }
    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Layer_Einsum, testing::ValuesIn(testEinsumConfigs));

}; //namespace
