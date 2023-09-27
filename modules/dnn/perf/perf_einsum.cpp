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
     (*os) << "\nEqiation: \t" << params.equation << "\n"
        << "InputSize: \t" << params.inputSize << "\n"
        << "OutputSize: \t" << params.outputSize << "\n";

        for(int i = 0; i < params.einsumInpShapes.size(); i++)
        {
            (*os) << "InputShape " << i << ": \t";
            for(int j = 0; j < params.einsumInpShapes[i].size(); j++)
            {
                (*os) << params.einsumInpShapes[i][j] << " ";
            }
            (*os) << std::endl;
        }
}

// test cases
static const EinsumParams testEinsumConfigs[] = {
    {"ij, jk -> ik", 2, 1,  {{2, 3}, {3, 2}}}

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

    // create inputs
    Mat input1(params.einsumInpShapes[0].size(), params.einsumInpShapes[0].data(), CV_32FC1);
    Mat input2(params.einsumInpShapes[1].size(), params.einsumInpShapes[1].data(), CV_32FC1);

    Net net;
    net.setInput(input1);
    net.setInput(input2);
    net.addLayerToPrev(lp.name, lp.type, lp);

    // Warm up
    std::vector<Mat> outputs;
    net.forward(outputs, "testEinsum");

    TEST_CYCLE()
    {
        net.forward(outputs, "testEinsum");
    }
    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Layer_Einsum, testing::ValuesIn(testEinsumConfigs));

}; //namespace