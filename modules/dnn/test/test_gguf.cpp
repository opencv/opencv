#include "test_precomp.hpp"
#include "opencv2/dnn.hpp"
#include <fstream>


namespace opencv_test { namespace {
// Helper function to mimic the _tf function.
template<typename TString>
static std::string _tf(TString filename, bool required = true)
{
    return findDataFile(std::string("dnn/gguf/") + filename, required);
}



class Test_GGUFImporter : public ::testing::Test
{
protected:
    // You can do common initialization here if needed.
};

// This test creates a new GGUFImporter, parses a file using _tf to locate it,
// then calls parse_attn_qkv() and checks that a Gemm layer has been added to the net.
TEST_F(Test_GGUFImporter, readNetFromGGUF)
{
    // Locate the GGUF file; this should be in the directory dnn/gguf/ (adjust the filename as needed)
    std::string ggufModelPath = _tf("mha.gguf", true);
    std::string onnxModelPath = _tf("mha.onnx", true);
    std::string inputTensorPath = _tf("input.pb", true);

    Net ggufnet = readNetFromGGUF(ggufModelPath.c_str());
    ASSERT_FALSE(ggufnet.empty());

    Net onnxnet = readNetFromONNX(onnxModelPath.c_str());
    ASSERT_FALSE(onnxnet.empty());

    std::vector<Mat> inps;
    inps.push_back( readTensorFromONNX(inputTensorPath.c_str()));

    std::vector<String> inputNames;
    inputNames.push_back("input");
    onnxnet.setInputsNames(inputNames);
    onnxnet.setInput(inps[0], "input");
    Mat ref = onnxnet.forward("");
    //ggufnet.setInputsNames(inputNames);
    ggufnet.setInput(inps[0], "globInput");
    Mat out = ggufnet.forward("");

    // limits thardcode as in 
    normAssert(ref, out, "", 1e-5, 1e-4);

}
    
// Main entry point for tests.
GTEST_API_ int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
}}