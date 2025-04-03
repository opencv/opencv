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





    // Check if the output is not empty





    
    
    //     CV_Error(Error::StsUnsupportedFormat, "Unsupported extension");

    // Net net = readNetFromONNX(onnxmodel);
    // ASSERT_FALSE(net.empty());


    // // Parse the file (throws if file reading or parsing fails)
    // ASSERT_NO_THROW(importer.parseFile(filePath.c_str()));
    
    // // Create a dummy LayerParams object for the attn_qkv layer
    // LayerParams lp;
    
    // // Call parse_attn_qkv to simulate the test parsing step
    // ASSERT_NO_THROW(importer.parse_attn_qkv(lp));
    
    // // After parse_attn_qkv, the importer.net should have a mainGraph set.
    // Ptr<Graph> graph = importer.net.getImpl()->mainGraph;
    // ASSERT_FALSE(graph.empty()) << "Graph should not be empty after parsing attn_qkv.";
    
    // // The graph should contain at least one layer.
    // std::vector<Ptr<Layer>> prog = graph->getProg();
    // ASSERT_FALSE(prog.empty()) << "Graph program should contain at least one layer.";
    
    // // We expect that parse_attn_qkv creates a Gemm layer.
    // EXPECT_EQ(prog[0]->type, "Gemm");
}
    
// Main entry point for tests.
GTEST_API_ int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
}}