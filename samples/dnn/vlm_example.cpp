#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <path_to_vlm_model>" << endl;
        return 1;
    }

    String modelPath = argv[1];

    try {
        // 1. Initialize the vLMSession with the model path
        // Ensure the model path points to a valid ONNX model file supported by OpenCV DNN
        cout << "Loading model from: " << modelPath << endl;
        vLMSession session(modelPath);
        session.isDummy = true;
        session.maxTokens = 20;

        // 2. Prepare input (Prompt)
        // Currently, setPrompt takes a Mat representing the input embedding.
        // Assuming the model expects a specific shape, e.g., (BatchSize, SequenceLength, EmbeddingDim)
        // This is just a dummy example. You need to know your model's embedding dimension.
        int batchSize = 1;
        int seqLen = 261 ;
        int embedDim = 2304; // Example dimension, replace with actual model requirement
        int sizes[] = {batchSize, seqLen, embedDim};

        Mat inputEmbedding(3, sizes, CV_32F);
        randu(inputEmbedding, Scalar(0), Scalar(1)); // Fill with random data for demonstration

        cout << "Setting prompt..." << endl;
        session.setPrompt(inputEmbedding);

        // 3. Generate tokens
        // This runs the generation process
        cout << "Generating..." << endl;
        session.generate();

        // 4. Retrieve results
        std::vector<int> tokens = session.getGeneratedTokens();
        std::vector<Mat> embeddings = session.getGeneratedTokenEmbeddings();

        cout << "Generation complete." << endl;
        cout << "Generated " << tokens.size() << " tokens." << endl;

        for (size_t i = 0; i < tokens.size(); ++i) {
            cout << "Token[" << i << "]: " << tokens[i] << endl;
        }

    } catch (const cv::Exception& e) {
        cerr << "OpenCV Exception: " << e.what() << endl;
        return 1;
    }

    return 0;
}
