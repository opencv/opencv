#include <iostream>
#include <fstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>

using namespace cv;
using namespace cv::dnn;

String keys =
        "{ help  h                          | | Print help message. }"
        "{ inputImage i                     | | Path to an input image. Skip this argument to capture frames from a camera. }"
        "{ modelPath mp                     | | Path to a binary .onnx file contains trained CRNN text recognition model. "
            "Download links are provided in doc/tutorials/dnn/dnn_text_spotting/dnn_text_spotting.markdown}"
        "{ RGBInput rgb                     |0| 0: imread with flags=IMREAD_GRAYSCALE; 1: imread with flags=IMREAD_COLOR. }"
        "{ evaluate e                       |false| false: predict with input images; true: evaluate on benchmarks. }"
        "{ evalDataPath edp                 | | Path to benchmarks for evaluation. "
            "Download links are provided in doc/tutorials/dnn/dnn_text_spotting/dnn_text_spotting.markdown}"
        "{ vocabularyPath vp                | alphabet_36.txt | Path to recognition vocabulary. "
            "Download links are provided in doc/tutorials/dnn/dnn_text_spotting/dnn_text_spotting.markdown}";

String convertForEval(String &input);

int main(int argc, char** argv)
{
    // Parse arguments
    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run the PyTorch implementation of "
                 "An End-to-End Trainable Neural Network for Image-based SequenceRecognition and Its Application to Scene Text Recognition "
                 "(https://arxiv.org/abs/1507.05717)");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    String modelPath = parser.get<String>("modelPath");
    String vocPath = parser.get<String>("vocabularyPath");
    int imreadRGB = parser.get<int>("RGBInput");

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    // Load the network
    CV_Assert(!modelPath.empty());
    TextRecognitionModel recognizer(modelPath);

    // Load vocabulary
    CV_Assert(!vocPath.empty());
    std::ifstream vocFile;
    vocFile.open(samples::findFile(vocPath));
    CV_Assert(vocFile.is_open());
    String vocLine;
    std::vector<String> vocabulary;
    while (std::getline(vocFile, vocLine)) {
        vocabulary.push_back(vocLine);
    }
    recognizer.setVocabulary(vocabulary);
    recognizer.setDecodeType("CTC-greedy");

    // Set parameters
    double scale = 1.0 / 127.5;
    Scalar mean = Scalar(127.5, 127.5, 127.5);
    Size inputSize = Size(100, 32);
    recognizer.setInputParams(scale, inputSize, mean);

    if (parser.get<bool>("evaluate"))
    {
        // For evaluation
        String evalDataPath = parser.get<String>("evalDataPath");
        CV_Assert(!evalDataPath.empty());
        String gtPath = evalDataPath + "/test_gts.txt";
        std::ifstream evalGts;
        evalGts.open(gtPath);
        CV_Assert(evalGts.is_open());

        String gtLine;
        int cntRight=0, cntAll=0;
        TickMeter timer;
        timer.reset();

        while (std::getline(evalGts, gtLine)) {
            size_t splitLoc = gtLine.find_first_of(' ');
            String imgPath = evalDataPath + '/' + gtLine.substr(0, splitLoc);
            String gt = gtLine.substr(splitLoc+1);

            // Inference
            Mat frame = imread(samples::findFile(imgPath), imreadRGB);
            CV_Assert(!frame.empty());
            timer.start();
            std::string recognitionResult = recognizer.recognize(frame);
            timer.stop();

            if (gt == convertForEval(recognitionResult))
                cntRight++;

            cntAll++;
        }
        std::cout << "Accuracy(%): " << (double)(cntRight) / (double)(cntAll) << std::endl;
        std::cout << "Average Inference Time(ms): " << timer.getTimeMilli() / (double)(cntAll) << std::endl;
    }
    else
    {
        // Create a window
        static const std::string winName = "Input Cropped Image";

        // Open an image file
        CV_Assert(parser.has("inputImage"));
        Mat frame = imread(samples::findFile(parser.get<String>("inputImage")), imreadRGB);
        CV_Assert(!frame.empty());

        // Recognition
        std::string recognitionResult = recognizer.recognize(frame);

        imshow(winName, frame);
        std::cout << "Predition: '" << recognitionResult << "'" << std::endl;
        waitKey();
    }

    return 0;
}

// Convert the predictions to lower case, and remove other characters.
// Only for Evaluation
String convertForEval(String & input)
{
    String output;
    for (uint i = 0; i < input.length(); i++){
        char ch = input[i];
        if ((int)ch >= 97 && (int)ch <= 122) {
            output.push_back(ch);
        } else if ((int)ch >= 65 && (int)ch <= 90) {
            output.push_back((char)(ch + 32));
        } else {
            continue;
        }
    }

    return output;
}
