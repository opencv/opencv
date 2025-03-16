//! [tutorial]
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include "../dnn/common.hpp"

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace cv::ccm;
using namespace mcc;

const string about =
    "This sample detects Macbeth color checker using DNN or thresholding and applies color correction."
    "To run default:\n"
    "\t ./example_cpp_color_correction_model --input=path/to/your/input/image\n"
    "With DNN model:\n"
    "\t ./example_cpp_color_correction_model mcc --input=path/to/your/input/image/\n\n"
    "Model path can also be specified using --model argument. And config path can be specified using --config. Download it using python download_models.py mcc from dnn samples directory\n\n";

const string param_keys =
    "{ help h          |                   | Print help message. }"
    "{ @alias          |                   | An alias name of model to extract preprocessing parameters from models.yml file. }"
    "{ zoo             | ../dnn/models.yml | An optional path to file with preprocessing parameters }"
    "{ input i         |                   | Path to input image for computing CCM. Skip to use device camera. }"
    "{ query q         |                   | Path to query image to apply color correction. If not provided, input image will be used. }"
    "{ type            |         0         | chartType: 0-Standard, 1-DigitalSG, 2-Vinyl }"
    "{ num_charts      |         1         | Maximum number of charts in the image }"
    "{ ccm_file        |                   | Path to YAML file containing pre-computed CCM parameters }";

const string backend_keys = format(
    "{ backend          | default | Choose one of computation backends: "
                            "default: automatically (by default), "
                            "openvino: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                            "opencv: OpenCV implementation, "
                            "vkcom: VKCOM, "
                            "cuda: CUDA, "
                            "webnn: WebNN }");

const string target_keys = format(
    "{ target           | cpu | Choose one of target computation devices: "
                            "cpu: CPU target (by default), "
                            "opencl: OpenCL, "
                            "opencl_fp16: OpenCL fp16 (half-float precision), "
                            "vpu: VPU, "
                            "vulkan: Vulkan, "
                            "cuda: CUDA, "
                            "cuda_fp16: CUDA fp16 (half-float preprocess) }");

string keys = param_keys + backend_keys + target_keys;

static bool processFrame(const Mat& frame, Ptr<CCheckerDetector> detector, Mat& src, int nc){
    if (!detector->process(frame, nc))
    {
        return false;
    }
    vector<Ptr<CChecker>> checkers = detector->getListColorChecker();
    detector->draw(checkers, frame);
    src = checkers[0]->getChartsRGB(false);

    return true;
}

int main(int argc, char* argv[]) {
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if (parser.has("help")) {
        cout << about << endl;
        parser.printMessage();
        return 0;
    }

    string modelName = parser.get<String>("@alias");
    string zooFile = parser.get<String>("zoo");
    const char* path = getenv("OPENCV_SAMPLES_DATA_PATH");

    if ((path != NULL) || parser.has("@alias")) {
        zooFile = findFile(zooFile);
    }
    else{
        cout<<"[WARN] set the environment variables or pass the arguments --model, --config and models.yml file using --zoo for using dnn based detector. Continuing with default detector.\n\n";
    }
    keys += genPreprocArguments(modelName, zooFile);
    parser = CommandLineParser(argc, argv, keys);

    int t = parser.get<int>("type");
    CV_Assert(0 <= t && t <= 2);
    ColorChart chartType = ColorChart(t);

    const string sha1 = parser.get<String>("sha1");
    const string modelPath = findModel(parser.get<string>("model"), sha1);
    const string config_sha1 = parser.get<String>("config_sha1");
    const string configPath = findModel(parser.get<string>("config"), config_sha1);
    const string backend = parser.get<String>("backend");
    const string target = parser.get<String>("target");

    int nc = parser.get<int>("num_charts");

    // Get input and target image paths
    const string inputFile = parser.get<String>("input");
    const string queryFile = parser.get<String>("query");
    const string ccmFile = parser.get<String>("ccm_file");

    if (!ccmFile.empty()) {
        // When ccm_file is provided, only query is required
        if (queryFile.empty()) {
            cout << "Error: Query image path must be provided when using pre-computed CCM." << endl;
            parser.printMessage();
            return -1;
        }
    } else {
        // Original validation for when computing new CCM
        if (inputFile.empty()) {
            cout << "Error: Input image path must be provided." << endl;
            parser.printMessage();
            return -1;
        }
    }

    cv::ccm::ColorCorrectionModel model;
    Mat queryImage;

    if (!ccmFile.empty()) {
        // Load CCM from YAML file
        FileStorage fs(ccmFile, FileStorage::READ);
        if (!fs.isOpened()) {
            cout << "Error: Unable to open CCM file: " << ccmFile << endl;
            return -1;
        }
        model.read(fs["ColorCorrectionModel"]);
        fs.release();
        cout << "Loaded CCM from file: " << ccmFile << endl;

        // Read query image when using pre-computed CCM
        queryImage = imread(findFile(queryFile));
        if (queryImage.empty()) {
            cout << "Error: Unable to read query image." << endl;
            return -1;
        }
    } else {
        // Read input image for computing new CCM
        Mat originalImage = imread(findFile(inputFile));
        if (originalImage.empty()) {
            cout << "Error: Unable to read input image." << endl;
            return -1;
        }

        // Process first image to compute CCM
        Mat image = originalImage.clone();
        Mat src;

        Ptr<CCheckerDetector> detector;
        if (!modelPath.empty() && !configPath.empty()) {
            Net net = readNetFromTensorflow(modelPath, configPath);
            net.setPreferableBackend(getBackendID(backend));
            net.setPreferableTarget(getTargetID(target));
            detector = CCheckerDetector::create(net);
            cout << "Using DNN-based checker detector." << endl;
        } else {
            detector = CCheckerDetector::create();
            cout << "Using thresholding-based checker detector." << endl;
        }
        detector->setColorChartType(chartType);

        if (!processFrame(image, detector, src, nc)) {
            cout << "No chart detected in the input image!" << endl;
            return -1;
        }

        cout << "Actual colors: " << src << endl << endl;

        // Convert to double and normalize
        src.convertTo(src, CV_64F, 1.0/255.0);

        // Color correction model
        model = cv::ccm::ColorCorrectionModel(src, cv::ccm::COLORCHECKER_MACBETH);
        model.setCCMType(CCM_LINEAR);
        model.setDistance(DISTANCE_CIE2000);
        model.setLinearization(LINEARIZATION_GAMMA);
        model.setLinearizationGamma(2.2);

        Mat ccm = model.compute();
        cout << "Computed CCM Matrix:\n" << ccm << endl;
        cout << "Loss: " << model.getLoss() << endl;

        // Save model parameters to YAML file
        FileStorage fs("ccm_output.yaml", FileStorage::WRITE);
        model.write(fs);
        fs.release();
        cout << "Model parameters saved to ccm_output.yaml" << endl;

        // Set query image for correction
        if (queryFile.empty()) {
            cout << "[WARN] No query image provided, applying color correction on input image" << endl;
            queryImage = originalImage.clone();
        } else {
            queryImage = imread(findFile(queryFile));
            if (queryImage.empty()) {
                cout << "Error: Unable to read query image." << endl;
                return -1;
            }
        }
    }

    // Apply correction to query image
    Mat calibratedImage, normalizedImage;
    cvtColor(queryImage, normalizedImage, COLOR_BGR2RGB);
    normalizedImage.convertTo(normalizedImage, CV_64F, 1.0/255.0);  // Convert to double and normalize
    model.correctImage(normalizedImage, calibratedImage);

    // Convert back to 8-bit
    calibratedImage *= 255.0;
    calibratedImage.convertTo(calibratedImage, CV_8UC3);
    cvtColor(calibratedImage, calibratedImage, COLOR_RGB2BGR);

    string outputFile = "corrected_output.png";
    imwrite(outputFile, calibratedImage);
    cout << "Corrected image saved to: " << outputFile << endl;

    return 0;
}
//! [tutorial]
