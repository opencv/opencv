#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "common.hpp"

using namespace cv;
using namespace std;
using namespace dnn;

const string about =
        "Use this script to run semantic segmentation deep learning networks using OpenCV.\n\n"
        "Firstly, download required models using `download_models.py` (if not already done). Set environment variable OPENCV_DOWNLOAD_CACHE_DIR to specify where models should be downloaded. Also, point OPENCV_SAMPLES_DATA_PATH to opencv/samples/data.\n"
        "To run:\n"
        "\t ./example_dnn_classification modelName(e.g. u2netp) --input=$OPENCV_SAMPLES_DATA_PATH/butterfly.jpg (or ignore this argument to use device camera)\n"
        "Model path can also be specified using --model argument.";

const string param_keys =
    "{ help  h    |                   | Print help message. }"
    "{ @alias     |                   | An alias name of model to extract preprocessing parameters from models.yml file. }"
    "{ zoo        | ../dnn/models.yml | An optional path to file with preprocessing parameters }"
    "{ device     |         0         | camera device number. }"
    "{ input i    |                   | Path to input image or video file. Skip this argument to capture frames from a camera. }"
    "{ colors     |                   | Optional path to a text file with colors for an every class. "
    "Every color is represented with three values from 0 to 255 in BGR channels order. }";

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
vector<string> labels;
vector<Vec3b> colors;


static void colorizeSegmentation(const Mat &score, Mat &segm)
{
    const int rows = score.size[2];
    const int cols = score.size[3];
    const int chns = score.size[1];

    if (colors.empty())
    {
        // Generate colors.
        colors.push_back(Vec3b());
        for (int i = 1; i < chns; ++i)
        {
            Vec3b color;
            for (int j = 0; j < 3; ++j)
                color[j] = (colors[i - 1][j] + rand() % 256) / 2;
            colors.push_back(color);
        }
    }
    else if (chns != (int)colors.size())
    {
        CV_Error(Error::StsError, format("Number of output labels does not match "
                                         "number of colors (%d != %zu)",
                                         chns, colors.size()));
    }

    Mat maxCl = Mat::zeros(rows, cols, CV_8UC1);
    Mat maxVal(rows, cols, CV_32FC1, score.data);
    for (int ch = 1; ch < chns; ch++)
    {
        for (int row = 0; row < rows; row++)
        {
            const float *ptrScore = score.ptr<float>(0, ch, row);
            uint8_t *ptrMaxCl = maxCl.ptr<uint8_t>(row);
            float *ptrMaxVal = maxVal.ptr<float>(row);
            for (int col = 0; col < cols; col++)
            {
                if (ptrScore[col] > ptrMaxVal[col])
                {
                    ptrMaxVal[col] = ptrScore[col];
                    ptrMaxCl[col] = (uchar)ch;
                }
            }
        }
    }
    segm.create(rows, cols, CV_8UC3);
    for (int row = 0; row < rows; row++)
    {
        const uchar *ptrMaxCl = maxCl.ptr<uchar>(row);
        Vec3b *ptrSegm = segm.ptr<Vec3b>(row);
        for (int col = 0; col < cols; col++)
        {
            ptrSegm[col] = colors[ptrMaxCl[col]];
        }
    }
}

static void showLegend(FontFace fontFace)
{
    static const int kBlockHeight = 30;
    static Mat legend;
    if (legend.empty())
    {
        const int numClasses = (int)labels.size();
        if ((int)colors.size() != numClasses)
        {
            CV_Error(Error::StsError, format("Number of output labels does not match "
                                             "number of labels (%zu != %zu)",
                                             colors.size(), labels.size()));
        }
        legend.create(kBlockHeight * numClasses, 200, CV_8UC3);
        for (int i = 0; i < numClasses; i++)
        {
            Mat block = legend.rowRange(i * kBlockHeight, (i + 1) * kBlockHeight);
            block.setTo(colors[i]);
            Rect r = getTextSize(Size(), labels[i], Point(), fontFace, 15, 400);
            r.height += 15; // padding
            r.width += 10; // padding
            rectangle(block, r, Scalar::all(255), FILLED);
            putText(block, labels[i], Point(10, kBlockHeight/2), Scalar(0,0,0), fontFace, 15, 400);
        }
        namedWindow("Legend", WINDOW_AUTOSIZE);
        imshow("Legend", legend);
    }
}

int main(int argc, char **argv)
{
    CommandLineParser parser(argc, argv, keys);

    const string modelName = parser.get<String>("@alias");
    const string zooFile = findFile(parser.get<String>("zoo"));

    keys += genPreprocArguments(modelName, zooFile);

    parser = CommandLineParser(argc, argv, keys);
    parser.about(about);
    if (!parser.has("@alias") || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    string sha1 = parser.get<String>("sha1");
    float scale = parser.get<float>("scale");
    Scalar mean = parser.get<Scalar>("mean");
    bool swapRB = parser.get<bool>("rgb");
    int inpWidth = parser.get<int>("width");
    int inpHeight = parser.get<int>("height");
    String model = findModel(parser.get<String>("model"), sha1);
    const string backend = parser.get<String>("backend");
    const string target = parser.get<String>("target");
    int stdSize = 20;
    int stdWeight = 400;
    int stdImgSize = 512;
    int imgWidth = -1; // Initialization
    int fontSize = 50;
    int fontWeight = 500;
    FontFace fontFace("sans");

    // Open file with labels names.
    if (parser.has("labels"))
    {
        string file = findFile(parser.get<String>("labels"));
        ifstream ifs(file.c_str());
        if (!ifs.is_open())
            CV_Error(Error::StsError, "File " + file + " not found");
        string line;
        while (getline(ifs, line))
        {
            labels.push_back(line);
        }
    }
    // Open file with colors.
    if (parser.has("colors"))
    {
        string file = findFile(parser.get<String>("colors"));
        ifstream ifs(file.c_str());
        if (!ifs.is_open())
            CV_Error(Error::StsError, "File " + file + " not found");
        string line;
        while (getline(ifs, line))
        {
            istringstream colorStr(line.c_str());

            Vec3b color;
            for (int i = 0; i < 3 && !colorStr.eof(); ++i)
                colorStr >> color[i];
            colors.push_back(color);
        }
    }

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    CV_Assert(!model.empty());
    //! [Read and initialize network]
    EngineType engine = ENGINE_AUTO;
    if (backend != "default" || target != "cpu"){
        engine = ENGINE_CLASSIC;
    }
    Net net = readNetFromONNX(model, engine);
    net.setPreferableBackend(getBackendID(backend));
    net.setPreferableTarget(getTargetID(target));
    //! [Read and initialize network]
    // Create a window
    static const string kWinName = "Deep learning semantic segmentation in OpenCV";
    namedWindow(kWinName, WINDOW_AUTOSIZE);

    //! [Open a video file or an image file or a camera stream]
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(findFile(parser.get<String>("input")));
    else
        cap.open(parser.get<int>("device"));

    if (!cap.isOpened()) {
        cerr << "Error: Video could not be opened." << endl;
        return -1;
    }
    //! [Open a video file or an image file or a camera stream]
    // Process frames.
    Mat frame, blob;
    while (waitKey(1) < 0)
    {
        cap >> frame;
        if (frame.empty())
        {
            waitKey();
            break;
        }
        if (imgWidth == -1){
            imgWidth = max(frame.rows, frame.cols);
            fontSize = min(fontSize, (stdSize*imgWidth)/stdImgSize);
            fontWeight = min(fontWeight, (stdWeight*imgWidth)/stdImgSize);
        }
        imshow("Original Image", frame);
        //! [Create a 4D blob from a frame]
        blobFromImage(frame, blob, scale, Size(inpWidth, inpHeight), mean, swapRB, false);
        //! [Set input blob]
        net.setInput(blob);
        //! [Set input blob]

        if (modelName == "u2netp")
        {
            vector<Mat> output;
            net.forward(output, net.getUnconnectedOutLayersNames());

            Mat pred = output[0].reshape(1, output[0].size[2]);
            pred.convertTo(pred, CV_8U, 255.0);
            Mat mask;
            resize(pred, mask, Size(frame.cols, frame.rows), 0, 0, INTER_AREA);

            // Create overlays for foreground and background
            Mat foreground_overlay;

            // Set foreground (object) to red
            Mat all_zeros = Mat::zeros(frame.size(), CV_8UC1);
            vector<Mat> channels = {all_zeros, all_zeros, mask};
            merge(channels, foreground_overlay);

            // Blend the overlays with the original frame
            addWeighted(frame, 0.25, foreground_overlay, 0.75, 0, frame);
        }
        else
        {
            //! [Make forward pass]
            Mat score = net.forward();
            //! [Make forward pass]
            Mat segm;
            colorizeSegmentation(score, segm);
            resize(segm, segm, frame.size(), 0, 0, INTER_NEAREST);
            addWeighted(frame, 0.1, segm, 0.9, 0.0, frame);
        }

        // Put efficiency information.
        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        string label = format("Inference time: %.2f ms", t);
        Rect r = getTextSize(Size(), label, Point(), fontFace, fontSize, fontWeight);
        r.height += fontSize; // padding
        r.width += 10; // padding
        rectangle(frame, r, Scalar::all(255), FILLED);
        putText(frame, label, Point(10, fontSize), Scalar(0,0,0), fontFace, fontSize, fontWeight);

        imshow(kWinName, frame);
        if (!labels.empty())
            showLegend(fontFace);
    }
    return 0;
}
