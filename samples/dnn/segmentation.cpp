#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "common.hpp"

using namespace cv;
using namespace std;
using namespace dnn;

const string about =
        "Use this script to run semantic segmentation deep learning networks using OpenCV.\n\n"
        "Firstly, download required models using `download_models.py` (if not already done). Set environment variable OPENCV_DOWNLOAD_CACHE_DIR to specify where models should be downloaded. Also, point OPENCV_SAMPLES_DATA_PATH to opencv/samples/data.\n"
        "To run:\n"
        "\t ./example_dnn_classification modelName(e.g. u2netp) --input=$OPENCV_SAMPLES_DATA_PATH/butterfly.jpg (or ignore this argument to use device camera)\n"
        "Model path can also be specified using --model argument.\n"
        "For promptable segmentation, pass a foreground point with --point=x,y (defaults to the image centre):\n"
        "\t ./example_dnn_segmentation sam --input=$OPENCV_SAMPLES_DATA_PATH/butterfly.jpg (or ignore this argument to use device camera) --point=320,240\n";

const string param_keys =
    "{ help  h    |                   | Print help message. }"
    "{ @alias     |                   | An alias name of model to extract preprocessing parameters from models.yml file. }"
    "{ zoo        | ../dnn/models.yml | An optional path to file with preprocessing parameters }"
    "{ device     |         0         | camera device number. }"
    "{ input i    |                   | Path to input image or video file. Skip this argument to capture frames from a camera. }"
    "{ colors     |                   | Optional path to a text file with colors for an every class. "
    "Every color is represented with three values from 0 to 255 in BGR channels order. }"
    "{ point      |                   | Foreground point prompt as 'x,y' in input image coordinates, "
    "used by promptable models (sam). Defaults to the image centre. }";

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


// SAM input: longest edge scaled to target, per-channel normalized, zero-padded to a square.
// Not expressible with blobFromImage; newH/newW report the unpadded extent for mask cropping.
static Mat samPreprocess(const Mat &bgr, int target, int &newH, int &newW)
{
    static const float samMean[3] = {0.485f, 0.456f, 0.406f};
    static const float samStd[3] = {0.229f, 0.224f, 0.225f};

    const double s = (double)target / max(bgr.rows, bgr.cols);
    newH = (int)(bgr.rows * s + 0.5);
    newW = (int)(bgr.cols * s + 0.5);

    Mat rgb, img;
    cvtColor(bgr, rgb, COLOR_BGR2RGB);
    resize(rgb, img, Size(newW, newH), 0, 0, INTER_LINEAR);
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    const int sizes[4] = {1, 3, target, target};
    Mat blob(4, sizes, CV_32F, Scalar(0));
    for (int y = 0; y < newH; y++)
    {
        const float *srow = img.ptr<float>(y);
        for (int x = 0; x < newW; x++)
            for (int ch = 0; ch < 3; ch++)
                blob.ptr<float>(0, ch, y)[x] = (srow[x * 3 + ch] - samMean[ch]) / samStd[ch];
    }
    return blob;
}

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
    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_INFO);

    CommandLineParser parser(argc, argv, keys);

    const string modelName = parser.get<String>("@alias");
    const string zooFile = findFile(parser.get<String>("zoo"));

    keys += genPreprocArguments(modelName, zooFile);
    keys += genPreprocArguments(modelName, zooFile, "decoder_");

    parser = CommandLineParser(argc, argv, keys);
    parser.about(about);
    if (!parser.has("@alias") || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    string sha1 = parser.get<String>("sha1");
    // Models that build their own blob (sam) carry no mean/scale in models.yml.
    float scale = parser.has("scale") ? parser.get<float>("scale") : 1.f;
    Scalar mean = parser.has("mean") ? parser.get<Scalar>("mean") : Scalar();
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

    Point promptPoint(-1, -1); // negative = fall back to the image centre
    if (parser.has("point"))
    {
        stringstream ss(parser.get<String>("point"));
        string xs, ys;
        if (!getline(ss, xs, ',') || !getline(ss, ys, ','))
            CV_Error(Error::StsBadArg, "Point prompt must be given as 'x,y'");
        promptPoint = Point(stoi(xs), stoi(ys));
    }

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    CV_Assert(!model.empty());
    //! [Read and initialize network]
    EngineType engine = ENGINE_OPENCV;
    Net net = readNetFromONNX(model, engine);
    net.setPreferableBackend(getBackendID(backend));
    net.setPreferableTarget(getTargetID(target));
    net.setProfilingMode(DNN_PROFILE_SUMMARY);
     //! [Read and initialize network]
    // Promptable models split into an image encoder (the primary model) and a prompt/mask decoder.
    Net decoder;
    if (modelName == "sam")
    {
        String decoderModel = findModel(parser.get<String>("decoder_model"), parser.get<String>("decoder_sha1"));
        CV_Assert(!decoderModel.empty());
        decoder = readNetFromONNX(decoderModel, engine);
        decoder.setPreferableBackend(getBackendID(backend));
        decoder.setPreferableTarget(getTargetID(target));
    }
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
        const bool promptable = (modelName == "sam"); // builds its own blob and uses named inputs
        //! [Create a 4D blob from a frame]
        if (!promptable)
            blobFromImage(frame, blob, scale, Size(inpWidth, inpHeight), mean, swapRB, false);
        //! [Set input blob]
        if (!promptable)
            net.setInput(blob);
        //! [Set input blob]
        int64 t0 = getTickCount();

        if (modelName == "sam")
        {
            int newH = 0, newW = 0;
            net.setInput(samPreprocess(frame, inpWidth, newH, newW), "pixel_values");
            vector<Mat> encOuts;
            net.forward(encOuts, vector<String>{"image_embeddings", "image_positional_embeddings"});
            net.printPerfProfile();

            // The prompt is given in input image coordinates, so scale it into the padded frame.
            Point pt = promptPoint.x < 0 ? Point(frame.cols / 2, frame.rows / 2) : promptPoint;
            const double s = (double)inpWidth / max(frame.rows, frame.cols);
            const float ptData[2] = {(float)(pt.x * s), (float)(pt.y * s)};
            const int ptSizes[4] = {1, 1, 1, 2};
            Mat inputPoints(4, ptSizes, CV_32F);
            memcpy(inputPoints.ptr<float>(), ptData, sizeof(ptData));
            const int lbSizes[3] = {1, 1, 1};
            Mat inputLabels(3, lbSizes, CV_64S, Scalar(1)); // 1 = foreground point

            decoder.setInput(inputPoints, "input_points");
            decoder.setInput(inputLabels, "input_labels");
            decoder.setInput(encOuts[0], "image_embeddings");
            decoder.setInput(encOuts[1], "image_positional_embeddings");
            vector<Mat> decOuts;
            decoder.forward(decOuts, vector<String>{"iou_scores", "pred_masks"});

            // The decoder proposes several masks per prompt; keep the highest scoring one.
            const Mat &iouScores = decOuts[0], &predMasks = decOuts[1];
            const float *scorePtr = iouScores.ptr<float>();
            const int numMasks = iouScores.size[iouScores.dims - 1];
            int best = 0;
            for (int i = 1; i < numMasks; i++)
            {
                if (scorePtr[i] > scorePtr[best])
                    best = i;
            }

            // Mask logits cover the padded square: upsample, crop the unpadded extent, then
            // resize to the frame. A logit above zero belongs to the object.
            const int maskH = predMasks.size[predMasks.dims - 2];
            const int maskW = predMasks.size[predMasks.dims - 1];
            Mat lowRes(maskH, maskW, CV_32F, (void*)predMasks.ptr<float>(0, 0, best)), padded, logits;
            resize(lowRes, padded, Size(inpWidth, inpHeight), 0, 0, INTER_LINEAR);
            resize(padded(Rect(0, 0, newW, newH)), logits, frame.size(), 0, 0, INTER_LINEAR);

            Mat overlay = Mat::zeros(frame.size(), CV_8UC3);
            overlay.setTo(Scalar(0, 0, 255), logits > 0.f);
            addWeighted(frame, 0.6, overlay, 0.4, 0.0, frame);
            circle(frame, pt, 5, Scalar(0, 255, 0), FILLED);
        }
        else if (modelName == "u2netp")
        {
            vector<Mat> output;
            net.forward(output, net.getUnconnectedOutLayersNames());
            net.printPerfProfile();

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
            net.printPerfProfile();
            //! [Make forward pass]
            Mat segm;
            colorizeSegmentation(score, segm);
            resize(segm, segm, frame.size(), 0, 0, INTER_NEAREST);
            addWeighted(frame, 0.1, segm, 0.9, 0.0, frame);
        }

        // Put efficiency information.
        double t = (getTickCount() - t0) * 1000.0 / getTickFrequency();
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
