//![includes]
#include <fstream>
#include <sstream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <mutex>
#include <thread>
#include <queue>

#include "iostream"
#include "common.hpp"
//![includes]

using namespace cv;
using namespace dnn;
using namespace std;

const string about =
        "Firstly, download required models using `download_models.py` (if not already done). Set environment variable OPENCV_DOWNLOAD_CACHE_DIR to specify where models should be downloaded. Also, point OPENCV_SAMPLES_DATA_PATH to opencv/samples/data.\n"
        "To run:\n"
        "\t ./example_dnn_object_detection model_name --input=path/to/your/input/image/or/video (don't give --input flag if want to use device camera)\n"
        "Sample command:\n"
        "\t ./example_dnn_object_detection yolov8 --input=$OPENCV_SAMPLES_DATA_PATH/baboon.jpg\n"

        "Model path can also be specified using --model argument. ";

const string param_keys =
    "{ help  h     |                   | Print help message. }"
    "{ @alias      |                   | An alias name of model to extract preprocessing parameters from models.yml file. }"
    "{ zoo         | ../dnn/models.yml | An optional path to file with preprocessing parameters }"
    "{ device      |         0         | camera device number. }"
    "{ input i     |                   | Path to input image or video file. Skip this argument to capture frames from a camera. }"
    "{ thr         |        .5         | Confidence threshold. }"
    "{ nms         |        .4         | Non-maximum suppression threshold. }"
    "{ async       |         0         | Number of asynchronous forwards at the same time. "
                        "Choose 0 for synchronous mode }"
    "{ padvalue    |       114.0       | padding value. }"
    "{ paddingmode |         2         | Choose one of padding modes: "
                         "0: resize to required input size without extra processing, "
                         "1: Image will be cropped after resize, "
                         "2: Resize image to the desired size while preserving the aspect ratio of original image }";

const string backend_keys = format(
    "{ backend | default | Choose one of computation backends: "
                        "default: automatically (by default), "
                        "openvino: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                        "opencv: OpenCV implementation, "
                        "vkcom: VKCOM, "
                        "cuda: CUDA, "
                        "webnn: WebNN }");

const string target_keys = format(
    "{ target | cpu | Choose one of target computation devices: "
                    "cpu: CPU target (by default), "
                    "opencl: OpenCL, "
                    "opencl_fp16: OpenCL fp16 (half-float precision), "
                    "vpu: VPU, "
                    "vulkan: Vulkan, "
                    "cuda: CUDA, "
                    "cuda_fp16: CUDA fp16 (half-float preprocess) }");

string keys = param_keys + backend_keys + target_keys;

float confThreshold, nmsThreshold, scale, paddingValue;
vector<string> labels;
Scalar meanv;
bool swapRB;
int inpWidth, inpHeight;
size_t asyncNumReq = 0;
ImagePaddingMode paddingMode;
string modelName, framework;

static void preprocess(const Mat& frame, Net& net, Size inpSize);

static void postprocess(Mat& frame, const vector<Mat>& outs, Net& net, int backend, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes, const string yolo_name);

static void drawPred(vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes, Mat& frame, FontFace& sans, int stdSize, int stdWeight, int stdImgSize, int stdThickness);

static void callback(int pos, void* userdata);

static Scalar getColor(int classId);

static void yoloPostProcessing(
    const vector<Mat>& outs,
    vector<int>& keep_classIds,
    vector<float>& keep_confidences,
    vector<Rect2d>& keep_boxes,
    float conf_threshold,
    float iou_threshold,
    const string& yolo_name);

static void printAliases(string& zooFile){
    vector<string> aliases = findAliases(zooFile, "object_detection");

    cout<<"Alias choices: [ ";
    for (auto it: aliases){
        cout<<"'"<<it<<"' ";
    }
    cout<<"]"<<endl;
}

static Scalar getTextColor(Scalar bgColor) {
    double luminance = 0.299 * bgColor[2] + 0.587 * bgColor[1] + 0.114 * bgColor[0];

    return luminance > 128 ? Scalar(0, 0, 0) : Scalar(255, 255, 255);
}

template <typename T>
class QueueFPS : public std::queue<T>
{
public:
    QueueFPS() : counter(0) {}

    void push(const T& entry)
    {
        std::lock_guard<std::mutex> lock(mutex);

        std::queue<T>::push(entry);
        counter += 1;
        if (counter == 1)
        {
            // Start counting from a second frame (warmup).
            tm.reset();
            tm.start();
        }
    }

    T get()
    {
        std::lock_guard<std::mutex> lock(mutex);
        T entry = this->front();
        this->pop();
        return entry;
    }

    float getFPS()
    {
        tm.stop();
        double fps = counter / tm.getTimeSec();
        tm.start();
        return static_cast<float>(fps);
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex);
        while (!this->empty())
            this->pop();
    }

    unsigned int counter;

private:
    TickMeter tm;
    std::mutex mutex;
};

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);

    string zooFile = parser.get<String>("zoo");
    if (!parser.has("@alias") || parser.has("help"))
    {
        cout << about << endl;
        parser.printMessage();
        printAliases(zooFile);
        return -1;
    }
    zooFile = findFile(zooFile);
    modelName = parser.get<String>("@alias");

    keys += genPreprocArguments(modelName, zooFile);

    parser = CommandLineParser(argc, argv, keys);

    if (!parser.has("model"))
    {
        cout << "Path to model is not provided in command line or model alias is not correct" << endl;
        printAliases(zooFile);
        return -1;
    }

    confThreshold = parser.get<float>("thr");
    nmsThreshold = parser.get<float>("nms");
    //![preprocess_params]
    scale = parser.get<float>("scale");
    meanv = parser.get<Scalar>("mean");
    swapRB = parser.get<bool>("rgb");
    inpWidth = parser.get<int>("width");
    inpHeight = parser.get<int>("height");
    int async = parser.get<int>("async");
    paddingValue = parser.get<float>("padvalue");
    const string yolo_name = parser.get<String>("postprocessing");
    paddingMode = static_cast<ImagePaddingMode>(parser.get<int>("paddingmode"));
    //![preprocess_params]
    String sha1 = parser.get<String>("sha1");
    const string modelPath = findModel(parser.get<String>("model"), sha1);
    const string configPath = findFile(parser.get<String>("config"));
    framework = modelPath.substr(modelPath.rfind('.') + 1);

    if (parser.has("labels"))
    {
        const string file = findFile(parser.get<String>("labels"));
        ifstream ifs(file.c_str());
        if (!ifs.is_open())
            CV_Error(Error::StsError, "File " + file + " not found");
        string line;
        while (getline(ifs, line))
        {
            labels.push_back(line);
        }
    }
    //![read_net]
    Net net = readNet(modelPath, configPath);
    int backend = getBackendID(parser.get<String>("backend"));
    net.setPreferableBackend(backend);
    net.setPreferableTarget(getTargetID(parser.get<String>("target")));
    //![read_net]

    // Create a window
    static const string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, WINDOW_AUTOSIZE);
    int initialConf = (int)(confThreshold * 100);
    createTrackbar("Confidence threshold, %", kWinName, &initialConf, 99, callback, &net);

    // Open a video file or an image file or a camera stream.
    VideoCapture cap;
    bool openSuccess = parser.has("input") ? cap.open(parser.get<String>("input")) : cap.open(parser.get<int>("device"));
    if (!openSuccess){
        cout << "Could not open input file or camera device" << endl;
        return 0;
    }

    FontFace sans("sans");

    int stdSize = 15;
    int stdWeight = 150;
    int stdImgSize = 512;
    int stdThickness = 2;
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    if (async > 0 && backend == DNN_BACKEND_INFERENCE_ENGINE){
        asyncNumReq = async;
    }

    if (async != 0) {
        // Threading is enabled
        bool process = true;

        // Frames capturing thread
        QueueFPS<Mat> framesQueue;
        std::thread framesThread([&]() {
            Mat frame;
            while (process) {
                cap >> frame;
                if (!frame.empty())
                    framesQueue.push(frame.clone());
                else
                    break;
            }
        });

        // Frames processing thread
        QueueFPS<Mat> processedFramesQueue;
        QueueFPS<std::vector<Mat>> predictionsQueue;
        std::thread processingThread([&]() {
            std::queue<AsyncArray> futureOutputs;
            Mat blob;
            while (process) {
                // Get the next frame
                Mat frame;
                {
                    if (!framesQueue.empty()) {
                        frame = framesQueue.get();
                        if (asyncNumReq) {
                            if (futureOutputs.size() == asyncNumReq)
                                frame = Mat();
                        }
                    }
                }

                // Process the frame
                if (!frame.empty()) {
                    preprocess(frame, net, Size(inpWidth, inpHeight));
                    processedFramesQueue.push(frame);

                    if (asyncNumReq) {
                        futureOutputs.push(net.forwardAsync());
                    } else {
                        //![forward]
                        vector<Mat> outs;
                        net.forward(outs, net.getUnconnectedOutLayersNames());
                        predictionsQueue.push(outs);
                        //![forward]
                    }
                }

                while (!futureOutputs.empty() &&
                    futureOutputs.front().wait_for(std::chrono::seconds(0))) {
                    AsyncArray async_out = futureOutputs.front();
                    futureOutputs.pop();
                    Mat out;
                    async_out.get(out);
                    predictionsQueue.push({out});
                }
            }
        });

        // Postprocessing and rendering loop
        while (waitKey(100) < 0) {
            if (predictionsQueue.empty())
                continue;

            vector<Mat> outs = predictionsQueue.get();
            Mat frame = processedFramesQueue.get();

            classIds.clear();
            confidences.clear();
            boxes.clear();
            postprocess(frame, outs, net, backend, classIds, confidences, boxes, yolo_name);

            drawPred(classIds, confidences, boxes, frame, sans, stdSize, stdWeight, stdImgSize, stdThickness);

            int imgWidth = max(frame.rows, frame.cols);
            int size = static_cast<int>((stdSize * imgWidth) / (stdImgSize * 1.5));
            int weight = static_cast<int>((stdWeight * imgWidth) / (stdImgSize * 1.5));

            if (predictionsQueue.counter > 1) {
                string label = format("Camera: %.2f FPS", framesQueue.getFPS());
                rectangle(frame, Point(0, 0), Point(10 * size, 3 * size + size / 4), Scalar::all(255), FILLED);
                putText(frame, label, Point(0, size), Scalar::all(0), sans, size, weight);

                label = format("Network: %.2f FPS", predictionsQueue.getFPS());
                putText(frame, label, Point(0, 2 * size), Scalar::all(0), sans, size, weight);

                label = format("Skipped frames: %d", framesQueue.counter - predictionsQueue.counter);
                putText(frame, label, Point(0, 3 * size), Scalar::all(0), sans, size, weight);
            }
            imshow(kWinName, frame);
        }

        process = false;
        framesThread.join();
        processingThread.join();
    } else {
        if (asyncNumReq)
            CV_Error(Error::StsNotImplemented, "Asynchronous forward is supported only with Inference Engine backend.");
        // Threading is disabled, run synchronously
        Mat frame, blob;
        while (waitKey(100) < 0) {
            cap >> frame;
            if (frame.empty()) {
                waitKey();
                break;
            }
            preprocess(frame, net, Size(inpWidth, inpHeight));

            vector<Mat> outs;
            net.forward(outs, net.getUnconnectedOutLayersNames());

            classIds.clear();
            confidences.clear();
            boxes.clear();

            postprocess(frame, outs, net, backend, classIds, confidences, boxes, yolo_name);

            drawPred(classIds, confidences, boxes, frame, sans, stdSize, stdWeight, stdImgSize, stdThickness);

            vector<double> layersTimes;
            int imgWidth = max(frame.rows, frame.cols);
            int size = static_cast<int>((stdSize * imgWidth) / (stdImgSize * 1.5));
            int weight = static_cast<int>((stdWeight * imgWidth) / (stdImgSize * 1.5));
            double freq = getTickFrequency() / 1000;
            double t = net.getPerfProfile(layersTimes) / freq;
            string label = format("Inference time: %.2f ms", t);
            putText(frame, label, Point(0, size), Scalar(0, 255, 0), sans, size, weight);
            imshow(kWinName, frame);
        }
    }
    return 0;
}

void preprocess(const Mat& frame, Net& net, Size inpSize)
{
    Size size(inpSize.width <= 0 ? frame.cols : inpSize.width, inpSize.height <= 0 ? frame.rows : inpSize.height);

    // Prepare the blob from the image
    Mat inp;
    if(framework == "weights"){ // checks whether model is darknet
        blobFromImage(frame, inp, scale, size, meanv, swapRB, false, CV_32F);
    }
    else{
        //![preprocess_call]
        Image2BlobParams imgParams(
            Scalar::all(scale),
            size,
            meanv,
            swapRB,
            CV_32F,
            DNN_LAYOUT_NCHW,
            paddingMode,
            paddingValue);

        inp = blobFromImageWithParams(frame, imgParams);
        //![preprocess_call]
    }

    // Set the blob as the network input
    net.setInput(inp);

    // Check if the model is Faster-RCNN or R-FCN
    if (net.getLayer(0)->outputNameToIndex("im_info") != -1)
    {
        // Resize the frame and prepare imInfo
        resize(frame, frame, size);
        Mat imInfo = (Mat_<float>(1, 3) << size.height, size.width, 1.6f);
        net.setInput(imInfo, "im_info");
    }
}

void yoloPostProcessing(
    const vector<Mat>& outs,
    vector<int>& keep_classIds,
    vector<float>& keep_confidences,
    vector<Rect2d>& keep_boxes,
    float conf_threshold,
    float iou_threshold,
    const string& yolo_name)
{
    // Retrieve
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect2d> boxes;

    vector<Mat> outs_copy = outs;

    if (yolo_name == "yolov8")
    {
        transposeND(outs_copy[0], {0, 2, 1}, outs_copy[0]);
    }

    if (yolo_name == "yolonas")
    {
        // outs contains 2 elements of shape [1, 8400, 80] and [1, 8400, 4]. Concat them to get [1, 8400, 84]
        Mat concat_out;
        // squeeze the first dimension
        outs_copy[0] = outs_copy[0].reshape(1, outs_copy[0].size[1]);
        outs_copy[1] = outs_copy[1].reshape(1, outs_copy[1].size[1]);
        hconcat(outs_copy[1], outs_copy[0], concat_out);
        outs_copy[0] = concat_out;
        // remove the second element
        outs_copy.pop_back();
        // unsqueeze the first dimension
        outs_copy[0] = outs_copy[0].reshape(0, vector<int>{1, 8400, 84});
    }

    for (auto preds : outs_copy)
    {
        preds = preds.reshape(1, preds.size[1]); // [1, 8400, 85] -> [8400, 85]
        for (int i = 0; i < preds.rows; ++i)
        {
            // filter out non-object
            float obj_conf = (yolo_name == "yolov8" || yolo_name == "yolonas") ? 1.0f : preds.at<float>(i, 4);
            if (obj_conf < conf_threshold)
                continue;

            Mat scores = preds.row(i).colRange((yolo_name == "yolov8" || yolo_name == "yolonas") ? 4 : 5, preds.cols);
            double conf;
            Point maxLoc;
            minMaxLoc(scores, 0, &conf, 0, &maxLoc);

            conf = (yolo_name == "yolov8" || yolo_name == "yolonas") ? conf : conf * obj_conf;
            if (conf < conf_threshold)
                continue;

            // get bbox coords
            float* det = preds.ptr<float>(i);
            double cx = det[0];
            double cy = det[1];
            double w = det[2];
            double h = det[3];

            // [x1, y1, x2, y2]
            if (yolo_name == "yolonas") {
                boxes.push_back(Rect2d(cx, cy, w, h));
            } else {
                boxes.push_back(Rect2d(cx - 0.5 * w, cy - 0.5 * h,
                                        cx + 0.5 * w, cy + 0.5 * h));
            }
            classIds.push_back(maxLoc.x);
            confidences.push_back(static_cast<float>(conf));
        }
    }

    // NMS
    vector<int> keep_idx;
    NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, keep_idx);

    for (auto i : keep_idx)
    {
        keep_classIds.push_back(classIds[i]);
        keep_confidences.push_back(confidences[i]);
        keep_boxes.push_back(boxes[i]);
    }
}

void postprocess(Mat& frame, const vector<Mat>& outs, Net& net, int backend, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes, const string yolo_name)
{
    static vector<int> outLayers = net.getUnconnectedOutLayers();
    static string outLayerType = net.getLayer(outLayers[0])->type;

    if (outLayerType == "DetectionOutput")
    {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        CV_Assert(outs.size() > 0);
        for (size_t k = 0; k < outs.size(); k++)
        {
            float* data = (float*)outs[k].data;
            for (size_t i = 0; i < outs[k].total(); i += 7)
            {
                float confidence = data[i + 2];
                if (confidence > confThreshold)
                {
                    int left   = (int)data[i + 3];
                    int top    = (int)data[i + 4];
                    int right  = (int)data[i + 5];
                    int bottom = (int)data[i + 6];
                    int width  = right - left + 1;
                    int height = bottom - top + 1;
                    if (width <= 2 || height <= 2)
                    {
                        left   = (int)(data[i + 3] * frame.cols);
                        top    = (int)(data[i + 4] * frame.rows);
                        right  = (int)(data[i + 5] * frame.cols);
                        bottom = (int)(data[i + 6] * frame.rows);
                        width  = right - left + 1;
                        height = bottom - top + 1;
                    }
                    classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
                    boxes.push_back(Rect(left, top, width, height));
                    confidences.push_back(confidence);
                }
            }
        }
    }
    else if (outLayerType == "Region")
    {
        for (size_t i = 0; i < outs.size(); ++i)
        {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }
    }
    else if (outLayerType == "Identity")
    {
        //![forward_buffers]
        vector<int> keep_classIds;
        vector<float> keep_confidences;
        vector<Rect2d> keep_boxes;
        //![forward_buffers]

        //![postprocess]
        yoloPostProcessing(outs, keep_classIds, keep_confidences, keep_boxes, confThreshold, nmsThreshold, yolo_name);
        //![postprocess]

        for (size_t i = 0; i < keep_classIds.size(); ++i)
        {
            classIds.push_back(keep_classIds[i]);
            confidences.push_back(keep_confidences[i]);
            Rect2d box = keep_boxes[i];
            boxes.push_back(Rect(cvFloor(box.x), cvFloor(box.y), cvFloor(box.width-box.x), cvFloor(box.height-box.y)));
        }
        if (framework == "onnx"){
            Image2BlobParams paramNet;
                paramNet.scalefactor = Scalar::all(scale);
                paramNet.size = Size(inpWidth, inpHeight);
                paramNet.mean = meanv;
                paramNet.swapRB = swapRB;
                paramNet.paddingmode = paddingMode;

            paramNet.blobRectsToImageRects(boxes, boxes, frame.size());
        }
    }
    else
    {
        CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);
    }

    // NMS is used inside Region layer only on DNN_BACKEND_OPENCV for other backends we need NMS in sample
    // or NMS is required if the number of outputs > 1
    if (outLayers.size() > 1 || (outLayerType == "Region" && backend != DNN_BACKEND_OPENCV))
    {
        map<int, vector<size_t> > class2indices;
        for (size_t i = 0; i < classIds.size(); i++)
        {
            if (confidences[i] >= confThreshold)
            {
                class2indices[classIds[i]].push_back(i);
            }
        }
        vector<Rect> nmsBoxes;
        vector<float> nmsConfidences;
        vector<int> nmsClassIds;
        for (map<int, vector<size_t> >::iterator it = class2indices.begin(); it != class2indices.end(); ++it)
        {
            vector<Rect> localBoxes;
            vector<float> localConfidences;
            vector<size_t> classIndices = it->second;
            for (size_t i = 0; i < classIndices.size(); i++)
            {
                localBoxes.push_back(boxes[classIndices[i]]);
                localConfidences.push_back(confidences[classIndices[i]]);
            }
            vector<int> nmsIndices;
            NMSBoxes(localBoxes, localConfidences, confThreshold, nmsThreshold, nmsIndices);
            for (size_t i = 0; i < nmsIndices.size(); i++)
            {
                size_t idx = nmsIndices[i];
                nmsBoxes.push_back(localBoxes[idx]);
                nmsConfidences.push_back(localConfidences[idx]);
                nmsClassIds.push_back(it->first);
            }
        }
        boxes = nmsBoxes;
        classIds = nmsClassIds;
        confidences = nmsConfidences;
    }
}

void drawPred(vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes, Mat& frame, FontFace& sans, int stdSize, int stdWeight, int stdImgSize, int stdThickness)
{
    //![draw_boxes]
    int imgWidth = max(frame.rows, frame.cols);
    int size = (stdSize*imgWidth)/stdImgSize;
    int weight = (stdWeight*imgWidth)/stdImgSize;
    int thickness = (stdThickness*imgWidth)/stdImgSize;

    for (size_t idx = 0; idx < boxes.size(); ++idx){
        Scalar boxColor = getColor(classIds[idx]);
        int left = boxes[idx].x;
        int top = boxes[idx].y;
        int right = boxes[idx].x + boxes[idx].width;
        int bottom = boxes[idx].y + boxes[idx].height;
        rectangle(frame, Point(left, top), Point(right, bottom), boxColor, thickness);

        string label = format("%.2f", confidences[idx]);
        if (!labels.empty())
        {
            CV_Assert(classIds[idx] < (int)labels.size());
            label = labels[classIds[idx]] + ": " + label;
        }

        Rect r = getTextSize(Size(), label, Point(), sans, size, weight);
        int baseline = r.y + r.height;
        Size labelSize = Size(r.width, r.height + size/4 - baseline);

        top = max(top-thickness/2, labelSize.height);
        rectangle(frame, Point(left-thickness/2, top-(labelSize.height)),
                Point(left + labelSize.width, top), boxColor, FILLED);
        putText(frame, label, Point(left, top-size/4), getTextColor(boxColor), sans, size, weight);
    }
    //![draw_boxes]
}

void callback(int pos, void*)
{
    confThreshold = pos * 0.01f;
}

Scalar getColor(int classId) {
    int r = min((classId >> 0 & 1) * 128 + (classId >> 3 & 1) * 64 + (classId >> 6 & 1) * 32 + 80, 255);
    int g = min((classId >> 1 & 1) * 128 + (classId >> 4 & 1) * 64 + (classId >> 7 & 1) * 32 + 40, 255);
    int b = min((classId >> 2 & 1) * 128 + (classId >> 5 & 1) * 64 + (classId >> 8 & 1) * 32 + 40, 255);
    return Scalar(b, g, r);
}
