// DaSiamRPN tracker.
// Original paper: https://arxiv.org/abs/1808.06048
// Link to original repo: https://github.com/foolwood/DaSiamRPN
// Links to onnx models:
// - network:     https://www.dropbox.com/s/rr1lk9355vzolqv/dasiamrpn_model.onnx?dl=0
// - kernel_r1:   https://www.dropbox.com/s/999cqx5zrfi7w4p/dasiamrpn_kernel_r1.onnx?dl=0
// - kernel_cls1: https://www.dropbox.com/s/qvmtszx5h339a0w/dasiamrpn_kernel_cls1.onnx?dl=0

#include <iostream>
#include <cmath>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace cv::dnn;

const char *keys =
        "{ help     h  |   | Print help message }"
        "{ input    i  |   | Full path to input video folder, the specific camera index. (empty for camera 0) }"
        "{ net         | dasiamrpn_model.onnx | Path to onnx model of net}"
        "{ kernel_cls1 | dasiamrpn_kernel_cls1.onnx | Path to onnx model of kernel_r1 }"
        "{ kernel_r1   | dasiamrpn_kernel_r1.onnx | Path to onnx model of kernel_cls1 }"
        "{ backend     | 0 | Choose one of computation backends: "
                            "0: automatically (by default), "
                            "1: Halide language (http://halide-lang.org/), "
                            "2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                            "3: OpenCV implementation }"
        "{ target      | 0 | Choose one of target computation devices: "
                            "0: CPU target (by default), "
                            "1: OpenCL, "
                            "2: OpenCL fp16 (half-float precision), "
                            "3: VPU }"
;

// Initial parameters of the model
struct trackerConfig
{
    float windowInfluence = 0.43f;
    float lr = 0.4f;
    int scale = 8;
    bool swapRB = false;
    int totalStride = 8;
    float penaltyK = 0.055f;
    int exemplarSize = 127;
    int instanceSize = 271;
    float contextAmount = 0.5f;
    std::vector<float> ratios = { 0.33f, 0.5f, 1.0f, 2.0f, 3.0f };
    int anchorNum = int(ratios.size());
    Mat anchors;
    Mat windows;
    Scalar avgChans;
    Size imgSize = { 0, 0 };
    Rect2f targetBox = { 0, 0, 0, 0 };
    int scoreSize = (instanceSize - exemplarSize) / totalStride + 1;

    void update_scoreSize()
    {
        scoreSize = int((instanceSize - exemplarSize) / totalStride + 1);
    }
};

static void softmax(const Mat& src, Mat& dst);
static void elementMax(Mat& src);
static Mat generateHanningWindow(const trackerConfig& trackState);
static Mat generateAnchors(trackerConfig& trackState);
static Mat getSubwindow(Mat& img, const Rect2f& targetBox, float originalSize, Scalar avgChans);
static float trackerEval(Mat img, trackerConfig& trackState, Net& siamRPN);
static void trackerInit(Mat img, trackerConfig& trackState, Net& siamRPN, Net& siamKernelR1, Net& siamKernelCL1);

template <typename T> static
T sizeCal(const T& w, const T& h)
{
    T pad = (w + h) * T(0.5);
    T sz2 = (w + pad) * (h + pad);
    return sqrt(sz2);
}

template <>
Mat sizeCal(const Mat& w, const Mat& h)
{
    Mat pad = (w + h) * 0.5;
    Mat sz2 = (w + pad).mul((h + pad));

    cv::sqrt(sz2, sz2);
    return sz2;
}

static
int run(int argc, char** argv)
{
    // Parse command line arguments.
    CommandLineParser parser(argc, argv, keys);

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    std::string inputName = parser.get<String>("input");
    std::string net = parser.get<String>("net");
    std::string kernel_cls1 = parser.get<String>("kernel_cls1");
    std::string kernel_r1 = parser.get<String>("kernel_r1");
    int backend = parser.get<int>("backend");
    int target = parser.get<int>("target");

    // Read nets.
    Net siamRPN, siamKernelCL1, siamKernelR1;
    try
    {
        siamRPN = readNet(samples::findFile(net));
        siamKernelCL1 = readNet(samples::findFile(kernel_cls1));
        siamKernelR1 = readNet(samples::findFile(kernel_r1));
    }
    catch (const cv::Exception& ee)
    {
        std::cerr << "Exception: " << ee.what() << std::endl;
        std::cout << "Can't load the network by using the following files:" << std::endl;
        std::cout << "siamRPN : " << net << std::endl;
        std::cout << "siamKernelCL1 : " << kernel_cls1 << std::endl;
        std::cout << "siamKernelR1 : " << kernel_r1 << std::endl;
        return 2;
    }

    // Set model backend.
    siamRPN.setPreferableBackend(backend);
    siamRPN.setPreferableTarget(target);
    siamKernelR1.setPreferableBackend(backend);
    siamKernelR1.setPreferableTarget(target);
    siamKernelCL1.setPreferableBackend(backend);
    siamKernelCL1.setPreferableTarget(target);

    const std::string winName = "DaSiamRPN";
    namedWindow(winName, WINDOW_AUTOSIZE);

    // Open a video file or an image file or a camera stream.
    VideoCapture cap;

    if (inputName.empty() || (isdigit(inputName[0]) && inputName.size() == 1))
    {
        int c = inputName.empty() ? 0 : inputName[0] - '0';
        std::cout << "Trying to open camera #" << c << " ..." << std::endl;
        if (!cap.open(c))
        {
            std::cout << "Capture from camera #" << c << " didn't work. Specify -i=<video> parameter to read from video file" << std::endl;
            return 2;
        }
    }
    else if (inputName.size())
    {
        inputName = samples::findFileOrKeep(inputName);
        if (!cap.open(inputName))
        {
            std::cout << "Could not open: " << inputName << std::endl;
            return 2;
        }
    }

    // Read the first image.
    Mat image;
    cap >> image;
    if (image.empty())
    {
        std::cerr << "Can't capture frame!" << std::endl;
        return 2;
    }

    Mat image_select = image.clone();
    putText(image_select, "Select initial bounding box you want to track.", Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
    putText(image_select, "And Press the ENTER key.", Point(0, 35), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

    Rect selectRect = selectROI(winName, image_select);
    std::cout << "ROI=" << selectRect << std::endl;

    trackerConfig trackState;
    trackState.update_scoreSize();
    trackState.targetBox = Rect2f(
        float(selectRect.x) + float(selectRect.width) * 0.5f,  // FIXIT don't use center in Rect structures, it is confusing
        float(selectRect.y) + float(selectRect.height) * 0.5f,
        float(selectRect.width),
        float(selectRect.height)
    );

    // Set tracking template.
    trackerInit(image, trackState, siamRPN, siamKernelR1, siamKernelCL1);

    TickMeter tickMeter;

    for (int count = 0; ; ++count)
    {
        cap >> image;
        if (image.empty())
        {
            std::cerr << "Can't capture frame " << count << ". End of video stream?" << std::endl;
            break;
        }

        tickMeter.start();
        float score = trackerEval(image, trackState, siamRPN);
        tickMeter.stop();

        Rect rect = {
            int(trackState.targetBox.x - int(trackState.targetBox.width / 2)),
            int(trackState.targetBox.y - int(trackState.targetBox.height / 2)),
            int(trackState.targetBox.width),
            int(trackState.targetBox.height)
        };
        std::cout << "frame " << count <<
            ": predicted score=" << score <<
            "  rect=" << rect <<
            "  time=" << tickMeter.getTimeMilli() << "ms" <<
            std::endl;

        Mat render_image = image.clone();
        rectangle(render_image, rect, Scalar(0, 255, 0), 2);

        std::string timeLabel = format("Inference time: %.2f ms", tickMeter.getTimeMilli());
        std::string scoreLabel = format("Score: %f", score);
        putText(render_image, timeLabel, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        putText(render_image, scoreLabel, Point(0, 35), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

        imshow(winName, render_image);

        tickMeter.reset();

        int c = waitKey(1);
        if (c == 27 /*ESC*/)
            break;
    }

    std::cout << "Exit" << std::endl;
    return 0;
}

Mat generateHanningWindow(const trackerConfig& trackState)
{
    Mat baseWindows, HanningWindows;

    createHanningWindow(baseWindows, Size(trackState.scoreSize, trackState.scoreSize), CV_32F);
    baseWindows = baseWindows.reshape(0, { 1, trackState.scoreSize, trackState.scoreSize });
    HanningWindows = baseWindows.clone();
    for (int i = 1; i < trackState.anchorNum; i++)
    {
        HanningWindows.push_back(baseWindows);
    }

    return HanningWindows;
}

Mat generateAnchors(trackerConfig& trackState)
{
    int totalStride = trackState.totalStride, scales = trackState.scale, scoreSize = trackState.scoreSize;
    std::vector<float> ratios = trackState.ratios;
    std::vector<Rect2f> baseAnchors;
    int anchorNum = int(ratios.size());
    int size = totalStride * totalStride;

    float ori = -(float(scoreSize / 2)) * float(totalStride);

    for (auto i = 0; i < anchorNum; i++)
    {
        int ws = int(sqrt(size / ratios[i]));
        int hs = int(ws * ratios[i]);

        float wws = float(ws) * scales;
        float hhs = float(hs) * scales;
        Rect2f anchor = { 0, 0, wws, hhs };
        baseAnchors.push_back(anchor);
    }

    int anchorIndex[] = { 0, 0, 0, 0 };
    const int sizes[] = { 4, (int)ratios.size(), scoreSize, scoreSize };
    Mat anchors(4, sizes, CV_32F);

    for (auto i = 0; i < scoreSize; i++)
    {
        for (auto j = 0; j < scoreSize; j++)
        {
            for (auto k = 0; k < anchorNum; k++)
            {
                anchorIndex[0] = 1, anchorIndex[1] = k, anchorIndex[2] = i, anchorIndex[3] = j;
                anchors.at<float>(anchorIndex) = ori + totalStride * i;

                anchorIndex[0] = 0;
                anchors.at<float>(anchorIndex) = ori + totalStride * j;

                anchorIndex[0] = 2;
                anchors.at<float>(anchorIndex) = baseAnchors[k].width;

                anchorIndex[0] = 3;
                anchors.at<float>(anchorIndex) = baseAnchors[k].height;
            }
        }
    }

    return anchors;
}

Mat getSubwindow(Mat& img, const Rect2f& targetBox, float originalSize, Scalar avgChans)
{
    Mat zCrop, dst;
    Size imgSize = img.size();
    float c = (originalSize + 1) / 2;
    float xMin = (float)cvRound(targetBox.x - c);
    float xMax = xMin + originalSize - 1;
    float yMin = (float)cvRound(targetBox.y - c);
    float yMax = yMin + originalSize - 1;

    int leftPad = (int)(fmax(0., -xMin));
    int topPad = (int)(fmax(0., -yMin));
    int rightPad = (int)(fmax(0., xMax - imgSize.width + 1));
    int bottomPad = (int)(fmax(0., yMax - imgSize.height + 1));

    xMin = xMin + leftPad;
    xMax = xMax + leftPad;
    yMax = yMax + topPad;
    yMin = yMin + topPad;

    if (topPad == 0 && bottomPad == 0 && leftPad == 0 && rightPad == 0)
    {
        img(Rect(int(xMin), int(yMin), int(xMax - xMin + 1), int(yMax - yMin + 1))).copyTo(zCrop);
    }
    else
    {
        copyMakeBorder(img, dst, topPad, bottomPad, leftPad, rightPad, BORDER_CONSTANT, avgChans);
        dst(Rect(int(xMin), int(yMin), int(xMax - xMin + 1), int(yMax - yMin + 1))).copyTo(zCrop);
    }

    return zCrop;
}

void softmax(const Mat& src, Mat& dst)
{
    Mat maxVal;
    cv::max(src.row(1), src.row(0), maxVal);

    src.row(1) -= maxVal;
    src.row(0) -= maxVal;

    exp(src, dst);

    Mat sumVal = dst.row(0) + dst.row(1);
    dst.row(0) = dst.row(0) / sumVal;
    dst.row(1) = dst.row(1) / sumVal;
}

void elementMax(Mat& src)
{
    int* p = src.size.p;
    int index[] = { 0, 0, 0, 0 };
    for (int n = 0; n < *p; n++)
    {
        for (int k = 0; k < *(p + 1); k++)
        {
            for (int i = 0; i < *(p + 2); i++)
            {
                for (int j = 0; j < *(p + 3); j++)
                {
                    index[0] = n, index[1] = k, index[2] = i, index[3] = j;
                    float& v = src.at<float>(index);
                    v = fmax(v, 1.0f / v);
                }
            }
        }
    }
}

float trackerEval(Mat img, trackerConfig& trackState, Net& siamRPN)
{
    Rect2f targetBox = trackState.targetBox;

    float wc = targetBox.height + trackState.contextAmount * (targetBox.width + targetBox.height);
    float hc = targetBox.width + trackState.contextAmount * (targetBox.width + targetBox.height);

    float sz = sqrt(wc * hc);
    float scaleZ = trackState.exemplarSize / sz;

    float searchSize = float((trackState.instanceSize - trackState.exemplarSize) / 2);
    float pad = searchSize / scaleZ;
    float sx = sz + 2 * pad;

    Mat xCrop = getSubwindow(img, targetBox, (float)cvRound(sx), trackState.avgChans);

    static Mat blob;
    std::vector<Mat> outs;
    std::vector<String> outNames;
    Mat delta, score;
    Mat sc, rc, penalty, pscore;

    blobFromImage(xCrop, blob, 1.0, Size(trackState.instanceSize, trackState.instanceSize), Scalar(), trackState.swapRB, false, CV_32F);

    siamRPN.setInput(blob);

    outNames = siamRPN.getUnconnectedOutLayersNames();
    siamRPN.forward(outs, outNames);

    delta = outs[0];
    score = outs[1];

    score = score.reshape(0, { 2, trackState.anchorNum, trackState.scoreSize, trackState.scoreSize });
    delta = delta.reshape(0, { 4, trackState.anchorNum, trackState.scoreSize, trackState.scoreSize });

    softmax(score, score);

    targetBox.width *= scaleZ;
    targetBox.height *= scaleZ;

    score = score.row(1);
    score = score.reshape(0, { 5, 19, 19 });

    // Post processing
    delta.row(0) = delta.row(0).mul(trackState.anchors.row(2)) + trackState.anchors.row(0);
    delta.row(1) = delta.row(1).mul(trackState.anchors.row(3)) + trackState.anchors.row(1);
    exp(delta.row(2), delta.row(2));
    delta.row(2) = delta.row(2).mul(trackState.anchors.row(2));
    exp(delta.row(3), delta.row(3));
    delta.row(3) = delta.row(3).mul(trackState.anchors.row(3));

    sc = sizeCal(delta.row(2), delta.row(3)) / sizeCal(targetBox.width, targetBox.height);
    elementMax(sc);

    rc = delta.row(2).mul(1 / delta.row(3));
    rc = (targetBox.width / targetBox.height) / rc;
    elementMax(rc);

    // Calculating the penalty
    exp(((rc.mul(sc) - 1.) * trackState.penaltyK * (-1.0)), penalty);
    penalty = penalty.reshape(0, { trackState.anchorNum, trackState.scoreSize, trackState.scoreSize });

    pscore = penalty.mul(score);
    pscore = pscore * (1.0 - trackState.windowInfluence) + trackState.windows * trackState.windowInfluence;

    int bestID[] = { 0 };
    // Find the index of best score.
    minMaxIdx(pscore.reshape(0, { trackState.anchorNum * trackState.scoreSize * trackState.scoreSize, 1 }), 0, 0, 0, bestID);
    delta = delta.reshape(0, { 4, trackState.anchorNum * trackState.scoreSize * trackState.scoreSize });
    penalty = penalty.reshape(0, { trackState.anchorNum * trackState.scoreSize * trackState.scoreSize, 1 });
    score = score.reshape(0, { trackState.anchorNum * trackState.scoreSize * trackState.scoreSize, 1 });

    int index[] = { 0, bestID[0] };
    Rect2f resBox = { 0, 0, 0, 0 };

    resBox.x = delta.at<float>(index) / scaleZ;
    index[0] = 1;
    resBox.y = delta.at<float>(index) / scaleZ;
    index[0] = 2;
    resBox.width = delta.at<float>(index) / scaleZ;
    index[0] = 3;
    resBox.height = delta.at<float>(index) / scaleZ;

    float lr = penalty.at<float>(bestID) * score.at<float>(bestID) * trackState.lr;

    resBox.x = resBox.x + targetBox.x;
    resBox.y = resBox.y + targetBox.y;
    targetBox.width /= scaleZ;
    targetBox.height /= scaleZ;

    resBox.width = targetBox.width * (1 - lr) + resBox.width * lr;
    resBox.height = targetBox.height * (1 - lr) + resBox.height * lr;

    resBox.x = float(fmax(0., fmin(float(trackState.imgSize.width), resBox.x)));
    resBox.y = float(fmax(0., fmin(float(trackState.imgSize.height), resBox.y)));
    resBox.width = float(fmax(10., fmin(float(trackState.imgSize.width), resBox.width)));
    resBox.height = float(fmax(10., fmin(float(trackState.imgSize.height), resBox.height)));

    trackState.targetBox = resBox;
    return score.at<float>(bestID);
}

void trackerInit(Mat img, trackerConfig& trackState, Net& siamRPN, Net& siamKernelR1, Net& siamKernelCL1)
{
    Rect2f targetBox = trackState.targetBox;
    Mat anchors = generateAnchors(trackState);
    trackState.anchors = anchors;

    Mat windows = generateHanningWindow(trackState);

    trackState.windows = windows;
    trackState.imgSize = img.size();

    trackState.avgChans = mean(img);
    float wc = targetBox.width + trackState.contextAmount * (targetBox.width + targetBox.height);
    float hc = targetBox.height + trackState.contextAmount * (targetBox.width + targetBox.height);
    float sz = (float)cvRound(sqrt(wc * hc));

    Mat zCrop = getSubwindow(img, targetBox, sz, trackState.avgChans);
    static Mat blob;

    blobFromImage(zCrop, blob, 1.0, Size(trackState.exemplarSize, trackState.exemplarSize), Scalar(), trackState.swapRB, false, CV_32F);
    siamRPN.setInput(blob);
    Mat out1;
    siamRPN.forward(out1, "63");

    siamKernelCL1.setInput(out1);
    siamKernelR1.setInput(out1);

    Mat cls1 = siamKernelCL1.forward();
    Mat r1 = siamKernelR1.forward();
    std::vector<int> r1_shape = { 20, 256, 4, 4 }, cls1_shape = { 10, 256, 4, 4 };

    siamRPN.setParam(siamRPN.getLayerId("65"), 0, r1.reshape(0, r1_shape));
    siamRPN.setParam(siamRPN.getLayerId("68"), 0, cls1.reshape(0, cls1_shape));
}

int main(int argc, char **argv)
{
    try
    {
        return run(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::cerr << "FATAL: C++ exception: " << e.what() << std::endl;
        return 1;
    }
}
