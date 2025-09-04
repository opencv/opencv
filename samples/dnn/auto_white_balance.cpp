// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

/*
Auto white balance using FC4: https://github.com/yuanming-hu/fc4

Given an image, the FC4 model predicts scene illuminant (R,G,B). We then apply
the illuminant to the image, applying the correction in the linear RGB space.

Yuanming Hu, Baoyuan Wang, and Stephen Lin. “FC⁴: Fully Convolutional Color
Constancy with Confidence-Weighted Pooling.” CVPR, 2017, pp. 4085–4094.
*/

#include <iostream>
#include <numeric>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "common.hpp"

using namespace cv;
using namespace cv::dnn;
using namespace std;

const string param_keys =
    "{ help h          |                   | Print help message }"
    "{ @alias          | fc4               | Model alias from models.yml "
    "(optional) }"
    "{ zoo             | ../dnn/models.yml | Path to models.yml file "
    "(optional) }"
    "{ input i         |  chicky_512.png   | Path to input image }"
    "{ model           |                   | Path to ONNX model file }";

const string backend_keys =
    format("{ backend          | default | Choose backend: "
           "default: auto, openvino, opencv, vkcom, cuda, webnn }");

const string target_keys =
    format("{ target           | cpu | Choose target: "
           "cpu, opencl, opencl_fp16, vpu, vulkan, cuda, cuda_fp16 }");

static cv::Vec3f extractIlluminant(const cv::Mat &out) {
    CV_Assert(out.total() >= 3);

    cv::Mat f32;
    if (out.depth() == CV_32F)
        f32 = out;
    else
        out.convertTo(f32, CV_32F);

    cv::Mat flat = f32.reshape(1, 1);
    const float *p = flat.ptr<float>(0);

    return cv::Vec3f(p[0], p[1], p[2]);
}

static Mat srgbToLinear(const Mat &srgb32f) {
    CV_Assert(srgb32f.type() == CV_32FC3);
    const float a = 0.055f;

    Mat y = srgb32f;
    Mat mask_low;
    compare(y, 0.04045f, mask_low, CMP_LE);

    Mat low = y / 12.92f;

    Mat t = (y + a) / (1.0f + a);
    Mat high;
    pow(t, 2.4, high);

    Mat lin(y.size(), y.type(), Scalar(0, 0, 0));
    low.copyTo(lin, mask_low);
    Mat mask_high;
    bitwise_not(mask_low, mask_high);
    high.copyTo(lin, mask_high);

    return lin;
}

static Mat linearToSrgb(const Mat &lin32f) {
    CV_Assert(lin32f.type() == CV_32FC3);
    const float a = 0.055f;

    Mat x = lin32f;
    Mat mask_low;
    compare(x, 0.0031308f, mask_low, CMP_LE);
    Mat low = x * 12.92f;
    Mat powPart;
    pow(x, 1.0 / 2.4, powPart);
    Mat high = (1.0f + a) * powPart - a;

    Mat srgb(x.size(), x.type(), Scalar(0, 0, 0));
    low.copyTo(srgb, mask_low);
    Mat mask_high;
    bitwise_not(mask_low, mask_high);
    high.copyTo(srgb, mask_high);

    return srgb;
}

static Mat correct(const Mat &bgr8u, const Vec3f &illumRGB_linear) {
    CV_Assert(bgr8u.type() == CV_8UC3);

    Mat f32;
    bgr8u.convertTo(f32, CV_32F, 1.0 / 255.0);
    Mat rgb;
    cvtColor(f32, rgb, COLOR_BGR2RGB);
    Mat lin = srgbToLinear(rgb);

    const float s3 = std::sqrt(3.0f);
    Scalar corr(illumRGB_linear[0] * s3 + 1e-10f,
                illumRGB_linear[1] * s3 + 1e-10f,
                illumRGB_linear[2] * s3 + 1e-10f);
    Mat corrected;
    divide(lin, corr, corrected);

    std::vector<Mat> ch;
    split(corrected, ch);
    double m0, m1, m2;
    minMaxLoc(ch[0], nullptr, &m0);
    minMaxLoc(ch[1], nullptr, &m1);
    minMaxLoc(ch[2], nullptr, &m2);
    float maxVal = static_cast<float>(std::max({m0, m1, m2})) + 1e-10f;
    Mat normalized = corrected / maxVal;

    Mat srgb = linearToSrgb(normalized);
    cv::min(srgb, 1.0, srgb);
    cv::max(srgb, 0.0, srgb);

    Mat rgb8u, bgrOut;
    srgb.convertTo(rgb8u, CV_8U, 255.0);
    cvtColor(rgb8u, bgrOut, COLOR_RGB2BGR);
    return bgrOut;
}

static void annotate(Mat &img, const string &title) {
    double fs = std::max(0.5, std::min(img.cols, img.rows) / 800.0);
    int th = std::max(1, (int)std::round(fs * 2));
    putText(img, title, Point(10, 30), FONT_HERSHEY_SIMPLEX, fs,
            Scalar(0, 255, 0), th);
}

int main(int argc, char **argv) {
    const string about =
        "FC4 Color Constancy (ONNX) sample.\n"
        "Predicts scene illuminant and corrects the white "
        "balance of the image.\n\n"
        "Example:\n"
        "\t./auto_white_balance --model=path/to/fc4_fold_0.onnx "
        "--input=image.jpg\n";

    string keys = param_keys + backend_keys + target_keys;

    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        cout << about << endl;
        parser.printMessage();
        return 0;
    }

    string modelName = parser.get<String>("@alias");
    string zooFile = samples::findFile(parser.get<String>("zoo"));
    keys += genPreprocArguments(modelName, zooFile);
    parser = CommandLineParser(argc, argv, keys);

    float scale = parser.get<float>("scale");
    Scalar mean = parser.get<Scalar>("mean");
    bool swapRB = parser.get<bool>("rgb");
    String backend = parser.get<String>("backend");
    String target = parser.get<String>("target");
    String sha1 = parser.get<String>("sha1");
    string model = findModel(parser.get<String>("model"), sha1);
    string inputPath = findFile(parser.get<String>("input"));

    if (model.empty()) {
        cerr << "Model file not found\n";
        return -1;
    }

    Net net;
    try {
        net = readNetFromONNX(model);
        net.setPreferableBackend(getBackendID(backend));
        net.setPreferableTarget(getTargetID(target));
    } catch (const Exception &e) {
        cerr << "Error loading model: " << e.what() << endl;
        return -1;
    }
    Mat img = imread(inputPath);
    if (img.empty()) {
        cerr << "Cannot load image: " << inputPath << endl;
        return -1;
    }

    Mat blob =
        blobFromImage(img, scale, img.size(), mean, swapRB, false, CV_32F);
    net.setInput(blob);

    Mat out;
    try {
        out = net.forward();
    } catch (const Exception &e) {
        cerr << "Forward error: " << e.what() << endl;
        return -1;
    }

    Vec3f illum = extractIlluminant(out);
    Mat corrected = correct(img, illum);

    Mat origVis = img.clone();
    Mat corrVis = corrected.clone();
    annotate(origVis, "Original");
    annotate(corrVis, "FC4-corrected");

    Mat stacked;
    hconcat(origVis, corrVis, stacked);
    imshow("Original and Corrected Images", stacked);
    waitKey(0);
    destroyAllWindows();
    return 0;
}
