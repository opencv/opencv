// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

/*
Auto white balance using FC4: https://github.com/yuanming-hu/fc4

Color constancy is a method to make colors of objects render correctly on a photo.
White balance aims to make white objects appear white on an image and not a shade of any
other color, independent of the actual light setting. White balance correction creates
a neutral looking coloring of the objects, and generally makes colors look more similar
to their 'true' colors under different light conditions.

Given an RGB image, the FC4 model predicts scene illuminant (R,G,B). We then apply
the illuminant to the image, applying the correction in the linear RGB space.
The transformation between linear and sRGB spaces is done as described in the sRGB standard,
which is a nonlinear Gamma correction with exponent 2.4 and extra handling of very small values.
This sample is written for 8bit images. The FC4 model accepts RGB images with applied Gamma scaling.

The training of the FC4 model was done on the Gehler-Shi dataset. The dataset includes
568 images and ground truth corrections, as well as ground truth illuminants. The linear
RGB images from the dataset were used with Gamma correction of 2.2 applied.

The model is a pretrained fold 0 of a training pipeline on the Gehler-Shi dataset, from the PyTorch
implementation of the FC4 algorithm by Mateo Rizzo. The model was converted from a .pth file to onnx
using torch.onnx.export. The model can be downloaded in the following link:
https://raw.githubusercontent.com/MykhailoTrushch/opencv/d6ab21353a87e4c527e38e464384c7ee78e96e22/samples/dnn/models/fc4_fold_0.onnx

Copyright (c) 2017 Yuanming Hu, Baoyuan Wang, Stephen Lin
Copyright (c) 2021 Matteo Rizzo

Licensed under the MIT license.

References:

Yuanming Hu, Baoyuan Wang, and Stephen Lin. “FC⁴: Fully Convolutional Color
Constancy with Confidence-Weighted Pooling.” CVPR, 2017, pp. 4085–4094.

Implementations of FC4:
https://github.com/yuanming-hu/fc4/
https://github.com/matteo-rizzo/fc4-pytorch

Lilong Shi and Brian Funt, "Re-processed Version of the Gehler Color Constancy Dataset of 568 Images,"
accessed from http://www.cs.sfu.ca/~colour/data/

“IEC 61966-2-1:1999 – Multimedia Systems and Equipment – Colour Measurement and Management – Part 2-1: Colour Management – Default RGB Colour Space – sRGB.” IEC Standard, 1999.
*/

#include <iostream>
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
    "{ input i         |  castle.png       | Path to input image }";
;

const string backend_keys =
    format("{ backend | default | Choose one of computation backends: "
           "default: automatically (by default), "
           "openvino: Intel's Deep Learning Inference Engine "
           "(https://software.intel.com/openvino-toolkit), "
           "opencv: OpenCV implementation, "
           "vkcom: VKCOM, "
           "cuda: CUDA, "
           "webnn: WebNN }");

const string target_keys =
    format("{ target | cpu | Choose one of target computation devices: "
           "cpu: CPU target (by default), "
           "opencl: OpenCL, "
           "opencl_fp16: OpenCL fp16 (half-float precision), "
           "vpu: VPU, "
           "vulkan: Vulkan, "
           "cuda: CUDA, "
           "cuda_fp16: CUDA fp16 (half-float preprocess) }");

// Normalization constant for 8bit values
const float NORMALIZE_FACTOR = 1.0f / 255.0f;

// sRGB to linear conversion constants (or vice versa):
// SRGB_THRESHOLD / LINEAR_THRESHOLD: breakpoints between linear and gamma regions
// SRGB_SLOPE: slope of the linear segment near black
// SRGB_ALPHA: offset to ensure continuity at the threshold
// SRGB_EXP: gamma exponent
const float SRGB_THRESHOLD = 0.04045f;
const float SRGB_ALPHA = 0.055f;
const float SRGB_SLOPE = 12.92f;
const float SRGB_EXP = 2.4f;
const float LINEAR_THRESHOLD = 0.0031308f;
const float EPS = 1e-10f;

static Mat srgbToLinear(const Mat &srgb32f) {
    CV_Assert(srgb32f.type() == CV_32FC3);
    const float a = SRGB_ALPHA;

    Mat y = srgb32f;
    Mat mask_low;
    compare(y, SRGB_THRESHOLD, mask_low, CMP_LE);

    Mat low = y / SRGB_SLOPE;

    Mat t = (y + a) / (1.0f + a);
    Mat high;
    pow(t, SRGB_EXP, high);

    Mat lin(y.size(), y.type(), Scalar(0, 0, 0));
    low.copyTo(lin, mask_low);
    Mat mask_high;
    bitwise_not(mask_low, mask_high);
    high.copyTo(lin, mask_high);

    return lin;
}

static Mat linearToSrgb(const Mat &lin32f) {
    CV_Assert(lin32f.type() == CV_32FC3);
    const float a = SRGB_ALPHA;

    Mat x = lin32f;
    Mat mask_low;
    compare(x, LINEAR_THRESHOLD, mask_low, CMP_LE);
    Mat low = x * SRGB_SLOPE;
    Mat powPart;
    pow(x, 1.0 / SRGB_EXP, powPart);
    Mat high = (1.0f + a) * powPart - a;

    Mat srgb(x.size(), x.type(), Scalar(0, 0, 0));
    low.copyTo(srgb, mask_low);
    Mat mask_high;
    bitwise_not(mask_low, mask_high);
    high.copyTo(srgb, mask_high);

    return srgb;
}

static Mat correct(const Mat &bgr8u, const Vec3f &illumRGB_linear) {
    Mat f32;
    bgr8u.convertTo(f32, CV_32F, NORMALIZE_FACTOR);

    Mat lin = srgbToLinear(f32);

    const float eR = std::max(illumRGB_linear[0], EPS);
    const float eG = std::max(illumRGB_linear[1], EPS);
    const float eB = std::max(illumRGB_linear[2], EPS);

    float s3 = std::sqrt(3.0f);
    Scalar corr(eB * s3 + EPS, eG * s3 + EPS, eR * s3 + EPS);

    Mat corrected;
    divide(lin, corr, corrected);

    std::vector<Mat> ch;
    split(corrected, ch);
    double m0, m1, m2;
    minMaxLoc(ch[0], nullptr, &m0);
    minMaxLoc(ch[1], nullptr, &m1);
    minMaxLoc(ch[2], nullptr, &m2);
    float maxVal = static_cast<float>(std::max({m0, m1, m2})) + EPS;
    corrected /= maxVal;
    min(corrected, 1.0, corrected);
    max(corrected, 0.0, corrected);

    Mat srgb = linearToSrgb(corrected);

    Mat out;
    srgb.convertTo(out, CV_8U, 255.0);
    return out;
}

static void annotate(Mat &img, const string &title) {
    double fs = std::max(0.5, std::min(img.cols, img.rows) / 800.0);
    int th = std::max(1, (int)std::round(fs * 2));
    putText(img, title, Point(10, 30), FONT_HERSHEY_SIMPLEX, fs,
            Scalar(0, 255, 0), th);
}

int main(int argc, char **argv) {
    const string about = "FC4 Color Constancy (ONNX) sample.\n"
                         "Predicts scene illuminant and corrects the white "
                         "balance of the image.\n";

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
    Mat img = imread(inputPath, IMREAD_COLOR);
    if (img.empty()) {
        cerr << "Cannot load image: " << inputPath << endl;
        return -1;
    }
    Mat blob;
    blob = blobFromImage(img, scale, img.size(), mean, swapRB, /*crop=*/false,
                         /*type=*/CV_32F);
    net.setInput(blob);

    Mat out;
    try {
        out = net.forward();
    } catch (const Exception &e) {
        cerr << "Forward error: " << e.what() << endl;
        return -1;
    }

    const float *p = out.ptr<float>(0);
    CV_Assert(out.total() == 3);
    Vec3f illum = Vec3f(p[0], p[1], p[2]);

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
