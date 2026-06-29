// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Learned feature usage examples.
// Demonstrates ALIKED + LightGlue matching and XFeat feature extraction.

#include <opencv2/features.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>

using namespace cv;
using namespace std;

namespace
{
static Mat toGray(const Mat& image)
{
    if (image.channels() == 1)
        return image;

    Mat gray;
    if (image.channels() == 3)
        cvtColor(image, gray, COLOR_BGR2GRAY);
    else if (image.channels() == 4)
        cvtColor(image, gray, COLOR_BGRA2GRAY);
    else
        CV_Error(Error::StsBadArg, "XFeat expects grayscale, BGR, or BGRA image");
    return gray;
}

static Mat toNCHW(const Mat& blob, int channelsHint = -1)
{
    CV_Assert(blob.dims == 4);
    if (channelsHint > 0 && blob.size[1] == channelsHint)
        return blob;

    const bool isNHWC = (channelsHint > 0) ? (blob.size[3] == channelsHint) : (blob.size[3] < blob.size[1]);
    if (!isNHWC)
        return blob;

    const int H = blob.size[1], W = blob.size[2], C = blob.size[3];
    int outSize[] = {1, C, H, W};
    Mat out(4, outSize, CV_32F);
    const float* src = blob.ptr<float>();
    float* dst = out.ptr<float>();
    for (int c = 0; c < C; ++c)
    {
        for (int y = 0; y < H; ++y)
        {
            for (int x = 0; x < W; ++x)
            {
                const int srcIdx = (y * W + x) * C + c;
                const int dstIdx = c * H * W + y * W + x;
                dst[dstIdx] = src[srcIdx];
            }
        }
    }
    return out;
}

static float sampleNearest(const Mat& map, float x, float y, int normW, int normH)
{
    CV_Assert(map.type() == CV_32F);
    const int H = map.rows;
    const int W = map.cols;
    if (H <= 0 || W <= 0)
        return 0.0f;
    if (normW <= 1 || normH <= 1)
        return map.at<float>(0, 0);

    const float fx = x * static_cast<float>(W - 1) / static_cast<float>(normW - 1);
    const float fy = y * static_cast<float>(H - 1) / static_cast<float>(normH - 1);
    const int ix = std::max(0, std::min(W - 1, cvRound(fx)));
    const int iy = std::max(0, std::min(H - 1, cvRound(fy)));
    return map.at<float>(iy, ix);
}

static float sampleBilinear(const Mat& map, float x, float y, int normW, int normH)
{
    CV_Assert(map.type() == CV_32F);
    const int H = map.rows;
    const int W = map.cols;
    if (H <= 0 || W <= 0)
        return 0.0f;
    if (W == 1 && H == 1)
        return map.at<float>(0, 0);
    if (normW <= 1 || normH <= 1)
        return map.at<float>(0, 0);

    float fx = x * static_cast<float>(W - 1) / static_cast<float>(normW - 1);
    float fy = y * static_cast<float>(H - 1) / static_cast<float>(normH - 1);
    fx = std::max(0.0f, std::min(fx, static_cast<float>(W - 1)));
    fy = std::max(0.0f, std::min(fy, static_cast<float>(H - 1)));

    const int x0 = cvFloor(fx);
    const int y0 = cvFloor(fy);
    const int x1 = std::min(x0 + 1, W - 1);
    const int y1 = std::min(y0 + 1, H - 1);
    const float dx = fx - x0;
    const float dy = fy - y0;

    const float v00 = map.at<float>(y0, x0);
    const float v01 = map.at<float>(y0, x1);
    const float v10 = map.at<float>(y1, x0);
    const float v11 = map.at<float>(y1, x1);
    return (1.f - dx) * (1.f - dy) * v00 +
           dx * (1.f - dy) * v01 +
           (1.f - dx) * dy * v10 +
           dx * dy * v11;
}

static bool detectAndComputeXFeatPostprocess(const Mat& image, const Mat& mask, dnn::Net& net,
                                             vector<KeyPoint>& keypoints, Mat& descriptors,
                                             int maxKeypoints = 2000,
                                             float detectionThreshold = 0.05f,
                                             int inputSize = 640)
{
    keypoints.clear();
    descriptors.release();
    if (image.empty())
        return false;

    Mat gray = toGray(image);
    const float scale = static_cast<float>(inputSize) /
                        static_cast<float>(std::max(gray.cols, gray.rows));
    const int resizedW = std::max(1, cvRound(gray.cols * scale));
    const int resizedH = std::max(1, cvRound(gray.rows * scale));
    Mat resized;
    resize(gray, resized, Size(resizedW, resizedH), 0, 0, INTER_LINEAR_EXACT);

    Mat padded = Mat::zeros(inputSize, inputSize, CV_8UC1);
    resized.copyTo(padded(Rect(0, 0, resizedW, resizedH)));

    Mat blob;
    dnn::blobFromImage(padded, blob, 1.0 / 255.0, Size(inputSize, inputSize), Scalar(), false, false, CV_32F);
    net.setInput(blob);
    vector<Mat> outs;
    net.forward(outs, {"output_feats", "output_keypoints", "output_heatmap"});
    CV_Assert(outs.size() == 3);

    Mat featBlob = toNCHW(outs[0], 64);
    Mat kptBlob = toNCHW(outs[1], 65);
    Mat relBlob = toNCHW(outs[2], 1);
    CV_Assert(featBlob.dims == 4 && kptBlob.dims == 4 && relBlob.dims == 4);

    const int featC = featBlob.size[1];
    const int featH = featBlob.size[2];
    const int featW = featBlob.size[3];
    const int kptC = kptBlob.size[1];
    const int kptH = kptBlob.size[2];
    const int kptW = kptBlob.size[3];
    CV_Assert(featC == 64);
    CV_Assert(kptC >= 64);

    Mat reliability(relBlob.size[2], relBlob.size[3], CV_32F, relBlob.ptr<float>());
    Mat heatmap = Mat::zeros(kptH * 8, kptW * 8, CV_32F);
    const float* kptPtr = kptBlob.ptr<float>();
    const int kptHW = kptH * kptW;

    for (int y = 0; y < kptH; ++y)
    {
        for (int x = 0; x < kptW; ++x)
        {
            const int offset = y * kptW + x;
            float maxLogit = -FLT_MAX;
            for (int c = 0; c < kptC; ++c)
                maxLogit = std::max(maxLogit, kptPtr[c * kptHW + offset]);

            float sumExp = 0.f;
            float probs[64];
            for (int c = 0; c < 64; ++c)
            {
                probs[c] = std::exp(kptPtr[c * kptHW + offset] - maxLogit);
                sumExp += probs[c];
            }
            if (kptC > 64)
                sumExp += std::exp(kptPtr[64 * kptHW + offset] - maxLogit);
            if (sumExp <= 0.f)
                continue;

            for (int c = 0; c < 64; ++c)
            {
                const int dy = c / 8;
                const int dx = c % 8;
                heatmap.at<float>(y * 8 + dy, x * 8 + dx) = probs[c] / sumExp;
            }
        }
    }

    Mat localMax;
    dilate(heatmap, localMax, getStructuringElement(MORPH_RECT, Size(5, 5)));

    struct Candidate
    {
        Point2f ptPadded;
        float score;
    };
    vector<Candidate> candidates;
    candidates.reserve(4096);

    for (int y = 0; y < heatmap.rows; ++y)
    {
        const float* hm = heatmap.ptr<float>(y);
        const float* mx = localMax.ptr<float>(y);
        for (int x = 0; x < heatmap.cols; ++x)
        {
            const float h = hm[x];
            if (h <= detectionThreshold || h != mx[x])
                continue;
            const float xf = static_cast<float>(x);
            const float yf = static_cast<float>(y);
            const float score = sampleNearest(heatmap, xf, yf, inputSize, inputSize) *
                                sampleBilinear(reliability, xf, yf, inputSize, inputSize);
            if (score <= 0.f)
                continue;
            candidates.push_back({Point2f(xf, yf), score});
        }
    }

    if (candidates.empty())
        return true;

    const int keep = maxKeypoints > 0 ? std::min(maxKeypoints, static_cast<int>(candidates.size()))
                                      : static_cast<int>(candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + keep, candidates.end(),
                      [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
    candidates.resize(keep);

    descriptors.create(keep, 64, CV_32F);
    keypoints.reserve(keep);
    const float* featPtr = featBlob.ptr<float>();
    const int featHW = featH * featW;

    for (int i = 0; i < keep; ++i)
    {
        const float xp = candidates[i].ptPadded.x;
        const float yp = candidates[i].ptPadded.y;
        const float xOrig = xp / scale;
        const float yOrig = yp / scale;
        const int ix = cvFloor(xOrig);
        const int iy = cvFloor(yOrig);
        if (ix < 0 || iy < 0 || ix >= image.cols || iy >= image.rows)
            continue;
        if (!mask.empty() && mask.at<uchar>(iy, ix) == 0)
            continue;

        float* dst = descriptors.ptr<float>(static_cast<int>(keypoints.size()));
        for (int c = 0; c < 64; ++c)
        {
            Mat channel(featH, featW, CV_32F, const_cast<float*>(featPtr + c * featHW));
            dst[c] = sampleBilinear(channel, xp, yp, inputSize, inputSize);
        }
        Mat row(1, 64, CV_32F, dst);
        normalize(row, row, 1.0, 0.0, NORM_L2);
        keypoints.emplace_back(Point2f(xOrig, yOrig), 1.0f, -1.0f, candidates[i].score);
    }

    descriptors = descriptors.rowRange(0, static_cast<int>(keypoints.size())).clone();
    return true;
}
} // namespace

static int runXFeatExample(const String& imgPath, const String& xfeatModel, const String& outputPath)
{
    Mat img = imread(imgPath);
    if (img.empty())
    {
        cerr << "Error: cannot load image: " << imgPath << endl;
        return -1;
    }

    dnn::Net net = dnn::readNetFromONNX(xfeatModel);

    vector<KeyPoint> keypoints;
    Mat descriptors;
    detectAndComputeXFeatPostprocess(img, noArray(), net, keypoints, descriptors,
                                     /*maxKeypoints=*/2000,
                                     /*detectionThreshold=*/0.05f,
                                     /*inputSize=*/640);

    cout << "XFeat descriptor size: " << 64 << endl;
    cout << "XFeat descriptor type: " << CV_32F << endl;
    cout << "XFeat default norm: " << NORM_L2 << endl;
    cout << "Image: " << keypoints.size() << " keypoints, descriptors "
         << descriptors.rows << "x" << descriptors.cols << endl;

    Mat canvas;
    drawKeypoints(img, keypoints, canvas, Scalar(0, 255, 0),
                  DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    if (!outputPath.empty())
    {
        imwrite(outputPath, canvas);
        cout << "Saved XFeat keypoint visualization to: " << outputPath << endl;
    }

    imshow("XFeat Keypoints", canvas);
    cout << "Press any key to exit..." << endl;
    waitKey(0);
    return 0;
}

int main(int argc, char** argv)
{
    // ---- Parse arguments ----
    String alikedModel, lightglueModel, imgPath1, imgPath2;

    if (argc >= 4 && String(argv[1]) == "--xfeat")
    {
        const String outputPath = argc >= 5 ? argv[4] : String();
        return runXFeatExample(argv[2], argv[3], outputPath);
    }

    if (argc >= 5)
    {
        imgPath1 = argv[1];
        imgPath2 = argv[2];
        alikedModel = argv[3];
        lightglueModel = argv[4];
    }
    else
    {
        cout << "Usage:" << endl;
        cout << "  " << argv[0] << " <image1> <image2> <aliked_model> <lightglue_model>" << endl;
        cout << "  " << argv[0] << " --xfeat <image> <xfeat_model> [output_image]" << endl;
        cout << endl;
        cout << "Examples:" << endl;
        cout << "  " << argv[0] << " img1.jpg img2.jpg aliked-n16rot-top1k-640.onnx aliked_lightglue.onnx" << endl;
        cout << "  " << argv[0] << " --xfeat img.jpg xfeat.onnx xfeat_keypoints.jpg" << endl;
        return 0;
    }

    // ---- Load images ----
    Mat img1 = imread(imgPath1);
    Mat img2 = imread(imgPath2);
    if (img1.empty() || img2.empty())
    {
        cerr << "Error: cannot load images." << endl;
        return -1;
    }

    // ================================================================
    //  1. Create ALIKED feature extractor
    // ================================================================
    // Method A: From ONNX model file
    Ptr<ALIKED> aliked = ALIKED::create(alikedModel);

    // Method B: Customize parameters
    // ALIKED::Params params;
    // params.inputSize = Size(640, 640);       // Network input resolution
    // params.normalizeDescriptors = true;       // L2-normalize descriptors
    // Ptr<ALIKED> aliked = ALIKED::create(alikedModel, params);

    // Method C: From in-memory model data
    // vector<uchar> modelData = readFile(alikedModel);
    // Ptr<ALIKED> aliked = ALIKED::create(modelData);

    cout << "Descriptor size: " << aliked->descriptorSize() << endl;  // 128
    cout << "Descriptor type: " << aliked->descriptorType() << endl;  // CV_32F
    cout << "Default norm: " << aliked->defaultNorm() << endl;        // NORM_L2

    // ================================================================
    //  2. Detect keypoints and compute descriptors
    // ================================================================
    vector<KeyPoint> kpts1, kpts2;
    Mat descs1, descs2;

    // Method A: detect + compute in one call (recommended)
    aliked->detectAndCompute(img1, Mat(), kpts1, descs1);
    aliked->detectAndCompute(img2, Mat(), kpts2, descs2);

    // Method B: detect only (no descriptors)
    // vector<KeyPoint> kpts;
    // aliked->detect(img, kpts);

    // Method C: compute only (from existing keypoints)
    // Mat descs;
    // aliked->compute(img, kpts, descs);

    cout << "Image 1: " << kpts1.size() << " keypoints, descriptors " << descs1.rows << "x" << descs1.cols << endl;
    cout << "Image 2: " << kpts2.size() << " keypoints, descriptors " << descs2.rows << "x" << descs2.cols << endl;

    // ================================================================
    //  3. Create LightGlueMatcher
    // ================================================================
    // Method A: From ONNX model file
    Ptr<LightGlueMatcher> lg = LightGlueMatcher::create(lightglueModel);

    // Method B: Customize parameters
    // LightGlueMatcher::Params lgParams;
    // lgParams.scoreThreshold = 0.1f;    // Filter low-confidence matches
    // lgParams.disableWinograd = false;   // Keep Winograd convolution
    // Ptr<LightGlueMatcher> lg = LightGlueMatcher::create(lightglueModel, lgParams);

    // Method C: From in-memory model data
    // vector<uchar> lgData = readFile(lightglueModel);
    // Ptr<LightGlueMatcher> lg = LightGlueMatcher::create(lgData);

    // ================================================================
    //  4. Set keypoint context for LightGlue
    // ================================================================
    // LightGlue needs keypoint coordinates + image sizes for spatial reasoning.
    // Build Nx2 float matrices with pixel coordinates.

    Mat kpts1Mat((int)kpts1.size(), 2, CV_32F);
    Mat kpts2Mat((int)kpts2.size(), 2, CV_32F);
    for (size_t i = 0; i < kpts1.size(); i++)
    {
        kpts1Mat.at<float>((int)i, 0) = kpts1[i].pt.x;
        kpts1Mat.at<float>((int)i, 1) = kpts1[i].pt.y;
    }
    for (size_t i = 0; i < kpts2.size(); i++)
    {
        kpts2Mat.at<float>((int)i, 0) = kpts2[i].pt.x;
        kpts2Mat.at<float>((int)i, 1) = kpts2[i].pt.y;
    }

    // setPairInfo must be called before match()/knnMatch()
    lg->setPairInfo(kpts1Mat, kpts2Mat, img1.size(), img2.size());

    // ================================================================
    //  5. Match descriptors
    // ================================================================

    // Method A: 1-to-1 matching (returns best match per query keypoint)
    vector<DMatch> matches;
    lg->match(descs1, descs2, matches);

    cout << "1-to-1 matches: " << matches.size() << endl;

    // Method B: kNN matching (k=1 only for LightGlue)
    // vector<vector<DMatch>> knnMatches;
    // lg->knnMatch(descs1, descs2, knnMatches, 1);
    // // knnMatches[i] contains matches for query keypoint i

    // ================================================================
    //  6. Filter matches by confidence (optional)
    // ================================================================
    // DMatch distance = 1.0 - confidence_score
    // Lower distance = better match

    vector<DMatch> goodMatches;
    float distanceThreshold = 0.9f;  // confidence > 0.1
    for (const auto& m : matches)
    {
        if (m.distance < distanceThreshold)
            goodMatches.push_back(m);
    }
    cout << "Good matches (distance < " << distanceThreshold << "): " << goodMatches.size() << endl;

    // ================================================================
    //  7. Visualize results
    // ================================================================
    Mat canvas;
    cv::drawMatches(img1, kpts1, img2, kpts2, goodMatches, canvas,
                    Scalar::all(-1), Scalar::all(-1), vector<char>(),
                    DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    imshow("ALIKED + LightGlue Matches", canvas);
    cout << "Press any key to exit..." << endl;
    waitKey(0);

    return 0;
}
