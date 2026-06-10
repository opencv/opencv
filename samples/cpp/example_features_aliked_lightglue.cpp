// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// ALIKED + LightGlueMatcher usage example
// Demonstrates feature detection, extraction, and matching using ALIKED and LightGlue.

#include <opencv2/features.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    // ---- Parse arguments ----
    String alikedModel, lightglueModel, imgPath1, imgPath2;

    if (argc >= 5)
    {
        imgPath1 = argv[1];
        imgPath2 = argv[2];
        alikedModel = argv[3];
        lightglueModel = argv[4];
    }
else
    {
        cout << "Usage: " << argv[0] << " <image1> <image2> <aliked_model> <lightglue_model>" << endl;
        cout << endl;
        cout << "Example:" << endl;
        cout << "  " << argv[0] << " img1.jpg img2.jpg aliked-n16rot-top1k-640.onnx aliked_lightglue.onnx" << endl;
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
