// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Learned feature usage examples.
// Demonstrates ALIKED/DISK + LightGlueMatcher matching and XFeat feature extraction.

#include <opencv2/features.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include "../dnn/common.hpp"

#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

const string about =
    "This sample demonstrates feature detection, extraction, and matching using ALIKED or DISK with LightGlueMatcher.\n"
    "It also demonstrates XFeat keypoint extraction on a single image.\n\n"
    "To run with ALIKED:\n"
    "\t ./example_cpp_features_lightglue <image1> <image2> aliked <aliked_model> <aliked_lightglue_model> [output] [--backend=<backend>]\n"
    "To run with DISK:\n"
    "\t ./example_cpp_features_lightglue <image1> <image2> disk <disk_model> <disk_lightglue_model> [output] [--backend=<backend>]\n"
    "To run the XFeat keypoint demo:\n"
    "\t ./example_cpp_features_lightglue <image> --xfeat_model=<xfeat_model> [--xfeat_output=<output>] [--backend=<backend>]\n";

const string param_keys =
    "{ help h           |         | Print help message. }"
    "{ @image1          |         | Path to the first input image. }"
    "{ @image2          |         | Path to the second input image. }"
    "{ @feature         |         | Feature type: aliked or disk. }"
    "{ @feature_model   |         | Path to the ALIKED or DISK model. }"
    "{ @lightglue_model |         | Path to the corresponding LightGlue model. }"
    "{ @output          |         | Optional path to save the match visualization. }"
    "{ xfeat_model      |         | Path to the XFeat model. When set, runs the XFeat keypoint "
                                   "demo on the first image instead of LightGlue matching. }"
    "{ xfeat_output     |         | Optional path to save the XFeat keypoint visualization. }";

const string backend_keys = format(
    "{ backend          | default | Choose one of computation backends: "
                              "default: automatically (by default), "
                              "openvino: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                              "opencv: OpenCV implementation, "
                              "vkcom: VKCOM, "
                              "cuda: CUDA, "
                              "webnn: WebNN }");

const string keys = param_keys + backend_keys;

static Mat keypointsToMat(const vector<KeyPoint>& keypoints)
{
    Mat points((int)keypoints.size(), 2, CV_32F);
    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        points.at<float>((int)i, 0) = keypoints[i].pt.x;
        points.at<float>((int)i, 1) = keypoints[i].pt.y;
    }
    return points;
}

// XFeat extracts keypoints and descriptors on its own; it does not use LightGlue.
static int runXFeatDemo(const String& imagePath, const String& xfeatModel, const String& outputPath,
                        int backendId, int targetId)
{
    Mat image = imread(imagePath);
    if (image.empty())
    {
        cerr << "Error: cannot load input image: " << imagePath << endl;
        return 1;
    }

    Ptr<XFeat> xfeat = XFeat::create(xfeatModel, 2000, 0.05f, Size(640, 640), backendId, targetId);

    vector<KeyPoint> keypoints;
    Mat descriptors;
    xfeat->detectAndCompute(image, noArray(), keypoints, descriptors);

    cout << "XFeat: " << keypoints.size() << " keypoints, descriptors "
         << descriptors.rows << "x" << descriptors.cols << endl;

    Mat visualization;
    drawKeypoints(image, keypoints, visualization, Scalar(0, 255, 0),
                  DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    if (!outputPath.empty())
    {
        if (!imwrite(outputPath, visualization))
        {
            cerr << "Error: cannot write output image: " << outputPath << endl;
            return 1;
        }
        cout << "Output written to " << outputPath << endl;
    }

    imshow("XFeat Keypoints", visualization);
    cout << "Press any key to exit..." << endl;
    waitKey(0);

    return 0;
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    const String imagePath1 = parser.get<String>("@image1");
    const String imagePath2 = parser.get<String>("@image2");
    const String featureType = parser.get<String>("@feature");
    const String featureModel = parser.get<String>("@feature_model");
    const String lightglueModel = parser.get<String>("@lightglue_model");
    const String outputPath = parser.get<String>("@output");
    const String xfeatModel = parser.get<String>("xfeat_model");
    const String xfeatOutput = parser.get<String>("xfeat_output");
    const String backend = parser.get<String>("backend");

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    const int backendId = getBackendID(backend);
    const int targetId = dnn::DNN_TARGET_CPU;

    if (!xfeatModel.empty())
    {
        if (imagePath1.empty())
        {
            cerr << "Error: the XFeat demo requires an input image." << endl;
            parser.printMessage();
            return 1;
        }
        return runXFeatDemo(imagePath1, xfeatModel, xfeatOutput, backendId, targetId);
    }

    if (imagePath1.empty() || imagePath2.empty() || featureType.empty() ||
        featureModel.empty() || lightglueModel.empty())
    {
        cerr << "Error: missing required arguments." << endl;
        parser.printMessage();
        return 1;
    }

    Mat image1 = imread(imagePath1);
    Mat image2 = imread(imagePath2);
    if (image1.empty() || image2.empty())
    {
        cerr << "Error: cannot load input images." << endl;
        return 1;
    }

    Ptr<Feature2D> detector;
    Ptr<LightGlueMatcher> matcher;

    if (featureType == "aliked")
    {
        ALIKED::Params params;
        params.backend = backendId;
        params.target = targetId;
        detector = ALIKED::create(featureModel, params);
        matcher = LightGlueMatcher::create(lightglueModel, 0.0f,
                                           backendId, targetId, LG_ALIKED);
    }
    else if (featureType == "disk")
    {
        Ptr<DISK> disk = DISK::create(featureModel, 1024, 0.0f, Size(),
                                      backendId, targetId);
        detector = disk;
        matcher = LightGlueMatcher::create(lightglueModel, 0.0f,
                                           backendId, targetId, LG_DISK);
    }
    else
    {
        cerr << "Error: feature type must be 'aliked' or 'disk'." << endl;
        parser.printMessage();
        return 1;
    }

    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute(image1, noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(image2, noArray(), keypoints2, descriptors2);

    cout << "Image 1: " << keypoints1.size() << " keypoints, descriptors "
         << descriptors1.rows << "x" << descriptors1.cols << endl;
    cout << "Image 2: " << keypoints2.size() << " keypoints, descriptors "
         << descriptors2.rows << "x" << descriptors2.cols << endl;

    if (descriptors1.empty() || descriptors2.empty())
    {
        cerr << "Error: no descriptors found in one or both images." << endl;
        return 1;
    }

    matcher->setPairInfo(keypointsToMat(keypoints1), keypointsToMat(keypoints2),
                         image1.size(), image2.size());

    vector<DMatch> matches;
    matcher->match(descriptors1, descriptors2, matches);

    cout << "Matches: " << matches.size() << endl;

    if (!outputPath.empty())
    {
        Mat visualization;
        drawMatches(image1, keypoints1, image2, keypoints2, matches, visualization,
                    Scalar::all(-1), Scalar::all(-1), vector<char>(),
                    DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        if (!imwrite(outputPath, visualization))
        {
            cerr << "Error: cannot write output image: " << outputPath << endl;
            return 1;
        }
        cout << "Output written to " << outputPath << endl;
    }

    return 0;
}
