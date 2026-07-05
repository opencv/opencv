// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// ALIKED/DISK + LightGlueMatcher usage example.
// Demonstrates feature detection, extraction, and matching using a LightGlue model.

#include <opencv2/features.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

static void printUsage(const char* programName)
{
    cout << "Usage:" << endl;
    cout << "  " << programName << " <image1> <image2> aliked <aliked_model> <aliked_lightglue_model> [output]" << endl;
    cout << "  " << programName << " <image1> <image2> disk <disk_model> <disk_lightglue_model> [output]" << endl;
}

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

int main(int argc, char** argv)
{
    if (argc != 6 && argc != 7)
    {
        printUsage(argv[0]);
        return 1;
    }

    const String imagePath1 = argv[1];
    const String imagePath2 = argv[2];
    const String featureType = argv[3];
    const String featureModel = argv[4];
    const String lightglueModel = argv[5];
    const String outputPath = argc == 7 ? argv[6] : String();

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
        detector = ALIKED::create(featureModel);
        matcher = LightGlueMatcher::create(lightglueModel);
    }
    else if (featureType == "disk")
    {
        Ptr<DISK> disk = DISK::create(featureModel);
        disk->setMaxKeypoints(1024);
        detector = disk;
        matcher = LightGlueMatcher::create(lightglueModel, 0.0f, 0, 0, LG_DISK);
    }
    else
    {
        cerr << "Error: feature type must be 'aliked' or 'disk'." << endl;
        printUsage(argv[0]);
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
