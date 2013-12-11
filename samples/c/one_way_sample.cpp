/*
 *  one_way_sample.cpp
 *  outlet_detection
 *
 *  Created by Victor  Eruhimov on 8/5/09.
 *  Copyright 2009 Argus Corp. All rights reserved.
 *
 */

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/legacy/compat.hpp"

#include <string>
#include <stdio.h>

static void help()
{
    printf("\nThis program demonstrates the one way interest point descriptor found in features2d.hpp\n"
            "Correspondences are drawn\n");
    printf("Format: \n./one_way_sample <path_to_samples> <image1> <image2>\n");
    printf("For example: ./one_way_sample . ../c/scene_l.bmp ../c/scene_r.bmp\n");
}

using namespace std;
using namespace cv;

Mat DrawCorrespondences(const Mat& img1, const vector<KeyPoint>& features1, const Mat& img2,
                        const vector<KeyPoint>& features2, const vector<int>& desc_idx);

int main(int argc, char** argv)
{
    const char images_list[] = "one_way_train_images.txt";
    const CvSize patch_size = cvSize(24, 24);
    const int pose_count = 50;

    if (argc != 4)
    {
        help();
        return 0;
    }

    std::string path_name = argv[1];
    std::string img1_name = path_name + "/" + std::string(argv[2]);
    std::string img2_name = path_name + "/" + std::string(argv[3]);

    printf("Reading the images...\n");
    Mat img1 = imread(img1_name, IMREAD_GRAYSCALE);
    Mat img2 = imread(img2_name, IMREAD_GRAYSCALE);

    // extract keypoints from the first image
    SURF surf_extractor(5.0e3);
    vector<KeyPoint> keypoints1;

    // printf("Extracting keypoints\n");
    surf_extractor(img1, Mat(), keypoints1);

    printf("Extracted %d keypoints...\n", (int)keypoints1.size());

    printf("Training one way descriptors... \n");
    // create descriptors
    OneWayDescriptorBase descriptors(patch_size, pose_count, OneWayDescriptorBase::GetPCAFilename(), path_name,
                                     images_list);
    IplImage img1_c = img1;
    IplImage img2_c = img2;
    descriptors.CreateDescriptorsFromImage(&img1_c, keypoints1);
    printf("done\n");

    // extract keypoints from the second image
    vector<KeyPoint> keypoints2;
    surf_extractor(img2, Mat(), keypoints2);
    printf("Extracted %d keypoints from the second image...\n", (int)keypoints2.size());

    printf("Finding nearest neighbors...");
    // find NN for each of keypoints2 in keypoints1
    vector<int> desc_idx;
    desc_idx.resize(keypoints2.size());
    for (size_t i = 0; i < keypoints2.size(); i++)
    {
        int pose_idx = 0;
        float distance = 0;
        descriptors.FindDescriptor(&img2_c, keypoints2[i].pt, desc_idx[i], pose_idx, distance);
    }
    printf("done\n");

    Mat img_corr = DrawCorrespondences(img1, keypoints1, img2, keypoints2, desc_idx);

    imshow("correspondences", img_corr);
    waitKey(0);
}

Mat DrawCorrespondences(const Mat& img1, const vector<KeyPoint>& features1, const Mat& img2,
                        const vector<KeyPoint>& features2, const vector<int>& desc_idx)
{
    Mat part, img_corr(Size(img1.cols + img2.cols, MAX(img1.rows, img2.rows)), CV_8UC3);
    img_corr = Scalar::all(0);
    part = img_corr(Rect(0, 0, img1.cols, img1.rows));
    cvtColor(img1, part, COLOR_GRAY2RGB);
    part = img_corr(Rect(img1.cols, 0, img2.cols, img2.rows));
    cvtColor(img1, part, COLOR_GRAY2RGB);

    for (size_t i = 0; i < features1.size(); i++)
    {
        circle(img_corr, features1[i].pt, 3, CV_RGB(255, 0, 0));
    }

    for (size_t i = 0; i < features2.size(); i++)
    {
        Point pt((int)features2[i].pt.x + img1.cols, (int)features2[i].pt.y);
        circle(img_corr, pt, 3, Scalar(0, 0, 255));
        line(img_corr, features1[desc_idx[i]].pt, pt, Scalar(0, 255, 0));
    }

    return img_corr;
}
