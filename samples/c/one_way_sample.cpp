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

#include <string>
void help()
{
	printf("\nThis program demonstrates the one way interest point descriptor found in features2d.hpp\n"
			"Correspondences are drawn\n");
    printf("Format: \n./one_way_sample [path_to_samples] [image1] [image2]\n");
    printf("For example: ./one_way_sample ../../../opencv/samples/c scene_l.bmp scene_r.bmp\n");
}

using namespace cv;

IplImage* DrawCorrespondences(IplImage* img1, const vector<KeyPoint>& features1, IplImage* img2,
                              const vector<KeyPoint>& features2, const vector<int>& desc_idx);

int main(int argc, char** argv)
{
    const char images_list[] = "one_way_train_images.txt";
    const CvSize patch_size = cvSize(24, 24);
    const int pose_count = 50;

    if (argc != 3 && argc != 4)
    {
    	help();
        return 0;
    }

    std::string path_name = argv[1];
    std::string img1_name = path_name + "/" + std::string(argv[2]);
    std::string img2_name = path_name + "/" + std::string(argv[3]);

    printf("Reading the images...\n");
    IplImage* img1 = cvLoadImage(img1_name.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    IplImage* img2 = cvLoadImage(img2_name.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    
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
    descriptors.CreateDescriptorsFromImage(img1, keypoints1);
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
        descriptors.FindDescriptor(img2, keypoints2[i].pt, desc_idx[i], pose_idx, distance);
    }
    printf("done\n");

    IplImage* img_corr = DrawCorrespondences(img1, keypoints1, img2, keypoints2, desc_idx);

    cvNamedWindow("correspondences", 1);
    cvShowImage("correspondences", img_corr);
    cvWaitKey(0);

    cvReleaseImage(&img1);
    cvReleaseImage(&img2);
    cvReleaseImage(&img_corr);
}

IplImage* DrawCorrespondences(IplImage* img1, const vector<KeyPoint>& features1, IplImage* img2,
                              const vector<KeyPoint>& features2, const vector<int>& desc_idx)
{
    IplImage* img_corr = cvCreateImage(cvSize(img1->width + img2->width, MAX(img1->height, img2->height)),
                                       IPL_DEPTH_8U, 3);
    cvSetImageROI(img_corr, cvRect(0, 0, img1->width, img1->height));
    cvCvtColor(img1, img_corr, CV_GRAY2RGB);
    cvSetImageROI(img_corr, cvRect(img1->width, 0, img2->width, img2->height));
    cvCvtColor(img2, img_corr, CV_GRAY2RGB);
    cvResetImageROI(img_corr);

    for (size_t i = 0; i < features1.size(); i++)
    {
        cvCircle(img_corr, features1[i].pt, 3, CV_RGB(255, 0, 0));
    }

    for (size_t i = 0; i < features2.size(); i++)
    {
        CvPoint pt = cvPoint(features2[i].pt.x + img1->width, features2[i].pt.y);
        cvCircle(img_corr, pt, 3, CV_RGB(255, 0, 0));
        cvLine(img_corr, features1[desc_idx[i]].pt, pt, CV_RGB(0, 255, 0));
    }

    return img_corr;
}
