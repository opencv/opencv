#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <highgui.h>
#include <cstdio>

using namespace cv;

IplImage* DrawCorrespondences(IplImage* img1, const vector<KeyPoint>& features1, IplImage* img2,
                              const vector<KeyPoint>& features2, const vector<int>& desc_idx);

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        printf("Format: \n./match_sample [image1] [image2] [algorithm] [XML params]\n");
        printf("For example: ./match_sample scene_l.bmp scene_r.bmp fern fern_params.xml\n");
        return 0;
    }

    std::string img1_name = std::string(argv[1]);
    std::string img2_name = std::string(argv[2]);
    std::string alg_name = std::string(argv[3]);
    std::string params_filename = std::string(argv[4]);

    GenericDescriptorMatch *descriptorMatcher = createGenericDescriptorMatcher(alg_name, params_filename);
    if( descriptorMatcher == 0 )
    {
        printf ("Cannot create descriptor\n");
        return 0;
    }

    //printf("Reading the images...\n");
    IplImage* img1 = cvLoadImage(img1_name.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    IplImage* img2 = cvLoadImage(img2_name.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    
    // extract keypoints from the first image
    SURF surf_extractor(5.0e3);
    vector<KeyPoint> keypoints1;

    // printf("Extracting keypoints\n");
    surf_extractor(img1, Mat(), keypoints1);
    
    printf("Extracted %d keypoints from the first image\n", (int)keypoints1.size());

    vector<KeyPoint> keypoints2;
    surf_extractor(img2, Mat(), keypoints2);
    printf("Extracted %d keypoints from the second image\n", (int)keypoints2.size());

    printf("Finding nearest neighbors... \n");
    // find NN for each of keypoints2 in keypoints1
    descriptorMatcher->add( img1, keypoints1 );
    vector<int> matches2to1;
    matches2to1.resize(keypoints2.size());
    descriptorMatcher->match( img2, keypoints2, matches2to1 );
    printf("Done\n");

    IplImage* img_corr = DrawCorrespondences(img1, keypoints1, img2, keypoints2, matches2to1);

    cvNamedWindow("correspondences", 1);
    cvShowImage("correspondences", img_corr);
    cvWaitKey(0);

    cvReleaseImage(&img1);
    cvReleaseImage(&img2);
    cvReleaseImage(&img_corr);
    delete descriptorMatcher;
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
        CvPoint pt = cvPoint(cvRound(features2[i].pt.x + img1->width), cvRound(features2[i].pt.y));
        cvCircle(img_corr, pt, 3, CV_RGB(255, 0, 0));
        cvLine(img_corr, features1[desc_idx[i]].pt, pt, CV_RGB(0, 255, 0));
    }

    return img_corr;
}
