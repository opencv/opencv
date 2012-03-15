#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#include <cstdio>

using namespace cv;

void help()
{
    printf("Use the SURF descriptor for matching keypoints between 2 images\n");
    printf("Format: \n./generic_descriptor_match <image1> <image2> <algorithm> <XML params>\n");
    printf("For example: ./generic_descriptor_match ../c/scene_l.bmp ../c/scene_r.bmp FERN fern_params.xml\n");
}

Mat DrawCorrespondences(const Mat& img1, const vector<KeyPoint>& features1, const Mat& img2,
                        const vector<KeyPoint>& features2, const vector<DMatch>& desc_idx);

int main(int argc, char** argv)
{
    if (argc != 5)
    {
    	help();
        return 0;
    }

    std::string img1_name = std::string(argv[1]);
    std::string img2_name = std::string(argv[2]);
    std::string alg_name = std::string(argv[3]);
    std::string params_filename = std::string(argv[4]);

    Ptr<GenericDescriptorMatcher> descriptorMatcher = GenericDescriptorMatcher::create(alg_name, params_filename);
    if( descriptorMatcher == 0 )
    {
        printf ("Cannot create descriptor\n");
        return 0;
    }

    //printf("Reading the images...\n");
    Mat img1 = imread(img1_name, CV_LOAD_IMAGE_GRAYSCALE);
    Mat img2 = imread(img2_name, CV_LOAD_IMAGE_GRAYSCALE);
    
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
    vector<DMatch> matches2to1;
    descriptorMatcher->match( img2, keypoints2, img1, keypoints1, matches2to1 );
    printf("Done\n");

    Mat img_corr = DrawCorrespondences(img1, keypoints1, img2, keypoints2, matches2to1);

    imshow("correspondences", img_corr);
    waitKey(0);
}

Mat DrawCorrespondences(const Mat& img1, const vector<KeyPoint>& features1, const Mat& img2,
                        const vector<KeyPoint>& features2, const vector<DMatch>& desc_idx)
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
        Point pt(cvRound(features2[i].pt.x + img1.cols), cvRound(features2[i].pt.y));
        circle(img_corr, pt, 3, Scalar(0, 0, 255));
        line(img_corr, features1[desc_idx[i].trainIdx].pt, pt, Scalar(0, 255, 0));
    }

    return img_corr;
}
