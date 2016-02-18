#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <vector>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

static void help()
{
    printf("\nThis program demonstrates using features2d detector, descriptor extractor and simple matcher\n"
            "Using the SURF desriptor:\n"
            "\n"
            "Usage:\n matcher_simple <image1> <image2>\n");
}

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        help();
        return -1;
    }

    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    Mat img2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    if(img1.empty() || img2.empty())
    {
        printf("Can't read one of the images\n");
        return -1;
    }

    // detecting keypoints
    Ptr<SURF> surf = SURF::create(400);
    vector<KeyPoint> keypoints1, keypoints2;
    surf->detect(img1, keypoints1);
    surf->detect(img2, keypoints2);

    // computing descriptors
    Mat descriptors1, descriptors2;
    surf->compute(img1, keypoints1, descriptors1);
    surf->compute(img2, keypoints2, descriptors2);

    // matching descriptors
    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // drawing the results
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
    imshow("matches", img_matches);
    waitKey(0);

    return 0;
}
