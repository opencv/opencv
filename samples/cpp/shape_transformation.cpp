/*
 * shape_context.cpp -- Shape context demo for shape matching
 */

#include "opencv2/shape.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/core/utility.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

static void help()
{
    printf("\nThis program demonstrates how to use common interface for shape transformers\n"
           "Call\n"
           "shape_transformation [image1] [image2]\n");
}

int main(int argc, char** argv)
{
    help();
    Mat img1 = imread(argv[1], IMREAD_GRAYSCALE);
    Mat img2 = imread(argv[2], IMREAD_GRAYSCALE);
    if(img1.empty() || img2.empty() || argc<2)
    {
        printf("Can't read one of the images\n");
        return -1;
    }

    // detecting keypoints
    SurfFeatureDetector detector(5000);
    vector<KeyPoint> keypoints1, keypoints2;
    detector.detect(img1, keypoints1);
    detector.detect(img2, keypoints2);

    // computing descriptors
    SurfDescriptorExtractor extractor;
    Mat descriptors1, descriptors2;
    extractor.compute(img1, keypoints1, descriptors1);
    extractor.compute(img2, keypoints2, descriptors2);

    // matching descriptors
    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // drawing the results
    namedWindow("matches", 1);
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
    imshow("matches", img_matches);

    // extract points
    vector<Point2f> pts1, pts2;
    for (size_t ii=0; ii<keypoints1.size(); ii++)
        pts1.push_back( keypoints1[ii].pt );
    for (size_t ii=0; ii<keypoints2.size(); ii++)
        pts2.push_back( keypoints2[ii].pt );

    // Apply TPS
    Ptr<ThinPlateSplineShapeTransformer> mytps = createThinPlateSplineShapeTransformer(25000); //TPS with a relaxed constraint
    mytps->estimateTransformation(pts1, pts2, matches);
    mytps->warpImage(img2, img2);

    imshow("Tranformed", img2);
    waitKey(0);

    return 0;
}
