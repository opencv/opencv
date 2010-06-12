#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

int main(int argc, char** argv)
{
	if(argc != 3)
	{
		printf("Usage: matches_simple <image1> <image2>\n");
		return -1;
	}

	Mat img1 = imread(argv[1]);
	Mat img2 = imread(argv[2]);
	if(img1.empty() || img2.empty())
	{
		printf("Can't read one of the images\n");
		return -1;
	}

	// detecting keypoints
	SurfFeatureDetector detector(400);
	vector<KeyPoint> keypoints1, keypoints2;
	detector.detect(img1, keypoints1);
	detector.detect(img2, keypoints2);

	// computing descriptors
	SurfDescriptorExtractor extractor;
	Mat descriptors1, descriptors2;
	extractor.compute(img1, keypoints1, descriptors1);
	extractor.compute(img2, keypoints2, descriptors2);

	// matching descriptors
	BruteForceMatcher<L2<float> > matcher;
	vector<int> matches;
	matcher.add(descriptors1);
	matcher.match(descriptors2, matches);

	// drawing the results
	namedWindow("matches", 1);
	Mat img_matches;
	drawMatches(img1, img2, keypoints1, keypoints2, matches, vector<char>(), img_matches);
	imshow("matches", img_matches);
	waitKey(0);

	return 0;
}
