#include <iostream>

#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_XFEATURES2D

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

static void help()
{
    cout << "\nThis program demonstrates using SURF_CUDA features detector, descriptor extractor and BruteForceMatcher_CUDA" << endl;
    cout << "\nUsage:\n\tsurf_keypoint_matcher --left <image1> --right <image2>" << endl;
}

int main(int argc, char* argv[])
{
    if (argc != 5)
    {
        help();
        return -1;
    }

    GpuMat img1, img2;
    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--left")
        {
            img1.upload(imread(argv[++i], IMREAD_GRAYSCALE));
            CV_Assert(!img1.empty());
        }
        else if (string(argv[i]) == "--right")
        {
            img2.upload(imread(argv[++i], IMREAD_GRAYSCALE));
            CV_Assert(!img2.empty());
        }
        else if (string(argv[i]) == "--help")
        {
            help();
            return -1;
        }
    }

    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    SURF_CUDA surf;

    // detecting keypoints & computing descriptors
    GpuMat keypoints1GPU, keypoints2GPU;
    GpuMat descriptors1GPU, descriptors2GPU;
    surf(img1, GpuMat(), keypoints1GPU, descriptors1GPU);
    surf(img2, GpuMat(), keypoints2GPU, descriptors2GPU);

    cout << "FOUND " << keypoints1GPU.cols << " keypoints on first image" << endl;
    cout << "FOUND " << keypoints2GPU.cols << " keypoints on second image" << endl;

    // matching descriptors
    Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());
    vector<DMatch> matches;
    matcher->match(descriptors1GPU, descriptors2GPU, matches);

    // downloading results
    vector<KeyPoint> keypoints1, keypoints2;
    vector<float> descriptors1, descriptors2;
    surf.downloadKeypoints(keypoints1GPU, keypoints1);
    surf.downloadKeypoints(keypoints2GPU, keypoints2);
    surf.downloadDescriptors(descriptors1GPU, descriptors1);
    surf.downloadDescriptors(descriptors2GPU, descriptors2);

    // drawing the results
    Mat img_matches;
    drawMatches(Mat(img1), keypoints1, Mat(img2), keypoints2, matches, img_matches);

    namedWindow("matches", 0);
    imshow("matches", img_matches);
    waitKey(0);

    return 0;
}

#else

int main()
{
    std::cerr << "OpenCV was built without xfeatures2d module" << std::endl;
    return 0;
}

#endif
