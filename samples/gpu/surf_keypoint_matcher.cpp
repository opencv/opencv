#include <iostream>

#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_NONFREE

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/nonfree/gpu.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

static void help()
{
    cout << "\nThis program demonstrates using SURF_GPU features detector, descriptor extractor and BruteForceMatcher_GPU" << endl;
    cout << "\nUsage:\n\tmatcher_simple_gpu --left <image1> --right <image2>" << endl;
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
            img1.upload(imread(argv[++i], CV_LOAD_IMAGE_GRAYSCALE));
            CV_Assert(!img1.empty());
        }
        else if (string(argv[i]) == "--right")
        {
            img2.upload(imread(argv[++i], CV_LOAD_IMAGE_GRAYSCALE));
            CV_Assert(!img2.empty());
        }
        else if (string(argv[i]) == "--help")
        {
            help();
            return -1;
        }
    }

    cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());

    SURF_GPU surf;

    // detecting keypoints & computing descriptors
    GpuMat keypoints1GPU, keypoints2GPU;
    GpuMat descriptors1GPU, descriptors2GPU;
    surf(img1, GpuMat(), keypoints1GPU, descriptors1GPU);
    surf(img2, GpuMat(), keypoints2GPU, descriptors2GPU);

    cout << "FOUND " << keypoints1GPU.cols << " keypoints on first image" << endl;
    cout << "FOUND " << keypoints2GPU.cols << " keypoints on second image" << endl;

    // matching descriptors
    BFMatcher_GPU matcher(NORM_L2);
    GpuMat trainIdx, distance;
    matcher.matchSingle(descriptors1GPU, descriptors2GPU, trainIdx, distance);

    // downloading results
    vector<KeyPoint> keypoints1, keypoints2;
    vector<float> descriptors1, descriptors2;
    vector<DMatch> matches;
    surf.downloadKeypoints(keypoints1GPU, keypoints1);
    surf.downloadKeypoints(keypoints2GPU, keypoints2);
    surf.downloadDescriptors(descriptors1GPU, descriptors1);
    surf.downloadDescriptors(descriptors2GPU, descriptors2);
    BFMatcher_GPU::matchDownload(trainIdx, distance, matches);

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
    std::cerr << "OpenCV was built without nonfree module" << std::endl;
    return 0;
}

#endif
