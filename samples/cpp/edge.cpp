#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <stdio.h>

using namespace cv;
using namespace std;

int edgeThresh = 1;
int edgeThreshScharr = 1;

Mat image, gray, blurImage, edge1, edge2, cedge;

// define a trackbar callback
static void onTrackbar()
{
    blur(gray, blurImage, Size(3,3));

    // Run the edge detector on grayscale
    Canny(blurImage, edge1, edgeThresh, edgeThresh * 3, 3);
    cedge = Scalar::all(0);

    image.copyTo(cedge, edge1);
    // imshow(window_name1, cedge); // 注释掉图像显示

    string filename1 = "canny_default.png";
    imwrite(filename1, cedge);
    printf("Result image saved as: %s\n", filename1.c_str());

    /// Canny detector with scharr
    Mat dx, dy;
    Scharr(blurImage, dx, CV_16S, 1, 0);
    Scharr(blurImage, dy, CV_16S, 0, 1);
    Canny(dx, dy, edge2, edgeThreshScharr, edgeThreshScharr * 3);
    /// Using Canny's output as a mask, we display our result
    cedge = Scalar::all(0);
    image.copyTo(cedge, edge2);
    // imshow(window_name2, cedge); // 注释掉图像显示

    string filename2 = "canny_scharr.png";
    imwrite(filename2, cedge);
    printf("Result image saved as: %s\n", filename2.c_str());
}

static void help(const char** argv)
{
    printf("\nThis sample demonstrates Canny edge detection\n"
           "Call:\n"
           "    %s [image_name -- Default is fruits.jpg]\n\n", argv[0]);
}

const char* keys =
{
    "{help h||}{@image |fruits.jpg|input image name}"
};

int main(int argc, const char** argv)
{
    help(argv);
    CommandLineParser parser(argc, argv, keys);
    string filename = parser.get<string>(0);

    image = imread(samples::findFile(filename), IMREAD_COLOR);
    if (image.empty())
    {
        printf("Cannot read image file: %s\n", filename.c_str());
        help(argv);
        return -1;
    }
    cedge.create(image.size(), image.type());
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Create a window
    // namedWindow(window_name1, 1); // 注释掉窗口创建
    // namedWindow(window_name2, 1); // 注释掉窗口创建

    // create a toolbar
    // createTrackbar("Canny threshold default", window_name1, &edgeThresh, 100, onTrackbar);
    // createTrackbar("Canny threshold Scharr", window_name2, &edgeThreshScharr, 400, onTrackbar);

    // Show the image
    onTrackbar();

    // Wait for a key stroke; the same function arranges events processing
    // waitKey(0); // 注释掉等待按键

    return 0;
}

