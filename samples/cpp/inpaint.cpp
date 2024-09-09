#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/photo.hpp"

#include <iostream>

using namespace cv;
using namespace std;

static void help(char** argv)
{
    cout << "\nCool inpainting demo. Inpainting repairs damage to images by floodfilling the damage \n"
            << "with surrounding image areas.\n"
            "Using OpenCV version " << CV_VERSION << "\n"
            "Usage:\n" << argv[0] << " [image_name -- Default fruits.jpg]\n" << endl;
}

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, "{@image|fruits.jpg|}");
    help(argv);

    string filename = samples::findFile(parser.get<string>("@image"));
    Mat img0 = imread(filename, IMREAD_COLOR);
    if (img0.empty())
    {
        cout << "Couldn't open the image " << filename << ". Usage: inpaint <image_name>\n" << endl;
        return 0;
    }

    Mat img = img0.clone();
    Mat inpaintMask = Mat::zeros(img.size(), CV_8U);

    // Simulate drawing on the mask for demo purposes
    rectangle(inpaintMask, Point(50, 50), Point(150, 150), Scalar::all(255), FILLED);
    rectangle(img, Point(50, 50), Point(150, 150), Scalar::all(255), FILLED);

    // Run inpainting algorithm
    Mat inpainted;
    inpaint(img, inpaintMask, inpainted, 3, INPAINT_TELEA);

    // Save the inpainted image
    imwrite("inpainted_image.jpg", inpainted);
    cout << "Inpainted image saved as: inpainted_image.jpg" << endl;

    return 0;
}

