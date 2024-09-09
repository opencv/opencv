#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <ctype.h>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

// static void help(char** argv)
// {
//     cout <<
//             "\nThis program demonstrates Laplace point/edge detection using OpenCV function Laplacian()\n"
//             "It captures from the camera of your choice: 0, 1, ... default 0\n"
//             "Call:\n"
//          <<  argv[0] << " -c=<camera #, default 0> -p=<index of the frame to be decoded/captured next>\n" << endl;
// }

enum {GAUSSIAN, BLUR, MEDIAN};

int sigma = 3;
int smoothType = GAUSSIAN;

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, "{ i | fruits.jpg | }");
    // help(argv);

    string input = parser.get<string>("i");
    Mat frame = imread(samples::findFileOrKeep(input));
    if (frame.empty())
    {
        cerr << "Can't open image: " << input << endl;
        return 1;
    }
    cout << "Image " << parser.get<string>("i") <<
        ": width=" << frame.cols <<
        ", height=" << frame.rows << endl;

    // namedWindow("Laplacian", WINDOW_AUTOSIZE);
    // createTrackbar("Sigma", "Laplacian", &sigma, 15, 0);

    Mat smoothed, laplace, result;
    const int max_iterations = 100; // 设置最大迭代次数
    int iteration_count = 0;

    while(iteration_count < max_iterations)
    {
        if(frame.empty())
            break;

        int ksize = (sigma*5)|1;
        if(smoothType == GAUSSIAN)
            GaussianBlur(frame, smoothed, Size(ksize, ksize), sigma, sigma);
        else if(smoothType == BLUR)
            blur(frame, smoothed, Size(ksize, ksize));
        else
            medianBlur(frame, smoothed, ksize);

        Laplacian(smoothed, laplace, CV_16S, 5);
        convertScaleAbs(laplace, result, (sigma+1)*0.25);

        // imshow("Laplacian", result);
        imwrite("laplacian_" + to_string(iteration_count) + ".png", result); // 保存图像

        // char c = (char)waitKey(30);
        // if(c == ' ')
        //     smoothType = smoothType == GAUSSIAN ? BLUR : smoothType == BLUR ? MEDIAN : GAUSSIAN;
        // if(c == 'q' || c == 'Q' || c == 27)
        //     break;

        iteration_count++;
    }

    return 0;
}

