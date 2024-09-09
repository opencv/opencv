#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>

using namespace cv;

static void help(char** argv)
{
    printf("\nShow off image morphology: erosion, dilation, open and close\n"
           "Call:\n   %s [image]\n"
           "This program also shows use of rect, ellipse and cross kernels\n\n", argv[0]);
    printf("Hot keys: \n"
           "\tESC - quit the program\n"
           "\tr - use rectangle structuring element\n"
           "\te - use elliptic structuring element\n"
           "\tc - use cross-shaped structuring element\n"
           "\tSPACE - loop through all the options\n");
}

Mat src, dst;
int element_shape = MORPH_RECT;
int max_iters = 10;
int open_close_pos = 0;
int erode_dilate_pos = 0;

static void OpenClose(int, void*)
{
    int n = open_close_pos;
    int an = abs(n);
    Mat element = getStructuringElement(element_shape, Size(an * 2 + 1, an * 2 + 1), Point(an, an));
    if (n < 0)
        morphologyEx(src, dst, MORPH_OPEN, element);
    else
        morphologyEx(src, dst, MORPH_CLOSE, element);
    // 注释掉显示相关代码
    // imshow("Open/Close", dst);
}

static void ErodeDilate(int, void*)
{
    int n = erode_dilate_pos;
    int an = abs(n);
    Mat element = getStructuringElement(element_shape, Size(an * 2 + 1, an * 2 + 1), Point(an, an));
    if (n < 0)
        erode(src, dst, element);
    else
        dilate(src, dst, element);
    // 注释掉显示相关代码
    // imshow("Erode/Dilate", dst);
}

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, "{help h||}{ @image | baboon.jpg | }{output o|output.jpg|}");
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }
    std::string filename = samples::findFile(parser.get<std::string>("@image"));
    std::string outputFilename = parser.get<std::string>("output");

    if ((src = imread(filename, IMREAD_COLOR)).empty())
    {
        help(argv);
        return -1;
    }

    open_close_pos = erode_dilate_pos = max_iters;

    for (;;)
    {
        OpenClose(open_close_pos, 0);
        ErodeDilate(erode_dilate_pos, 0);

        // 保存处理后的图像
        std::string openCloseOutput = "open_close_" + outputFilename;
        std::string erodeDilateOutput = "erode_dilate_" + outputFilename;

        imwrite(openCloseOutput, dst);
        std::cout << "Open/Close result saved to " << openCloseOutput << std::endl;

        imwrite(erodeDilateOutput, dst);
        std::cout << "Erode/Dilate result saved to " << erodeDilateOutput << std::endl;

        // 注释掉等待键输入的代码
        // char c = (char)waitKey(0);
        // if (c == 27)
        //     break;
        // if (c == 'e')
        //     element_shape = MORPH_ELLIPSE;
        // else if (c == 'r')
        //     element_shape = MORPH_RECT;
        // else if (c == 'c')
        //     element_shape = MORPH_CROSS;
        // else if (c == ' ')
        //     element_shape = (element_shape + 1) % 3;
        break; // 在保存结果后退出循环
    }

    return 0;
}

