/**
  @file ela.cpp
  @brief Error Level Analysis (ELA) permits identifying areas within an image that are at different compression levels. With JPEG images, the entire picture should be at roughly the same level. If a section of the image is at a significantly different error level, then it likely indicates a digital modification. This example allows to see visually the changes made in a JPG image based in it's compression error analysis.
*/

#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;

int scale_value = 7;
int quality = 95;
Mat image;
Mat compressed_img;

static void processImage(int, void*)
{
    Mat Ela;

    // Compression jpeg
    std::vector<int> compressing_factor;
    std::vector<uchar> buf;

    compressing_factor.push_back(IMWRITE_JPEG_QUALITY);
    compressing_factor.push_back(quality);

    imencode(".jpg", image, buf, compressing_factor);

    compressed_img = imdecode(buf, 1);

    Mat output;
    absdiff(image, compressed_img, output);
    output.convertTo(Ela, CV_8UC3, scale_value);

    // Shows processed image
    // imshow(decodedwin, compressed_img); // 注释掉图像显示
    // imshow(diffwin, Ela); // 注释掉图像显示

    // 保存处理后的图像
    std::string compressed_filename = "compressed_image.jpg";
    imwrite(compressed_filename, compressed_img);
    printf("Compressed image saved as: %s\n", compressed_filename.c_str());

    std::string ela_filename = "ela_result.png";
    imwrite(ela_filename, Ela);
    printf("ELA result image saved as: %s\n", ela_filename.c_str());
}

int main(int argc, char* argv[])
{
    CommandLineParser parser(argc, argv, "{ input i | ela_modified.jpg | Input image to calculate ELA algorithm. }");
    parser.about("\nJpeg Recompression Example:\n");
    parser.printMessage();

    // Read the new image
    image = imread(samples::findFile(parser.get<String>("input")));

    // Check image
    if (!image.empty())
    {
        processImage(0, 0);
        // createTrackbar("Scale", diffwin, &scale_value, 100, processImage); // 注释掉创建滑动条
        // createTrackbar("Quality", diffwin, &quality, 100, processImage); // 注释掉创建滑动条
        // waitKey(0); // 注释掉等待按键
    }
    else
    {
        std::cout << "> Error in load image\n";
    }

    return 0;
}

