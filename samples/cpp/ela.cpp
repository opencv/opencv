/**
  @file ela.cpp
  @author Alessandro de Oliveira Faria (A.K.A. CABELO)
  @brief Error Level Analysis (ELA) permits identifying areas within an image that are at different compression levels. With JPEG images, the entire picture should be at roughly the same level. If a section of the image is at a significantly different error level, then it likely indicates a digital modification. This example allows to see visually the changes made in a JPG image based in it's compression error analysis. Questions and suggestions email to: Alessandro de Oliveira Faria cabelo[at]opensuse[dot]org or OpenCV Team.
  @date Jun 24, 2018
*/

#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;

int scale_value = 7;
int quality = 95;
Mat image;
Mat compressed_img;
const char* decodedwin = "the recompressed image";
const char* diffwin = "scaled difference between the original and recompressed images";

static void processImage(int , void*)
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
    absdiff(image,compressed_img,output);
    output.convertTo(Ela, CV_8UC3, scale_value);

    // Shows processed image
    imshow(decodedwin, compressed_img);
    imshow(diffwin, Ela);
}

int main (int argc, char* argv[])
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
        createTrackbar("Scale", diffwin, &scale_value, 100, processImage);
        createTrackbar("Quality", diffwin, &quality, 100, processImage);
        waitKey(0);
    }
    else
    {
        std::cout << "> Error in load image\n";
    }

    return 0;
}
