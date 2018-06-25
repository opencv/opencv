/**
  @file ela.cpp
  @
  @brief ELA allows to see visually the changes made in a JPG image based in it's compression error analysis. Based in Eliezer Bernart example.
  @date Jun 24, 2018
*/

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <iostream>
#include <vector> 

const char* keys =
    "{ help h      | | Print help message. }"
    "{ input i     | | Input image to calc ELA algorithm. }";


using namespace cv;

int scale_value = 7;
int quality = 95;
Mat image;
Mat compressed_img;

static void processImage(int , void* )
{
    // Compression jpeg
    std::vector<int> compressing_factor;
    compressing_factor.push_back(CV_IMWRITE_JPEG_QUALITY);
    compressing_factor.push_back(quality);

    imwrite("temp.jpg", image, compressing_factor);
    compressed_img = imread("temp.jpg");

    if (compressed_img.empty())
    {
        std::cout << "> Error in load file" << std::endl;
        exit(EXIT_FAILURE);
    }

    Mat output = Mat::zeros(image.size(), CV_8UC3);

    // Compare values through matrices
    for (int row = 0; row < image.rows; ++row)
    {
        const uchar* ptr_input = image.ptr<uchar>(row);
        const uchar* ptr_compress = compressed_img.ptr<uchar>(row);
        uchar* ptr_out = output.ptr<uchar>(row);

        for (int column = 0; column < image.cols; column++)
        {
            // Calc absolute difference between images
            ptr_out[0] = abs(ptr_input[0] - ptr_compress[0]) * scale_value;
            ptr_out[1] = abs(ptr_input[1] - ptr_compress[1]) * scale_value;
            ptr_out[2] = abs(ptr_input[2] - ptr_compress[2]) * scale_value;

            ptr_input += 3;
            ptr_compress += 3;
            ptr_out += 3;
        }
    }

    // Shows processed image
    cv::imshow("Error Level Analysis", output);
} 

int main (int argc, char* argv[])
{

    CommandLineParser parser(argc, argv, keys);
    if(argc == 1 || parser.has("help"))
    {
        parser.printMessage();
	std::cout<<std::endl<<"Example: "<<std::endl<<argv[0]<< " -input=../../data/ela_modified.jpg"<<std::endl;
        return 0;
    }

    if(parser.has("input"))
    {
        // Read the new image
        image = cv::imread(parser.get<String>("input"));
    }
    // Check image
    if (!image.empty())
    {
        cv::namedWindow("Error Level Analysis");
        cv::imshow("Error Level Analysis", image);
        cv::createTrackbar("Scale", "Error Level Analysis", &scale_value, 100, processImage);
        cv::createTrackbar("Quality", "Error Level Analysis", &quality, 100, processImage);
    }

    cv::waitKey(0);

    return 0;
} 
