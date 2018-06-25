/**
  @file ela.cpp
  @author cabelo@opensuse.org
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
    Mat Ela;
 
    // Compression jpeg
    std::vector<int> compressing_factor;
    std::vector<uchar> buf;

    compressing_factor.push_back(IMWRITE_JPEG_QUALITY);
    compressing_factor.push_back(quality);

    imencode(".jpg", image, buf, compressing_factor );

    compressed_img = imdecode(buf, CV_LOAD_IMAGE_COLOR);

    Mat output = Mat::zeros(image.size(), CV_8UC3);
    absdiff(image,compressed_img,output);
    output.convertTo(Ela, CV_8UC3, scale_value);

    // Shows processed image
    imshow("diff between the original and compressed images", Ela);
} 

int main (int argc, char* argv[])
{

    CommandLineParser parser(argc, argv, keys);
    if(argc == 1 || parser.has("help"))
    {
        parser.printMessage();
	std::cout<<std::endl<<"Example: "<<std::endl<<argv[0]<< " --input=../../data/ela_modified.jpg"<<std::endl;
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
        namedWindow("diff between the original and compressed images");
        imshow("diff between the original and compressed images", image);
        createTrackbar("Scale", "diff between the original and compressed images", &scale_value, 100, processImage);
        createTrackbar("Quality", "diff between the original and compressed images", &quality, 100, processImage);
    	cv::waitKey(0);
    }
    else
    {
        std::cout << "> Error in load image" << std::endl;
        exit(EXIT_FAILURE);
    }

    return 0;
} 
