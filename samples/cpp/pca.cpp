/*
* pca.cpp
*
*  Author: 
*  Kevin Hughes <kevinhughes27[at]gmail[dot]com>
*
*  Special Thanks to:
*  Philipp Wagner <bytefish[at]gmx[dot]de>
*
* This program demonstrates how to use OpenCV PCA with a 
* specified amount of variance to retain. The effect
* is illustrated further by using a trackbar to
* change the value for retained varaince.
*
*/

#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;


///////////////////////
// Global Variables
vector<Mat> images;
Mat data;
PCA pca;
string winName = "Reconstruction | press 'q' to quit";


///////////////////////
// Functions
void read_imgList(const string& filename, vector<Mat>& images) {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line;
    while (getline(file, line)) {
        images.push_back(imread(line, 0));
    }
}

Mat formatImagesForPCA(const vector<Mat> &data)
{
    Mat dst(data.size(), data[0].rows*data[0].cols, CV_32F);
    for(unsigned int i = 0; i < data.size(); i++)
    {
        Mat image_row = data[i].clone().reshape(1,1);
        Mat row_i = dst.row(i);
        image_row.convertTo(row_i,CV_32F);    
    }
    return dst;
}

Mat toGrayscale(InputArray _src) {
    Mat src = _src.getMat();
    // only allow one channel
    if(src.channels() != 1) {
        CV_Error(CV_StsBadArg, "Only Matrices with one channel are supported");
    }
    // create and return normalized image
    Mat dst;
    cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    return dst;
}

void onTrackbar(int pos, void* ptr) 
{
    cout << "Retained Variance = " << pos << "%" << endl;
    cout << "re-calculating PCA..." << std::flush;
    
    double var = pos / 100.0;
    pca = PCA(data, cv::Mat(), CV_PCA_DATA_AS_ROW, var);
    
    Mat point = pca.project(data.row(0));
    Mat reconstruction = pca.backProject(point);
    reconstruction = reconstruction.reshape(images[0].channels(), images[0].rows);
    reconstruction = toGrayscale(reconstruction);
    
    imshow(winName, reconstruction);
    cout << "done!" << endl;
}


///////////////////////
// Main
int main(int argc, char** argv) 
{
    if (argc != 2) {
        cout << "usage: " << argv[0] << " <image_list.txt>" << endl;
        exit(1);
    }
    
    // Get the path to your CSV.
    string imgList = string(argv[1]);
    
    // Read in the data. This can fail if not valid
    try {
        read_imgList(imgList, images);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << imgList << "\". Reason: " << e.msg << endl;
        exit(1);
    }
    
    // Quit if there are not enough images for this demo.
    if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }
    
    // Reshape and stack images into a rowMatrix
    data = formatImagesForPCA(images);
    
    // perform PCA
    pca = PCA(data, cv::Mat(), CV_PCA_DATA_AS_ROW, 0.95); // trackbar is initially set here, also this is a common value for retainedVariance
    
    // Demonstration of the effect of retainedVariance on the first image 
    Mat point = pca.project(data.row(0)); // project into the eigenspace, thus the image becomes a "point"
    Mat reconstruction = pca.backProject(point); // re-create the image from the "point"
    reconstruction = reconstruction.reshape(images[0].channels(), images[0].rows); // reshape from a row vector into image shape
    reconstruction = toGrayscale(reconstruction); // re-scale for displaying purposes
    
    namedWindow(winName, CV_WINDOW_NORMAL);
    int pos = 95;
    createTrackbar("Retained Variance (%)", winName, &pos, 100, onTrackbar); 
    
    imshow(winName, reconstruction);
    char key = 0;
    while(key != 'q' || key == 27)
        key = waitKey();
   
   return 0; 
}
