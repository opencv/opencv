//! [tutorial]
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace ccm;

const char *about = "Color correction sample";
const char *keys =
    "{ help h  |                                                     | show this message }"
    "{ input   |   opencv_extra/testdata/cv/mcc/mcc_ccm_test.jpg     | Path of the image file to process }"
    "{ colors  |             samples/data/ccm_test_data.txt               | Path to the txt file containing color values }";

int main(int argc, char *argv[])
{
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);
    if (argc==1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    string filepath = parser.get<string>("input");
    string colorFile = parser.get<string>("colors");

    if (!parser.check())
    {
        cout << "Usage: " << argv[0] << " <input_image> <color_values.yml>" << endl;
        return 1;
    }

    Mat image = imread(filepath, IMREAD_COLOR);
    if (image.empty())
    {
        cout << "Failed to open image file!" << endl;
        return 1;
    }

    // Read color values from YAML file
    FileStorage fs(colorFile, FileStorage::READ);
    if (!fs.isOpened())
    {
        cout << "Failed to open color values file!" << endl;
        return 1;
    }

    Mat src;
    fs["color_values"] >> src;
    fs.release();

    if (src.empty())
    {
        cout << "Failed to read color values from file!" << endl;
        return 1;
    }

    // Convert to double and normalize
    src.convertTo(src, CV_64F, 1.0/255.0);

    // Create and train the model
    cv::ccm::ColorCorrectionModel model(src, cv::ccm::COLORCHECKER_Macbeth);
    model.setColorSpace(cv::ccm::COLOR_SPACE_SRGB);
    model.setCCMType(cv::ccm::CCM_LINEAR);
    model.setDistance(cv::ccm::DISTANCE_CIE2000);
    model.setLinear(cv::ccm::LINEARIZATION_GAMMA);
    model.setLinearGamma(2.2);
    model.computeCCM();

    // Save the model parameters
    FileStorage modelFs("model.yml", FileStorage::WRITE);
    modelFs << "ccm" << model.getCCM();
    modelFs << "loss" << model.getLoss();
    modelFs.release();

    Mat calibratedImage = model.infer(image);
    Mat out_ = calibratedImage * 255.0;

    out_.convertTo(out_, CV_8UC3);
    Mat img_out = min(max(out_, 0), 255);
    Mat out_img;
    cvtColor(img_out, out_img, COLOR_RGB2BGR);
    string filename = filepath.substr(filepath.find_last_of('/')+1);
    size_t dotIndex = filename.find_last_of('.');
    string baseName = filename.substr(0, dotIndex);
    string ext = filename.substr(dotIndex+1, filename.length()-dotIndex);
    string calibratedFilePath = baseName + ".calibrated." + ext;
    imwrite(calibratedFilePath, out_img);

    return 0;
}
//! [tutorial]
