//! [tutorial]
#include <opencv2/core.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ccm.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace ccm;

const char *about = "Basic chart detection";
const char *keys =
    "{ help h  |    | show this message }"
    "{ f       | 1    | Path of the file to process (-v) }";

int main(int argc, char *argv[])
{


    // ----------------------------------------------------------
    // Scroll down a bit (~40 lines) to find actual relevant code
    // ----------------------------------------------------------
    //! [get_messages_of_image]
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);
    if (argc==1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    string filepath = parser.get<string>("f");

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    Mat image = imread(filepath, IMREAD_COLOR);
    if (!image.data)
    {
        cout << "Invalid Image!" << endl;
        return 1;
    }
    //! [get_messages_of_image]

    Mat src(24, 1, CV_64FC3);

    // Hardcoded values. Image used: opencv_extra/testdata/cv/mcc/mcc_ccm_test.jpg
    double values[24][3] = {
        {0.380463, 0.31696, 0.210053},
        {0.649781, 0.520561, 0.452553},
        {0.323114, 0.37593, 0.50123},
        {0.314785, 0.396522, 0.258116},
        {0.452971, 0.418602, 0.578767},
        {0.34908, 0.608649, 0.652283},
        {0.691127, 0.517818, 0.144984},
        {0.208668, 0.224391, 0.485851},
        {0.657849, 0.378126, 0.304115},
        {0.285762, 0.229671, 0.31913},
        {0.513422, 0.685031, 0.337381},
        {0.786459, 0.676133, 0.246303},
        {0.11751, 0.135079, 0.383441},
        {0.190745, 0.470513, 0.296844},
        {0.587832, 0.299132, 0.196117},
        {0.783908, 0.746261, 0.294357},
        {0.615481, 0.359983, 0.471403},
        {0.107095, 0.370516, 0.573142},
        {0.708598, 0.718936, 0.740915},
        {0.593812, 0.612474, 0.63222},
        {0.489774, 0.510077, 0.521757},
        {0.380591, 0.398499, 0.393662},
        {0.27461, 0.293267, 0.275244},
        {0.180753, 0.194968, 0.145006}
    };

    // Assign values to src
    for (int i = 0; i < 24; i++) {
        src.at<cv::Vec3d>(i, 0) = cv::Vec3d(values[i][0], values[i][1], values[i][2]);
    }

    //compte color correction matrix
    //! [get_ccm_Matrix]
    ColorCorrectionModel model1(src, COLORCHECKER_Macbeth);
    model1.run();
    Mat ccm = model1.getCCM();
    std::cout<<"ccm "<<ccm<<std::endl;
    double loss = model1.getLoss();
    std::cout<<"loss "<<loss<<std::endl;
    //! [get_ccm_Matrix]
        /* brief More models with different parameters, try it & check the document for details.
    */
    // model1.setColorSpace(COLOR_SPACE_sRGB);
    // model1.setCCM_TYPE(CCM_3x3);
    // model1.setDistance(DISTANCE_CIE2000);
    // model1.setLinear(LINEARIZATION_GAMMA);
    // model1.setLinearGamma(2.2);
    // model1.setLinearDegree(3);
    // model1.setSaturatedThreshold(0, 0.98);

    /* If you use a customized ColorChecker, you can use your own reference color values and corresponding color space in a way like:
    */
    //! [reference_color_values]
    // cv::Mat ref = (Mat_<Vec3d>(18, 1) <<
    // Vec3d(100, 0.00520000001, -0.0104),
    // Vec3d(73.0833969, -0.819999993, -2.02099991),
    // Vec3d(62.493, 0.425999999, -2.23099995),
    // Vec3d(50.4640007, 0.446999997, -2.32399988),
    // Vec3d(37.7970009, 0.0359999985, -1.29700005),
    // Vec3d(0, 0, 0),
    // Vec3d(51.5880013, 73.5179977, 51.5690002),
    // Vec3d(93.6989975, -15.7340002, 91.9420013),
    // Vec3d(69.4079971, -46.5940018, 50.4869995),
    // Vec3d(66.61000060000001, -13.6789999, -43.1720009),
    // Vec3d(11.7110004, 16.9799995, -37.1759987),
    // Vec3d(51.973999, 81.9440002, -8.40699959),
    // Vec3d(40.5489998, 50.4399986, 24.8490009),
    // Vec3d(60.8160019, 26.0690002, 49.4420013),
    // Vec3d(52.2529984, -19.9500008, -23.9960003),
    // Vec3d(51.2859993, 48.4700012, -15.0579996),
    // Vec3d(68.70700069999999, 12.2959995, 16.2129993),
    // Vec3d(63.6839981, 10.2930002, 16.7639999));

    // ColorCorrectionModel model8(src,ref,COLOR_SPACE_Lab_D50_2);
    // model8.run();
    //! [reference_color_values]

    //! [make_color_correction]
    Mat img_;
    cvtColor(image, img_, COLOR_BGR2RGB);
    img_.convertTo(img_, CV_64F);
    const int inp_size = 255;
    const int out_size = 255;
    img_ = img_ / inp_size;
    Mat calibratedImage= model1.infer(img_);
    Mat out_ = calibratedImage * out_size;
    //! [make_color_correction]

    //! [Save_calibrated_image]
    // Save the calibrated image to {FILE_NAME}.calibrated.{FILE_EXT}
    out_.convertTo(out_, CV_8UC3);
    Mat img_out = min(max(out_, 0), out_size);
    Mat out_img;
    cvtColor(img_out, out_img, COLOR_RGB2BGR);
    string filename = filepath.substr(filepath.find_last_of('/')+1);
    size_t dotIndex = filename.find_last_of('.');
    string baseName = filename.substr(0, dotIndex);
    string ext = filename.substr(dotIndex+1, filename.length()-dotIndex);
    string calibratedFilePath = baseName + ".calibrated." + ext;
    imwrite(calibratedFilePath, out_img);
    //! [Save_calibrated_image]

    return 0;
}
//! [tutorial]