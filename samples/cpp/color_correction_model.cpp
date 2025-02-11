//! [tutorial]
#include <opencv2/core.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/mcc.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace mcc;
using namespace ccm;
using namespace std;

const char *about = "Basic chart detection";
const char *keys =
    "{ help h  |    | show this message }"
    "{t        |      |  chartType: 0-Standard, 1-DigitalSG, 2-Vinyl }"
    "{v        |      | Input from video file, if ommited, input comes from camera }"
    "{ci       | 0    | Camera id if input doesnt come from video (-v) }"
    "{f        | 1    | Path of the file to process (-v) }"
    "{nc       | 1    | Maximum number of charts in the image }";

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

    int t = parser.get<int>("t");
    int nc = parser.get<int>("nc");
    string filepath = parser.get<string>("f");

    CV_Assert(0 <= t && t <= 2);
    TYPECHART chartType = TYPECHART(t);


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

    Mat imageCopy = image.clone();
    Ptr<CCheckerDetector> detector = CCheckerDetector::create();
    // Marker type to detect
    if (!detector->process(image, chartType, nc))
    {
        printf("ChartColor not detected \n");
        return 2;
    }
    //! [get_color_checker]
    vector<Ptr<mcc::CChecker>> checkers = detector->getListColorChecker();
    //! [get_color_checker]
    for (Ptr<mcc::CChecker> checker : checkers)
    {
        //! [create]
        Ptr<CCheckerDraw> cdraw = CCheckerDraw::create(checker);
        cdraw->draw(image);
        Mat chartsRGB = checker->getChartsRGB();
        Mat src = chartsRGB.col(1).clone().reshape(3, chartsRGB.rows/3);
        src /= 255.0;
        //! [create]

        //compte color correction matrix
        //! [get_ccm_Matrix]
        ColorCorrectionModel model1(src, COLORCHECKER_Vinyl);
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

    }

    return 0;
}
//! [tutorial]