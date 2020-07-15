#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

// we're NOT "using namespace std;" here, to avoid collisions between the beta variable and std::beta in c++17
using std::cout;
using std::endl;
using namespace cv;

namespace
{
/** Global Variables */
int alpha = 100;
int beta = 100;
int gamma_cor = 100;
Mat img_original, img_corrected, img_gamma_corrected;

void basicLinearTransform(const Mat &img, const double alpha_, const int beta_)
{
    Mat res;
    img.convertTo(res, -1, alpha_, beta_);

    hconcat(img, res, img_corrected);
    imshow("Brightness and contrast adjustments", img_corrected);
}

void gammaCorrection(const Mat &img, const double gamma_)
{
    CV_Assert(gamma_ >= 0);
    //! [changing-contrast-brightness-gamma-correction]
    Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for( int i = 0; i < 256; ++i)
        p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma_) * 255.0);

    Mat res = img.clone();
    LUT(img, lookUpTable, res);
    //! [changing-contrast-brightness-gamma-correction]

    hconcat(img, res, img_gamma_corrected);
    imshow("Gamma correction", img_gamma_corrected);
}

void on_linear_transform_alpha_trackbar(int, void *)
{
    double alpha_value = alpha / 100.0;
    int beta_value = beta - 100;
    basicLinearTransform(img_original, alpha_value, beta_value);
}

void on_linear_transform_beta_trackbar(int, void *)
{
    double alpha_value = alpha / 100.0;
    int beta_value = beta - 100;
    basicLinearTransform(img_original, alpha_value, beta_value);
}

void on_gamma_correction_trackbar(int, void *)
{
    double gamma_value = gamma_cor / 100.0;
    gammaCorrection(img_original, gamma_value);
}
}

int main( int argc, char** argv )
{
    CommandLineParser parser( argc, argv, "{@input | lena.jpg | input image}" );
    img_original = imread( samples::findFile( parser.get<String>( "@input" ) ) );
    if( img_original.empty() )
    {
      cout << "Could not open or find the image!\n" << endl;
      cout << "Usage: " << argv[0] << " <Input image>" << endl;
      return -1;
    }

    img_corrected = Mat(img_original.rows, img_original.cols*2, img_original.type());
    img_gamma_corrected = Mat(img_original.rows, img_original.cols*2, img_original.type());

    hconcat(img_original, img_original, img_corrected);
    hconcat(img_original, img_original, img_gamma_corrected);

    namedWindow("Brightness and contrast adjustments");
    namedWindow("Gamma correction");

    createTrackbar("Alpha gain (contrast)", "Brightness and contrast adjustments", &alpha, 500, on_linear_transform_alpha_trackbar);
    createTrackbar("Beta bias (brightness)", "Brightness and contrast adjustments", &beta, 200, on_linear_transform_beta_trackbar);
    createTrackbar("Gamma correction", "Gamma correction", &gamma_cor, 200, on_gamma_correction_trackbar);

    on_linear_transform_alpha_trackbar(0, 0);
    on_gamma_correction_trackbar(0, 0);

    waitKey();

    imwrite("linear_transform_correction.png", img_corrected);
    imwrite("gamma_correction.png", img_gamma_corrected);

    return 0;
}
