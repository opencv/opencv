#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp" // 无头环境不需要

#include <iostream>

using namespace cv;
using namespace std;

int _brightness = 100;
int _contrast = 100;

Mat image;

/* brightness/contrast callback function */
static void updateBrightnessContrast( int /*arg*/, void* )
{
    int histSize = 64;
    int brightness = _brightness - 100;
    int contrast = _contrast - 100;

    /*
     * The algorithm is by Werner D. Streidt
     * (http://visca.com/ffactory/archives/5-99/msg00021.html)
     */
    double a, b;
    if( contrast > 0 )
    {
        double delta = 127.*contrast/100;
        a = 255./(255. - delta*2);
        b = a*(brightness - delta);
    }
    else
    {
        double delta = -128.*contrast/100;
        a = (256.-delta*2)/255.;
        b = a*brightness + delta;
    }

    Mat dst, hist;
    image.convertTo(dst, CV_8U, a, b);
    // imshow("image", dst); // 注释掉显示图像的代码

    calcHist(&dst, 1, 0, Mat(), hist, 1, &histSize, 0);
    Mat histImage = Mat::ones(200, 320, CV_8U)*255;

    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, CV_32F);

    histImage = Scalar::all(255);
    int binW = cvRound((double)histImage.cols/histSize);

    for( int i = 0; i < histSize; i++ )
        rectangle( histImage, Point(i*binW, histImage.rows),
                   Point((i+1)*binW, histImage.rows - cvRound(hist.at<float>(i))),
                   Scalar::all(0), -1, 8, 0 );
    // imshow("histogram", histImage); // 注释掉显示直方图的代码

    // 保存处理后的图像
    imwrite("result.png", dst);
    imwrite("histogram.png", histImage);

    cout << "Result image saved as: result.png" << endl;
    cout << "Histogram image saved as: histogram.png" << endl;
}

const char* keys =
{
    "{help h||}{@image|baboon.jpg|input image file}"
};

int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("\nThis program demonstrates the use of calcHist() -- histogram creation.\n");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    string inputImage = parser.get<string>(0);

    // Load the source image. HighGUI use.
    image = imread(samples::findFile(inputImage), IMREAD_GRAYSCALE);
    if(image.empty())
    {
        std::cerr << "Cannot read image file: " << inputImage << std::endl;
        return -1;
    }

    // 无头环境不需要创建窗口和轨迹条
    // namedWindow("image", 0);
    // namedWindow("histogram", 0);
    // createTrackbar("brightness", "image", &_brightness, 200, updateBrightnessContrast);
    // createTrackbar("contrast", "image", &_contrast, 200, updateBrightnessContrast);

    updateBrightnessContrast(0, 0);
    // waitKey(); // 注释掉等待按键的代码

    return 0;
}

