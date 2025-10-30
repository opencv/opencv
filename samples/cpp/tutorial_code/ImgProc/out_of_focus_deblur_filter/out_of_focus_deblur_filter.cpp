/**
* @brief You will learn how to recover an out-of-focus image by Wiener filter
* @author Karpushin Vladislav, karpushin@ngs.ru, https://github.com/VladKarpushin
*/
#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

using namespace cv;
using namespace std;

void help();
void calcPSF(Mat& outputImg, Size filterSize, int R);
void fftshift(const Mat& inputImg, Mat& outputImg);
void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H);
void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr);

const String keys =
"{help h usage ? |             | print this message   }"
"{image          |original.jpg | input image name     }"
"{R              |5           | radius               }"
"{SNR            |100         | signal to noise ratio}"
;

int main(int argc, char *argv[])
{
    help();
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    int R = parser.get<int>("R");
    int snr = parser.get<int>("SNR");
    string strInFileName = parser.get<String>("image");
    samples::addSamplesDataSearchSubDirectory("doc/tutorials/imgproc/out_of_focus_deblur_filter/images");

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    Mat imgIn;
    imgIn = imread(samples::findFile( strInFileName ), IMREAD_GRAYSCALE);
    if (imgIn.empty()) //check whether the image is loaded or not
    {
        cout << "ERROR : Image cannot be loaded..!!" << endl;
        return -1;
    }

    Mat imgOut;

//! [main]
    // it needs to process even image only
    Rect roi = Rect(0, 0, imgIn.cols & -2, imgIn.rows & -2);

    //Hw calculation (start)
    Mat Hw, h;
    calcPSF(h, roi.size(), R);
    calcWnrFilter(h, Hw, 1.0 / double(snr));
    //Hw calculation (stop)

    // filtering (start)
    filter2DFreq(imgIn(roi), imgOut, Hw);
    // filtering (stop)
//! [main]

    imgOut.convertTo(imgOut, CV_8U);
    normalize(imgOut, imgOut, 0, 255, NORM_MINMAX);
    imshow("Original", imgIn);
    imshow("Debluring", imgOut);
    imwrite("result.jpg", imgOut);
    waitKey(0);
    return 0;
}

void help()
{
    cout << "2018-07-12" << endl;
    cout << "DeBlur_v8" << endl;
    cout << "You will learn how to recover an out-of-focus image by Wiener filter" << endl;
}

//! [calcPSF]
void calcPSF(Mat& outputImg, Size filterSize, int R)
{
    Mat h(filterSize, CV_32F, Scalar(0));
    Point point(filterSize.width / 2, filterSize.height / 2);
    circle(h, point, R, 255, -1, 8);
    Scalar summa = sum(h);
    outputImg = h / summa[0];
}
//! [calcPSF]

//! [fftshift]
void fftshift(const Mat& inputImg, Mat& outputImg)
{
    outputImg = inputImg.clone();
    int cx = outputImg.cols / 2;
    int cy = outputImg.rows / 2;
    Mat q0(outputImg, Rect(0, 0, cx, cy));
    Mat q1(outputImg, Rect(cx, 0, cx, cy));
    Mat q2(outputImg, Rect(0, cy, cx, cy));
    Mat q3(outputImg, Rect(cx, cy, cx, cy));
    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}
//! [fftshift]

//! [filter2DFreq]
void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H)
{
    Mat planes[2] = { Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI, DFT_SCALE);

    Mat planesH[2] = { Mat_<float>(H.clone()), Mat::zeros(H.size(), CV_32F) };
    Mat complexH;
    merge(planesH, 2, complexH);
    Mat complexIH;
    mulSpectrums(complexI, complexH, complexIH, 0);

    idft(complexIH, complexIH);
    split(complexIH, planes);
    outputImg = planes[0];
}
//! [filter2DFreq]

//! [calcWnrFilter]
void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr)
{
    Mat h_PSF_shifted;
    fftshift(input_h_PSF, h_PSF_shifted);
    Mat planes[2] = { Mat_<float>(h_PSF_shifted.clone()), Mat::zeros(h_PSF_shifted.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI);
    split(complexI, planes);
    Mat denom;
    pow(abs(planes[0]), 2, denom);
    denom += nsr;
    divide(planes[0], denom, output_G);
}
//! [calcWnrFilter]
