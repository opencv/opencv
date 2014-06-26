#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <stdio.h>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

static void help()
{
    printf("\nThis program demonstrated the use of the discrete Fourier transform (dft)\n"
           "The dft of an image is taken and it's power spectrum is displayed.\n"
           "Usage:\n"
            "./dft [image_name -- default lena.jpg]\n");
}

const char* keys =
{
    "{@image|lena.jpg|input image file}"
};

int main(int argc, const char ** argv)
{
    //int cols = 4; 
    //int rows = 768;
    //srand(0);
    //Mat input(Size(cols, rows), CV_32FC2);
    //for (int i=0; i<cols; i++)
    //    for (int j=0; j<rows; j++)
    //        input.at<Vec2f>(j,i) = Vec2f((float) rand() / RAND_MAX, (float) rand() / RAND_MAX);
    //Mat dst;
    //
    //UMat gpu_input, gpu_dst;
    //input.copyTo(gpu_input);
    //auto start = std::chrono::system_clock::now();
    //dft(input, dst, DFT_ROWS);
    //auto cpu_duration = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start);
    //
    //start = std::chrono::system_clock::now();
    //dft(gpu_input, gpu_dst, DFT_ROWS);
    //auto gpu_duration = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start);

    //double n = norm(dst, gpu_dst);
    //cout << "norm = " << n << endl;
    //cout << "CPU time: " << cpu_duration.count() << "ms" << endl;
    //cout << "GPU time: " << gpu_duration.count() << "ms" << endl;


    help();
    CommandLineParser parser(argc, argv, keys);
    string filename = parser.get<string>(0);

    Mat img = imread(filename.c_str(), IMREAD_GRAYSCALE);
    if( img.empty() )
    {
        help();
        printf("Cannot read image file: %s\n", filename.c_str());
        return -1;
    }

    Mat small_img = img(Rect(0,0,6,6));

    int M = getOptimalDFTSize( small_img.rows );
    int N = getOptimalDFTSize( small_img.cols );
    Mat padded;
    copyMakeBorder(small_img, padded, 0, M - small_img.rows, 0, N - small_img.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {Mat_<float>(padded), Mat::ones(padded.size(), CV_32F)};
    Mat complexImg, complexImg1, complexInput;
    merge(planes, 2, complexImg);

    Mat realInput;
    padded.convertTo(realInput, CV_32F);
    complexInput = complexImg;
    //cout << complexImg << endl;
    //dft(complexImg, complexImg, DFT_REAL_OUTPUT);
    //cout << "Complex to Complex" << endl;
    //cout << complexImg << endl;
    cout << "Complex input" << endl << complexInput << endl;
    cout << "Real input" << endl << realInput << endl;

    dft(complexInput, complexImg1, DFT_COMPLEX_OUTPUT);
    cout << "Complex to Complex image: " << endl;
    cout << endl << complexImg1 << endl;
    
    Mat realImg1;
    dft(complexInput, realImg1, DFT_REAL_OUTPUT);
    cout << "Complex to Real image: " << endl;
    cout << endl << realImg1 << endl;

    Mat realOut;
    dft(complexImg1, realOut, DFT_INVERSE | DFT_COMPLEX_OUTPUT);
    cout << "Complex to Complex (inverse):" << endl;
    cout << realOut << endl;

    Mat complexOut;
    dft(realImg1, complexOut, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
    cout << "Complex to Real (inverse):" << endl;
    cout << complexOut << endl;

    // compute log(1 + sqrt(Re(DFT(img))**2 + Im(DFT(img))**2))
    split(complexImg, planes);
    magnitude(planes[0], planes[1], planes[0]);
    Mat mag = planes[0];
    mag += Scalar::all(1);
    log(mag, mag);

    // crop the spectrum, if it has an odd number of rows or columns
    mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));

    int cx = mag.cols/2;
    int cy = mag.rows/2;

    // rearrange the quadrants of Fourier image
    // so that the origin is at the image center
    Mat tmp;
    Mat q0(mag, Rect(0, 0, cx, cy));
    Mat q1(mag, Rect(cx, 0, cx, cy));
    Mat q2(mag, Rect(0, cy, cx, cy));
    Mat q3(mag, Rect(cx, cy, cx, cy));

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    normalize(mag, mag, 0, 1, NORM_MINMAX);

    imshow("spectrum magnitude", mag);
    waitKey();
    return 0;
}
