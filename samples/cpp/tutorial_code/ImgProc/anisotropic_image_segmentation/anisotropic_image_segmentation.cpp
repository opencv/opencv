/**
* @brief You will learn how to segment an anisotropic image with a single local orientation by a gradient structure tensor (GST)
* @author Karpushin Vladislav, karpushin@ngs.ru, https://github.com/VladKarpushin
*/

#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

using namespace cv;
using namespace std;

void calcGST(const Mat& inputImg, Mat& imgCoherencyOut, Mat& imgOrientationOut, int w);

int main()
{
    int W = 52;             // window size is WxW
    double C_Thr = 0.43;    // threshold for coherency
    int LowThr = 35;        // threshold1 for orientation, it ranges from 0 to 180
    int HighThr = 57;       // threshold2 for orientation, it ranges from 0 to 180

    Mat imgIn = imread("input.jpg", IMREAD_GRAYSCALE);
    if (imgIn.empty()) //check whether the image is loaded or not
    {
        cout << "ERROR : Image cannot be loaded..!!" << endl;
        return -1;
    }

    //! [main]
    Mat imgCoherency, imgOrientation;
    calcGST(imgIn, imgCoherency, imgOrientation, W);

    //! [thresholding]
    Mat imgCoherencyBin;
    imgCoherencyBin = imgCoherency > C_Thr;
    Mat imgOrientationBin;
    inRange(imgOrientation, Scalar(LowThr), Scalar(HighThr), imgOrientationBin);
    //! [thresholding]

    //! [combining]
    Mat imgBin;
    imgBin = imgCoherencyBin & imgOrientationBin;
    //! [combining]
    //! [main]

    normalize(imgCoherency, imgCoherency, 0, 255, NORM_MINMAX);
    normalize(imgOrientation, imgOrientation, 0, 255, NORM_MINMAX);
    imwrite("result.jpg", 0.5*(imgIn + imgBin));
    imwrite("Coherency.jpg", imgCoherency);
    imwrite("Orientation.jpg", imgOrientation);
    return 0;
}
//! [calcGST]
void calcGST(const Mat& inputImg, Mat& imgCoherencyOut, Mat& imgOrientationOut, int w)
{
    Mat img;
    inputImg.convertTo(img, CV_64F);

    // GST components calculation (start)
    // J =  (J11 J12; J12 J22) - GST
    Mat imgDiffX, imgDiffY, imgDiffXY;
    Sobel(img, imgDiffX, CV_64F, 1, 0, 3);
    Sobel(img, imgDiffY, CV_64F, 0, 1, 3);
    multiply(imgDiffX, imgDiffY, imgDiffXY);

    Mat imgDiffXX, imgDiffYY;
    multiply(imgDiffX, imgDiffX, imgDiffXX);
    multiply(imgDiffY, imgDiffY, imgDiffYY);

    Mat J11, J22, J12;      // J11, J22 and J12 are GST components
    boxFilter(imgDiffXX, J11, CV_64F, Size(w, w));
    boxFilter(imgDiffYY, J22, CV_64F, Size(w, w));
    boxFilter(imgDiffXY, J12, CV_64F, Size(w, w));
    // GST components calculation (stop)

    // eigenvalue calculation (start)
    // lambda1 = J11 + J22 + sqrt((J11-J22)^2 + 4*J12^2)
    // lambda2 = J11 + J22 - sqrt((J11-J22)^2 + 4*J12^2)
    Mat tmp1, tmp2, tmp3, tmp4;
    tmp1 = J11 + J22;
    tmp2 = J11 - J22;
    multiply(tmp2, tmp2, tmp2);
    multiply(J12, J12, tmp3);
    sqrt(tmp2 + 4.0 * tmp3, tmp4);

    Mat lambda1, lambda2;
    lambda1 = tmp1 + tmp4;      // biggest eigenvalue
    lambda2 = tmp1 - tmp4;      // smallest eigenvalue
    // eigenvalue calculation (stop)

    // Coherency calculation (start)
    // Coherency = (lambda1 - lambda2)/(lambda1 + lambda2)) - measure of anisotropism
    // Coherency is anisotropy degree (consistency of local orientation)
    divide(lambda1 - lambda2, lambda1 + lambda2, imgCoherencyOut);
    // Coherency calculation (stop)

    // orientation angle calculation (start)
    // tan(2*Alpha) = 2*J12/(J22 - J11)
    // Alpha = 0.5 atan2(2*J12/(J22 - J11))
    phase(J22 - J11, 2.0*J12, imgOrientationOut, true);
    imgOrientationOut = 0.5*imgOrientationOut;
    // orientation angle calculation (stop)
}
//! [calcGST]
