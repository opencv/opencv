/**
* @brief You will learn how port an existing algorithm to G-API
* @author Dmitry Matveev, dmitry.matveev@intel.com, based
*    on sample by Karpushin Vladislav, karpushin@ngs.ru
*/

#include <iostream>
#include <utility>

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/gapi.hpp"
#include "opencv2/gapi/core.hpp"
#include "opencv2/gapi/imgproc.hpp"

using namespace cv;
using namespace std;

//! [calcGST_proto]
void calcGST(const cv::GMat& inputImg, cv::GMat& imgCoherencyOut, cv::GMat& imgOrientationOut, int w);
//! [calcGST_proto]

void calcGST(const cv::Mat& inputImg, Mat& imgCoherencyOut, Mat& imgOrientationOut, int w);

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
    // Calculate Gradient Structure Tensor and post-process it for output with G-API
    cv::GMat in;
    cv::GMat imgCoherency, imgOrientation;
    calcGST(in, imgCoherency, imgOrientation, W);

    auto imgCoherencyBin = imgCoherency > C_Thr;
    auto imgOrientationBin = cv::gapi::inRange(imgOrientation, LowThr, HighThr);
    auto imgBin = imgCoherencyBin & imgOrientationBin;
    cv::GMat out = cv::gapi::addWeighted(in, 0.5, imgBin, 0.5, 0.0);

    // Capture the graph into object segm
    cv::GComputation segm(cv::GIn(in), cv::GOut(out, imgCoherency, imgOrientation));

    // Define cv::Mats for output data
    cv::Mat imgOut, imgOutCoherency, imgOutOrientation;

    // Run the graph
    segm.apply(cv::gin(imgIn), cv::gout(imgOut, imgOutCoherency, imgOutOrientation));

    // Normalize extra outputs (out of the graph)
    cv::normalize(imgOutCoherency, imgOutCoherency, 0, 255, NORM_MINMAX);
    cv::normalize(imgOutOrientation, imgOutOrientation, 0, 255, NORM_MINMAX);

    imwrite("result.jpg", imgOut);
    imwrite("Coherency.jpg", imgOutCoherency);
    imwrite("Orientation.jpg", imgOutOrientation);
    //! [main]

    return 0;
}
//! [calcGST]
//! [calcGST_header]
void calcGST(const cv::GMat& inputImg, cv::GMat& imgCoherencyOut, cv::GMat& imgOrientationOut, int w)
{
    auto img = cv::gapi::convertTo(inputImg, CV_32F);
    auto imgDiffX = cv::gapi::sobel(img, CV_32F, 1, 0, 3);
    auto imgDiffY = cv::gapi::sobel(img, CV_32F, 0, 1, 3);
    auto imgDiffXY = cv::gapi::mul(imgDiffX, imgDiffY, 1.0);
    //! [calcGST_header]

    auto imgDiffXX = cv::gapi::mul(imgDiffX, imgDiffX, 1.0);
    auto imgDiffYY = cv::gapi::mul(imgDiffY, imgDiffY, 1.0);

    auto J11 = cv::gapi::boxFilter(imgDiffXX, CV_32F, cv::Size(w, w));
    auto J22 = cv::gapi::boxFilter(imgDiffYY, CV_32F, cv::Size(w, w));
    auto J12 = cv::gapi::boxFilter(imgDiffXY, CV_32F, cv::Size(w, w));

    auto tmp1 = J11 + J22;
    auto tmp2 = J11 - J22;
    auto tmp22 = cv::gapi::mul(tmp2, tmp2, 1.0);
    auto tmp3 = cv::gapi::mul(J12, J12, 1.0);
    auto tmp4 = cv::gapi::sqrt(tmp22 + 4.0*tmp3);

    auto lambda1 = tmp1 + tmp4;
    auto lambda2 = tmp1 - tmp4;

    imgCoherencyOut = (lambda1 - lambda2) / (lambda1 + lambda2);
    imgOrientationOut = 0.5*cv::gapi::phase(J22 - J11, 2.0*J12, true);
}
//! [calcGST]

void calcGST(const Mat& inputImg, Mat& imgCoherencyOut, Mat& imgOrientationOut, int w)
{
    Mat img;
    inputImg.convertTo(img, CV_32F);

    // GST components calculation (start)
    // J =  (J11 J12; J12 J22) - GST
    Mat imgDiffX, imgDiffY, imgDiffXY;
    Sobel(img, imgDiffX, CV_32F, 1, 0, 3);
    Sobel(img, imgDiffY, CV_32F, 0, 1, 3);
    multiply(imgDiffX, imgDiffY, imgDiffXY);

    Mat imgDiffXX, imgDiffYY;
    multiply(imgDiffX, imgDiffX, imgDiffXX);
    multiply(imgDiffY, imgDiffY, imgDiffYY);

    Mat J11, J22, J12;      // J11, J22 and J12 are GST components
    boxFilter(imgDiffXX, J11, CV_32F, Size(w, w));
    boxFilter(imgDiffYY, J22, CV_32F, Size(w, w));
    boxFilter(imgDiffXY, J12, CV_32F, Size(w, w));
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
