/**
* @brief You will learn how port an existing algorithm to G-API
* @author Dmitry Matveev, dmitry.matveev@intel.com, based
*    on sample by Karpushin Vladislav, karpushin@ngs.ru
*/
#include "opencv2/opencv_modules.hpp"
#ifdef HAVE_OPENCV_GAPI

//! [full_sample]
#include <iostream>
#include <utility>

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/gapi.hpp"
#include "opencv2/gapi/core.hpp"
#include "opencv2/gapi/imgproc.hpp"
//! [fluid_includes]
#include "opencv2/gapi/fluid/core.hpp"            // Fluid Core kernel library
#include "opencv2/gapi/fluid/imgproc.hpp"         // Fluid ImgProc kernel library
//! [fluid_includes]
#include "opencv2/gapi/fluid/gfluidkernel.hpp"    // Fluid user kernel API

//! [calcGST_proto]
void calcGST(const cv::GMat& inputImg, cv::GMat& imgCoherencyOut, cv::GMat& imgOrientationOut, int w);
//! [calcGST_proto]

int main()
{
    int W = 52;             // window size is WxW
    double C_Thr = 0.43;    // threshold for coherency
    int LowThr = 35;        // threshold1 for orientation, it ranges from 0 to 180
    int HighThr = 57;       // threshold2 for orientation, it ranges from 0 to 180

    cv::Mat imgIn = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE);
    if (imgIn.empty()) //check whether the image is loaded or not
    {
        std::cout << "ERROR : Image cannot be loaded..!!" << std::endl;
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

    // Normalize extra outputs
    cv::GMat imgCoherencyNorm = cv::gapi::normalize(imgCoherency, 0, 255, cv::NORM_MINMAX);
    cv::GMat imgOrientationNorm = cv::gapi::normalize(imgOrientation, 0, 255, cv::NORM_MINMAX);

    // Capture the graph into object segm
    cv::GComputation segm(cv::GIn(in), cv::GOut(out, imgCoherencyNorm, imgOrientationNorm));

    // Define cv::Mats for output data
    cv::Mat imgOut, imgOutCoherency, imgOutOrientation;

    //! [kernel_pkg_proper]
    //! [kernel_pkg]
    // Prepare the kernel package and run the graph
    cv::gapi::GKernelPackage fluid_kernels = cv::gapi::combine        // Define a custom kernel package:
        (cv::gapi::core::fluid::kernels(),                            // ...with Fluid Core kernels
         cv::gapi::imgproc::fluid::kernels());                        // ...and Fluid ImgProc kernels
    //! [kernel_pkg]
    //! [kernel_hotfix]
    fluid_kernels.remove<cv::gapi::imgproc::GBoxFilter>();            // Remove Fluid Box filter as unsuitable,
                                                                      // G-API will fall-back to OpenCV there.
    //! [kernel_hotfix]
    //! [kernel_pkg_use]
    segm.apply(cv::gin(imgIn),                                        // Input data vector
               cv::gout(imgOut, imgOutCoherency, imgOutOrientation),  // Output data vector
               cv::compile_args(fluid_kernels));                      // Kernel package to use
    //! [kernel_pkg_use]
    //! [kernel_pkg_proper]

    cv::imwrite("result.jpg", imgOut);
    cv::imwrite("Coherency.jpg", imgOutCoherency);
    cv::imwrite("Orientation.jpg", imgOutOrientation);
    //! [main]

    return 0;
}
//! [calcGST]
//! [calcGST_header]
void calcGST(const cv::GMat& inputImg, cv::GMat& imgCoherencyOut, cv::GMat& imgOrientationOut, int w)
{
    auto img = cv::gapi::convertTo(inputImg, CV_32F);
    auto imgDiffX = cv::gapi::Sobel(img, CV_32F, 1, 0, 3);
    auto imgDiffY = cv::gapi::Sobel(img, CV_32F, 0, 1, 3);
    auto imgDiffXY = cv::gapi::mul(imgDiffX, imgDiffY);
    //! [calcGST_header]

    auto imgDiffXX = cv::gapi::mul(imgDiffX, imgDiffX);
    auto imgDiffYY = cv::gapi::mul(imgDiffY, imgDiffY);

    auto J11 = cv::gapi::boxFilter(imgDiffXX, CV_32F, cv::Size(w, w));
    auto J22 = cv::gapi::boxFilter(imgDiffYY, CV_32F, cv::Size(w, w));
    auto J12 = cv::gapi::boxFilter(imgDiffXY, CV_32F, cv::Size(w, w));

    auto tmp1 = J11 + J22;
    auto tmp2 = J11 - J22;
    auto tmp22 = cv::gapi::mul(tmp2, tmp2);
    auto tmp3 = cv::gapi::mul(J12, J12);
    auto tmp4 = cv::gapi::sqrt(tmp22 + 4.0*tmp3);

    auto lambda1 = tmp1 + tmp4;
    auto lambda2 = tmp1 - tmp4;

    imgCoherencyOut = (lambda1 - lambda2) / (lambda1 + lambda2);
    imgOrientationOut = 0.5*cv::gapi::phase(J22 - J11, 2.0*J12, true);
}
//! [calcGST]

//! [full_sample]

#else
#include <iostream>
int main()
{
    std::cerr << "This tutorial code requires G-API module to run" << std::endl;
}
#endif  // HAVE_OPECV_GAPI
