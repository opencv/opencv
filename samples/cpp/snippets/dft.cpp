#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

static void convolveDFT(InputArray A, InputArray B, OutputArray C) {
    // Calculate the size of the output array
    int outputRows = A.rows() + B.rows() - 1;
    int outputCols = A.cols() + B.cols() - 1;

    // Reallocate the output array if needed
    C.create(outputRows, outputCols, A.type());

    Size dftSize;
    // Calculate the size of DFT transform
    dftSize.width = getOptimalDFTSize(A.cols() + B.cols() - 1);
    dftSize.height = getOptimalDFTSize(A.rows() + B.rows() - 1);

    // Allocate temporary buffers and initialize them with 0's
    Mat tempA(dftSize, A.type(), Scalar::all(0));
    Mat tempB(dftSize, B.type(), Scalar::all(0));

    // Copy A and B to the top-left corners of tempA and tempB, respectively
    Mat roiA(tempA, Rect(0, 0, A.cols(), A.rows()));
    A.copyTo(roiA);
    Mat roiB(tempB, Rect(0, 0, B.cols(), B.rows()));
    B.copyTo(roiB);

    // Now transform the padded A & B in-place;
    // use "nonzeroRows" hint for faster processing
    dft(tempA, tempA, 0, A.rows());
    dft(tempB, tempB, 0, B.rows());

    // Multiply the spectrums;
    // the function handles packed spectrum representations well
    mulSpectrums(tempA, tempB, tempA, 0);

    // Transform the product back from the frequency domain.
    // Even though all the result rows will be non-zero,
    // you need only the first C.rows of them, and thus you
    // pass nonzeroRows == C.rows
    dft(tempA, tempA, DFT_INVERSE + DFT_SCALE, C.rows());

    // Now copy the result back to C.
    tempA(Rect(0, 0, C.cols(), C.rows())).copyTo(C);

    // All the temporary buffers will be deallocated automatically
}
static void help(const char ** argv)
{
    printf("\nThis program demonstrates the use of convolution using discrete Fourier transform (DFT)\n"
           "An image is convolved with kernel filter using DFT.\n"
           "Usage:\n %s [input -- default lena.jpg]\n", argv[0]);
}

const char* keys =
{
    "{help h||}{@input|lena.jpg|input image file}"
};

int main(int argc, const char** argv) {
    // Load the image in grayscale
    help(argv);
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }
    string filename = parser.get<string>(0);
    Mat img = imread(samples::findFile(filename), IMREAD_GRAYSCALE);

    // Check if the image is loaded successfully
    if (img.empty()) {
        std::cerr << "Error: Image not loaded!" << std::endl;
        return -1;
    }

    // Convert the image to CV_32F
    Mat img_32f;
    img.convertTo(img_32f, CV_32F);

    float kernelData[9] = { 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9 }; // example of blur filter, can be changed to other filter as well.
    Mat kernel(3, 3, CV_32F, kernelData);

    // Perform convolution of the image with the sharpening kernel
    Mat result;
    convolveDFT(img_32f, kernel, result);

    // Normalize the result for better visualization
    normalize(result, result, 0, 255, NORM_MINMAX);

    // Convert result back to 8-bit for display
    Mat result_8u;
    result.convertTo(result_8u, CV_8U);

    // Display the images
    imshow("Original Image", img);
    imshow("Output Image", result_8u);

    waitKey(0); // Wait for a key press to close the windows
    return 0;
}
