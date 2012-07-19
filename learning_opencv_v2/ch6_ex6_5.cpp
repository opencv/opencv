
//
// Example 6-5. Use of cvDFT() to accelerate the computation of convolutions
// Use DFT to accelerate the convolution of array A by kernel B.
// Place the result in array V.
//
/* License:
   July 20, 2011
   Standard BSD

   BOOK: It would be nice if you cited it:
   Learning OpenCV 2: Computer Vision with the OpenCV Library
     by Gary Bradski and Adrian Kaehler
     Published by O'Reilly Media
 
   AVAILABLE AT: 
     http://www.amazon.com/Learning-OpenCV-Computer-Vision-Library/dp/0596516134
     Or: http://oreilly.com/catalog/9780596516130/
     ISBN-10: 0596516134 or: ISBN-13: 978-0596516130    

   Main OpenCV site
   http://opencv.willowgarage.com/wiki/
   * An active user group is at:
     http://tech.groups.yahoo.com/group/OpenCV/
   * The minutes of weekly OpenCV development meetings are at:
     http://pr.willowgarage.com/wiki/OpenCV
*/
//

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    if(argc != 2) { cout << "Fourier Transform\nUsage: ch6_ex6_5 <imagename>" << endl; return -1; }
    
    Mat A = imread(argv[1],0);
    
    if( A.empty() ) { cout << "Can not load " << argv[1] << endl; return -1; } 
    
    Size patchSize(100, 100);
    Point topleft(A.cols/2, A.rows/2);
    Rect roi(topleft.x, topleft.y, patchSize.width, patchSize.height);
    Mat B = A(roi);
    
    int dft_M = getOptimalDFTSize( A.rows+B.rows-1 );
    int dft_N = getOptimalDFTSize( A.cols+B.cols-1 );

    Mat dft_A = Mat::zeros(dft_M, dft_N, CV_32F);
    Mat dft_B = Mat::zeros(dft_M, dft_N, CV_32F);
    
    Mat dft_A_part = dft_A(Rect(0, 0, A.cols,A.rows));
    A.convertTo(dft_A_part, dft_A_part.type(), 1, -mean(A)[0]);
    Mat dft_B_part = dft_B(Rect(0, 0, B.cols,B.rows));
    B.convertTo(dft_B_part, dft_B_part.type(), 1, -mean(B)[0]);
    
    dft(dft_A, dft_A, 0, A.rows);
    dft(dft_B, dft_B, 0, B.rows);
    
    // set the last parameter to false to compute convolution instead of correlation
    mulSpectrums( dft_A, dft_B, dft_A, 0, true );
    idft(dft_A, dft_A, DFT_SCALE, A.rows + B.rows - 1 );
    
    Mat corr = dft_A(Rect(0, 0, A.cols + B.cols - 1, A.rows + B.rows - 1));
    normalize(corr, corr, 0, 1, NORM_MINMAX, corr.type());
    pow(corr, 3., corr);
    
    B ^= Scalar::all(255);

    imshow("Image", A);
    imshow("Correlation", corr);
    waitKey();
    return 0;
}

