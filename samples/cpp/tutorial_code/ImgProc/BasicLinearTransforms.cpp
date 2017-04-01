/**
 * @file BasicLinearTransforms.cpp
 * @brief Simple program to change contrast and brightness
 * @author OpenCV team
 */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

/**
 * @function main
 * @brief Main function
 */
int main( int, char** argv )
{
    //! [basic-linear-transform-parameters]
    double alpha = 1.0; /*< Simple contrast control */
    int beta = 0;       /*< Simple brightness control */
    //! [basic-linear-transform-parameters]

    /// Read image given by user
    //! [basic-linear-transform-load]
    Mat image = imread( argv[1] );
    //! [basic-linear-transform-load]
    //! [basic-linear-transform-output]
    Mat new_image = Mat::zeros( image.size(), image.type() );
    //! [basic-linear-transform-output]

    /// Initialize values
    cout << " Basic Linear Transforms " << endl;
    cout << "-------------------------" << endl;
    cout << "* Enter the alpha value [1.0-3.0]: "; cin >> alpha;
    cout << "* Enter the beta value [0-100]: ";    cin >> beta;

    /// Do the operation new_image(i,j) = alpha*image(i,j) + beta
    /// Instead of these 'for' loops we could have used simply:
    /// image.convertTo(new_image, -1, alpha, beta);
    /// but we wanted to show you how to access the pixels :)
    //! [basic-linear-transform-operation]
    for( int y = 0; y < image.rows; y++ ) {
        for( int x = 0; x < image.cols; x++ ) {
            for( int c = 0; c < 3; c++ ) {
                new_image.at<Vec3b>(y,x)[c] =
                  saturate_cast<uchar>( alpha*( image.at<Vec3b>(y,x)[c] ) + beta );
            }
        }
    }
    //! [basic-linear-transform-operation]

    //! [basic-linear-transform-display]
    /// Create Windows
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    namedWindow("New Image", WINDOW_AUTOSIZE);

    /// Show stuff
    imshow("Original Image", image);
    imshow("New Image", new_image);

    /// Wait until user press some key
    waitKey();
    //! [basic-linear-transform-display]
    return 0;
}
