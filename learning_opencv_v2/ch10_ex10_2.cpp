// Example 10-2. Kalman filter sample code
//
//  Use Kalman Filter to model particle in circular trajectory.
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
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;


#define phi2xy(mat)                                             \
  Point( cvRound(img.cols/2 + img.cols/3*cos(mat.at<float>(0))),\
    cvRound( img.rows/2 - img.cols/3*sin(mat.at<float>(0))))

void help()
{
	cout << "Demonstrate use of Kalman filter" << endl;
	cout << "Usage: ./ch10_ex10_2\n" << endl;
}

int main(int argc, char** argv) {
	help();
    // Initialize, create Kalman Filter object, window, random number
    // generator etc.
    //
    Mat img(500, 500, CV_8UC3);
    KalmanFilter kalman(2, 1, 0);
    // state is (phi, delta_phi) - angle and angular velocity
    // Initialize with random guess.
    //
    Mat x_k(2, 1, CV_32F);
    randn(x_k, 0., 0.1);

    // process noise
    //
    Mat w_k(2, 1, CV_32F);
    
    // measurements, only one parameter for angle
    //
    Mat z_k = Mat::zeros(1, 1, CV_32F);

    // Transition matrix 'F' describes relationship between
    // model parameters at step k and at step k+1 (this is 
    // the "dynamics" in our model.
    //
    float F[] = { 1, 1, 0, 1 };
    kalman.transitionMatrix = Mat(2, 2, CV_32F, F).clone();
    // Initialize other Kalman filter parameters.
    //
    setIdentity( kalman.measurementMatrix,   Scalar(1) );
    setIdentity( kalman.processNoiseCov,     Scalar(1e-5) );
    setIdentity( kalman.measurementNoiseCov, Scalar(1e-1) );
    setIdentity( kalman.errorCovPost,        Scalar(1));

    // choose random initial state
    //
    randn(kalman.statePost, 0., 0.1);

    for(;;) {
        // predict point position
        Mat y_k = kalman.predict();

        // generate measurement (z_k)
        //
        randn(z_k, 0., sqrt((double)kalman.measurementNoiseCov.at<float>(0,0)));
        z_k = kalman.measurementMatrix*x_k + z_k;
        // plot points (eg convert to planar co-ordinates and draw)
        //
        img = Scalar::all(0);
        circle( img, phi2xy(z_k), 4, Scalar(128,255,255) );   // observed state
        circle( img, phi2xy(y_k), 4, Scalar(255,255,255), 2 ); // "predicted" state
        circle( img, phi2xy(x_k), 4, Scalar(0,0,255) );      // real state
        imshow( "Kalman", img );
        // adjust Kalman filter state
        //
        kalman.correct( z_k );

        // Apply the transition matrix 'F' (eg, step time forward)
        // and also apply the "process" noise w_k.
        //
        randn(w_k, 0., sqrt((double)kalman.processNoiseCov.at<float>(0,0)));
        x_k = kalman.transitionMatrix*x_k + w_k;
        
        // exit if user hits 'Esc'
        if( (waitKey( 100 )&255) == 27 ) break;
    }

    return 0;
}
