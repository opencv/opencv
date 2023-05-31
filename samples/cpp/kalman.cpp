#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/cvdef.h"
#include <stdio.h>

using namespace cv;

static inline Point calcPoint(Point2f center, double R, double angle)
{
    return center + Point2f((float)cos(angle), (float)-sin(angle))*(float)R;
}

static void help()
{
    printf( "\nExample of c calls to OpenCV's Kalman filter.\n"
"   Tracking of rotating point.\n"
"   Point moves in a circle and is characterized by a 1D state.\n"
"   state_k+1 = state_k + speed + process_noise N(0, 1e-5)\n"
"   The speed is constant.\n"
"   Both state and measurements vectors are 1D (a point angle),\n"
"   Measurement is the real state + gaussian noise N(0, 1e-1).\n"
"   The real and the measured points are connected with red line segment,\n"
"   the real and the estimated points are connected with yellow line segment,\n"
"   the real and the corrected estimated points are connected with green line segment.\n"
"   (if Kalman filter works correctly,\n"
"    the yellow segment should be shorter than the red one and\n"
"    the green segment should be shorter than the yellow one)."
            "\n"
"   Pressing any key (except ESC) will reset the tracking.\n"
"   Pressing ESC will stop the program.\n"
            );
}

int main(int, char**)
{
    help();
    Mat img(500, 500, CV_8UC3);
    KalmanFilter KF(2, 1, 0);
    Mat state(2, 1, CV_32F); /* (phi, delta_phi) */
    Mat processNoise(2, 1, CV_32F);
    Mat measurement = Mat::zeros(1, 1, CV_32F);
    char code = (char)-1;

    for(;;)
    {
        img = Scalar::all(0);
        state.at<float>(0) = 0.0f;
        state.at<float>(1) = 2.f * (float)CV_PI / 6;
        KF.transitionMatrix = (Mat_<float>(2, 2) << 1, 1, 0, 1);

        setIdentity(KF.measurementMatrix);
        setIdentity(KF.processNoiseCov, Scalar::all(1e-5));
        setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
        setIdentity(KF.errorCovPost, Scalar::all(1));

        randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));

        for(;;)
        {
            Point2f center(img.cols*0.5f, img.rows*0.5f);
            float R = img.cols/3.f;
            double stateAngle = state.at<float>(0);
            Point statePt = calcPoint(center, R, stateAngle);

            Mat prediction = KF.predict();
            double predictAngle = prediction.at<float>(0);
            Point predictPt = calcPoint(center, R, predictAngle);

            // generate measurement
            randn( measurement, Scalar::all(0), Scalar::all(KF.measurementNoiseCov.at<float>(0)));
            measurement += KF.measurementMatrix*state;

            double measAngle = measurement.at<float>(0);
            Point measPt = calcPoint(center, R, measAngle);

            // correct the state estimates based on measurements
            // updates statePost & errorCovPost
            KF.correct(measurement);
            double improvedAngle = KF.statePost.at<float>(0);
            Point improvedPt = calcPoint(center, R, improvedAngle);

            // plot points
            img = img * 0.2;
            drawMarker(img, measPt, Scalar(0, 0, 255), cv::MARKER_SQUARE, 5, 2);
            drawMarker(img, predictPt, Scalar(0, 255, 255), cv::MARKER_SQUARE, 5, 2);
            drawMarker(img, improvedPt, Scalar(0, 255, 0), cv::MARKER_SQUARE, 5, 2);
            drawMarker(img, statePt, Scalar(255, 255, 255), cv::MARKER_STAR, 10, 1);
            // forecast one step
            Mat test = Mat(KF.transitionMatrix*KF.statePost);
            drawMarker(img, calcPoint(center, R, Mat(KF.transitionMatrix*KF.statePost).at<float>(0)),
                       Scalar(255, 255, 0), cv::MARKER_SQUARE, 12, 1);

            line( img, statePt, measPt, Scalar(0,0,255), 1, LINE_AA, 0 );
            line( img, statePt, predictPt, Scalar(0,255,255), 1, LINE_AA, 0 );
            line( img, statePt, improvedPt, Scalar(0,255,0), 1, LINE_AA, 0 );


            randn( processNoise, Scalar(0), Scalar::all(sqrt(KF.processNoiseCov.at<float>(0, 0))));
            state = KF.transitionMatrix*state + processNoise;

            imshow( "Kalman", img );
            code = (char)waitKey(1000);

            if( code > 0 )
                break;
        }
        if( code == 27 || code == 'q' || code == 'Q' )
            break;
    }

    return 0;
}
