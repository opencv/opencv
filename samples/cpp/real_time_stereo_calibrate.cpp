/* This is a code by Team SAHE, India */

/* *************** Stereo Camera Calibration in Real Time **************

 This code can be used to calibrate stereo camera or two cameras to get the intrinsic
 and extrinsic files.
 
 This code also generated rectified image, and also shows RMS Error and Reprojection error
 to find the accuracy of calibration.
 
 This code captures stereo images from two different cameras (or stereo camera), whose
 index can be changed by changing the index in 'VideoCapture camLeft(0), camRight(2);'
 in the main() function.
 
 You can set no of stereo pairs you want to use bby editing 'noOfStereoPairs' global
 variable.
 
 Cheers
 Abhishek Upperwal
 
 ***********************************************************************/
 
 

/* *************** License:**************************
 3rd Oct, 2014
 Right to use this code in any way you want without warranty, support or any guarantee of it working.
 
 You can distribute, modify or regenerate this code without any warrenty or guarantee by the author.
 
 By: Abhishek Upperwal
 Country: India
 ************************************************** */

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>

using namespace cv;
using namespace std;

enum Modes { DETECTING, CAPTURING, CALIBRATING};
int mode = Modes::DETECTING;
const int noOfStereoPairs = 14;
int stereoPairIndex = 0, cornerImageIndex=0;
int goIn = 1;
Mat _leftOri, _rightOri;
int64 prevTickCount;

vector<Point2f> cornersLeft, cornersRight;

vector<vector<Point2f>> cameraImagePoints[2];

Mat displayCapturedImageIndex(Mat img) {
    std::ostringstream imageIndex;
    imageIndex<<stereoPairIndex<<"/"<<noOfStereoPairs;
    putText(img, imageIndex.str().c_str(), Point(50, 70), FONT_HERSHEY_PLAIN, 0.9, Scalar(0,0,255), 2);
    return img;
}

Mat displayMode(Mat img) {
    String modeString = "DETECTING";
    if (mode==Modes::CAPTURING) {
        modeString="CAPTURING";
    }
    else if (mode==Modes::CALIBRATING) {
        modeString="CALIBRATED";
    }
    putText(img, modeString, Point(50,50), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,0), 2);
    
    if (mode == Modes::CAPTURING) {
        img = displayCapturedImageIndex(img);
    }
    
    return img;
}

bool findChessboardCornersAndDraw(Mat inputLeft, Mat inputRight, Size boardSize) {
    
    
    
    _leftOri = inputLeft;
    _rightOri = inputRight;
    
    bool foundLeft = false, foundRight = false;
    
    cvtColor(inputLeft, inputLeft, COLOR_BGR2GRAY);
    cvtColor(inputRight, inputRight, COLOR_BGR2GRAY);
    foundLeft = findChessboardCorners(inputLeft, boardSize, cornersLeft, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
    foundRight = findChessboardCorners(inputRight, boardSize, cornersRight, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
    
    drawChessboardCorners(_leftOri, boardSize, cornersLeft, foundLeft);
    drawChessboardCorners(_rightOri, boardSize, cornersRight, foundRight);
    
    _leftOri = displayMode(_leftOri);
    _rightOri = displayMode(_rightOri);
    
    if (foundLeft && foundRight) {
        return true;
    }
    else {
        return false;
    }
}

void displayImages() {
    imshow("Left Image", _leftOri);
    imshow("Right Image", _rightOri);
}

void saveImages(Mat leftImage, Mat rightImage, int pairIndex) {
    
    cameraImagePoints[0].push_back(cornersLeft);
    cameraImagePoints[1].push_back(cornersRight);
    
    cvtColor(leftImage, leftImage, COLOR_BGR2GRAY);
    cvtColor(rightImage, rightImage, COLOR_BGR2GRAY);
    
    std::ostringstream leftString, rightString;
    
    leftString<<"left"<<pairIndex<<".jpg";
    rightString<<"right"<<pairIndex<<".jpg";
    
    imwrite(leftString.str().c_str(), leftImage);
    imwrite(rightString.str().c_str(), rightImage);
}

void calibrateStereoCamera(Size boardSize, Size imageSize) {
    
    vector<vector<Point3f>> objectPoints;
    
    objectPoints.resize(noOfStereoPairs);
    
    for (int i=0; i<noOfStereoPairs; i++) {
        for (int j=0; j<boardSize.height; j++) {
            for (int k=0; k<boardSize.width; k++) {
                objectPoints[i].push_back(Point3f(j,k,0));
            }
        }
    }
    
    Mat cameraMatrix[2], distCoeffs[2];
    cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
    cameraMatrix[1] = Mat::eye(3, 3, CV_64F);
    Mat R, T, E, F;
    double rms = stereoCalibrate(objectPoints, cameraImagePoints[0], cameraImagePoints[1],
                                 cameraMatrix[0], distCoeffs[0],
                                 cameraMatrix[1], distCoeffs[1],
                                 imageSize, R, T, E, F,
                                 CALIB_FIX_ASPECT_RATIO +
                                 CALIB_ZERO_TANGENT_DIST +
                                 CALIB_SAME_FOCAL_LENGTH +
                                 CALIB_RATIONAL_MODEL +
                                 CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
                                 TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-5) );
    
    cout<<"RMS Error: "<<rms<<"\n";
    
    
    double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];
    for(int i = 0; i < noOfStereoPairs; i++ )
    {
        int npt = (int)cameraImagePoints[0][i].size();
        Mat imgpt[2];
        for(int k = 0; k < 2; k++ )
        {
            imgpt[k] = Mat(cameraImagePoints[k][i]);
            undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
            computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
        }
        for(int j = 0; j < npt; j++ )
        {
            double errij = fabs(cameraImagePoints[0][i][j].x*lines[1][j][0] +
                                cameraImagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
            fabs(cameraImagePoints[1][i][j].x*lines[0][j][0] +
                 cameraImagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    cout << "Average Reprojection Error: " <<  err/npoints << endl;
    
    
    FileStorage fs("intrinsics.yml", FileStorage::WRITE);
    if (fs.isOpened()) {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
        "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
        fs.release();
    }
    else
        cout<<"Error: Could not open intrinsics file.";
    
    Mat R1, R2, P1, P2, Q;
    Rect validROI[2];
    
    stereoRectify(cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], imageSize, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, 1, imageSize, &validROI[0], &validROI[1]);
    
    fs.open("extrinsics.yml", FileStorage::WRITE);
    if (fs.isOpened()) {
        fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
        fs.release();
    }
    else
        cout<<"Error: Could not open extrinsics file";
    
    bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));
    
    Mat rmap[2][2];
    
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);
    
    
    Mat canvas;
    double sf;
    int w, h;
    
    if (!isVerticalStereo) {
        sf = 600./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h, w*2, CV_8UC3);
    }
    else {
        sf = 300./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h*2, w, CV_8UC3);
    }
    
    String file;
    
    namedWindow("rectified");
    
    for (int i=0; i < noOfStereoPairs; i++) {
        for (int j=0; j < 2; j++) {
            if (j==0) {
                file = "left";
            }
            else if (j==1) {
                file = "right";
            }
            ostringstream st;
            st<<file<<i+1<<".jpg";
            Mat img = imread(st.str().c_str()), rimg, cimg;
            remap(img, rimg, rmap[j][0], rmap[j][1], INTER_LINEAR);
            cimg=rimg;
            
            Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*j, 0, w, h)) : canvas(Rect(0, h*j, w, h));
            resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
        
            Rect vroi(cvRound(validROI[j].x*sf), cvRound(validROI[j].y*sf),
                      cvRound(validROI[j].width*sf), cvRound(validROI[j].height*sf));
            rectangle(canvasPart, vroi, Scalar(0,0,255), 3, 8);
        }
        if( !isVerticalStereo )
            for(int j = 0; j < canvas.rows; j += 16 )
                line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
        else
            for(int j = 0; j < canvas.cols; j += 16 )
                line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
        imshow("rectified", canvas);
    }
}

int main(int argc, char** argv) {
    
    VideoCapture camLeft(0), camRight(2);
    Size boardSize = Size(7,6);
    
    Mat inputLeft, inputRight, copyImageLeft, copyImageRight;
    bool foundCornersInBothImage = false;
    
    namedWindow("Left Image");
    namedWindow("Right Image");
    while (true) {
        camLeft>>inputLeft;
        camRight>>inputRight;
        
        inputLeft.copyTo(copyImageLeft);
        inputRight.copyTo(copyImageRight);
        
        foundCornersInBothImage = findChessboardCornersAndDraw(inputLeft, inputRight, boardSize);
        if (foundCornersInBothImage && mode == Modes::CAPTURING && stereoPairIndex<14) {
            int64 thisTick = getTickCount();
            int64 diff = thisTick - prevTickCount;
            if (goIn==1 || diff >= 3000000000) {
                goIn=0;
                saveImages(copyImageLeft, copyImageRight, ++stereoPairIndex);
                prevTickCount = getTickCount();
            }
        }
        displayImages();
        
        if (mode == Modes::CALIBRATING) {
            calibrateStereoCamera(boardSize, inputLeft.size());
            waitKey();
        }
        
        char keyBoardInput = (char)waitKey(50);
        if (keyBoardInput == 'q' || keyBoardInput == 'Q') {
            exit(-1);
        }
        else if(keyBoardInput == 'c' || keyBoardInput == 'C') {
            mode = Modes::CAPTURING;
        }
        else if (keyBoardInput == 'p' || keyBoardInput == 'P') {
            mode = Modes::CALIBRATING;
        }
    }
    return 0;
}