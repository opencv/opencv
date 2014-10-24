/*
 * Author: Samyak Datta (datta[dot]samyak[at]gmail.com)
 * 
 * A program to detect facial feature points using 
 * Haarcascade classifiers for face, eyes, nose and mouth
 *
 */

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
 
#include <iostream>
#include <cstdio>
 
using namespace std;
using namespace cv;

static void help();
static void detectFaces(Mat&, vector<Rect_<int> >&, string);
static void detectEyes(Mat&, vector<Rect_<int> >&, string);
static void detectNose(Mat&, vector<Rect_<int> >&, string);
static void detectMouth(Mat&, vector<Rect_<int> >&, string);
static void detectFacialFeaures(Mat&, const vector<Rect_<int> >, string, string, string);

string input_image_path;
string face_cascade_path = "../../data/haarcascades/haarcascade_frontalface_default.xml"; 
string eye_cascade_path  = "../../data/haarcascades/haarcascade_eye.xml";
string nose_cascade_path, mouth_cascade_path;

int main(int argc, char** argv)
{
    /* Parse command line arguments: 
     *  (1) Path to input image
     *  (2) Path to cascade file for nose detection
     *  (3) Path to cascade file for mouth detection
     */
    
    if(argc < 4) 
    {
        help();
        exit(1);
    }
    
    input_image_path = argv[1];
    nose_cascade_path = argv[2];
    mouth_cascade_path = argv[3];
    
    // Load image and cascade classifier files
    Mat image;
    image = imread(input_image_path, CV_LOAD_IMAGE_UNCHANGED); 
 
    // Detect faces and facial features
    vector<Rect_<int> > faces;
    detectFaces(image, faces, face_cascade_path);
    detectFacialFeaures(image, faces, eye_cascade_path, nose_cascade_path, mouth_cascade_path);
    
    imshow("Result", image);
 
    waitKey(0);                  
    return 0;
}

static void help()
{
    cout << "\nThis file demonstrates facial feature points detection using Haarcascade classifiers."
        " The program detects a face and eyes, nose and mouth inside the face. The code has been"
        " tested on the Japanese Female Facial Expression (JAFFE) database and found to give"
        " reasonably accurate results. \n";
    
    cout << "\nUsage: ./face_detector <input_image> <nose_cascade> <mouth_cascade>\n"
        "<nose_cascade> and <mouth_cascade> are paths to the Haarcascade classifiers for"
        " nose and mouth detection respectively. They have been removed from the "
        " data/haarcascade/ directory and can be downloaded separately from: "
        " https://github.com/Itseez/opencv_contrib/tree/master/modules/face/data/cascades\n";
}

static void detectFaces(Mat& img, vector<Rect_<int> >& faces, string cascade_path) 
{
    CascadeClassifier face_cascade;
    face_cascade.load(cascade_path);

    face_cascade.detectMultiScale(img, faces, 1.15, 3, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30));
    return;
}

static void detectFacialFeaures(Mat& img, const vector<Rect_<int> > faces, string eye_cascade_path,
        string nose_cascade_path, string mouth_cascade_path) 
{
    for(int i = 0; i < faces.size(); ++i) 
    {
        Rect face = faces[i];
        rectangle(img, Point(face.x, face.y), Point(face.x+face.width, face.y+face.height),
                Scalar(255, 0, 0), 1, 4);
        
        Mat ROI = img(Rect(face.x, face.y, face.width, face.height));
        
        vector<Rect_<int> > eyes, nose, mouth;
        detectEyes(ROI, eyes, eye_cascade_path);
        detectNose(ROI, nose, nose_cascade_path);
        detectMouth(ROI, mouth, mouth_cascade_path);
        
        // Mark points corresponding to the centre of the eyes
        for(int j = 0; j < eyes.size(); ++j) 
        {
            Rect e = eyes[j];
            circle(ROI, Point(e.x+e.width/2, e.y+e.height/2), 3.0, Scalar(0, 255, 0), -1, 8);
            /* rectangle(ROI, Point(e.x, e.y), Point(e.x+e.width, e.y+e.height), 
                    Scalar(0, 255, 0), 1, 4); */
        }
        
        // Mark points corresponding to the centre (tip) of the nose
        int nose_center_height;
        for(int j = 0; j < nose.size(); ++j) 
        {
            Rect n = nose[j];
            nose_center_height = (n.y + n.height/2);
            circle(ROI, Point(n.x+n.width/2, n.y+n.height/2), 3.0, Scalar(0, 255, 0), -1, 8);
        }

        int mouth_center_height;
        for(int j = 0; j < mouth.size(); ++j) 
        {
            Rect m = mouth[j];
            mouth_center_height = (m.y + m.height/2);
            if(mouth_center_height > nose_center_height)
                rectangle(ROI, Point(m.x, m.y), Point(m.x+m.width, m.y+m.height), Scalar(0, 255, 0), 1, 4);
        }
    }
    return;
}

static void detectEyes(Mat& img, vector<Rect_<int> >& eyes, string cascade_path) 
{
    CascadeClassifier eyes_cascade;
    eyes_cascade.load(cascade_path);

    eyes_cascade.detectMultiScale(img, eyes, 1.20, 5, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30));
    return;
}

static void detectNose(Mat& img, vector<Rect_<int> >& nose, string cascade_path) 
{
    CascadeClassifier nose_cascade;
    nose_cascade.load(cascade_path);

    nose_cascade.detectMultiScale(img, nose, 1.20, 5, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30));
    return;
}

static void detectMouth(Mat& img, vector<Rect_<int> >& mouth, string cascade_path) 
{
    CascadeClassifier mouth_cascade;
    mouth_cascade.load(cascade_path);

    mouth_cascade.detectMultiScale(img, mouth, 1.20, 5, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30));
    return;
}

