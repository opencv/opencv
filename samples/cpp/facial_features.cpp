/*
 * Author: Samyak Datta (datta[dot]samyak[at]gmail.com)
 *
 * A program to detect facial feature points using
 * Haarcascade classifiers for face, eyes, nose and mouth
 *
 */

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <cstdio>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

// Functions for facial feature detection
static void help(char** argv);
static void detectFaces(Mat&, vector<Rect_<int> >&, string);
static void detectEyes(Mat&, vector<Rect_<int> >&, string);
static void detectNose(Mat&, vector<Rect_<int> >&, string);
static void detectMouth(Mat&, vector<Rect_<int> >&, string);
static void detectFacialFeaures(Mat&, const vector<Rect_<int> >, string, string, string);

string input_image_path;
string face_cascade_path, eye_cascade_path, nose_cascade_path, mouth_cascade_path;

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv,
            "{eyes||}{nose||}{mouth||}{help h||}{@image||}{@facexml||}");
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }
    input_image_path = parser.get<string>("@image");
    face_cascade_path = parser.get<string>("@facexml");
    eye_cascade_path = parser.has("eyes") ? parser.get<string>("eyes") : "";
    nose_cascade_path = parser.has("nose") ? parser.get<string>("nose") : "";
    mouth_cascade_path = parser.has("mouth") ? parser.get<string>("mouth") : "";
    if (input_image_path.empty() || face_cascade_path.empty())
    {
        cout << "IMAGE or FACE_CASCADE are not specified";
        return 1;
    }
    // Load image and cascade classifier files
    Mat image;
    image = imread(samples::findFile(input_image_path));

    // Detect faces and facial features
    vector<Rect_<int> > faces;
    detectFaces(image, faces, face_cascade_path);
    detectFacialFeaures(image, faces, eye_cascade_path, nose_cascade_path, mouth_cascade_path);

    imshow("Result", image);

    waitKey(0);
    return 0;
}

static void help(char** argv)
{
    cout << "\nThis file demonstrates facial feature points detection using Haarcascade classifiers.\n"
        "The program detects a face and eyes, nose and mouth inside the face."
        "The code has been tested on the Japanese Female Facial Expression (JAFFE) database and found"
        "to give reasonably accurate results. \n";

    cout << "\nUSAGE: " << argv[0] << " [IMAGE] [FACE_CASCADE] [OPTIONS]\n"
        "IMAGE\n\tPath to the image of a face taken as input.\n"
        "FACE_CASCSDE\n\t Path to a haarcascade classifier for face detection.\n"
        "OPTIONS: \nThere are 3 options available which are described in detail. There must be a "
        "space between the option and it's argument (All three options accept arguments).\n"
        "\t-eyes=<eyes_cascade> : Specify the haarcascade classifier for eye detection.\n"
        "\t-nose=<nose_cascade> : Specify the haarcascade classifier for nose detection.\n"
        "\t-mouth=<mouth-cascade> : Specify the haarcascade classifier for mouth detection.\n";


    cout << "EXAMPLE:\n"
        "(1) " << argv[0] << " image.jpg face.xml -eyes=eyes.xml -mouth=mouth.xml\n"
        "\tThis will detect the face, eyes and mouth in image.jpg.\n"
        "(2) " << argv[0] << " image.jpg face.xml -nose=nose.xml\n"
        "\tThis will detect the face and nose in image.jpg.\n"
        "(3) " << argv[0] << " image.jpg face.xml\n"
        "\tThis will detect only the face in image.jpg.\n";

    cout << " \n\nThe classifiers for face and eyes can be downloaded from : "
        " \nhttps://github.com/opencv/opencv/tree/master/data/haarcascades";

    cout << "\n\nThe classifiers for nose and mouth can be downloaded from : "
        " \nhttps://github.com/opencv/opencv_contrib/tree/master/modules/face/data/cascades\n";
}

static void detectFaces(Mat& img, vector<Rect_<int> >& faces, string cascade_path)
{
    CascadeClassifier face_cascade;
    face_cascade.load(samples::findFile(cascade_path));

    if (!face_cascade.empty())
        face_cascade.detectMultiScale(img, faces, 1.15, 3, 0|CASCADE_SCALE_IMAGE, Size(30, 30));
    return;
}

static void detectFacialFeaures(Mat& img, const vector<Rect_<int> > faces, string eye_cascade,
        string nose_cascade, string mouth_cascade)
{
    for(unsigned int i = 0; i < faces.size(); ++i)
    {
        // Mark the bounding box enclosing the face
        Rect face = faces[i];
        rectangle(img, Point(face.x, face.y), Point(face.x+face.width, face.y+face.height),
                Scalar(255, 0, 0), 1, 4);

        // Eyes, nose and mouth will be detected inside the face (region of interest)
        Mat ROI = img(Rect(face.x, face.y, face.width, face.height));

        // Check if all features (eyes, nose and mouth) are being detected
        bool is_full_detection = false;
        if( (!eye_cascade.empty()) && (!nose_cascade.empty()) && (!mouth_cascade.empty()) )
            is_full_detection = true;

        // Detect eyes if classifier provided by the user
        if(!eye_cascade.empty())
        {
            vector<Rect_<int> > eyes;
            detectEyes(ROI, eyes, eye_cascade);

            // Mark points corresponding to the centre of the eyes
            for(unsigned int j = 0; j < eyes.size(); ++j)
            {
                Rect e = eyes[j];
                circle(ROI, Point(e.x+e.width/2, e.y+e.height/2), 3, Scalar(0, 255, 0), -1, 8);
                /* rectangle(ROI, Point(e.x, e.y), Point(e.x+e.width, e.y+e.height),
                    Scalar(0, 255, 0), 1, 4); */
            }
        }

        // Detect nose if classifier provided by the user
        double nose_center_height = 0.0;
        if(!nose_cascade.empty())
        {
            vector<Rect_<int> > nose;
            detectNose(ROI, nose, nose_cascade);

            // Mark points corresponding to the centre (tip) of the nose
            for(unsigned int j = 0; j < nose.size(); ++j)
            {
                Rect n = nose[j];
                circle(ROI, Point(n.x+n.width/2, n.y+n.height/2), 3, Scalar(0, 255, 0), -1, 8);
                nose_center_height = (n.y + n.height/2);
            }
        }

        // Detect mouth if classifier provided by the user
        double mouth_center_height = 0.0;
        if(!mouth_cascade.empty())
        {
            vector<Rect_<int> > mouth;
            detectMouth(ROI, mouth, mouth_cascade);

            for(unsigned int j = 0; j < mouth.size(); ++j)
            {
                Rect m = mouth[j];
                mouth_center_height = (m.y + m.height/2);

                // The mouth should lie below the nose
                if( (is_full_detection) && (mouth_center_height > nose_center_height) )
                {
                    rectangle(ROI, Point(m.x, m.y), Point(m.x+m.width, m.y+m.height), Scalar(0, 255, 0), 1, 4);
                }
                else if( (is_full_detection) && (mouth_center_height <= nose_center_height) )
                    continue;
                else
                    rectangle(ROI, Point(m.x, m.y), Point(m.x+m.width, m.y+m.height), Scalar(0, 255, 0), 1, 4);
            }
        }

    }

    return;
}

static void detectEyes(Mat& img, vector<Rect_<int> >& eyes, string cascade_path)
{
    CascadeClassifier eyes_cascade;
    eyes_cascade.load(samples::findFile(cascade_path, !cascade_path.empty()));

    if (!eyes_cascade.empty())
        eyes_cascade.detectMultiScale(img, eyes, 1.20, 5, 0|CASCADE_SCALE_IMAGE, Size(30, 30));
    return;
}

static void detectNose(Mat& img, vector<Rect_<int> >& nose, string cascade_path)
{
    CascadeClassifier nose_cascade;
    nose_cascade.load(samples::findFile(cascade_path, !cascade_path.empty()));

    if (!nose_cascade.empty())
        nose_cascade.detectMultiScale(img, nose, 1.20, 5, 0|CASCADE_SCALE_IMAGE, Size(30, 30));
    return;
}

static void detectMouth(Mat& img, vector<Rect_<int> >& mouth, string cascade_path)
{
    CascadeClassifier mouth_cascade;
    mouth_cascade.load(samples::findFile(cascade_path, !cascade_path.empty()));

    if (!mouth_cascade.empty())
        mouth_cascade.detectMultiScale(img, mouth, 1.20, 5, 0|CASCADE_SCALE_IMAGE, Size(30, 30));
    return;
}
