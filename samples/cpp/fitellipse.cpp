/********************************************************************************
*
*
*  This program is demonstration for ellipse fitting. Program finds
*  contours and approximate it by ellipses.
*
*  Trackbar specify threshold parametr.
*
*  White lines is contours. Red lines is fitting ellipses.
*
*
*  Autor:  Denis Burenkov.
*
*
*
********************************************************************************/
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

static void help()
{
    cout <<
        "\nThis program is demonstration for ellipse fitting. The program finds\n"
        "contours and approximate it by ellipses. Three methods are used to find the \n"
        "elliptical fits: fitEllipse, fitEllipseAMS and fitEllipseDirect.\n"
        "Call:\n"
        "./fitellipse [image_name -- Default ../data/stuff.jpg]\n" << endl;
}

int sliderPos = 70;

Mat image;

bool fitElipseQ, fitElipseAMSQ, fitElipseDirectQ;
cv::Scalar fitElipseColor, fitElipseAMSColor, fitElipseDirectColor;

void drawEllipseWithBox(cv::Mat cimage, cv::RotatedRect box, cv::Scalar color);

void processImage(int, void*);

int main( int argc, char** argv )
{
    fitElipseQ       = true;
    fitElipseAMSQ    = true;
    fitElipseDirectQ = true;
    fitElipseColor       = Scalar(255,  0,  0, 200);
    fitElipseAMSColor    = Scalar(  0,255,  0, 200);
    fitElipseDirectColor = Scalar(  0,  0,255, 200);

    cv::CommandLineParser parser(argc, argv,
        "{help h||}{@image|../data/stuff.jpg|}"
    );
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    string filename = parser.get<string>("@image");
    image = imread(filename, 0);
    if( image.empty() )
    {
        cout << "Couldn't open image " << filename << "\n";
        return 0;
    }

    imshow("source", image);
    namedWindow("result", CV_WINDOW_NORMAL );

    // Create toolbars. HighGUI use.
    createTrackbar( "threshold", "result", &sliderPos, 255, processImage );

    processImage(0, 0);

    // Wait for a key stroke; the same function arranges events processing
    waitKey();
    return 0;
}

void drawEllipseWithBox(cv::Mat cimage, cv::RotatedRect box, cv::Scalar color)
{
    ellipse(cimage, box, color, 1, LINE_AA);
    //       ellipse(cimage, box.center, box.size*0.5f, box.angle, 0, 360, color, 1, LINE_AA);

    Point2f vtx[4];
    box.points(vtx);
    for( int j = 0; j < 4; j++ ){
        line(cimage, vtx[j], vtx[(j+1)%4], color, 1, LINE_AA);
    }
}

// Define trackbar callback functon. This function find contours,
// draw it and approximate it by ellipses.
void processImage(int /*h*/, void*)
{

    RotatedRect box, boxAMS, boxDirect;
    vector<vector<Point> > contours;
    Mat bimage = image >= sliderPos;

    findContours(bimage, contours, RETR_LIST, CHAIN_APPROX_NONE);

    Mat cimage = Mat::zeros(bimage.size(), CV_8UC3);

    Size textsize1 = getTextSize("openCV", FONT_HERSHEY_COMPLEX, 1, 1, 0);
    Size textsize2 = getTextSize("AMS", FONT_HERSHEY_COMPLEX, 1, 1, 0);
    Size textsize3 = getTextSize("Direct", FONT_HERSHEY_COMPLEX, 1, 1, 0);
    Point org1((cimage.cols - textsize1.width), (int)(1.3 * textsize1.height));
    Point org2((cimage.cols - textsize2.width), (int)(1.3 * textsize1.height + 1.3 * textsize2.height));
    Point org3((cimage.cols - textsize3.width), (int)(1.3 * textsize1.height + 1.3 * textsize2.height + 1.3 * textsize3.height));

    putText(cimage, "openCV", org1, FONT_HERSHEY_COMPLEX, 1, fitElipseColor, 1, LINE_8);
    putText(cimage, "AMS",    org2, FONT_HERSHEY_COMPLEX, 1, fitElipseAMSColor, 1, LINE_8);
    putText(cimage, "Direct", org3, FONT_HERSHEY_COMPLEX, 1, fitElipseDirectColor, 1, LINE_8);

    for(size_t i = 0; i < contours.size(); i++)
    {
        size_t count = contours[i].size();
        if( count < 6 )
            continue;

        Mat pointsf;
        Mat(contours[i]).convertTo(pointsf, CV_32F);
        if (fitElipseQ) {
            box = fitEllipse(pointsf);
            if( MAX(box.size.width, box.size.height) > MIN(box.size.width, box.size.height)*30 ){continue;};
        }
        if (fitElipseAMSQ) {
            boxAMS = fitEllipseAMS(pointsf);
            if( MAX(boxAMS.size.width, boxAMS.size.height) > MIN(boxAMS.size.width, boxAMS.size.height)*30 ){continue;};
        }
        if (fitElipseDirectQ) {
            boxDirect = fitEllipseDirect(pointsf);
            if( MAX(boxDirect.size.width, boxDirect.size.height) > MIN(boxDirect.size.width, boxDirect.size.height)*30 ){continue;};
        }

        drawContours(cimage, contours, (int)i, Scalar::all(255), 1, 8);

        if (fitElipseQ) {
            drawEllipseWithBox(cimage, box, fitElipseColor);
        }
        if (fitElipseAMSQ) {
            drawEllipseWithBox(cimage, boxAMS, fitElipseAMSColor);
        }
        if (fitElipseDirectQ) {
            drawEllipseWithBox(cimage, boxDirect, fitElipseDirectColor);
        }

    }

    imshow("result", cimage);
}
