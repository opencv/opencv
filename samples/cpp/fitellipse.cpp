/********************************************************************************
 *
 *
 *  This program is demonstration for ellipse fitting. Program finds
 *  contours and approximate it by ellipses using three methods.
 *  1: OpenCV's original method fitEllipse which implements Fitzgibbon 1995 method.
 *  2: The Approximate Mean Square (AMS) method fitEllipseAMS  proposed by Taubin 1991
 *  3: The Direct least square (Direct) method fitEllipseDirect proposed by oy1998NumericallySD.
 *
 *  Trackbar specify threshold parameter.
 *
 *  White lines is contours/input points and the true ellipse used to generate the data.
 *  1: Blue lines is fitting ellipses using openCV's original method.
 *  2: Green lines is fitting ellipses using the AMS method.
 *  3: Red lines is fitting ellipses using the Direct method.
 *
 *
 *  Original Author:  Denis Burenkov
 *  AMS and Direct Methods Author:  Jasper Shemilt
 *
 *
 ********************************************************************************/
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

class canvas{
public:
    bool setupQ;
    cv::Point origin;
    cv::Point corner;
    int minDims,maxDims;
    double scale;
    int rows, cols;
    cv::Mat img;

    void init(int minD, int maxD){
        // Initialise the canvas with minimum and maximum rows and column sizes.
        minDims = minD; maxDims = maxD;
        origin = cv::Point(0,0);
        corner = cv::Point(0,0);
        scale = 1.0;
        rows = 0;
        cols = 0;
        setupQ = false;
    }

    void stretch(cv::Point2f min, cv::Point2f max){
        // Stretch the canvas to include the points min and max.
        if(setupQ){
            if(corner.x < max.x){corner.x = (int)(max.x + 1.0);};
            if(corner.y < max.y){corner.y = (int)(max.y + 1.0);};
            if(origin.x > min.x){origin.x = (int) min.x;};
            if(origin.y > min.y){origin.y = (int) min.y;};
        } else {
            origin = cv::Point((int)min.x, (int)min.y);
            corner = cv::Point((int)(max.x + 1.0), (int)(max.y + 1.0));
        }

        int c = (int)(scale*((corner.x + 1.0) - origin.x));
        if(c<minDims){
            scale = scale * (double)minDims/(double)c;
        } else {
            if(c>maxDims){
                scale = scale * (double)maxDims/(double)c;
            }
        }
        int r = (int)(scale*((corner.y + 1.0) - origin.y));
        if(r<minDims){
            scale = scale * (double)minDims/(double)r;
        } else {
            if(r>maxDims){
                scale = scale * (double)maxDims/(double)r;
            }
        }
        cols = (int)(scale*((corner.x + 1.0) - origin.x));
        rows = (int)(scale*((corner.y + 1.0) - origin.y));
        setupQ = true;
    }

    void stretch(vector<Point2f> pts)
    {   // Stretch the canvas so all the points pts are on the canvas.
        cv::Point2f min = pts[0];
        cv::Point2f max = pts[0];
        for(size_t i=1; i < pts.size(); i++){
            Point2f pnt = pts[i];
            if(max.x < pnt.x){max.x = pnt.x;};
            if(max.y < pnt.y){max.y = pnt.y;};
            if(min.x > pnt.x){min.x = pnt.x;};
            if(min.y > pnt.y){min.y = pnt.y;};
        };
        stretch(min, max);
    }

    void stretch(cv::RotatedRect box)
    {   // Stretch the canvas so that the rectangle box is on the canvas.
        cv::Point2f min = box.center;
        cv::Point2f max = box.center;
        cv::Point2f vtx[4];
        box.points(vtx);
        for( int i = 0; i < 4; i++ ){
            cv::Point2f pnt = vtx[i];
            if(max.x < pnt.x){max.x = pnt.x;};
            if(max.y < pnt.y){max.y = pnt.y;};
            if(min.x > pnt.x){min.x = pnt.x;};
            if(min.y > pnt.y){min.y = pnt.y;};
        }
        stretch(min, max);
    }

    void drawEllipseWithBox(cv::RotatedRect box, cv::Scalar color, int lineThickness)
    {
        if(img.empty()){
            stretch(box);
            img = cv::Mat::zeros(rows,cols,CV_8UC3);
        }

        box.center = scale * cv::Point2f(box.center.x - origin.x, box.center.y - origin.y);
        box.size.width  = (float)(scale * box.size.width);
        box.size.height = (float)(scale * box.size.height);

        ellipse(img, box, color, lineThickness, LINE_AA);

        Point2f vtx[4];
        box.points(vtx);
        for( int j = 0; j < 4; j++ ){
            line(img, vtx[j], vtx[(j+1)%4], color, lineThickness, LINE_AA);
        }
    }

    void drawPoints(vector<Point2f> pts, cv::Scalar color)
    {
        if(img.empty()){
            stretch(pts);
            img = cv::Mat::zeros(rows,cols,CV_8UC3);
        }
        for(size_t i=0; i < pts.size(); i++){
            Point2f pnt = scale * cv::Point2f(pts[i].x - origin.x, pts[i].y - origin.y);
            img.at<cv::Vec3b>(int(pnt.y), int(pnt.x))[0] = (uchar)color[0];
            img.at<cv::Vec3b>(int(pnt.y), int(pnt.x))[1] = (uchar)color[1];
            img.at<cv::Vec3b>(int(pnt.y), int(pnt.x))[2] = (uchar)color[2];
        };
    }

    void drawLabels( std::vector<std::string> text, std::vector<cv::Scalar> colors)
    {
        if(img.empty()){
            img = cv::Mat::zeros(rows,cols,CV_8UC3);
        }
        int vPos = 0;
        for (size_t i=0; i < text.size(); i++) {
            cv::Scalar color = colors[i];
            std::string txt = text[i];
            Size textsize = getTextSize(txt, FONT_HERSHEY_COMPLEX, 1, 1, 0);
            vPos += (int)(1.3 * textsize.height);
            Point org((img.cols - textsize.width), vPos);
            cv::putText(img, txt, org, FONT_HERSHEY_COMPLEX, 1, color, 1, LINE_8);
        }
    }

};

static void help(char** argv)
{
    cout << "\nThis program is demonstration for ellipse fitting. The program finds\n"
            "contours and approximate it by ellipses. Three methods are used to find the \n"
            "elliptical fits: fitEllipse, fitEllipseAMS and fitEllipseDirect.\n"
            "Call:\n"
        << argv[0] << " [image_name -- Default ellipses.jpg]\n" << endl;
}

int sliderPos = 70;

Mat image;

bool fitEllipseQ, fitEllipseAMSQ, fitEllipseDirectQ;
cv::Scalar fitEllipseColor       = Scalar(255,  0,  0);
cv::Scalar fitEllipseAMSColor    = Scalar(  0,255,  0);
cv::Scalar fitEllipseDirectColor = Scalar(  0,  0,255);
cv::Scalar fitEllipseTrueColor   = Scalar(255,255,255);

void processImage(int, void*);

int main( int argc, char** argv )
{
    fitEllipseQ       = true;
    fitEllipseAMSQ    = true;
    fitEllipseDirectQ = true;

    cv::CommandLineParser parser(argc, argv,"{help h||}{@image|ellipses.jpg|}");
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }
    string filename = parser.get<string>("@image");
    image = imread(samples::findFile(filename), 0);
    if( image.empty() )
    {
        cout << "Couldn't open image " << filename << "\n";
        return 0;
    }

    imshow("source", image);
    namedWindow("result", WINDOW_NORMAL );

    // Create toolbars. HighGUI use.
    createTrackbar( "threshold", "result", &sliderPos, 255, processImage );

    processImage(0, 0);

    // Wait for a key stroke; the same function arranges events processing
    waitKey();
    return 0;
}

inline static bool isGoodBox(const RotatedRect& box) {
    //size.height >= size.width awalys,only if the pts are on a line or at the same point,size.width=0
    return (box.size.height <= box.size.width * 30) && (box.size.width > 0);
}

// Define trackbar callback function. This function finds contours,
// draws them, and approximates by ellipses.
void processImage(int /*h*/, void*)
{
    RotatedRect box, boxAMS, boxDirect;
    vector<vector<Point> > contours;
    Mat bimage = image >= sliderPos;

    findContours(bimage, contours, RETR_LIST, CHAIN_APPROX_NONE);

    canvas paper;
    paper.init(int(0.8*MIN(bimage.rows, bimage.cols)), int(1.2*MAX(bimage.rows, bimage.cols)));
    paper.stretch(cv::Point2f(0.0f, 0.0f), cv::Point2f((float)(bimage.cols+2.0), (float)(bimage.rows+2.0)));

    std::vector<std::string> text;
    std::vector<cv::Scalar> color;

    if (fitEllipseQ) {
        text.push_back("OpenCV");
        color.push_back(fitEllipseColor);
    }
    if (fitEllipseAMSQ) {
        text.push_back("AMS");
        color.push_back(fitEllipseAMSColor);
    }
    if (fitEllipseDirectQ) {
        text.push_back("Direct");
        color.push_back(fitEllipseDirectColor);
    }
    paper.drawLabels(text, color);

    int margin = 2;
    vector< vector<Point2f> > points;
    for(size_t i = 0; i < contours.size(); i++)
    {
        size_t count = contours[i].size();
        if( count < 6 )
            continue;

        Mat pointsf;
        Mat(contours[i]).convertTo(pointsf, CV_32F);

        vector<Point2f>pts;
        for (int j = 0; j < pointsf.rows; j++) {
            Point2f pnt = Point2f(pointsf.at<float>(j,0), pointsf.at<float>(j,1));
            if ((pnt.x > margin && pnt.y > margin && pnt.x < bimage.cols-margin && pnt.y < bimage.rows-margin)) {
                if(j%20==0){
                    pts.push_back(pnt);
                }
            }
        }
        points.push_back(pts);
    }

    for(size_t i = 0; i < points.size(); i++)
    {
        vector<Point2f> pts = points[i];

        //At least 5 points can fit an ellipse
        if (pts.size()<5) {
            continue;
        }
        if (fitEllipseQ) {
            box = fitEllipse(pts);
            if (isGoodBox(box)) {
                paper.drawEllipseWithBox(box, fitEllipseColor, 3);
            }
        }
        if (fitEllipseAMSQ) {
            boxAMS = fitEllipseAMS(pts);
            if (isGoodBox(boxAMS)) {
                paper.drawEllipseWithBox(boxAMS, fitEllipseAMSColor, 2);
            }
        }
        if (fitEllipseDirectQ) {
            boxDirect = fitEllipseDirect(pts);
            if (isGoodBox(boxDirect)){
                paper.drawEllipseWithBox(boxDirect, fitEllipseDirectColor, 1);
            }
        }

        paper.drawPoints(pts, fitEllipseTrueColor);
    }

    imshow("result", paper.img);
}
