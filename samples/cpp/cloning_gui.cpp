/*
* cloning.cpp
*
* Author:
* Siddharth Kherada <siddharthkherada27[at]gmail[dot]com>
*
* This tutorial demonstrates how to use OpenCV seamless cloning
* module.
*
* 1- Normal Cloning
* 2- Mixed Cloning
* 3- Monochrome Transfer
* 4- Color Change
* 5- Illumination change
* 6- Texture Flattening

* The program takes as input a source and a destination image (for 1-3 methods)
* and outputs the cloned image.

* Step 1:
* -> In the source image, select the region of interest by left click mouse button. A Polygon ROI will be created by left clicking mouse button.
* -> To set the Polygon ROI, click the right mouse button or 'd' key.
* -> To reset the region selected, click the middle mouse button or 'r' key.

* Step 2:
* -> In the destination image, select the point where you want to place the ROI in the image by left clicking mouse button.
* -> To get the cloned result, click the right mouse button or 'c' key.
* -> To quit the program, use 'q' key.
*
* Result: The cloned image will be displayed.
*/

#include "opencv2/photo.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include <iostream>
#include <climits>

// we're NOT "using namespace std;" here, to avoid collisions between the beta variable and std::beta in c++17
using std::cin;
using std::cout;
using std::endl;
using std::string;

using namespace cv;

Mat img0, img1, img2, res, res1, final, final1, blend;

Point point;
int drag = 0;
int destx, desty;

int numpts = 100;
Point* pts = new Point[100];
Point* pts2 = new Point[100];
Point* pts_diff = new Point[100];

int var = 0;
int flag = 0, flag1 = 0, flag4 = 0;

int minx, miny, maxx, maxy, lenx, leny;
int minxd, minyd, maxxd, maxyd, lenxd, lenyd;

int channel, num, kernel_size;

float alpha,beta;

float red, green, blue;

float low_t, high_t;

void source(int, int, int, int, void*);
void destination(int, int, int, int, void*);
void checkfile(char*);

void source(int event, int x, int y, int, void*)
{

    // if (event == EVENT_LBUTTONDOWN && !drag)
    // {
    //     if(flag1 == 0)
    //     {
    //         if(var==0)
    //             img1 = img0.clone();
    //         point = Point(x, y);
    //         circle(img1,point,2,Scalar(0, 0, 255),-1, 8, 0);
    //         pts[var] = point;
    //         var++;
    //         drag  = 1;
    //         if(var>1)
    //             line(img1,pts[var-2], point, Scalar(0, 0, 255), 2, 8, 0);

    //         imshow("Source", img1);
    //     }
    // }

    // if (event == EVENT_LBUTTONUP && drag)
    // {
    //     imshow("Source", img1);

    //     drag = 0;
    // }
    // if (event == EVENT_RBUTTONDOWN)
    // {
    //     flag1 = 1;
    //     img1 = img0.clone();
    //     for(int i = var; i < numpts ; i++)
    //         pts[i] = point;

    //     if(var!=0)
    //     {
    //         const Point* pts3[1] = {&pts[0]};
    //         polylines( img1, pts3, &numpts,1, 1, Scalar(0,0,0), 2, 8, 0);
    //     }

    //     for(int i=0;i<var;i++)
    //     {
    //         minx = min(minx,pts[i].x);
    //         maxx = max(maxx,pts[i].x);
    //         miny = min(miny,pts[i].y);
    //         maxy = max(maxy,pts[i].y);
    //     }
    //     lenx = maxx - minx;
    //     leny = maxy - miny;

    //     int mid_pointx = minx + lenx/2;
    //     int mid_pointy = miny + leny/2;

    //     for(int i=0;i<var;i++)
    //     {
    //         pts_diff[i].x = pts[i].x - mid_pointx;
    //         pts_diff[i].y = pts[i].y - mid_pointy;
    //     }

    //     imshow("Source", img1);
    // }

    // if (event == EVENT_RBUTTONUP)
    // {
    //     flag = var;

    //     final = Mat::zeros(img0.size(),CV_8UC3);
    //     res1 = Mat::zeros(img0.size(),CV_8UC1);
    //     const Point* pts4[1] = {&pts[0]};

    //     fillPoly(res1, pts4,&numpts, 1, Scalar(255, 255, 255), 8, 0);
    //     bitwise_and(img0, img0, final,res1);

    //     imshow("Source", img1);

    //     if(num == 4)
    //     {
    //         colorChange(img0,res1,blend,red,green,blue);
    //         imshow("Color Change Image", blend);
    //         waitKey(0);

    //     }
    //     else if(num == 5)
    //     {
    //         illuminationChange(img0,res1,blend,alpha,beta);
    //         imshow("Illum Change Image", blend);
    //         waitKey(0);
    //     }
    //     else if(num == 6)
    //     {
    //         textureFlattening(img0,res1,blend,low_t,high_t,kernel_size);
    //         imshow("Texture Flattened", blend);
    //         waitKey(0);
    //     }

    // }
    // if (event == EVENT_MBUTTONDOWN)
    // {
    //     for(int i = 0; i < numpts ; i++)
    //     {
    //         pts[i].x=0;
    //         pts[i].y=0;
    //     }
    //     var = 0;
    //     flag1 = 0;
    //     minx = INT_MAX; miny = INT_MAX; maxx = INT_MIN; maxy = INT_MIN;
    //     imshow("Source", img0);
    //     if(num == 1 || num == 2 || num == 3)
    //         imshow("Destination",img2);
    //     drag = 0;
    // }
}

void destination(int event, int x, int y, int, void*)
{

    // Mat im1;
    // minxd = INT_MAX; minyd = INT_MAX; maxxd = INT_MIN; maxyd = INT_MIN;
    // im1 = img2.clone();
    // if (event == EVENT_LBUTTONDOWN)
    // {
    //     flag4 = 1;
    //     if(flag1 == 1)
    //     {
    //         point = Point(x, y);

    //         for(int i=0;i<var;i++)
    //         {
    //             pts2[i].x = point.x + pts_diff[i].x;
    //             pts2[i].y = point.y + pts_diff[i].y;
    //         }

    //         for(int i=var;i<numpts;i++)
    //         {
    //             pts2[i].x = point.x + pts_diff[0].x;
    //             pts2[i].y = point.y + pts_diff[0].y;
    //         }

    //         const Point* pts5[1] = {&pts2[0]};
    //         polylines( im1, pts5, &numpts,1, 1, Scalar(0,0,255), 2, 8, 0);

    //         destx = x;
    //         desty = y;

    //         imshow("Destination", im1);
    //     }
    // }
    // if (event == EVENT_RBUTTONUP)
    // {
    //     for(int i=0;i<flag;i++)
    //     {
    //         minxd = min(minxd,pts2[i].x);
    //         maxxd = max(maxxd,pts2[i].x);
    //         minyd = min(minyd,pts2[i].y);
    //         maxyd = max(maxyd,pts2[i].y);
    //     }

    //     if(maxxd > im1.size().width || maxyd > im1.size().height || minxd < 0 || minyd < 0)
    //     {
    //         cout << "Index out of range" << endl;
    //         exit(1);
    //     }

    //     final1 = Mat::zeros(img2.size(),CV_8UC3);
    //     res = Mat::zeros(img2.size(),CV_8UC1);
    //     for(int i=miny, k=minyd;i<(miny+leny);i++,k++)
    //         for(int j=minx,l=minxd ;j<(minx+lenx);j++,l++)
    //         {
    //             for(int c=0;c<channel;c++)
    //             {
    //                 final1.at<uchar>(k,l*channel+c) = final.at<uchar>(i,j*channel+c);

    //             }
    //         }

    //     const Point* pts6[1] = {&pts2[0]};
    //     fillPoly(res, pts6, &numpts, 1, Scalar(255, 255, 255), 8, 0);

    //     if(num == 1 || num == 2 || num == 3)
    //     {
    //         seamlessClone(img0,img2,res1,point,blend,num);
    //         imshow("Cloned Image", blend);
    //         imwrite("cloned.png",blend);
    //         waitKey(0);
    //     }

    //     for(int i = 0; i < flag ; i++)
    //     {
    //         pts2[i].x=0;
    //         pts2[i].y=0;
    //     }

    //     minxd = INT_MAX; minyd = INT_MAX; maxxd = INT_MIN; maxyd = INT_MIN;
    // }

    // im1.release();
}

int main() {
    cout << endl;
    cout << "Cloning Module" << endl;
    cout << "---------------" << endl;
    cout << "Options: " << endl;
    cout << endl;
    cout << "1) Normal Cloning " << endl;
    cout << "2) Mixed Cloning " << endl;
    cout << "3) Monochrome Transfer " << endl;
    cout << "4) Local Color Change " << endl;
    cout << "5) Local Illumination Change " << endl;
    cout << "6) Texture Flattening " << endl;

    cout << endl;
    cout << "Press number 1-6 to choose from above techniques: ";
    cin >> num;
    cout << endl;

    minx = INT_MAX; miny = INT_MAX; maxx = INT_MIN; maxy = INT_MIN;
    minxd = INT_MAX; minyd = INT_MAX; maxxd = INT_MIN; maxyd = INT_MIN;

    if(num == 1 || num == 2 || num == 3) {
        string src, dest;
        cout << "Enter Source Image: ";
        cin >> src;
        cout << "Enter Destination Image: ";
        cin >> dest;

        img0 = imread(samples::findFile(src));
        img2 = imread(samples::findFile(dest));

        if(img0.empty()) {
            cout << "Source Image does not exist" << endl;
            exit(2);
        }
        if(img2.empty()) {
            cout << "Destination Image does not exist" << endl;
            exit(2);
        }

        channel = img0.channels();

        // For simplicity, we will use predefined points for source and destination
        var = 4;
        pts[0] = Point(50, 50);
        pts[1] = Point(200, 50);
        pts[2] = Point(200, 200);
        pts[3] = Point(50, 200);
        for(int i = var; i < numpts ; i++) pts[i] = pts[0];

        minx = 50; miny = 50; maxx = 200; maxy = 200;
        lenx = maxx - minx;
        leny = maxy - miny;
        int mid_pointx = minx + lenx/2;
        int mid_pointy = miny + leny/2;
        for(int i=0;i<var;i++) {
            pts_diff[i].x = pts[i].x - mid_pointx;
            pts_diff[i].y = pts[i].y - mid_pointy;
        }

        point = Point(img2.cols / 2, img2.rows / 2);
        for(int i=0;i<var;i++) {
            pts2[i] = Point(point.x + pts_diff[i].x, point.y + pts_diff[i].y);
        }
        for(int i=var;i<numpts;i++) {
            pts2[i] = pts2[0];
        }

        res = Mat::zeros(img2.size(), CV_8UC1);
        res1 = Mat::zeros(img0.size(), CV_8UC1);
        final = Mat::zeros(img0.size(), CV_8UC3);
        final1 = Mat::zeros(img2.size(), CV_8UC3);

        const Point* pts4[1] = {&pts[0]};
        fillPoly(res1, pts4, &numpts, 1, Scalar(255, 255, 255), 8, 0);
        bitwise_and(img0, img0, final, res1);

        seamlessClone(img0, img2, res1, point, blend, num);
        imwrite("cloned.png", blend);
        cout << "Cloned image saved as cloned.png" << endl;
    }
    else if(num == 4) {
        string src;
        cout << "Enter Source Image: ";
        cin >> src;

        cout << "Enter RGB values: " << endl;
        cout << "Red: ";
        cin >> red;
        cout << "Green: ";
        cin >> green;
        cout << "Blue: ";
        cin >> blue;

        img0 = imread(samples::findFile(src));
        if(img0.empty()) {
            cout << "Source Image does not exist" << endl;
            exit(2);
        }

        res1 = Mat::zeros(img0.size(), CV_8UC1);
        final = Mat::zeros(img0.size(), CV_8UC3);

        // For simplicity, we will use predefined points for source and destination
        var = 4;
        pts[0] = Point(50, 50);
        pts[1] = Point(200, 50);
        pts[2] = Point(200, 200);
        pts[3] = Point(50, 200);
        for(int i = var; i < numpts ; i++) pts[i] = pts[0];

        minx = 50; miny = 50; maxx = 200; maxy = 200;
        lenx = maxx - minx;
        leny = maxy - miny;

        const Point* pts4[1] = {&pts[0]};
        fillPoly(res1, pts4, &numpts, 1, Scalar(255, 255, 255), 8, 0);
        bitwise_and(img0, img0, final, res1);

        colorChange(img0, res1, blend, red, green, blue);
        imwrite("cloned_color_change.png", blend);
        cout << "Color change image saved as cloned_color_change.png" << endl;
    }
    else if(num == 5) {
        string src;
        cout << "Enter Source Image: ";
        cin >> src;

        cout << "alpha: ";
        cin >> alpha;
        cout << "beta: ";
        cin >> beta;

        img0 = imread(samples::findFile(src));
        if(img0.empty()) {
            cout << "Source Image does not exist" << endl;
            exit(2);
        }

        res1 = Mat::zeros(img0.size(), CV_8UC1);
        final = Mat::zeros(img0.size(), CV_8UC3);

        // For simplicity, we will use predefined points for source and destination
        var = 4;
        pts[0] = Point(50, 50);
        pts[1] = Point(200, 50);
        pts[2] = Point(200, 200);
        pts[3] = Point(50, 200);
        for(int i = var; i < numpts ; i++) pts[i] = pts[0];

        minx = 50; miny = 50; maxx = 200; maxy = 200;
        lenx = maxx - minx;
        leny = maxy - miny;

        const Point* pts4[1] = {&pts[0]};
        fillPoly(res1, pts4, &numpts, 1, Scalar(255, 255, 255), 8, 0);
        bitwise_and(img0, img0, final, res1);

        illuminationChange(img0, res1, blend, alpha, beta);
        imwrite("cloned_illumination_change.png", blend);
        cout << "Illumination change image saved as cloned_illumination_change.png" << endl;
    }
    else if(num == 6) {
        string src;
        cout << "Enter Source Image: ";
        cin >> src;

        cout << "low_threshold: ";
        cin >> low_t;
        cout << "high_threshold: ";
        cin >> high_t;
        cout << "kernel_size: ";
        cin >> kernel_size;

        img0 = imread(samples::findFile(src));
        if(img0.empty()) {
            cout << "Source Image does not exist" << endl;
            exit(2);
        }

        res1 = Mat::zeros(img0.size(), CV_8UC1);
        final = Mat::zeros(img0.size(), CV_8UC3);

        // For simplicity, we will use predefined points for source and destination
        var = 4;
        pts[0] = Point(50, 50);
        pts[1] = Point(200, 50);
        pts[2] = Point(200, 200);
        pts[3] = Point(50, 200);
        for(int i = var; i < numpts ; i++) pts[i] = pts[0];

        minx = 50; miny = 50; maxx = 200; maxy = 200;
        lenx = maxx - minx;
        leny = maxy - miny;

        const Point* pts4[1] = {&pts[0]};
        fillPoly(res1, pts4, &numpts, 1, Scalar(255, 255, 255), 8, 0);
        bitwise_and(img0, img0, final, res1);

        textureFlattening(img0, res1, blend, low_t, high_t, kernel_size);
        imwrite("cloned_texture_flattening.png", blend);
        cout << "Texture flattened image saved as cloned_texture_flattening.png" << endl;
    }
    else {
        cout << "Wrong Option Chosen" << endl;
        exit(1);
    }

    return 0;
}
