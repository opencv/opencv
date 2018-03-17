#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

enum MyShape{MyCIRCLE=0,MyRECTANGLE,MyELLIPSE};

struct ParamColorMap {
    int iColormap;
    Mat img;
};

String winName="False color";
static const String ColorMaps[] = { "Autumn", "Bone", "Jet", "Winter", "Rainbow", "Ocean", "Summer",
                                    "Spring", "Cool", "HSV", "Pink", "Hot", "Parula", "User defined (random)"};

static void TrackColorMap(int x, void *r)
{
    ParamColorMap *p = (ParamColorMap*)r;
    Mat dst;
    p->iColormap= x;
    if (x == COLORMAP_PARULA + 1)
    {
        Mat lutRND(256, 1, CV_8UC3);
        randu(lutRND, Scalar(0, 0, 0), Scalar(255, 255, 255));
        applyColorMap(p->img, dst, lutRND);
    }
    else
        applyColorMap(p->img,dst,p->iColormap);

    putText(dst, "Colormap : "+ColorMaps[p->iColormap], Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255),2);
    imshow(winName, dst);
}


static Mat DrawMyImage(int thickness,int nbShape)
{
    Mat img=Mat::zeros(500,256*thickness+100,CV_8UC1);
    int offsetx = 50, offsety = 25;
    int lineLenght = 50;

    for (int i=0;i<256;i++)
        line(img,Point(thickness*i+ offsetx, offsety),Point(thickness*i+ offsetx, offsety+ lineLenght),Scalar(i), thickness);
    RNG r;
    Point center;
    int radius;
    int width,height;
    int angle;
    Rect rc;

    for (int i=1;i<=nbShape;i++)
    {
        int typeShape = r.uniform(MyCIRCLE, MyELLIPSE+1);
        switch (typeShape) {
        case MyCIRCLE:
            center = Point(r.uniform(offsetx,img.cols- offsetx), r.uniform(offsety + lineLenght, img.rows - offsety));
            radius = r.uniform(1, min(offsetx, offsety));
            circle(img,center,radius,Scalar(i),-1);
            break;
        case MyRECTANGLE:
            center = Point(r.uniform(offsetx, img.cols - offsetx), r.uniform(offsety + lineLenght, img.rows - offsety));
            width = r.uniform(1, min(offsetx, offsety));
            height = r.uniform(1, min(offsetx, offsety));
            rc = Rect(center-Point(width ,height )/2, center + Point(width , height )/2);
            rectangle(img,rc, Scalar(i), -1);
            break;
        case MyELLIPSE:
            center = Point(r.uniform(offsetx, img.cols - offsetx), r.uniform(offsety + lineLenght, img.rows - offsety));
            width = r.uniform(1, min(offsetx, offsety));
            height = r.uniform(1, min(offsetx, offsety));
            angle = r.uniform(0, 180);
            ellipse(img, center,Size(width/2,height/2),angle,0,360, Scalar(i), -1);
            break;
        }
    }
    return img;
}

int main(int argc, char** argv)
{
    cout << "This program demonstrates the use of applyColorMap function.\n\n";

    ParamColorMap  p;
    Mat img;

    if (argc > 1)
        img = imread(argv[1], IMREAD_GRAYSCALE);
    else
        img = DrawMyImage(2,256);

    p.img=img;
    p.iColormap=0;

    imshow("Gray image",img);
    namedWindow(winName);
    createTrackbar("colormap", winName,&p.iColormap,1,TrackColorMap,(void*)&p);
    setTrackbarMin("colormap", winName, COLORMAP_AUTUMN);
    setTrackbarMax("colormap", winName, COLORMAP_PARULA+1);
    setTrackbarPos("colormap", winName, -1);

    TrackColorMap(0, (void*)&p);

    cout << "Press a key to exit" << endl;
    waitKey(0);
    return 0;
}
