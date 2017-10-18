#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

enum MyShape{MyCIRCLE=0,MyRECTANGLE,MyELLIPSE};

struct ParamColorMar {
    int iColormap;
    Mat img;
};

Ptr<Mat> lutRND;
String winName="False color";

static void TrackColorMap(int x, void *r)
{
    std::cout << "selected: " << x << std::endl;
    ParamColorMar *p = (ParamColorMar*)r;
    Mat dst;
    p->iColormap= x;
    if (x == cv::COLORMAP_PARULA + 1)
    {
        if (!lutRND)
        {
            RNG ra;
            lutRND = makePtr<Mat>(256, 1, CV_8UC3);
            ra.fill(*lutRND, RNG::UNIFORM, 0, 256);
        }
        applyColorMap(p->img, dst, *lutRND.get());
    }
    else
        applyColorMap(p->img,dst,p->iColormap);
    String colorMapName;

    switch (p->iColormap) {
    case COLORMAP_AUTUMN :
        colorMapName = "Colormap : Autumn";
        break;
    case COLORMAP_BONE :
        colorMapName = "Colormap : Bone";
        break;
    case COLORMAP_JET :
        colorMapName = "Colormap : Jet";
        break;
    case COLORMAP_WINTER :
        colorMapName = "Colormap : Winter";
        break;
    case COLORMAP_RAINBOW :
        colorMapName = "Colormap : Rainbow";
        break;
    case COLORMAP_OCEAN :
        colorMapName = "Colormap : Ocean";
        break;
    case COLORMAP_SUMMER :
        colorMapName = "Colormap : Summer";
        break;
    case COLORMAP_SPRING :
        colorMapName = "Colormap : Spring";
        break;
    case COLORMAP_COOL :
        colorMapName = "Colormap : Cool";
        break;
    case COLORMAP_HSV :
        colorMapName = "Colormap : HSV";
        break;
    case COLORMAP_PINK :
        colorMapName = "Colormap : Pink";
        break;
    case COLORMAP_HOT :
        colorMapName = "Colormap : Hot";
        break;
    case COLORMAP_PARULA :
        colorMapName = "Colormap : Parula";
        break;
    default:
        colorMapName = "User colormap : random";
        break;
    }
    std::cout << "> " << colorMapName << std::endl;
    putText(dst, colorMapName, Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255));
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
    ParamColorMar  p;

    Mat img;
    if (argc > 1)
        img = imread(argv[1], 0);
    else
        img = DrawMyImage(2,256);
    p.img=img;
    p.iColormap=0;

    imshow("Gray image",img);
    namedWindow(winName);
    createTrackbar("colormap", winName,&p.iColormap,1,TrackColorMap,(void*)&p);
    setTrackbarMin("colormap", winName, cv::COLORMAP_AUTUMN);
    setTrackbarMax("colormap", winName, cv::COLORMAP_PARULA+1);
    setTrackbarPos("colormap", winName, -1);

    TrackColorMap((int)getTrackbarPos("colormap", winName),(void*)&p);
    while (waitKey(0) != 27)
    {
        std::cout << "Press 'ESC' to exit" << std::endl;
    }
    return 0;
}
