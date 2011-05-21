#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
using namespace cv;

void help()
{
	std::cout
	<< "\nThis program demonstrates OpenCV drawing and text output functions.\n"
	"Call:\n"
	"./drawing\n" << std::endl;
}
static Scalar randomColor(RNG& rng)
{
    int icolor = (unsigned)rng;
    return Scalar(icolor&255, (icolor>>8)&255, (icolor>>16)&255);
}

int main( int, char** )
{
    char wndname[] = "Drawing Demo";
    const int NUMBER = 100;
    const int DELAY = 5;
    int lineType = CV_AA; // change it to 8 to see non-antialiased graphics
    int i, width = 1000, height = 700;
    int x1 = -width/2, x2 = width*3/2, y1 = -height/2, y2 = height*3/2;
    RNG rng(0xFFFFFFFF);

    Mat image = Mat::zeros(height, width, CV_8UC3);
    imshow(wndname, image);
    waitKey(DELAY);

    for (i = 0; i < NUMBER; i++)
    {
        Point pt1, pt2;
        pt1.x = rng.uniform(x1, x2);
        pt1.y = rng.uniform(y1, y2);
        pt2.x = rng.uniform(x1, x2);
        pt2.y = rng.uniform(y1, y2);

        line( image, pt1, pt2, randomColor(rng), rng.uniform(1,10), lineType );
        
        imshow(wndname, image);
        if(waitKey(DELAY) >= 0)
            return 0;
    }

    for (i = 0; i < NUMBER; i++)
    {
        Point pt1, pt2;
        pt1.x = rng.uniform(x1, x2);
        pt1.y = rng.uniform(y1, y2);
        pt2.x = rng.uniform(x1, x2);
        pt2.y = rng.uniform(y1, y2);
        int thickness = rng.uniform(-3, 10);
        
        rectangle( image, pt1, pt2, randomColor(rng), MAX(thickness, -1), lineType );
        
        imshow(wndname, image);
        if(waitKey(DELAY) >= 0)
            return 0;
    }
    
    for (i = 0; i < NUMBER; i++)
    {
        Point center;
        center.x = rng.uniform(x1, x2);
        center.y = rng.uniform(y1, y2);
        Size axes;
        axes.width = rng.uniform(0, 200);
        axes.height = rng.uniform(0, 200);
        double angle = rng.uniform(0, 180);

        ellipse( image, center, axes, angle, angle - 100, angle + 200,
                 randomColor(rng), rng.uniform(-1,9), lineType );
        
        imshow(wndname, image);
        if(waitKey(DELAY) >= 0)
            return 0;
    }

    for (i = 0; i< NUMBER; i++)
    {
        Point pt[2][3];
        pt[0][0].x = rng.uniform(x1, x2);
        pt[0][0].y = rng.uniform(y1, y2);
        pt[0][1].x = rng.uniform(x1, x2); 
        pt[0][1].y = rng.uniform(y1, y2); 
        pt[0][2].x = rng.uniform(x1, x2);
        pt[0][2].y = rng.uniform(y1, y2);
        pt[1][0].x = rng.uniform(x1, x2); 
        pt[1][0].y = rng.uniform(y1, y2);
        pt[1][1].x = rng.uniform(x1, x2); 
        pt[1][1].y = rng.uniform(y1, y2);
        pt[1][2].x = rng.uniform(x1, x2); 
        pt[1][2].y = rng.uniform(y1, y2);
        const Point* ppt[2] = {pt[0], pt[1]};
        int npt[] = {3, 3};
        
        polylines(image, ppt, npt, 2, true, randomColor(rng), rng.uniform(1,10), lineType);
        
        imshow(wndname, image);
        if(waitKey(DELAY) >= 0)
            return 0;
    }
    
    for (i = 0; i< NUMBER; i++)
    {
        Point pt[2][3];
        pt[0][0].x = rng.uniform(x1, x2);
        pt[0][0].y = rng.uniform(y1, y2);
        pt[0][1].x = rng.uniform(x1, x2); 
        pt[0][1].y = rng.uniform(y1, y2); 
        pt[0][2].x = rng.uniform(x1, x2);
        pt[0][2].y = rng.uniform(y1, y2);
        pt[1][0].x = rng.uniform(x1, x2); 
        pt[1][0].y = rng.uniform(y1, y2);
        pt[1][1].x = rng.uniform(x1, x2); 
        pt[1][1].y = rng.uniform(y1, y2);
        pt[1][2].x = rng.uniform(x1, x2); 
        pt[1][2].y = rng.uniform(y1, y2);
        const Point* ppt[2] = {pt[0], pt[1]};
        int npt[] = {3, 3};
        
        fillPoly(image, ppt, npt, 2, randomColor(rng), lineType);
        
        imshow(wndname, image);
        if(waitKey(DELAY) >= 0)
            return 0;
    }

    for (i = 0; i < NUMBER; i++)
    {
        Point center;
        center.x = rng.uniform(x1, x2);
        center.y = rng.uniform(y1, y2);
        
        circle(image, center, rng.uniform(0, 300), randomColor(rng),
               rng.uniform(-1, 9), lineType);
        
        imshow(wndname, image);
        if(waitKey(DELAY) >= 0)
            return 0;
    }

    for (i = 1; i < NUMBER; i++)
    {
        Point org;
        org.x = rng.uniform(x1, x2);
        org.y = rng.uniform(y1, y2);

        putText(image, "Testing text rendering", org, rng.uniform(0,8),
                rng.uniform(0,100)*0.05+0.1, randomColor(rng), rng.uniform(1, 10), lineType);
        
        imshow(wndname, image);
        if(waitKey(DELAY) >= 0)
            return 0;
    }

    Size textsize = getTextSize("OpenCV forever!", CV_FONT_HERSHEY_COMPLEX, 3, 5, 0);
    Point org((width - textsize.width)/2, (height - textsize.height)/2);
    
    Mat image2;
    for( i = 0; i < 255; i += 2 )
    {
        image2 = image - Scalar::all(i);
        putText(image2, "OpenCV forever!", org, CV_FONT_HERSHEY_COMPLEX, 3,
                Scalar(i, i, 255), 5, lineType);
        
        imshow(wndname, image2);
        if(waitKey(DELAY) >= 0)
            return 0;
    }

    waitKey();
    return 0;
}

#ifdef _EiC
main(1,"drawing.c");
#endif
