/*
* cloning_demo.cpp
*
* Author:
* Siddharth Kherada <siddharthkherada27[at]gmail[dot]com>
*
* This tutorial demonstrates how to use OpenCV seamless cloning
* module without GUI.
*
* 1- Normal Cloning
* 2- Mixed Cloning
* 3- Monochrome Transfer
* 4- Color Change
* 5- Illumination change
* 6- Texture Flattening

* The program takes as input a source and a destination image (for 1-3 methods)
* and ouputs the cloned image.
*
* Download test images from opencv_extra folder @github.
*
*/

#include "opencv2/photo.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include <iostream>
#include <stdlib.h>

using namespace std;
using namespace cv;

int main()
{
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
    int num = 1;
    cin >> num;
    cout << endl;

    if(num == 1)
    {
        string folder =  "cloning/Normal_Cloning/";
        string original_path1 = folder + "source1.png";
        string original_path2 = folder + "destination1.png";
        string original_path3 = folder + "mask.png";

        Mat source = imread(original_path1, IMREAD_COLOR);
        Mat destination = imread(original_path2, IMREAD_COLOR);
        Mat mask = imread(original_path3, IMREAD_COLOR);

        if(source.empty())
        {
            cout << "Could not load source image " << original_path1 << endl;
            exit(0);
        }
        if(destination.empty())
        {
            cout << "Could not load destination image " << original_path2 << endl;
            exit(0);
        }
        if(mask.empty())
        {
            cout << "Could not load mask image " << original_path3 << endl;
            exit(0);
        }

        Mat result;
        Point p;
        p.x = 400;
        p.y = 100;

        seamlessClone(source, destination, mask, p, result, 1);

        imshow("Output",result);
        imwrite(folder + "cloned.png", result);
    }
    else if(num == 2)
    {
        string folder = "cloning/Mixed_Cloning/";
        string original_path1 = folder + "source1.png";
        string original_path2 = folder + "destination1.png";
        string original_path3 = folder + "mask.png";

        Mat source = imread(original_path1, IMREAD_COLOR);
        Mat destination = imread(original_path2, IMREAD_COLOR);
        Mat mask = imread(original_path3, IMREAD_COLOR);

        if(source.empty())
        {
            cout << "Could not load source image " << original_path1 << endl;
            exit(0);
        }
        if(destination.empty())
        {
            cout << "Could not load destination image " << original_path2 << endl;
            exit(0);
        }
        if(mask.empty())
        {
            cout << "Could not load mask image " << original_path3 << endl;
            exit(0);
        }

        Mat result;
        Point p;
        p.x = destination.size().width/2;
        p.y = destination.size().height/2;

        seamlessClone(source, destination, mask, p, result, 2);

        imshow("Output",result);
        imwrite(folder + "cloned.png", result);
    }
    else if(num == 3)
    {
        string folder = "cloning/Monochrome_Transfer/";
        string original_path1 = folder + "source1.png";
        string original_path2 = folder + "destination1.png";
        string original_path3 = folder + "mask.png";

        Mat source = imread(original_path1, IMREAD_COLOR);
        Mat destination = imread(original_path2, IMREAD_COLOR);
        Mat mask = imread(original_path3, IMREAD_COLOR);

        if(source.empty())
        {
            cout << "Could not load source image " << original_path1 << endl;
            exit(0);
        }
        if(destination.empty())
        {
            cout << "Could not load destination image " << original_path2 << endl;
            exit(0);
        }
        if(mask.empty())
        {
            cout << "Could not load mask image " << original_path3 << endl;
            exit(0);
        }

        Mat result;
        Point p;
        p.x = destination.size().width/2;
        p.y = destination.size().height/2;

        seamlessClone(source, destination, mask, p, result, 3);

        imshow("Output",result);
        imwrite(folder + "cloned.png", result);
    }
    else if(num == 4)
    {
        string folder = "cloning/Color_Change/";
        string original_path1 = folder + "source1.png";
        string original_path2 = folder + "mask.png";

        Mat source = imread(original_path1, IMREAD_COLOR);
        Mat mask = imread(original_path2, IMREAD_COLOR);

        if(source.empty())
        {
            cout << "Could not load source image " << original_path1 << endl;
            exit(0);
        }
        if(mask.empty())
        {
            cout << "Could not load mask image " << original_path2 << endl;
            exit(0);
        }

        Mat result;

        colorChange(source, mask, result, 1.5, .5, .5);

        imshow("Output",result);
        imwrite(folder + "cloned.png", result);
    }
    else if(num == 5)
    {
        string folder = "cloning/Illumination_Change/";
        string original_path1 = folder + "source1.png";
        string original_path2 = folder + "mask.png";

        Mat source = imread(original_path1, IMREAD_COLOR);
        Mat mask = imread(original_path2, IMREAD_COLOR);

        if(source.empty())
        {
            cout << "Could not load source image " << original_path1 << endl;
            exit(0);
        }
        if(mask.empty())
        {
            cout << "Could not load mask image " << original_path2 << endl;
            exit(0);
        }

        Mat result;

        illuminationChange(source, mask, result, 0.2f, 0.4f);

        imshow("Output",result);
        imwrite(folder + "cloned.png", result);
    }
    else if(num == 6)
    {
        string folder = "cloning/Texture_Flattening/";
        string original_path1 = folder + "source1.png";
        string original_path2 = folder + "mask.png";

        Mat source = imread(original_path1, IMREAD_COLOR);
        Mat mask = imread(original_path2, IMREAD_COLOR);

        if(source.empty())
        {
            cout << "Could not load source image " << original_path1 << endl;
            exit(0);
        }
        if(mask.empty())
        {
            cout << "Could not load mask image " << original_path2 << endl;
            exit(0);
        }

        Mat result;

        textureFlattening(source, mask, result, 30, 45, 3);

        imshow("Output",result);
        imwrite(folder + "cloned.png", result);
    }
    waitKey(0);
}
