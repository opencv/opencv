/*
* layerModelBlending_demo.cpp
*
* Author:
* jsxyhelu <jsxyhelu[at]foxmail[dot]com>
*
* This tutorial demonstrates how to use OpenCV layerModelBlending
* module without GUI.
*
* 1- CV::BLEND_MODEL_DARKEN
* 2- CV::BLEND_MODEL_MULTIPY
* 3- CV::BLEND_MODEL_COLOR_BURN
* 4- CV::BLEND_MODEL_LINEAR_BRUN
* 5- CV::BLEND_MODEL_LIGHTEN
* 6- CV::BLEND_MODEL_SCREEN
* 7- CV::BLEND_MODEL_COLOR_DODGE
* 8- CV::BLEND_MODEL_LINEAR_DODGE
* 9- CV::BLEND_MODEL_OVERLAY
*10- CV::BLEND_MODEL_SOFT_LIGHT
*11- CV::BLEND_MODEL_HARD_LIGHT
*12- CV::BLEND_MODEL_VIVID_LIGHT
*13- CV::BLEND_MODEL_LINEAR_LIGHT
*14- CV::BLEND_MODEL_PIN_LIGHT
*15- CV::BLEND_MODEL_DIFFERENCE
*16- CV::BLEND_MODEL_EXCLUSION
*17- CV::BLEND_MODEL_DIVIDE
* The program takes as input a target and a blend image
* and outputs the layerModelBlended image.
*
* Download test images from opencv/sample/data folder @github.
*
*/

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    cout << endl;
    cout << "layerModelBlended Module" << endl;
    cout << "---------------" << endl;
    cout << "Options: " << endl;
    cout << endl;
    cout << "1) Darken " << endl;
    cout << "2) Multiply " << endl;
    cout << "3) Color Burn " << endl;
    cout << "4) Linear Burn " << endl;
    cout << "5) Lighten " << endl;
    cout << "6) Screen " << endl;
    cout << "7) Color Dodge " << endl;
    cout << "8) Linear Dodge " << endl;
    cout << "9) Overlay " << endl;
    cout << "10) Soft Light " << endl;
    cout << "11) Hard Light " << endl;
    cout << "12) Vivid Light " << endl;
    cout << "13) Linear Light " << endl;
    cout << "14) Pin Light " << endl;
    cout << "15) Difference " << endl;
    cout << "16) Exclusion " << endl;
    cout << "17) divide " << endl;
    cout << endl;
    cout << "Press number 1-17 to choose from above techniques: ";
    int num = BLEND_MODEL_DARKEN;
    cin >> num;
    cout << endl;
    Mat target = cv::imread("samples/cpp/lena.jpg");
    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
    GaussianBlur(target, blend, Size(33, 33), 0);
    Mat result(target.size(), CV_8UC3, Scalar::all(0));
    if (target.empty()) {
        std::cout << "Unable to load target!\n";
    }
    if (num == BLEND_MODEL_DARKEN)
    {
        layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);
        imwrite("samples/cpp/tutorial_code/photo/layerModelBlending/DARKEN_RESULT.jpg", result);
    }
    else if (num == BLEND_MODEL_MULTIPY)
    {
        layerModelBlending(target, blend, result, BLEND_MODEL_MULTIPY);
        imwrite("samples/cpp/tutorial_code/photo/layerModelBlending/MULTIPY_RESULT.jpg", result);
    }
    else if (num == BLEND_MODEL_COLOR_BURN)
    {
        layerModelBlending(target, blend, result, BLEND_MODEL_COLOR_BURN);
        imwrite("samples/cpp/tutorial_code/photo/layerModelBlending/COLOR_BURN.jpg", result);
    }
    else if (num == BLEND_MODEL_LINEAR_BRUN)
    {
        layerModelBlending(target, blend, result, BLEND_MODEL_LINEAR_BRUN);
        imwrite("samples/cpp/tutorial_code/photo/layerModelBlending/LINEAR_BRUN.jpg", result);
    }
    else if (num == BLEND_MODEL_LIGHTEN)
    {
        layerModelBlending(target, blend, result, BLEND_MODEL_LIGHTEN);
        imwrite("samples/cpp/tutorial_code/photo/layerModelBlending/LIGHTEN.jpg", result);
    }
    else if (num == BLEND_MODEL_SCREEN)
    {
        layerModelBlending(target, blend, result, BLEND_MODEL_SCREEN);
        imwrite("samples/cpp/tutorial_code/photo/layerModelBlending/SCREEN_RESULT.jpg", result);
    }
    else if (num == BLEND_MODEL_COLOR_DODGE)
    {
        layerModelBlending(target, blend, result, BLEND_MODEL_COLOR_DODGE);
        imwrite("samples/cpp/tutorial_code/photo/layerModelBlending/COLOR_DODGE_RESULT.jpg", result);
    }
    else if (num == BLEND_MODEL_LINEAR_DODGE)
    {
        layerModelBlending(target, blend, result, BLEND_MODEL_LINEAR_DODGE);
        imwrite("samples/cpp/tutorial_code/photo/layerModelBlending/LINEAR_DODGE_RESULT.jpg", result);
    }
    else if (num == BLEND_MODEL_OVERLAY)
    {
        layerModelBlending(target, blend, result, BLEND_MODEL_OVERLAY);
        imwrite("samples/cpp/tutorial_code/photo/layerModelBlending/OVERLAY_RESULT.jpg", result);
    }
    else if (num == BLEND_MODEL_SOFT_LIGHT)
    {
        layerModelBlending(target, blend, result, BLEND_MODEL_SOFT_LIGHT);
        imwrite("samples/cpp/tutorial_code/photo/layerModelBlending/SOFT_LIGHT_RESULT.jpg", result);
    }
    else if (num == BLEND_MODEL_HARD_LIGHT)
    {
        layerModelBlending(target, blend, result, BLEND_MODEL_HARD_LIGHT);
        imwrite("samples/cpp/tutorial_code/photo/layerModelBlending/HARD_LIGHT_RESULT.jpg", result);
    }
    else if (num == BLEND_MODEL_VIVID_LIGHT)
    {
        layerModelBlending(target, blend, result, BLEND_MODEL_VIVID_LIGHT);
        imwrite("samples/cpp/tutorial_code/photo/layerModelBlending/VIVID_LIGHT_RESULT.jpg", result);
    }
    else if (num == BLEND_MODEL_LINEAR_LIGHT)
    {
        layerModelBlending(target, blend, result, BLEND_MODEL_LINEAR_LIGHT);
        imwrite("samples/cpp/tutorial_code/photo/layerModelBlending/LINEAR_LIGHT_RESULT.jpg", result);
    }
    else if (num == BLEND_MODEL_PIN_LIGHT)
    {
        layerModelBlending(target, blend, result, BLEND_MODEL_PIN_LIGHT);
        imwrite("samples/cpp/tutorial_code/photo/layerModelBlending/PIN_LIGHT_RESULT.jpg", result);
    }
    else if (num == BLEND_MODEL_DIFFERENCE)
    {
        layerModelBlending(target, blend, result, BLEND_MODEL_DIFFERENCE);
        imwrite("samples/cpp/tutorial_code/photo/layerModelBlending/DIFFERENCE_RESULT.jpg", result);
    }
    else if (num == BLEND_MODEL_EXCLUSION)
    {
        layerModelBlending(target, blend, result, BLEND_MODEL_EXCLUSION);
        imwrite("samples/cpp/tutorial_code/photo/layerModelBlending/EXCLUSION_RESULT.jpg", result);
    }
    else if (num == BLEND_MODEL_DIVIDE)
    {
        layerModelBlending(target, blend, result, BLEND_MODEL_DIVIDE);
        imwrite("samples/cpp/tutorial_code/photo/layerModelBlending/DIVIDE_RESULT.jpg", result);
    }
    imshow("Output", result);
    waitKey(0);
}
