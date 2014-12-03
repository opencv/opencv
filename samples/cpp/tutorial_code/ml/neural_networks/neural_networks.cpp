#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "ann.hpp"

using namespace cv;
using namespace cv::ml;

int main(int argc, char** argv )
{
    printf("|--Setting--|----Training----|--------------------Randering--------------------|");
    // setting up network
    NN nn;
    int neuronsmat[1][5] = {{2, 150, 150, 150, 1}};// initiate network with 3 hidden layers,
    Mat neurons = Mat(1, 5, CV_32S, neuronsmat);   // 150 neurons for each layer
    nn.create(neurons, ANN::SIGMOID_SYM);

    // loading sample data. change the data below to see what will happen
    int n = 16;
    float inputmat[][2]  = {{-0.71,-0.42}, {-0.27,-0.58}, { 0.20,-0.85}, { 0.71,-0.84},
                            {-0.84,-0.78}, {-0.19,-0.23}, { 0.45,-0.03}, { 0.52,-0.69},
                            {-0.43, 0.81}, {-0.11, 0.15}, { 0.39, 0.35}, { 0.63, 0.36},
                            {-0.65, 0.65}, {-0.23, 0.64}, { 0.12, 0.87}, { 0.57, 0.80}};
    float outputmat[][1] = {         {-1},          {-1},          {-1},          { 1},
                                     {-1},          { 1},          { 1},          { 1},
                                     {-1},          { 1},          { 1},          { 1},
                                     {-1},          {-1},          {-1},          { 1}};
    Mat inputs  = Mat(n, 2, CV_32F, inputmat);
    Mat outputs = Mat(n, 1, CV_32F, outputmat);

    // training
    nn.setEpoches(300);// have the network well trained for 300 epoches
    nn.train(inputs, outputs);
    printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>"); fflush(stdout);

    // aha! that's right, nn is now trained and fully functional, we can proceed to predict

    // beautify graphical output. if prefer terminal, can simply call B = nn.predict(A),
    // just like how a function y = f(x) do
    int width = 512, height = 512;
    Mat image = Mat::zeros(width, height, CV_8UC3);

    // randering decision regions
    Vec3b green(0, 255, 0), blue(255, 0, 0);
    for(int i = 0; i < width; i++)
    {
        for(int j = 0; j < height; j++)
        {
            // normalizing to [-1, 1] for better performance
            float samplemat[][2] = {{(float)i / width * 2 - 1, (float)j / height * 2 - 1}};
            Mat sampleMat = Mat(1, 2, CV_32F, samplemat);

            // predicting user input
            Mat responseMat = nn.predict(sampleMat);
            float response = responseMat.at<float>(0, 0);

            // presenting
            if (response < 0)
                image.at<Vec3b>(i, j) = green;
            else
                image.at<Vec3b>(i, j) = blue;
        }
        if(i % 10 == 0) {printf(">"); fflush(stdout);}
    }

    // randering training samples
    int radius = 5;
    for(int i = 0; i < n; i++)
    {
        Point p = Point((inputs.at<float>(i, 1) + 1) * height / 2,
                        (inputs.at<float>(i, 0) + 1) *  width / 2);// invert for image portray
        Scalar s = outputs.at<float>(i, 0) < 0 ? Scalar(0, 0, 0) : Scalar(255, 255, 255);
        circle(image, p, radius, s);
    }

    // demonstration
    char demo[] = "demo: distinguish between two kinds of samples";
    imwrite("result.png", image);
    imshow(demo, image.t());
    resizeWindow(demo, width, height);
    waitKey(0);

    return 0;
}
