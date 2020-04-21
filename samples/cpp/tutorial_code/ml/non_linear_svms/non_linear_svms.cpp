#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

static void help()
{
    cout<< "\n--------------------------------------------------------------------------" << endl
        << "This program shows Support Vector Machines for Non-Linearly Separable Data. " << endl
        << "--------------------------------------------------------------------------"   << endl
        << endl;
}

int main()
{
    help();

    const int NTRAINING_SAMPLES = 100;         // Number of training samples per class
    const float FRAC_LINEAR_SEP = 0.9f;        // Fraction of samples which compose the linear separable part

    // Data for visual representation
    const int WIDTH = 512, HEIGHT = 512;
    Mat I = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);

    //--------------------- 1. Set up training data randomly ---------------------------------------
    Mat trainData(2*NTRAINING_SAMPLES, 2, CV_32F);
    Mat labels   (2*NTRAINING_SAMPLES, 1, CV_32S);

    RNG rng(100); // Random value generation class

    // Set up the linearly separable part of the training data
    int nLinearSamples = (int) (FRAC_LINEAR_SEP * NTRAINING_SAMPLES);

    //! [setup1]
    // Generate random points for the class 1
    Mat trainClass = trainData.rowRange(0, nLinearSamples);
    // The x coordinate of the points is in [0, 0.4)
    Mat c = trainClass.colRange(0, 1);
    rng.fill(c, RNG::UNIFORM, Scalar(0), Scalar(0.4 * WIDTH));
    // The y coordinate of the points is in [0, 1)
    c = trainClass.colRange(1,2);
    rng.fill(c, RNG::UNIFORM, Scalar(0), Scalar(HEIGHT));

    // Generate random points for the class 2
    trainClass = trainData.rowRange(2*NTRAINING_SAMPLES-nLinearSamples, 2*NTRAINING_SAMPLES);
    // The x coordinate of the points is in [0.6, 1]
    c = trainClass.colRange(0 , 1);
    rng.fill(c, RNG::UNIFORM, Scalar(0.6*WIDTH), Scalar(WIDTH));
    // The y coordinate of the points is in [0, 1)
    c = trainClass.colRange(1,2);
    rng.fill(c, RNG::UNIFORM, Scalar(0), Scalar(HEIGHT));
    //! [setup1]

    //------------------ Set up the non-linearly separable part of the training data ---------------
    //! [setup2]
    // Generate random points for the classes 1 and 2
    trainClass = trainData.rowRange(nLinearSamples, 2*NTRAINING_SAMPLES-nLinearSamples);
    // The x coordinate of the points is in [0.4, 0.6)
    c = trainClass.colRange(0,1);
    rng.fill(c, RNG::UNIFORM, Scalar(0.4*WIDTH), Scalar(0.6*WIDTH));
    // The y coordinate of the points is in [0, 1)
    c = trainClass.colRange(1,2);
    rng.fill(c, RNG::UNIFORM, Scalar(0), Scalar(HEIGHT));
    //! [setup2]

    //------------------------- Set up the labels for the classes ---------------------------------
    labels.rowRange(                0,   NTRAINING_SAMPLES).setTo(1);  // Class 1
    labels.rowRange(NTRAINING_SAMPLES, 2*NTRAINING_SAMPLES).setTo(2);  // Class 2

    //------------------------ 2. Set up the support vector machines parameters --------------------
    cout << "Starting training process" << endl;
    //! [init]
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setC(0.1);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)1e7, 1e-6));
    //! [init]

    //------------------------ 3. Train the svm ----------------------------------------------------
    //! [train]
    svm->train(trainData, ROW_SAMPLE, labels);
    //! [train]
    cout << "Finished training process" << endl;

    //------------------------ 4. Show the decision regions ----------------------------------------
    //! [show]
    Vec3b green(0,100,0), blue(100,0,0);
    for (int i = 0; i < I.rows; i++)
    {
        for (int j = 0; j < I.cols; j++)
        {
            Mat sampleMat = (Mat_<float>(1,2) << j, i);
            float response = svm->predict(sampleMat);

            if      (response == 1) I.at<Vec3b>(i,j) = green;
            else if (response == 2) I.at<Vec3b>(i,j) = blue;
        }
    }
    //! [show]

    //----------------------- 5. Show the training data --------------------------------------------
    //! [show_data]
    int thick = -1;
    float px, py;
    // Class 1
    for (int i = 0; i < NTRAINING_SAMPLES; i++)
    {
        px = trainData.at<float>(i,0);
        py = trainData.at<float>(i,1);
        circle(I, Point( (int) px,  (int) py ), 3, Scalar(0, 255, 0), thick);
    }
    // Class 2
    for (int i = NTRAINING_SAMPLES; i <2*NTRAINING_SAMPLES; i++)
    {
        px = trainData.at<float>(i,0);
        py = trainData.at<float>(i,1);
        circle(I, Point( (int) px, (int) py ), 3, Scalar(255, 0, 0), thick);
    }
    //! [show_data]

    //------------------------- 6. Show support vectors --------------------------------------------
    //! [show_vectors]
    thick = 2;
    Mat sv = svm->getUncompressedSupportVectors();

    for (int i = 0; i < sv.rows; i++)
    {
        const float* v = sv.ptr<float>(i);
        circle(I,  Point( (int) v[0], (int) v[1]), 6, Scalar(128, 128, 128), thick);
    }
    //! [show_vectors]

    imwrite("result.png", I);                      // save the Image
    imshow("SVM for Non-Linear Training Data", I); // show it to the user
    waitKey();
    return 0;
}
