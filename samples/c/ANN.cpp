#include "opencv2/core/core_c.h"
#include "opencv2/ml/ml.hpp"

#include <cstdio>
#include <ctime>

void trainMachine();
void Predict(float data1, float data2);

// The neural network
CvANN_MLP machineBrain;

CvRNG rng;

// Read the training data and train the network.
void trainMachine()
{
    int train_sample_count; //The number of training samples.

    //The training data matrix.
    //Note that we are limiting the number of training data samples to 1000 here.
    //The data sample consists of two inputs and an output. That's why 3.
    float td[1000][3];

    //Generate random train data, the output is the average of two input data
    train_sample_count = 1000;
    for (int i = 0; i < train_sample_count; i++)
    {
        td[i][0] = (float)(cvRandInt(&rng)%100);
        td[i][1] = (float)(cvRandInt(&rng)%100);
        td[i][2] = (float)((td[i][0] + td[i][1]) / 2.0);
    }

    //Create the matrices
    //Input data samples. Matrix of order (train_sample_count x 2)
    CvMat* trainData = cvCreateMat(train_sample_count, 2, CV_32FC1);

    //Output data samples. Matrix of order (train_sample_count x 1)
    CvMat* trainClasses = cvCreateMat(train_sample_count, 1, CV_32FC1);

    //The weight of each training data sample. We'll later set all to equal weights.
    CvMat* sampleWts = cvCreateMat(train_sample_count, 1, CV_32FC1);

    //The matrix representation of our ANN. We'll have four layers.
    CvMat* neuralLayers = cvCreateMat(4, 1, CV_32SC1);

    CvMat trainData1, trainClasses1, sampleWts1, neuralLayers1;

    cvGetRows(trainData, &trainData1, 0, train_sample_count);
    cvGetRows(trainClasses, &trainClasses1, 0, train_sample_count);
    cvGetRows(sampleWts, &sampleWts1, 0, train_sample_count);
    cvGetRows(neuralLayers, &neuralLayers1, 0, 4);

    //Setting the number of neurons on each layer of the ANN
    /*
    We have in  Layer 1: 2 neurons (2 inputs)
                Layer 2: 3 neurons (hidden layer)
                Layer 3: 3 neurons (hidden layer)
                Layer 4: 1 neurons (1 output)
    */
    cvSet1D(&neuralLayers1, 0, cvScalar(2));
    cvSet1D(&neuralLayers1, 1, cvScalar(3));
    cvSet1D(&neuralLayers1, 2, cvScalar(3));
    cvSet1D(&neuralLayers1, 3, cvScalar(1));

    //Assemble the ML training data.
    for (int i=0; i<train_sample_count; i++)
    {
        //Input 1
        cvSetReal2D(&trainData1, i, 0, td[i][0]);
        //Input 2
        cvSetReal2D(&trainData1, i, 1, td[i][1]);
        //Output
        cvSet1D(&trainClasses1, i, cvScalar(td[i][2]));
        //Weight (setting everything to 1)
        cvSet1D(&sampleWts1, i, cvScalar(1));
    }

    //Create our ANN.
    machineBrain.create(neuralLayers);

    //Train it with our data.
    machineBrain.train(
        trainData,
        trainClasses,
        sampleWts,
        0,
        CvANN_MLP_TrainParams(
            cvTermCriteria(
                CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,
                100000,
                1.0 ),
            CvANN_MLP_TrainParams::BACKPROP,
            0.01,
            0.05
        )
    );
}

// Predict the output with the trained ANN given the two inputs.
void Predict(float data1, float data2)
{
    float _sample[2];
    CvMat sample = cvMat(1, 2, CV_32FC1, _sample);
    float _predout[1];
    CvMat predout = cvMat(1, 1, CV_32FC1, _predout);
    sample.data.fl[0] = data1;
    sample.data.fl[1] = data2;

    machineBrain.predict(&sample, &predout);

    printf("(%.2f+%.2f)/2\t= %.2f -> %.2f\n", data1, data2, (data1+data2)/2, predout.data.fl[0]);
}

int main()
{
    rng = cvRNG((unsigned int)time(NULL));

    // Train the neural network with the samples
    trainMachine();

    // Now try predicting some values with the trained network
    for (int i=0; i<10; i++)
        Predict((float)(cvRandInt(&rng)%100), (float)(cvRandInt(&rng)%100));

    return 0;
}