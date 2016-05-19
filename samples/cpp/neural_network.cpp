#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

int main()
{
    //create random training data
    Mat_<float> data(100, 100);
    randn(data, Mat::zeros(1, 1, data.type()), Mat::ones(1, 1, data.type()));

    //half of the samples for each class
    Mat_<float> responses(data.rows, 2);
    for (int i = 0; i<data.rows; ++i)
    {
        if (i < data.rows/2)
        {
            data(i, 0) = 1;
            data(i, 1) = 0;
        }
        else
        {
            data(i, 0) = 0;
            data(i, 1) = 1;
        }
    }

    /*
    //example code for just a single response (regression)
    Mat_<float> responses(data.rows, 1);
    for (int i=0; i<responses.rows; ++i)
        responses(i, 0) = i < responses.rows / 2 ? 0 : 1;
    */

    //create the neural network
    Mat_<int> layerSizes(1, 3);
    layerSizes(0, 0) = data.cols;
    layerSizes(0, 1) = 20;
    layerSizes(0, 2) = responses.cols;

    Ptr<ANN_MLP> network = ANN_MLP::create();
    network->setLayerSizes(layerSizes);
    network->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0.1, 0.1);
    network->setTrainMethod(ANN_MLP::BACKPROP, 0.1, 0.1);
    Ptr<TrainData> trainData = TrainData::create(data, ROW_SAMPLE, responses);

    network->train(trainData);
    if (network->isTrained())
    {
        printf("Predict one-vector:\n");
        Mat result;
        network->predict(Mat::ones(1, data.cols, data.type()), result);
        cout << result << endl;

        printf("Predict training data:\n");
        for (int i=0; i<data.rows; ++i)
        {
            network->predict(data.row(i), result);
            cout << result << endl;
        }
    }

    return 0;
}
