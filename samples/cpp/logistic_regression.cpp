#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

// static void showImage(const Mat &data, int columns, const String &name)
// {
//     Mat bigImage;
//     for(int i = 0; i < data.rows; ++i)
//     {
//         bigImage.push_back(data.row(i).reshape(0, columns));
//     }
//     imshow(name, bigImage.t());
// }

static float calculateAccuracyPercent(const Mat &original, const Mat &predicted)
{
    return 100 * (float)countNonZero(original == predicted) / predicted.rows;
}

int main()
{
    const String filename = samples::findFile("data01.xml");
    cout << "**********************************************************************" << endl;
    cout << filename
         << " contains digits 0 and 1 of 20 samples each, collected on an Android device" << endl;
    cout << "Each of the collected images are of size 28 x 28 re-arranged to 1 x 784 matrix"
         << endl;
    cout << "**********************************************************************" << endl;

    Mat data, labels;
    {
        cout << "loading the dataset...";
        FileStorage f;
        if(f.open(filename, FileStorage::READ))
        {
            f["datamat"] >> data;
            f["labelsmat"] >> labels;
            f.release();
        }
        else
        {
            cerr << "file can not be opened: " << filename << endl;
            return 1;
        }
        data.convertTo(data, CV_32F);
        labels.convertTo(labels, CV_32F);
        cout << "read " << data.rows << " rows of data" << endl;
    }

    Mat data_train, data_test;
    Mat labels_train, labels_test;
    for(int i = 0; i < data.rows; i++)
    {
        if(i % 2 == 0)
        {
            data_train.push_back(data.row(i));
            labels_train.push_back(labels.row(i));
        }
        else
        {
            data_test.push_back(data.row(i));
            labels_test.push_back(labels.row(i));
        }
    }
    cout << "training/testing samples count: " << data_train.rows << "/" << data_test.rows << endl;

    // display sample image
    // showImage(data_train, 28, "train data");
    // showImage(data_test, 28, "test data");

    // simple case with batch gradient
    cout << "training...";
    //! [init]
    Ptr<LogisticRegression> lr1 = LogisticRegression::create();
    lr1->setLearningRate(0.001);
    lr1->setIterations(10);
    lr1->setRegularization(LogisticRegression::REG_L2);
    lr1->setTrainMethod(LogisticRegression::BATCH);
    lr1->setMiniBatchSize(1);
    //! [init]
    lr1->train(data_train, ROW_SAMPLE, labels_train);
    cout << "done!" << endl;

    cout << "predicting...";
    Mat responses;
    lr1->predict(data_test, responses);
    cout << "done!" << endl;

    // show prediction report
    cout << "original vs predicted:" << endl;
    labels_test.convertTo(labels_test, CV_32S);
    cout << labels_test.t() << endl;
    cout << responses.t() << endl;
    cout << "accuracy: " << calculateAccuracyPercent(labels_test, responses) << "%" << endl;

    // save the classifier
    const String saveFilename = "NewLR_Trained.xml";
    cout << "saving the classifier to " << saveFilename << endl;
    lr1->save(saveFilename);

    // load the classifier onto new object
    cout << "loading a new classifier from " << saveFilename << endl;
    Ptr<LogisticRegression> lr2 = StatModel::load<LogisticRegression>(saveFilename);

    // predict using loaded classifier
    cout << "predicting the dataset using the loaded classifier...";
    Mat responses2;
    lr2->predict(data_test, responses2);
    cout << "done!" << endl;

    // calculate accuracy
    cout << labels_test.t() << endl;
    cout << responses2.t() << endl;
    cout << "accuracy: " << calculateAccuracyPercent(labels_test, responses2) << "%" << endl;

    // waitKey(0);
    return 0;
}

