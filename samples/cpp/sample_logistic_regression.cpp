///////////////////////////////////////////////////////////////////////////////////////
// sample_logistic_regression.cpp
// IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.

//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.

// This is a sample program demostrating classification of digits 0 and 1 using Logistic Regression

// AUTHOR:
// Rahul Kavi rahulkavi[at]live[at]com
//

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;


int main()
{
    Mat data_temp, labels_temp;
    Mat data, labels;
    
    Mat data_train, data_test;
    Mat labels_train, labels_test;

    Mat responses, result;

    FileStorage f;

    cout<<"*****************************************************************************************"<<endl;
    cout<<"\"data01.xml\" contains digits 0 and 1 of 20 samples each, collected on an Android device"<<endl;
    cout<<"Each of the collected images are of size 28 x 28 re-arranged to 1 x 784 matrix"<<endl;
    cout<<"*****************************************************************************************\n\n"<<endl;

    cout<<"loading the dataset\n"<<endl;

    f.open("data01.xml", FileStorage::READ);

    f["datamat"] >> data_temp;
    f["labelsmat"] >> labels_temp;

    data_temp.convertTo(data, CV_32F);
    labels_temp.convertTo(labels, CV_32F);

    for(int i =0;i<data.rows;i++)
    {
        if(i%2 ==0)
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

    cout<<"training samples per class: "<<data_train.rows/2<<endl;
    cout<<"testing samples per class: "<<data_test.rows/2<<endl;
    
    // display sample image
    Mat img_disp1 = data_train.row(2).reshape(0,28).t();
    Mat img_disp2 = data_train.row(18).reshape(0,28).t();

    imshow("digit 0", img_disp1);
    imshow("digit 1", img_disp2);

    

    cout<<"initializing Logisitc Regression Parameters\n"<<endl;

    CvLR_TrainParams params = CvLR_TrainParams();

    params.alpha = 0.001;
    params.num_iters = 10;
    params.norm = CvLR::REG_L2;
    params.regularized = 1;
    params.train_method = CvLR::BATCH;

    cout<<"training Logisitc Regression classifier\n"<<endl;

    CvLR lr_(data_train, labels_train, params);
    lr_.predict(data_test, responses);
    labels_test.convertTo(labels_test, CV_32S);
    
    cout<<"Original Label ::  Predicted Label"<<endl;
    result = (labels_test == responses)/255;

    for(int i=0;i<labels_test.rows;i++)
    {
        cout<<labels_test.at<int>(i,0)<<" :: "<< responses.at<int>(i,0)<<endl;
    }
    
    // calculate accuracy
    cout<<"accuracy: "<<((double)cv::sum(result)[0]/result.rows)*100<<"%\n";
    cout<<"saving the classifier"<<endl;

    // save the classfier
    lr_.save("NewLR_Trained.xml");

    // load the classifier onto new object
    CvLR lr2;
    cout<<"loading a new classifier"<<endl;

    lr2.load("NewLR_Trained.xml");

    Mat responses2;

    // predict using loaded classifier
    cout<<"predicting the dataset using the loaded classfier\n"<<endl;

    lr2.predict(data_test, responses2);

    // calculate accuracy
    result = (labels_test == responses2)/255;
    cout<<"accuracy using loaded classifier: "<<((double)cv::sum(result)[0]/result.rows)*100<<"%\n";
    waitKey(0);

    return 0;
}
