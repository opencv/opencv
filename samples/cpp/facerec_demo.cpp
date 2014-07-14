/*
 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, std::map<int, string>& labelsInfo, char separator = ';') {
    ifstream csv(filename.c_str());
    if (!csv) CV_Error(CV_StsBadArg, "No valid input file was given, please check the given filename.");
    string line, path, classlabel, info;
    while (getline(csv, line)) {
        stringstream liness(line);
        path.clear(); classlabel.clear(); info.clear();
        getline(liness, path, separator);
        getline(liness, classlabel, separator);
        getline(liness, info, separator);
        if(!path.empty() && !classlabel.empty()) {
            cout << "Processing " << path << endl;
            int label = atoi(classlabel.c_str());
            if(!info.empty())
                labelsInfo.insert(std::make_pair(label, info));
            // 'path' can be file, dir or wildcard path
            String root(path.c_str());
            vector<String> files;
            glob(root, files, true);
            for(vector<String>::const_iterator f = files.begin(); f != files.end(); ++f) {
                cout << "\t" << *f << endl;
                Mat img = imread(*f, CV_LOAD_IMAGE_GRAYSCALE);
                static int w=-1, h=-1;
                static bool showSmallSizeWarning = true;
                if(w>0 && h>0 && (w!=img.cols || h!=img.rows)) cout << "\t* Warning: images should be of the same size!" << endl;
                if(showSmallSizeWarning && (img.cols<50 || img.rows<50)) {
                    cout << "* Warning: for better results images should be not smaller than 50x50!" << endl;
                    showSmallSizeWarning = false;
                }
                images.push_back(img);
                labels.push_back(label);
            }
        }
    }
}

int main(int argc, const char *argv[]) {
    // Check for valid command line arguments, print usage
    // if no arguments were given.
    if (argc != 2 && argc != 3) {
        cout << "Usage: " << argv[0] << " <csv> [arg2]\n"
             << "\t<csv> - path to config file in CSV format\n"
             << "\targ2 - if the 2nd argument is provided (with any value) "
             << "the advanced stuff is run and shown to console.\n"
             << "The CSV config file consists of the following lines:\n"
             << "<path>;<label>[;<comment>]\n"
             << "\t<path> - file, dir or wildcard path\n"
             << "\t<label> - non-negative integer person label\n"
             << "\t<comment> - optional comment string (e.g. person name)"
             << endl;
        exit(1);
    }
    // Get the path to your CSV.
    string fn_csv = string(argv[1]);
    // These vectors hold the images and corresponding labels.
    vector<Mat> images;
    vector<int> labels;
    std::map<int, string> labelsInfo;
    // Read in the data. This can fail if no valid
    // input filename is given.
    try {
        read_csv(fn_csv, images, labels, labelsInfo);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }

    // Quit if there are not enough images for this demo.
    if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }
    // The following lines simply get the last images from
    // your dataset and remove it from the vector. This is
    // done, so that the training data (which we learn the
    // cv::FaceRecognizer on) and the test data we test
    // the model with, do not overlap.
    Mat testSample = images[images.size() - 1];
    int testLabel = labels[labels.size() - 1];
    images.pop_back();
    labels.pop_back();
    // The following lines create an Eigenfaces model for
    // face recognition and train it with the images and
    // labels read from the given CSV file.
    // This here is a full PCA, if you just want to keep
    // 10 principal components (read Eigenfaces), then call
    // the factory method like this:
    //
    //      cv::createEigenFaceRecognizer(10);
    //
    // If you want to create a FaceRecognizer with a
    // confidennce threshold, call it with:
    //
    //      cv::createEigenFaceRecognizer(10, 123.0);
    //
    Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
    model->setLabelsInfo(labelsInfo);
    model->train(images, labels);
    string saveModelPath = "face-rec-model.txt";
    cout << "Saving the trained model to " << saveModelPath << endl;
    model->save(saveModelPath);

    // The following line predicts the label of a given
    // test image:
    int predictedLabel = model->predict(testSample);
    //
    // To get the confidence of a prediction call the model with:
    //
    //      int predictedLabel = -1;
    //      double confidence = 0.0;
    //      model->predict(testSample, predictedLabel, confidence);
    //
    string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
    cout << result_message << endl;
    if( (predictedLabel == testLabel) && !model->getLabelInfo(predictedLabel).empty() )
        cout << format("%d-th label's info: %s", predictedLabel, model->getLabelInfo(predictedLabel).c_str()) << endl;

    // advanced stuff
    if(argc>2) {
        // Sometimes you'll need to get/set internal model data,
        // which isn't exposed by the public cv::FaceRecognizer.
        // Since each cv::FaceRecognizer is derived from a
        // cv::Algorithm, you can query the data.
        //
        // First we'll use it to set the threshold of the FaceRecognizer
        // to 0.0 without retraining the model. This can be useful if
        // you are evaluating the model:
        //
        model->set("threshold", 0.0);
        // Now the threshold of this model is set to 0.0. A prediction
        // now returns -1, as it's impossible to have a distance below
        // it
        predictedLabel = model->predict(testSample);
        cout << "Predicted class = " << predictedLabel << endl;
        // Here is how to get the eigenvalues of this Eigenfaces model:
        Mat eigenvalues = model->getMat("eigenvalues");
        // And we can do the same to display the Eigenvectors (read Eigenfaces):
        Mat W = model->getMat("eigenvectors");
        // From this we will display the (at most) first 10 Eigenfaces:
        for (int i = 0; i < min(10, W.cols); i++) {
            string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
            cout << msg << endl;
            // get eigenvector #i
            Mat ev = W.col(i).clone();
            // Reshape to original size & normalize to [0...255] for imshow.
            Mat grayscale;
            normalize(ev.reshape(1), grayscale, 0, 255, NORM_MINMAX, CV_8UC1);
            // Show the image & apply a Jet colormap for better sensing.
            Mat cgrayscale;
            applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
            imshow(format("%d", i), cgrayscale);
        }
        waitKey(0);
    }
    return 0;
}
