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

#include "opencv2/opencv.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file)
        throw std::exception();
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        images.push_back(imread(path, 0));
        labels.push_back(atoi(classlabel.c_str()));
    }
}

int main(int argc, const char *argv[]) {
    // check for command line arguments
    if (argc != 2) {
        cout << "usage: " << argv[0] << " <csv.ext>" << endl;
        exit(1);
    }
    // path to your CSV
    string fn_csv = string(argv[1]);
    // images and corresponding labels
    vector<Mat> images;
    vector<int> labels;
    // read in the data
    try {
        read_csv(fn_csv, images, labels);
    } catch (exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\"." << endl;
        exit(1);
    }
    // get width and height
    //int width = images[0].cols;
    int height = images[0].rows;
    // get test instances
    Mat testSample = images[images.size() - 1];
    int testLabel = labels[labels.size() - 1];
    // ... and delete last element
    images.pop_back();
    labels.pop_back();
    // build the Fisherfaces model
    Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
    model->train(images, labels);
    // test model
    int predicted = model->predict(testSample);
    cout << "predicted class = " << predicted << endl;
    cout << "actual class = " << testLabel << endl;
    // get the eigenvectors
    Mat W = model->eigenvectors();
    // show first 10 fisherfaces
    for (int i = 0; i < min(10, W.cols); i++) {
        // get eigenvector #i
        Mat ev = W.col(i).clone();
        // reshape to original site
        Mat grayscale, cgrayscale;
        cvtColor(ev.reshape(1, height), grayscale, COLOR_BGR2GRAY);
        // show image (with Jet colormap)
        applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
        imshow(format("%d", i), cgrayscale);
    }
    waitKey(0);
    return 0;
}
