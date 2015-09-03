/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2014, Itseez Inc, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include "svmsgd.hpp"

/****************************************************************************************\
*                        Stochastic Gradient Descent SVM Classifier                      *
\****************************************************************************************/

namespace cv {
namespace ml {

SVMSGD::SVMSGD(float lambda, float learnRate, uint nIterations){

    // Initialize with random seed
    _randomNumber = 1;

    // Initialize constants
    _slidingWindowSize = 0;
    _nFeatures = 0;
    _predictSlidingWindowSize = 1;

    // Initialize sliderCounter at index 0
    _sliderCounter = 0;

    // Parameters for learning
    _lambda = lambda;  // regularization
    _learnRate = learnRate;  // learning rate (ideally should be large at beginning and decay each iteration)
    _nIterations = nIterations;  // number of training iterations

    // True only in the first predict iteration
    _initPredict = true;

    // Online update flag
    _onlineUpdate = false;
}

SVMSGD::SVMSGD(uint updateFrequency, float learnRateDecay, float lambda, float learnRate, uint nIterations){

    // Initialize with random seed
    _randomNumber = 1;

    // Initialize constants
    _slidingWindowSize = 0;
    _nFeatures = 0;
    _predictSlidingWindowSize = updateFrequency;

    // Initialize sliderCounter at index 0
    _sliderCounter = 0;

    // Parameters for learning
    _lambda = lambda;  // regularization
    _learnRate = learnRate;  // learning rate (ideally should be large at beginning and decay each iteration)
    _nIterations = nIterations;  // number of training iterations

    // True only in the first predict iteration
    _initPredict = true;

    // Online update flag
    _onlineUpdate = true;

    // Learn rate decay: _learnRate = _learnRate * _learnDecay
    _learnRateDecay = learnRateDecay;
}

SVMSGD::~SVMSGD(){

}

SVMSGD* SVMSGD::clone() const{
    return new SVMSGD(*this);
}

void SVMSGD::train(cv::Mat trainFeatures, cv::Mat labels){

    // Initialize _nFeatures
    _slidingWindowSize = trainFeatures.rows;
    _nFeatures = trainFeatures.cols;

    float innerProduct;
    // Initialize weights vector with zeros
    if (_weights.size()==0){
        _weights.reserve(_nFeatures);
        for (uint feat = 0; feat < _nFeatures; ++feat){
            _weights.push_back(0.0);
        }
    }

    // Stochastic gradient descent SVM
    for (uint iter = 0; iter < _nIterations; ++iter){
        generateRandomIndex();
        innerProduct = calcInnerProduct(trainFeatures.ptr<float>(_randomIndex));
        int label = (labels.at<int>(_randomIndex,0) > 0) ? 1 : -1; // ensure that labels are -1 or 1
        updateWeights(innerProduct, trainFeatures.ptr<float>(_randomIndex), label );
    }
}

float SVMSGD::predict(cv::Mat newFeature){
    float innerProduct;

    if (_initPredict){
        _nFeatures = newFeature.cols;
        _slidingWindowSize = _predictSlidingWindowSize;
        _featuresSlider = cv::Mat::zeros(_slidingWindowSize, _nFeatures, CV_32F);
        _initPredict = false;
        _labelSlider = new float[_predictSlidingWindowSize]();
        _learnRate = _learnRate * _learnRateDecay;
    }

    innerProduct = calcInnerProduct(newFeature.ptr<float>(0));

    // Resultant label (-1 or 1)
    int label = (innerProduct>=0) ? 1 : -1;

    if (_onlineUpdate){
        // Update the featuresSlider with newFeature and _labelSlider with label
        newFeature.row(0).copyTo(_featuresSlider.row(_sliderCounter));
        _labelSlider[_sliderCounter] = float(label);

        // Update weights with a random index
        if (_sliderCounter == _slidingWindowSize-1){
            generateRandomIndex();
            updateWeights(innerProduct, _featuresSlider.ptr<float>(_randomIndex), int(_labelSlider[_randomIndex]) ;
        }

        // _sliderCounter++ if < _slidingWindowSize
        _sliderCounter = (_sliderCounter == _slidingWindowSize-1) ? 0 : (_sliderCounter+1);
    }

    return float(label);
}

void SVMSGD::generateRandomIndex(){
    // Choose random sample, using Mikolov's fast almost-uniform random number
    _randomNumber = _randomNumber * (unsigned long long) 25214903917 + 11;
    _randomIndex = uint(_randomNumber % (unsigned long long) _slidingWindowSize);
}

float SVMSGD::calcInnerProduct(float *rowDataPointer){
    float innerProduct = 0;
    for (uint feat = 0; feat < _nFeatures; ++feat){
        innerProduct += _weights[feat] * rowDataPointer[feat];
    }
    return innerProduct;
}

void SVMSGD::updateWeights(float innerProduct, float *rowDataPointer, int label){
    if (label * innerProduct > 1) {
        // Not a support vector, only apply weight decay
        for (uint feat = 0; feat < _nFeatures; feat++) {
            _weights[feat] -= _learnRate * _lambda * _weights[feat];
        }
    } else {
        // It's a support vector, add it to the weights
        for (uint feat = 0; feat < _nFeatures; feat++) {
            _weights[feat] -= _learnRate * (_lambda * _weights[feat] - label * rowDataPointer[feat]);
        }
    }
}

}
}
