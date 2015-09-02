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

#ifndef SVMSGD_H
#define SVMSGD_H

#include "precomp.hpp"

/****************************************************************************************\
*                        Stochastic Gradient Descent SVM Classifier                      *
\****************************************************************************************/


/*!
Stochastic Gradient Descent SVM classifier

SVMSGD provides a fast and easy-to-use implementation of the SVM classifier using the Stochastic Gradient Descent approach, as presented in @cite bottou2010large.
The gradient descent show amazing performance for large-scale problems, reducing the computing time. This allows a fast and reliable online update of the classifier for each new feature which
is fundamental when dealing with different conditions over time (like weather and illumination changes, for example).

First, create the SVMSGD object. To enable the online update, a value for updateFrequency should be defined.

Then the SVM model can be trained using the train features and the correspondent labels.

After that, the label of a new feature vector can be predicted using the predict function. If the updateFrequency was defined in the constructor, the predict function will update the weights automatically.

// Initialize object
SvmSgd SVMSGD;

// Train the Stochastic Gradient Descent SVM
SVMSGD.train(trainFeatures, labels);

// Predict label for the new feature vector (1xM)
predictedLabel = SVMSGD.predict(newFeatureVector);

*/

namespace cv
{
namespace ml
{

class CV_EXPORTS_W SVMSGD {

    public:
        /** @brief SGDSVM constructor.

        @param lambda regularization
        @param learnRate learning rate
        @param nIterations number of training iterations

        */
        SVMSGD(float lambda = 0.000001, float learnRate = 2, uint nIterations = 100000);

        /** @brief SGDSVM constructor.

        @param updateFrequency online update frequency
        @param lambda regularization
        @param learnRate learning rate
        @param nIterations number of training iterations

        */
        SVMSGD(uint updateFrequency, float lambda = 0.000001, float learnRate = 2, uint nIterations = 100000);
        virtual ~SVMSGD();
        virtual SVMSGD* clone() const;

        /** @brief Train the SGDSVM classifier.

        The function trains the SGDSVM classifier using the train features and the correspondent labels (-1 or 1).

        @param trainFeatures features used for training. Each row is a new sample.
        @param labels mat (size Nx1 with N = number of features) with the label of each training feature.

        */
        virtual void train(cv::Mat trainFeatures, cv::Mat labels);

        /** @brief Predict the label of a new feature vector.

        The function predicts and returns the label of a new feature vector, using the previously trained SVM model.

        @param newFeature new feature vector used for prediction

        */
        virtual float predict(cv::Mat newFeature);

        /** @brief Returns the weights of the trained model.

        */
        virtual std::vector<float> getWeights(){ return _weights; };

        /** @brief Sets the weights of the trained model.

        @weights weights used to predict the label of a new feature vector.

        */
        virtual void setWeights(std::vector<float> weights){ _weights = weights; };

    private:
        void updateWeights();
        void generateRandomIndex();
        float calcInnerProduct(float *rowDataPointer);
        void updateWeights(float innerProduct, float *rowDataPointer, int label);

        // Vector with SVM weights
        std::vector<float> _weights;

        // Random index generation
        long long int _randomNumber;
        unsigned int _randomIndex;

        // Number of features and samples
        unsigned int _nFeatures;
        unsigned int _nTrainSamples;

        // Parameters for learning
        float _lambda;  //regularization
        float _learnRate;  //learning rate
        unsigned int _nIterations; //number of training iterations

        // Vars to control the features slider matrix
        bool _onlineUpdate;
        bool _initPredict;
        uint _slidingWindowSize;
        uint _predictSlidingWindowSize;
        float* _labelSlider;
        float _learnRateDecay;

        // Mat with features slider and correspondent counter
        unsigned int _sliderCounter;
        cv::Mat _featuresSlider;

};

}
}

#endif
