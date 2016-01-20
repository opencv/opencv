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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#ifndef __OPENCV_ML_SVMSGD_HPP__
#define __OPENCV_ML_SVMSGD_HPP__

#ifdef __cplusplus

#include "opencv2/ml.hpp"

namespace cv
{
namespace ml
{


/****************************************************************************************\
*                        Stochastic Gradient Descent SVM Classifier                      *
\****************************************************************************************/

/*!
@brief Stochastic Gradient Descent SVM classifier

SVMSGD provides a fast and easy-to-use implementation of the SVM classifier using the Stochastic Gradient Descent approach, as presented in @cite bottou2010large.
The gradient descent show amazing performance for large-scale problems, reducing the computing time. This allows a fast and reliable online update of the classifier for each new feature which
is fundamental when dealing with variations of data over time (like weather and illumination changes in videosurveillance, for example).

First, create the SVMSGD object. To enable the online update, a value for updateFrequency should be defined.

Then the SVM model can be trained using the train features and the correspondent labels.

After that, the label of a new feature vector can be predicted using the predict function. If the updateFrequency was defined in the constructor, the predict function will update the weights automatically.

@code
// Initialize object
SVMSGD SvmSgd;

// Train the Stochastic Gradient Descent SVM
SvmSgd.train(trainFeatures, labels);

// Predict label for the new feature vector (1xM)
predictedLabel = SvmSgd.predict(newFeatureVector);
@endcode

*/

class CV_EXPORTS_W SVMSGD : public cv::ml::StatModel
{
public:

    enum SvmsgdType
    {
        ILLEGAL_VALUE,
        SGD,                                     //Stochastic Gradient Descent
        ASGD                                     //Average Stochastic Gradient Descent
    };

    /**
     * @return the weights of the trained model.
    */
    CV_WRAP virtual Mat getWeights() = 0;

    CV_WRAP virtual float getShift() = 0;

    CV_WRAP static Ptr<SVMSGD> create();    

    CV_WRAP virtual void setOptimalParameters(int type = ASGD) = 0;

    CV_WRAP virtual int getType() const = 0;

    CV_WRAP virtual void setType(int type) = 0;

    CV_WRAP virtual float getLambda() const = 0;

    CV_WRAP virtual void setLambda(float lambda) = 0;

    CV_WRAP virtual float getGamma0() const = 0;

    CV_WRAP virtual void setGamma0(float gamma0) = 0;

    CV_WRAP virtual float getC() const = 0;

    CV_WRAP virtual void setC(float c) = 0;

    CV_WRAP virtual cv::TermCriteria getTermCriteria() const = 0;

    CV_WRAP virtual void setTermCriteria(const cv::TermCriteria &val) = 0;
};

} //ml
} //cv

#endif  // __clpusplus
#endif  // __OPENCV_ML_SVMSGD_HPP
