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
#include "limits"

#include <iostream>

using std::cout;
using std::endl;

/****************************************************************************************\
*                        Stochastic Gradient Descent SVM Classifier                      *
\****************************************************************************************/

namespace cv
{
namespace ml
{

class SVMSGDImpl : public SVMSGD
{

public:
    SVMSGDImpl();

    virtual ~SVMSGDImpl() {}

    virtual bool train(const Ptr<TrainData>& data, int);

    virtual float predict( InputArray samples, OutputArray results=noArray(), int flags = 0 ) const;

    virtual bool isClassifier() const;

    virtual bool isTrained() const;

    virtual void clear();

    virtual void write(FileStorage &fs) const;

    virtual void read(const FileNode &fn);

    virtual Mat getWeights(){ return weights_; }

    virtual float getShift(){ return shift_; }

    virtual int getVarCount() const { return weights_.cols; }

    virtual String getDefaultName() const {return "opencv_ml_svmsgd";}

    virtual void setOptimalParameters(int svmsgdType = ASGD, int marginType = SOFT_MARGIN);

    virtual int getSvmsgdType() const;

    virtual void setSvmsgdType(int svmsgdType);

    virtual int getMarginType() const;

    virtual void setMarginType(int marginType);

    CV_IMPL_PROPERTY(float, Lambda, params.lambda)
    CV_IMPL_PROPERTY(float, Gamma0, params.gamma0)
    CV_IMPL_PROPERTY(float, C, params.c)
    CV_IMPL_PROPERTY_S(cv::TermCriteria, TermCriteria, params.termCrit)

private:
    void updateWeights(InputArray sample, bool isFirstClass, float gamma, Mat &weights);

    std::pair<bool,bool> areClassesEmpty(Mat responses);

    void writeParams( FileStorage &fs ) const;

    void readParams( const FileNode &fn );

    static inline bool isFirstClass(float val) { return val > 0; }

    static void normalizeSamples(Mat &matrix, Mat &average, float &multiplier);

    float calcShift(InputArray _samples, InputArray _responses) const;

    static void makeExtendedTrainSamples(const Mat &trainSamples, Mat &extendedTrainSamples, Mat &average, float &multiplier);



    // Vector with SVM weights
    Mat weights_;
    float shift_;

    // Parameters for learning
    struct SVMSGDParams
    {
        float lambda;                             //regularization
        float gamma0;                             //learning rate
        float c;
        TermCriteria termCrit;
        SvmsgdType svmsgdType;
        MarginType marginType;
    };

    SVMSGDParams params;
};

Ptr<SVMSGD> SVMSGD::create()
{
    return makePtr<SVMSGDImpl>();
}

std::pair<bool,bool> SVMSGDImpl::areClassesEmpty(Mat responses)
{
    CV_Assert(responses.cols == 1 || responses.rows == 1);
    std::pair<bool,bool> emptyInClasses(true, true);
    int limit_index = responses.rows;

    for(int index = 0; index < limit_index; index++)
    {
        if (isFirstClass(responses.at<float>(index)))
            emptyInClasses.first = false;
        else
            emptyInClasses.second = false;

        if (!emptyInClasses.first && ! emptyInClasses.second)
            break;
    }

    return emptyInClasses;
}

void SVMSGDImpl::normalizeSamples(Mat &samples, Mat &average, float &multiplier)
{
    int featuresCount = samples.cols;
    int samplesCount = samples.rows;

    average = Mat(1, featuresCount, samples.type());
    for (int featureIndex = 0; featureIndex < featuresCount; featureIndex++)
    {
        average.at<float>(featureIndex) = mean(samples.col(featureIndex))[0];
    }

    for (int sampleIndex = 0; sampleIndex < samplesCount; sampleIndex++)
    {
        samples.row(sampleIndex) -= average;
    }

    double normValue = norm(samples);

    multiplier = sqrt(samples.total()) / normValue;

    samples *= multiplier;
}

void SVMSGDImpl::makeExtendedTrainSamples(const Mat &trainSamples, Mat &extendedTrainSamples, Mat &average, float &multiplier)
{
    Mat normalisedTrainSamples = trainSamples.clone();
    int samplesCount = normalisedTrainSamples.rows;

    normalizeSamples(normalisedTrainSamples, average, multiplier);

    Mat onesCol = Mat::ones(samplesCount, 1, CV_32F);
    cv::hconcat(normalisedTrainSamples, onesCol, extendedTrainSamples);
}

void SVMSGDImpl::updateWeights(InputArray _sample, bool firstClass, float gamma, Mat& weights)
{
    Mat sample = _sample.getMat();

    int response = firstClass ? 1 : -1; // ensure that trainResponses are -1 or 1

    if ( sample.dot(weights) * response > 1)
    {
        // Not a support vector, only apply weight decay
        weights *= (1.f - gamma * params.lambda);
    }
    else
    {
        // It's a support vector, add it to the weights
        weights -= (gamma * params.lambda) * weights - (gamma * response) * sample;
    }
}

float SVMSGDImpl::calcShift(InputArray _samples, InputArray _responses) const
{
    float distanceToClasses[2] = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };

    Mat trainSamples = _samples.getMat();
    int trainSamplesCount = trainSamples.rows;

    Mat trainResponses = _responses.getMat();

    for (int samplesIndex = 0; samplesIndex < trainSamplesCount; samplesIndex++)
    {
        Mat currentSample = trainSamples.row(samplesIndex);
        float dotProduct = currentSample.dot(weights_);

        bool firstClass = isFirstClass(trainResponses.at<float>(samplesIndex));
        int index = firstClass ? 0:1;
        float signToMul = firstClass ? 1 : -1;
        float curDistance = dotProduct * signToMul;

        if (curDistance < distanceToClasses[index])
        {
            distanceToClasses[index] = curDistance;
        }
    }

    return -(distanceToClasses[0] - distanceToClasses[1]) / 2.f;
}

bool SVMSGDImpl::train(const Ptr<TrainData>& data, int)
{
    clear();
    CV_Assert( isClassifier() );   //toDo: consider

    Mat trainSamples = data->getTrainSamples();

    int featureCount = trainSamples.cols;
    Mat trainResponses = data->getTrainResponses();        // (trainSamplesCount x 1) matrix

    std::pair<bool,bool> areEmpty = areClassesEmpty(trainResponses);

    if ( areEmpty.first && areEmpty.second )
    {
        return false;
    }
    if ( areEmpty.first || areEmpty.second )
    {
        weights_ = Mat::zeros(1, featureCount, CV_32F);
        shift_ = areEmpty.first ? -1 : 1;
        return true;
    }

    Mat extendedTrainSamples;
    Mat average;
    float multiplier = 0;
    makeExtendedTrainSamples(trainSamples, extendedTrainSamples, average, multiplier);

    int extendedTrainSamplesCount = extendedTrainSamples.rows;
    int extendedFeatureCount = extendedTrainSamples.cols;

    Mat extendedWeights = Mat::zeros(1, extendedFeatureCount, CV_32F);         // Initialize extendedWeights vector with zeros
    Mat previousWeights = Mat::zeros(1, extendedFeatureCount, CV_32F);     //extendedWeights vector for calculating terminal criterion
    Mat averageExtendedWeights;                                        //average extendedWeights vector for ASGD model
    if (params.svmsgdType == ASGD)
    {
        averageExtendedWeights = Mat::zeros(1, extendedFeatureCount, CV_32F);
    }

    RNG rng(0);

    CV_Assert (params.termCrit.type & TermCriteria::COUNT || params.termCrit.type & TermCriteria::EPS);
    int maxCount = (params.termCrit.type & TermCriteria::COUNT) ? params.termCrit.maxCount : INT_MAX;
    double epsilon = (params.termCrit.type & TermCriteria::EPS) ? params.termCrit.epsilon : 0;

    double err = DBL_MAX;
    // Stochastic gradient descent SVM
    for (int iter = 0; (iter < maxCount) && (err > epsilon); iter++)
    {
        int randomNumber = rng.uniform(0, extendedTrainSamplesCount);             //generate sample number

        Mat currentSample = extendedTrainSamples.row(randomNumber);
        bool firstClass = isFirstClass(trainResponses.at<float>(randomNumber));

        float gamma = params.gamma0 * std::pow((1 + params.lambda * params.gamma0 * (float)iter), (-params.c));    //update gamma

        updateWeights( currentSample, firstClass, gamma, extendedWeights );

        //average weights (only for ASGD model)
        if (params.svmsgdType == ASGD)
        {
            averageExtendedWeights = ((float)iter/ (1 + (float)iter)) * averageExtendedWeights  + extendedWeights / (1 + (float) iter);
            err = norm(averageExtendedWeights - previousWeights);
            averageExtendedWeights.copyTo(previousWeights);
        }
        else
        {
             err = norm(extendedWeights - previousWeights);
             extendedWeights.copyTo(previousWeights);
        }
    }

    if (params.svmsgdType == ASGD)
    {
        extendedWeights = averageExtendedWeights;
    }

    Rect roi(0, 0, featureCount, 1);
    weights_ = extendedWeights(roi);
    weights_ *= multiplier;

    CV_Assert(params.marginType == SOFT_MARGIN || params.marginType == HARD_MARGIN);

    if (params.marginType == SOFT_MARGIN)
    {
        shift_ = extendedWeights.at<float>(featureCount) - weights_.dot(average);
    }
    else
    {
        shift_ = calcShift(trainSamples, trainResponses);
    }

    return true;
}


float SVMSGDImpl::predict( InputArray _samples, OutputArray _results, int ) const
{
    float result = 0;
    cv::Mat samples = _samples.getMat();
    int nSamples = samples.rows;
    cv::Mat results;

    CV_Assert( samples.cols == weights_.cols && samples.type() == CV_32F );

    if( _results.needed() )
    {
        _results.create( nSamples, 1, samples.type() );
        results = _results.getMat();
    }
    else
    {
        CV_Assert( nSamples == 1 );
        results = Mat(1, 1, CV_32F, &result);
    }

    for (int sampleIndex = 0; sampleIndex < nSamples; sampleIndex++)
    {
        Mat currentSample = samples.row(sampleIndex);
        float criterion = currentSample.dot(weights_) + shift_;
        results.at<float>(sampleIndex) = (criterion >= 0) ? 1 : -1;
    }

    return result;
}

bool SVMSGDImpl::isClassifier() const
{
    return (params.svmsgdType == SGD || params.svmsgdType == ASGD)
            &&
            (params.marginType == SOFT_MARGIN || params.marginType == HARD_MARGIN)
            &&
            (params.lambda > 0) && (params.gamma0 > 0) && (params.c >= 0);
}

bool SVMSGDImpl::isTrained() const
{
    return !weights_.empty();
}

void SVMSGDImpl::write(FileStorage& fs) const
{
    if( !isTrained() )
        CV_Error( CV_StsParseError, "SVMSGD model data is invalid, it hasn't been trained" );

    writeParams( fs );

    fs << "weights" << weights_;
    fs << "shift" << shift_;
}

void SVMSGDImpl::writeParams( FileStorage& fs ) const
{
    String SvmsgdTypeStr;

    switch (params.svmsgdType)
    {
    case SGD:
        SvmsgdTypeStr = "SGD";
        break;
    case ASGD:
        SvmsgdTypeStr = "ASGD";
        break;
    case ILLEGAL_SVMSGD_TYPE:
        SvmsgdTypeStr = format("Unknown_%d", params.svmsgdType);
    default:
        std::cout << "params.svmsgdType isn't initialized" << std::endl;
    }

    fs << "svmsgdType" << SvmsgdTypeStr;

    String marginTypeStr;

    switch (params.marginType)
    {
    case SOFT_MARGIN:
        marginTypeStr = "SOFT_MARGIN";
        break;
    case HARD_MARGIN:
        marginTypeStr = "HARD_MARGIN";
        break;
    case ILLEGAL_MARGIN_TYPE:
        marginTypeStr = format("Unknown_%d", params.marginType);
    default:
        std::cout << "params.marginType isn't initialized" << std::endl;
    }

    fs << "marginType" << marginTypeStr;

    fs << "lambda" << params.lambda;
    fs << "gamma0" << params.gamma0;
    fs << "c" << params.c;

    fs << "term_criteria" << "{:";
    if( params.termCrit.type & TermCriteria::EPS )
        fs << "epsilon" << params.termCrit.epsilon;
    if( params.termCrit.type & TermCriteria::COUNT )
        fs << "iterations" << params.termCrit.maxCount;
    fs << "}";
}

void SVMSGDImpl::read(const FileNode& fn)
{
    clear();

    readParams(fn);

    fn["weights"] >> weights_;
    fn["shift"] >> shift_;
}

void SVMSGDImpl::readParams( const FileNode& fn )
{
    String svmsgdTypeStr = (String)fn["svmsgdType"];
    SvmsgdType svmsgdType =
            svmsgdTypeStr == "SGD" ? SGD :
                                     svmsgdTypeStr == "ASGD" ? ASGD : ILLEGAL_SVMSGD_TYPE;

    if( svmsgdType == ILLEGAL_SVMSGD_TYPE )
        CV_Error( CV_StsParseError, "Missing or invalid SVMSGD type" );

    params.svmsgdType = svmsgdType;

    String marginTypeStr = (String)fn["marginType"];
    MarginType marginType =
            marginTypeStr == "SOFT_MARGIN" ? SOFT_MARGIN :
                                     marginTypeStr == "HARD_MARGIN" ? HARD_MARGIN : ILLEGAL_MARGIN_TYPE;

    if( marginType == ILLEGAL_MARGIN_TYPE )
        CV_Error( CV_StsParseError, "Missing or invalid margin type" );

    params.marginType = marginType;

    CV_Assert ( fn["lambda"].isReal() );
    params.lambda = (float)fn["lambda"];

    CV_Assert ( fn["gamma0"].isReal() );
    params.gamma0 = (float)fn["gamma0"];

    CV_Assert ( fn["c"].isReal() );
    params.c = (float)fn["c"];

    FileNode tcnode = fn["term_criteria"];
    if( !tcnode.empty() )
    {
        params.termCrit.epsilon = (double)tcnode["epsilon"];
        params.termCrit.maxCount = (int)tcnode["iterations"];
        params.termCrit.type = (params.termCrit.epsilon > 0 ? TermCriteria::EPS : 0) +
                (params.termCrit.maxCount > 0 ? TermCriteria::COUNT : 0);
    }
    else
        params.termCrit = TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 100000, FLT_EPSILON );

}

void SVMSGDImpl::clear()
{
    weights_.release();
}


SVMSGDImpl::SVMSGDImpl()
{
    clear();

    params.svmsgdType = ILLEGAL_SVMSGD_TYPE;
    params.marginType = ILLEGAL_MARGIN_TYPE;

    // Parameters for learning
    params.lambda = 0;                              // regularization
    params.gamma0 = 0;                        // learning rate (ideally should be large at beginning and decay each iteration)
    params.c = 0;

    TermCriteria _termCrit(TermCriteria::COUNT + TermCriteria::EPS, 0, 0);
    params.termCrit = _termCrit;
}

void SVMSGDImpl::setOptimalParameters(int svmsgdType, int marginType)
{
    switch (svmsgdType)
    {
    case SGD:
        params.svmsgdType = SGD;
        params.marginType = (marginType == SOFT_MARGIN) ? SOFT_MARGIN :
                            (marginType == HARD_MARGIN) ? HARD_MARGIN : ILLEGAL_MARGIN_TYPE;
        params.lambda = 0.0001;
        params.gamma0 = 0.05;
        params.c = 1;
        params.termCrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100000, 0.00001);
        break;

    case ASGD:
        params.svmsgdType = ASGD;
        params.marginType = (marginType == SOFT_MARGIN) ? SOFT_MARGIN :
                            (marginType == HARD_MARGIN) ? HARD_MARGIN : ILLEGAL_MARGIN_TYPE;
        params.lambda = 0.00001;
        params.gamma0 = 0.05;
        params.c = 0.75;
        params.termCrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100000, 0.00001);
        break;

    default:
        CV_Error( CV_StsParseError, "SVMSGD model data is invalid" );
    }
}

void SVMSGDImpl::setSvmsgdType(int type)
{
    switch (type)
    {
    case SGD:
        params.svmsgdType = SGD;
        break;
    case ASGD:
        params.svmsgdType = ASGD;
        break;
    default:
        params.svmsgdType = ILLEGAL_SVMSGD_TYPE;
    }
}

int SVMSGDImpl::getSvmsgdType() const
{
    return params.svmsgdType;
}

void SVMSGDImpl::setMarginType(int type)
{
    switch (type)
    {
    case HARD_MARGIN:
        params.marginType = HARD_MARGIN;
        break;
    case SOFT_MARGIN:
        params.marginType = SOFT_MARGIN;
        break;
    default:
        params.marginType = ILLEGAL_MARGIN_TYPE;
    }
}

int SVMSGDImpl::getMarginType() const
{
    return params.marginType;
}
}   //ml
}   //cv
