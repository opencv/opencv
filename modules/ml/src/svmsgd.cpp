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
// Copyright (C) 2016, Itseez Inc, all rights reserved.
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

class SVMSGDImpl CV_FINAL : public SVMSGD
{

public:
    SVMSGDImpl();

    virtual ~SVMSGDImpl() {}

    virtual bool train(const Ptr<TrainData>& data, int) CV_OVERRIDE;

    virtual float predict( InputArray samples, OutputArray results=noArray(), int flags = 0 ) const CV_OVERRIDE;

    virtual bool isClassifier() const CV_OVERRIDE;

    virtual bool isTrained() const CV_OVERRIDE;

    virtual void clear() CV_OVERRIDE;

    virtual void write(FileStorage &fs) const CV_OVERRIDE;

    virtual void read(const FileNode &fn) CV_OVERRIDE;

    virtual Mat getWeights() CV_OVERRIDE { return weights_; }

    virtual float getShift() CV_OVERRIDE { return shift_; }

    virtual int getVarCount() const CV_OVERRIDE { return weights_.cols; }

    virtual String getDefaultName() const CV_OVERRIDE {return "opencv_ml_svmsgd";}

    virtual void setOptimalParameters(int svmsgdType = ASGD, int marginType = SOFT_MARGIN) CV_OVERRIDE;

    inline int getSvmsgdType() const CV_OVERRIDE { return params.svmsgdType; }
    inline void setSvmsgdType(int val) CV_OVERRIDE { params.svmsgdType = val; }
    inline int getMarginType() const CV_OVERRIDE { return params.marginType; }
    inline void setMarginType(int val) CV_OVERRIDE { params.marginType = val; }
    inline float getMarginRegularization() const CV_OVERRIDE { return params.marginRegularization; }
    inline void setMarginRegularization(float val) CV_OVERRIDE { params.marginRegularization = val; }
    inline float getInitialStepSize() const CV_OVERRIDE { return params.initialStepSize; }
    inline void setInitialStepSize(float val) CV_OVERRIDE { params.initialStepSize = val; }
    inline float getStepDecreasingPower() const CV_OVERRIDE { return params.stepDecreasingPower; }
    inline void setStepDecreasingPower(float val) CV_OVERRIDE { params.stepDecreasingPower = val; }
    inline cv::TermCriteria getTermCriteria() const CV_OVERRIDE { return params.termCrit; }
    inline void setTermCriteria(const cv::TermCriteria& val) CV_OVERRIDE { params.termCrit = val; }

private:
    void updateWeights(InputArray sample, bool positive, float stepSize, Mat &weights);

    void writeParams( FileStorage &fs ) const;

    void readParams( const FileNode &fn );

    static inline bool isPositive(float val) { return val > 0; }

    static void normalizeSamples(Mat &matrix, Mat &average, float &multiplier);

    float calcShift(InputArray _samples, InputArray _responses) const;

    static void makeExtendedTrainSamples(const Mat &trainSamples, Mat &extendedTrainSamples, Mat &average, float &multiplier);

    // Vector with SVM weights
    Mat weights_;
    float shift_;

    // Parameters for learning
    struct SVMSGDParams
    {
        float marginRegularization;
        float initialStepSize;
        float stepDecreasingPower;
        TermCriteria termCrit;
        int svmsgdType;
        int marginType;
    };

    SVMSGDParams params;
};

Ptr<SVMSGD> SVMSGD::create()
{
    return makePtr<SVMSGDImpl>();
}

Ptr<SVMSGD> SVMSGD::load(const String& filepath, const String& nodeName)
{
    return Algorithm::load<SVMSGD>(filepath, nodeName);
}


void SVMSGDImpl::normalizeSamples(Mat &samples, Mat &average, float &multiplier)
{
    int featuresCount = samples.cols;
    int samplesCount = samples.rows;

    average = Mat(1, featuresCount, samples.type());
    CV_Assert(average.type() ==  CV_32FC1);
    for (int featureIndex = 0; featureIndex < featuresCount; featureIndex++)
    {
        average.at<float>(featureIndex) = static_cast<float>(mean(samples.col(featureIndex))[0]);
    }

    for (int sampleIndex = 0; sampleIndex < samplesCount; sampleIndex++)
    {
        samples.row(sampleIndex) -= average;
    }

    double normValue = norm(samples);

    multiplier = static_cast<float>(sqrt(static_cast<double>(samples.total())) / normValue);

    samples *= multiplier;
}

void SVMSGDImpl::makeExtendedTrainSamples(const Mat &trainSamples, Mat &extendedTrainSamples, Mat &average, float &multiplier)
{
    Mat normalizedTrainSamples = trainSamples.clone();
    int samplesCount = normalizedTrainSamples.rows;

    normalizeSamples(normalizedTrainSamples, average, multiplier);

    Mat onesCol = Mat::ones(samplesCount, 1, CV_32F);
    cv::hconcat(normalizedTrainSamples, onesCol, extendedTrainSamples);
}

void SVMSGDImpl::updateWeights(InputArray _sample, bool positive, float stepSize, Mat& weights)
{
    Mat sample = _sample.getMat();

    int response = positive ? 1 : -1; // ensure that trainResponses are -1 or 1

    if ( sample.dot(weights) * response > 1)
    {
        // Not a support vector, only apply weight decay
        weights *= (1.f - stepSize * params.marginRegularization);
    }
    else
    {
        // It's a support vector, add it to the weights
        weights -= (stepSize * params.marginRegularization) * weights - (stepSize * response) * sample;
    }
}

float SVMSGDImpl::calcShift(InputArray _samples, InputArray _responses) const
{
    float margin[2] = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };

    Mat trainSamples = _samples.getMat();
    int trainSamplesCount = trainSamples.rows;

    Mat trainResponses = _responses.getMat();

    CV_Assert(trainResponses.type() ==  CV_32FC1);
    for (int samplesIndex = 0; samplesIndex < trainSamplesCount; samplesIndex++)
    {
        Mat currentSample = trainSamples.row(samplesIndex);
        float dotProduct = static_cast<float>(currentSample.dot(weights_));

        bool positive = isPositive(trainResponses.at<float>(samplesIndex));
        int index = positive ? 0 : 1;
        float signToMul = positive ? 1.f : -1.f;
        float curMargin = dotProduct * signToMul;

        if (curMargin < margin[index])
        {
            margin[index] = curMargin;
        }
    }

    return -(margin[0] - margin[1]) / 2.f;
}

bool SVMSGDImpl::train(const Ptr<TrainData>& data, int)
{
    CV_Assert(!data.empty());
    clear();
    CV_Assert( isClassifier() );   //toDo: consider

    Mat trainSamples = data->getTrainSamples();

    int featureCount = trainSamples.cols;
    Mat trainResponses = data->getTrainResponses();        // (trainSamplesCount x 1) matrix

    CV_Assert(trainResponses.rows == trainSamples.rows);

    if (trainResponses.empty())
    {
        return false;
    }

    int positiveCount = countNonZero(trainResponses >= 0);
    int negativeCount = countNonZero(trainResponses < 0);

    if ( positiveCount <= 0 || negativeCount <= 0 )
    {
        weights_ = Mat::zeros(1, featureCount, CV_32F);
        shift_ = (positiveCount > 0) ? 1.f : -1.f;
        return true;
    }

    Mat extendedTrainSamples;
    Mat average;
    float multiplier = 0;
    makeExtendedTrainSamples(trainSamples, extendedTrainSamples, average, multiplier);

    int extendedTrainSamplesCount = extendedTrainSamples.rows;
    int extendedFeatureCount = extendedTrainSamples.cols;

    Mat extendedWeights = Mat::zeros(1, extendedFeatureCount, CV_32F);
    Mat previousWeights = Mat::zeros(1, extendedFeatureCount, CV_32F);
    Mat averageExtendedWeights;
    if (params.svmsgdType == ASGD)
    {
        averageExtendedWeights = Mat::zeros(1, extendedFeatureCount, CV_32F);
    }

    RNG rng(0);

    CV_Assert (params.termCrit.type & TermCriteria::COUNT || params.termCrit.type & TermCriteria::EPS);
    int maxCount = (params.termCrit.type & TermCriteria::COUNT) ? params.termCrit.maxCount : INT_MAX;
    double epsilon = (params.termCrit.type & TermCriteria::EPS) ? params.termCrit.epsilon : 0;

    double err = DBL_MAX;
    CV_Assert (trainResponses.type() == CV_32FC1);
    // Stochastic gradient descent SVM
    for (int iter = 0; (iter < maxCount) && (err > epsilon); iter++)
    {
        int randomNumber = rng.uniform(0, extendedTrainSamplesCount);             //generate sample number

        Mat currentSample = extendedTrainSamples.row(randomNumber);

        float stepSize = params.initialStepSize * std::pow((1 + params.marginRegularization * params.initialStepSize * (float)iter), (-params.stepDecreasingPower));    //update stepSize

        updateWeights( currentSample, isPositive(trainResponses.at<float>(randomNumber)), stepSize, extendedWeights );

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

    CV_Assert((params.marginType == SOFT_MARGIN || params.marginType == HARD_MARGIN) && (extendedWeights.type() ==  CV_32FC1));

    if (params.marginType == SOFT_MARGIN)
    {
        shift_ = extendedWeights.at<float>(featureCount) - static_cast<float>(weights_.dot(average));
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

    CV_Assert( samples.cols == weights_.cols && samples.type() == CV_32FC1);

    if( _results.needed() )
    {
        _results.create( nSamples, 1, samples.type() );
        results = _results.getMat();
    }
    else
    {
        CV_Assert( nSamples == 1 );
        results = Mat(1, 1, CV_32FC1, &result);
    }

    for (int sampleIndex = 0; sampleIndex < nSamples; sampleIndex++)
    {
        Mat currentSample = samples.row(sampleIndex);
        float criterion = static_cast<float>(currentSample.dot(weights_)) + shift_;
        results.at<float>(sampleIndex) = (criterion >= 0) ? 1.f : -1.f;
    }

    return result;
}

bool SVMSGDImpl::isClassifier() const
{
    return (params.svmsgdType == SGD || params.svmsgdType == ASGD)
            &&
            (params.marginType == SOFT_MARGIN || params.marginType == HARD_MARGIN)
            &&
            (params.marginRegularization > 0) && (params.initialStepSize > 0) && (params.stepDecreasingPower >= 0);
}

bool SVMSGDImpl::isTrained() const
{
    return !weights_.empty();
}

void SVMSGDImpl::write(FileStorage& fs) const
{
    if( !isTrained() )
        CV_Error( cv::Error::StsParseError, "SVMSGD model data is invalid, it hasn't been trained" );

    writeFormat(fs);
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
    default:
        SvmsgdTypeStr = format("Unknown_%d", params.svmsgdType);
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
    default:
        marginTypeStr = format("Unknown_%d", params.marginType);
    }

    fs << "marginType" << marginTypeStr;

    fs << "marginRegularization" << params.marginRegularization;
    fs << "initialStepSize" << params.initialStepSize;
    fs << "stepDecreasingPower" << params.stepDecreasingPower;

    fs << "term_criteria" << "{:";
    if( params.termCrit.type & TermCriteria::EPS )
        fs << "epsilon" << params.termCrit.epsilon;
    if( params.termCrit.type & TermCriteria::COUNT )
        fs << "iterations" << params.termCrit.maxCount;
    fs << "}";
}
void SVMSGDImpl::readParams( const FileNode& fn )
{
    String svmsgdTypeStr = (String)fn["svmsgdType"];
    int svmsgdType =
            svmsgdTypeStr == "SGD" ? SGD :
                                     svmsgdTypeStr == "ASGD" ? ASGD : -1;

    if( svmsgdType < 0 )
        CV_Error( cv::Error::StsParseError, "Missing or invalid SVMSGD type" );

    params.svmsgdType = svmsgdType;

    String marginTypeStr = (String)fn["marginType"];
    int marginType =
            marginTypeStr == "SOFT_MARGIN" ? SOFT_MARGIN :
                                             marginTypeStr == "HARD_MARGIN" ? HARD_MARGIN : -1;

    if( marginType < 0 )
        CV_Error( cv::Error::StsParseError, "Missing or invalid margin type" );

    params.marginType = marginType;

    CV_Assert ( fn["marginRegularization"].isReal() );
    params.marginRegularization = (float)fn["marginRegularization"];

    CV_Assert ( fn["initialStepSize"].isReal() );
    params.initialStepSize = (float)fn["initialStepSize"];

    CV_Assert ( fn["stepDecreasingPower"].isReal() );
    params.stepDecreasingPower = (float)fn["stepDecreasingPower"];

    FileNode tcnode = fn["term_criteria"];
    CV_Assert(!tcnode.empty());
    params.termCrit.epsilon = (double)tcnode["epsilon"];
    params.termCrit.maxCount = (int)tcnode["iterations"];
    params.termCrit.type = (params.termCrit.epsilon > 0 ? TermCriteria::EPS : 0) +
            (params.termCrit.maxCount > 0 ? TermCriteria::COUNT : 0);
    CV_Assert ((params.termCrit.type & TermCriteria::COUNT || params.termCrit.type & TermCriteria::EPS));
}

void SVMSGDImpl::read(const FileNode& fn)
{
    clear();

    readParams(fn);

    fn["weights"] >> weights_;
    fn["shift"] >> shift_;
}

void SVMSGDImpl::clear()
{
    weights_.release();
    shift_ = 0;
}


SVMSGDImpl::SVMSGDImpl()
{
    clear();
    setOptimalParameters();
}

void SVMSGDImpl::setOptimalParameters(int svmsgdType, int marginType)
{
    switch (svmsgdType)
    {
    case SGD:
        params.svmsgdType = SGD;
        params.marginType = (marginType == SOFT_MARGIN) ? SOFT_MARGIN :
                                                          (marginType == HARD_MARGIN) ? HARD_MARGIN : -1;
        params.marginRegularization = 0.0001f;
        params.initialStepSize = 0.05f;
        params.stepDecreasingPower = 1.f;
        params.termCrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100000, 0.00001);
        break;

    case ASGD:
        params.svmsgdType = ASGD;
        params.marginType = (marginType == SOFT_MARGIN) ? SOFT_MARGIN :
                                                          (marginType == HARD_MARGIN) ? HARD_MARGIN : -1;
        params.marginRegularization = 0.00001f;
        params.initialStepSize = 0.05f;
        params.stepDecreasingPower = 0.75f;
        params.termCrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100000, 0.00001);
        break;

    default:
        CV_Error( cv::Error::StsParseError, "SVMSGD model data is invalid" );
    }
}
}   //ml
}   //cv
