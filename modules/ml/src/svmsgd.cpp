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

    virtual bool isClassifier() const { return params.svmsgdType == SGD || params.svmsgdType == ASGD; }

    virtual bool isTrained() const;

    virtual void clear();

    virtual void write(FileStorage& fs) const;

    virtual void read(const FileNode& fn);

    virtual Mat getWeights(){ return weights_; }

    virtual float getShift(){ return shift_; }

    virtual int getVarCount() const { return weights_.cols; }

    virtual String getDefaultName() const {return "opencv_ml_svmsgd";}

    virtual void setOptimalParameters(int type = ASGD);

    virtual int getType() const;

    virtual void setType(int type);

    CV_IMPL_PROPERTY(float, Lambda, params.lambda)
    CV_IMPL_PROPERTY(float, Gamma0, params.gamma0)
    CV_IMPL_PROPERTY(float, C, params.c)
    CV_IMPL_PROPERTY_S(cv::TermCriteria, TermCriteria, params.termCrit)

    private:
        void updateWeights(InputArray sample, bool is_first_class, float gamma);
    float calcShift(InputArray trainSamples, InputArray trainResponses) const;
    std::pair<bool,bool> areClassesEmpty(Mat responses);
    void writeParams( FileStorage& fs ) const;
    void readParams( const FileNode& fn );
    static inline bool isFirstClass(float val) { return val > 0; }


    // Vector with SVM weights
    Mat weights_;
    float shift_;

    // Random index generation
    RNG rng_;

    // Parameters for learning
    struct SVMSGDParams
    {
        float lambda;                             //regularization
        float gamma0;                             //learning rate
        float c;
        TermCriteria termCrit;
        SvmsgdType svmsgdType;
    };

    SVMSGDParams params;
};

Ptr<SVMSGD> SVMSGD::create()
{    
    return makePtr<SVMSGDImpl>();
}


bool SVMSGDImpl::train(const Ptr<TrainData>& data, int)
{
    clear();

    Mat trainSamples = data->getTrainSamples();

    // Initialize varCount
    int trainSamplesCount_ = trainSamples.rows;
    int varCount = trainSamples.cols;

    // Initialize weights vector with zeros
    weights_ = Mat::zeros(1, varCount, CV_32F);

    Mat trainResponses = data->getTrainResponses();        // (trainSamplesCount x 1) matrix

    std::pair<bool,bool> are_empty = areClassesEmpty(trainResponses);

    if ( are_empty.first && are_empty.second )
    {
        weights_.release();
        return false;
    }
    if ( are_empty.first || are_empty.second )
    {
        shift_ = are_empty.first ? -1 : 1;
        return true;
    }


    Mat currentSample;
    float gamma = 0;
    Mat lastWeights = Mat::zeros(1, varCount, CV_32F);     //weights vector for calculating terminal criterion
    Mat averageWeights;                                    //average weights vector for ASGD model
    double err = DBL_MAX;
    if (params.svmsgdType == ASGD)
    {
        averageWeights = Mat::zeros(1, varCount, CV_32F);
    }

    // Stochastic gradient descent SVM
    for (int iter = 0; (iter < params.termCrit.maxCount)&&(err > params.termCrit.epsilon); iter++)
    {
        //generate sample number
        int randomNumber = rng_.uniform(0, trainSamplesCount_);

        currentSample = trainSamples.row(randomNumber);

        //update gamma
        gamma = params.gamma0 * std::pow((1 + params.lambda * params.gamma0 * (float)iter), (-params.c));

        bool is_first_class = isFirstClass(trainResponses.at<float>(randomNumber));
        updateWeights( currentSample, is_first_class, gamma );

        //average weights (only for ASGD model)
        if (params.svmsgdType == ASGD)
        {
            averageWeights = ((float)iter/ (1 + (float)iter)) * averageWeights  + weights_ / (1 + (float) iter);
        }

        err = norm(weights_ - lastWeights);
        weights_.copyTo(lastWeights);
    }

    if (params.svmsgdType == ASGD)
    {
        weights_ = averageWeights;
    }

    shift_ = calcShift(trainSamples, trainResponses);

    return true;
}

std::pair<bool,bool> SVMSGDImpl::areClassesEmpty(Mat responses)
{
    std::pair<bool,bool> are_classes_empty(true, true);
    int limit_index = responses.rows;

    for(int index = 0; index < limit_index; index++)
    {
        if (isFirstClass(responses.at<float>(index,0)))
            are_classes_empty.first = false;
        else
            are_classes_empty.second = false;

        if (!are_classes_empty.first && ! are_classes_empty.second)
            break;
    }

    return are_classes_empty;
}

float SVMSGDImpl::calcShift(InputArray _samples, InputArray _responses) const
{
    float distance_to_classes[2] = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };

    Mat trainSamples = _samples.getMat();
    int trainSamplesCount = trainSamples.rows;

    Mat trainResponses = _responses.getMat();

    for (int samplesIndex = 0; samplesIndex < trainSamplesCount; samplesIndex++)
    {
        Mat currentSample = trainSamples.row(samplesIndex);
        float scalar_product = currentSample.dot(weights_);

        bool is_first_class = isFirstClass(trainResponses.at<float>(samplesIndex));
        int index = is_first_class ? 0:1;
        float sign_to_mul = is_first_class ? 1 : -1;
        float cur_distance = scalar_product * sign_to_mul ;

        if (cur_distance < distance_to_classes[index])
        {
            distance_to_classes[index] = cur_distance;
        }
    }

    //todo: areClassesEmpty(); make const;
    return -(distance_to_classes[0] - distance_to_classes[1]) / 2.f;
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

    Mat currentSample;
    float criterion = 0;

    for (int sampleIndex = 0; sampleIndex < nSamples; sampleIndex++)
    {
        currentSample = samples.row(sampleIndex);
        criterion = currentSample.dot(weights_) + shift_;
        results.at<float>(sampleIndex) = (criterion >= 0) ? 1 : -1;
    }

    return result;
}

void SVMSGDImpl::updateWeights(InputArray _sample, bool is_first_class, float gamma)
{
    Mat sample = _sample.getMat();

    int responce = is_first_class ? 1 : -1; // ensure that trainResponses are -1 or 1

    if ( sample.dot(weights_) * responce > 1)
    {
        // Not a support vector, only apply weight decay
        weights_ *= (1.f - gamma * params.lambda);
    }
    else
    {
        // It's a support vector, add it to the weights
        weights_ -= (gamma * params.lambda) * weights_ - gamma * responce * sample;
        //std::cout << "sample " << sample << std::endl;
        //std::cout << "weights_ " << weights_ << std::endl;
    }
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

    fs << "shift" << shift_;
    fs << "weights" << weights_;
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
    case ILLEGAL_VALUE:
        SvmsgdTypeStr = format("Uknown_%d", params.svmsgdType);
    default:
        std::cout << "params.svmsgdType isn't initialized" << std::endl;
    }


    fs << "svmsgdType" << SvmsgdTypeStr;

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

    shift_ = (float) fn["shift"];
    fn["weights"] >> weights_;
}

void SVMSGDImpl::readParams( const FileNode& fn )
{
    String svmsgdTypeStr = (String)fn["svmsgdType"];
    SvmsgdType svmsgdType =
            svmsgdTypeStr == "SGD" ? SGD :
                                     svmsgdTypeStr == "ASGD" ? ASGD : ILLEGAL_VALUE;

    if( svmsgdType == ILLEGAL_VALUE )
        CV_Error( CV_StsParseError, "Missing or invalid SVMSGD type" );

    params.svmsgdType = svmsgdType;

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
        params.termCrit = TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 1000, FLT_EPSILON );

}

void SVMSGDImpl::clear()
{
    weights_.release();
    shift_ = 0;
}


SVMSGDImpl::SVMSGDImpl()
{
    clear();
    rng_(0);

    params.svmsgdType = ILLEGAL_VALUE;

    // Parameters for learning
    params.lambda = 0;                              // regularization
    params.gamma0 = 0;                        // learning rate (ideally should be large at beginning and decay each iteration)
    params.c = 0;

    TermCriteria _termCrit(TermCriteria::COUNT + TermCriteria::EPS, 0, 0);
    params.termCrit = _termCrit;
}

void SVMSGDImpl::setOptimalParameters(int type)
{
    switch (type)
    {
    case SGD:
        params.svmsgdType = SGD;
        params.lambda = 0.00001;
        params.gamma0 = 0.05;
        params.c = 1;
        params.termCrit.maxCount = 50000;
        params.termCrit.epsilon = 0.00000001;
        break;

    case ASGD:
        params.svmsgdType = ASGD;
        params.lambda = 0.00001;
        params.gamma0 = 0.5;
        params.c = 0.75;
        params.termCrit.maxCount = 100000;
        params.termCrit.epsilon = 0.000001;
        break;

    default:
        CV_Error( CV_StsParseError, "SVMSGD model data is invalid" );
    }
}

void SVMSGDImpl::setType(int type)
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
        params.svmsgdType = ILLEGAL_VALUE;
    }
}

int SVMSGDImpl::getType() const
{
    return params.svmsgdType;
}
}   //ml
}   //cv
