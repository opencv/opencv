/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

/*
 * This class implements an algorithm described in "Visual Tracking of Human Visitors under
 * Variable-Lighting Conditions for a Responsive Audio Art Installation," A. Godbehere,
 * A. Matsukawa, K. Goldberg, American Control Conference, Montreal, June 2012.
 *
 * Prepared and integrated by Andrew B. Godbehere.
 */

#include "precomp.hpp"

using namespace std;

namespace cv
{

BackgroundSubtractorGMG::BackgroundSubtractorGMG()
{
    /*
     * Default Parameter Values. Override with algorithm "set" method.
     */
    maxFeatures = 64;
    learningRate = 0.025;
    numInitializationFrames = 120;
    quantizationLevels = 16;
    backgroundPrior = 0.8;
    decisionThreshold = 0.8;
    smoothingRadius = 7;
}

void BackgroundSubtractorGMG::initializeType(InputArray _image, double min, double max)
{
    minVal = min;
    maxVal = max;

    if (minVal == maxVal)
    {
        CV_Error_(CV_StsBadArg,("minVal and maxVal cannot be the same."));
    }

    /*
     * Parameter validation
     */
    if (maxFeatures <= 0)
    {
        CV_Error_(CV_StsBadArg,
                ("maxFeatures parameter must be 1 or greater. Instead, it is %d.",maxFeatures));
    }
    if (learningRate < 0.0 || learningRate > 1.0)
    {
        CV_Error_(CV_StsBadArg,
                ("learningRate parameter must be in the range [0.0,1.0]. Instead, it is %f.",
                learningRate));
    }
    if (numInitializationFrames < 1)
    {
        CV_Error_(CV_StsBadArg,
                ("numInitializationFrames must be at least 1. Instead, it is %d.",
                        numInitializationFrames));
    }
    if (quantizationLevels < 1)
    {
        CV_Error_(CV_StsBadArg,
                ("quantizationLevels must be at least 1 (preferably more). Instead it is %d.",
                        quantizationLevels));
    }
    if (backgroundPrior < 0.0 || backgroundPrior > 1.0)
    {
        CV_Error_(CV_StsBadArg,
                ("backgroundPrior must be a probability, between 0.0 and 1.0. Instead it is %f.",
                        backgroundPrior));
    }

    /*
     * Detect and accommodate the image depth
     */
    Mat image = _image.getMat();
    numChannels = image.channels();

    /*
     * Color quantization [0 | | | | max] --> [0 | | max]
     *  (0) Use double as intermediary to convert all types to int.
     *  (i) Shift min to 0,
     *  (ii) max/(num intervals) = factor.  x/factor * factor = quantized result, after integer operation.
     */

    /*
     * Data Structure Initialization
     */
    imWidth = image.cols;
    imHeight = image.rows;
    numPixels = image.total();
    pixels.resize(numPixels);
    frameNum = 0;

    // used to iterate through matrix of type unknown at compile time
    //elemSize = image.elemSize();
    //elemSize1 = image.elemSize1();

    vector<PixelModelGMG>::iterator pixel;
    vector<PixelModelGMG>::iterator pixel_end = pixels.end();
    for (pixel = pixels.begin(); pixel != pixel_end; ++pixel)
    {
        pixel->setMaxFeatures(maxFeatures);
    }

    fgMaskImage = Mat::zeros(imHeight, imWidth, CV_8UC1);  // 8-bit unsigned mask. 255 for FG, 0 for BG
    posteriorImage = Mat::zeros(imHeight, imWidth, CV_32FC1);  // float for storing probabilities. Can be viewed directly with imshow.
    isDataInitialized = true;
}

void BackgroundSubtractorGMG::operator()(InputArray _image, OutputArray _fgmask, double newLearningRate)
{
    if (!isDataInitialized)
    {
        CV_Error(CV_StsError,"BackgroundSubstractorGMG has not been initialized. Call initialize() first.\n");
    }

    /*
     * Update learning rate parameter, if desired
     */
    if (newLearningRate != -1.0)
    {
        if (newLearningRate < 0.0 || newLearningRate > 1.0)
        {
            CV_Error(CV_StsOutOfRange,"Learning rate for Operator () must be between 0.0 and 1.0.\n");
        }
        this->learningRate = newLearningRate;
    }

    Mat image = _image.getMat();

    _fgmask.create(imHeight,imWidth,CV_8U);
    fgMaskImage = _fgmask.getMat();  // 8-bit unsigned mask. 255 for FG, 0 for BG

    /*
     * Iterate over pixels in image
     */
    // grab data at each pixel (1,2,3 channels, int, float, etc.)
    // grab data as an array of bytes. Then, send that array to a function that reads data into vector of appropriate types... and quantizing... before saving as a feature, which is a vector of flexitypes, so code can be portable.
    // multiple channels do have sequential storage, use mat::elemSize() and mat::elemSize1()
    vector<PixelModelGMG>::iterator pixel;
    vector<PixelModelGMG>::iterator pixel_end = pixels.end();
    size_t i;
    //#pragma omp parallel
    for (i = 0, pixel=pixels.begin(); pixel != pixel_end; ++i,++pixel)
    {
        HistogramFeatureGMG newFeature;
        newFeature.color.clear();
        int irow = int(i / imWidth);
        int icol = i % imWidth;
        for (size_t c = 0; c < numChannels; ++c)
        {
            /*
             * Perform quantization. in each channel. (color-min)*(levels)/(max-min).
             * Shifts min to 0 and scales, finally casting to an int.
             */
            double color;
            switch(image.depth())
            {
                case CV_8U: color = image.ptr<uchar>(irow)[icol * numChannels + c]; break;
                case CV_8S: color = image.ptr<schar>(irow)[icol * numChannels + c]; break;
                case CV_16U: color = image.ptr<ushort>(irow)[icol * numChannels + c]; break;
                case CV_16S: color = image.ptr<short>(irow)[icol * numChannels + c]; break;
                case CV_32S: color = image.ptr<int>(irow)[icol * numChannels + c]; break;
                case CV_32F: color = image.ptr<float>(irow)[icol * numChannels + c]; break;
                case CV_64F: color = image.ptr<double>(irow)[icol * numChannels + c]; break;
                default: color = 0; break;
            }
            size_t quantizedColor = (size_t)((color-minVal)*quantizationLevels/(maxVal-minVal));
            newFeature.color.push_back(quantizedColor);
        }
        // now that the feature is ready for use, put it in the histogram

        if (frameNum > numInitializationFrames)  // typical operation
        {
            newFeature.likelihood = float(learningRate);
            /*
             * (1) Query histogram to find posterior probability of feature under model.
             */
            float likelihood = (float)pixel->getLikelihood(newFeature);

            // see Godbehere, Matsukawa, Goldberg (2012) for reasoning behind this implementation of Bayes rule
            float posterior = float((likelihood*backgroundPrior)/(likelihood*backgroundPrior+(1-likelihood)*(1-backgroundPrior)));

            /*
             * (2) feed posterior probability into the posterior image
             */
            int row,col;
            col = i%imWidth;
            row = int(i-col)/imWidth;
            posteriorImage.at<float>(row,col) = (1.0f-posterior);
        }
        pixel->setLastObservedFeature(newFeature);
    }
    /*
     * (3) Perform filtering and threshold operations to yield final mask image.
     *
     * 2 options. First is morphological open/close as before. Second is "median filtering" which Jon Barron says is good to remove noise
     */
    Mat thresholdedPosterior;
    threshold(posteriorImage,thresholdedPosterior,decisionThreshold,1.0,THRESH_BINARY);
    thresholdedPosterior.convertTo(fgMaskImage,CV_8U,255);  // convert image to integer space for further filtering and mask creation
    medianBlur(fgMaskImage,fgMaskImage,smoothingRadius);

    fgMaskImage.copyTo(_fgmask);

    ++frameNum;  // keep track of how many frames we have processed
}

void BackgroundSubtractorGMG::getPosteriorImage(OutputArray _img)
{
    _img.create(Size(imWidth,imHeight),CV_32F);
    Mat img = _img.getMat();
    posteriorImage.copyTo(img);
}

void BackgroundSubtractorGMG::updateBackgroundModel(InputArray _mask)
{
    CV_Assert(_mask.size() == Size(imWidth,imHeight));  // mask should be same size as image

    Mat maskImg = _mask.getMat();
//#pragma omp parallel
    for (int i = 0; i < imHeight; ++i)
    {
//#pragma omp parallel
        for (int j = 0; j < imWidth; ++j)
        {
            if (frameNum <= numInitializationFrames + 1)
            {
                // insert previously observed feature into the histogram. -1.0 parameter indicates training.
                pixels[i*imWidth+j].insertFeature(-1.0);
                if (frameNum >= numInitializationFrames+1)  // training is done, normalize
                {
                    pixels[i*imWidth+j].normalizeHistogram();
                }
            }
            // if mask is 0, pixel is identified as a background pixel, so update histogram.
            else if (maskImg.at<uchar>(i,j) == 0)
            {
                pixels[i*imWidth+j].insertFeature(learningRate); // updates the histogram for the next iteration.
            }
        }
    }
}

BackgroundSubtractorGMG::~BackgroundSubtractorGMG()
{

}

BackgroundSubtractorGMG::PixelModelGMG::PixelModelGMG()
{
    numFeatures = 0;
    maxFeatures = 0;
}

BackgroundSubtractorGMG::PixelModelGMG::~PixelModelGMG()
{

}

void BackgroundSubtractorGMG::PixelModelGMG::setLastObservedFeature(HistogramFeatureGMG f)
{
    this->lastObservedFeature = f;
}

double BackgroundSubtractorGMG::PixelModelGMG::getLikelihood(BackgroundSubtractorGMG::HistogramFeatureGMG f)
{
    std::list<HistogramFeatureGMG>::iterator feature = histogram.begin();
    std::list<HistogramFeatureGMG>::iterator feature_end = histogram.end();

    for (feature = histogram.begin(); feature != feature_end; ++feature)
    {
        // comparing only feature color, not likelihood. See equality operator for HistogramFeatureGMG
        if (f == *feature)
        {
            return feature->likelihood;
        }
    }

    return 0.0; // not in histogram, so return 0.
}

void BackgroundSubtractorGMG::PixelModelGMG::insertFeature(double learningRate)
{

    std::list<HistogramFeatureGMG>::iterator feature;
    std::list<HistogramFeatureGMG>::iterator swap_end;
    std::list<HistogramFeatureGMG>::iterator last_feature = histogram.end();
    /*
     * If feature is in histogram already, add the weights, and move feature to front.
     * If there are too many features, remove the end feature and push new feature to beginning
     */
    if (learningRate == -1.0) // then, this is a training-mode update.
    {
        /*
         * (1) Check if feature already represented in histogram
         */
        lastObservedFeature.likelihood = 1.0;

        for (feature = histogram.begin(); feature != last_feature; ++feature)
        {
            if (lastObservedFeature == *feature)  // feature in histogram
            {
                feature->likelihood += lastObservedFeature.likelihood;
                // now, move feature to beginning of list and break the loop
                HistogramFeatureGMG tomove = *feature;
                histogram.erase(feature);
                histogram.push_front(tomove);
                return;
            }
        }
        if (numFeatures == maxFeatures)
        {
            histogram.pop_back(); // discard oldest feature
            histogram.push_front(lastObservedFeature);
        }
        else
        {
            histogram.push_front(lastObservedFeature);
            ++numFeatures;
        }
    }
    else
    {
        /*
         * (1) Scale entire histogram by scaling factor
         * (2) Scale input feature.
         * (3) Check if feature already represented. If so, simply add.
         * (4) If feature is not represented, remove old feature, distribute weight evenly among existing features, add in new feature.
         */
        *this *= float(1.0-learningRate);
        lastObservedFeature.likelihood = float(learningRate);

        for (feature = histogram.begin(); feature != last_feature; ++feature)
        {
            if (lastObservedFeature == *feature)  // feature in histogram
            {
                lastObservedFeature.likelihood += feature->likelihood;
                histogram.erase(feature);
                histogram.push_front(lastObservedFeature);
                return;  // done with the update.
            }
        }
        if (numFeatures == maxFeatures)
        {
            histogram.pop_back(); // discard oldest feature
            histogram.push_front(lastObservedFeature);
            normalizeHistogram();
        }
        else
        {
            histogram.push_front(lastObservedFeature);
            ++numFeatures;
        }
    }
}

BackgroundSubtractorGMG::PixelModelGMG& BackgroundSubtractorGMG::PixelModelGMG::operator *=(const float &rhs)
{
    /*
     * Used to scale histogram by a constant factor
     */
    list<HistogramFeatureGMG>::iterator feature;
    list<HistogramFeatureGMG>::iterator last_feature = histogram.end();
    for (feature = histogram.begin(); feature != last_feature; ++feature)
    {
        feature->likelihood *= rhs;
    }
    return *this;
}

void BackgroundSubtractorGMG::PixelModelGMG::normalizeHistogram()
{
    /*
     * First, calculate the total weight in the histogram
     */
    list<HistogramFeatureGMG>::iterator feature;
    list<HistogramFeatureGMG>::iterator last_feature = histogram.end();
    double total = 0.0;
    for (feature = histogram.begin(); feature != last_feature; ++feature)
    {
        total += feature->likelihood;
    }

    /*
     * Then, if weight is not 0, divide every feature by the total likelihood to re-normalize.
     */
    for (feature = histogram.begin(); feature != last_feature; ++feature)
    {
        if (total != 0.0)
            feature->likelihood = float(feature->likelihood / total);
    }
}

bool BackgroundSubtractorGMG::HistogramFeatureGMG::operator ==(HistogramFeatureGMG &rhs)
{
    CV_Assert(color.size() == rhs.color.size());

    std::vector<size_t>::iterator color_a;
    std::vector<size_t>::iterator color_b;
    std::vector<size_t>::iterator color_a_end = this->color.end();
    std::vector<size_t>::iterator color_b_end = rhs.color.end();
    for (color_a = color.begin(),color_b =rhs.color.begin();color_a!=color_a_end;++color_a,++color_b)
    {
        if (*color_a != *color_b)
        {
            return false;
        }
    }
    return true;
}


}

