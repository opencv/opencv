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
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#ifndef __OPENCV_OBJDETECT_ERFILTER_HPP__
#define __OPENCV_OBJDETECT_ERFILTER_HPP__

#include "opencv2/core.hpp"
#include <vector>
#include <deque>

namespace cv
{

/*!
    Extremal Region Stat structure

    The ERStat structure represents a class-specific Extremal Region (ER).

    An ER is a 4-connected set of pixels with all its grey-level values smaller than the values
    in its outer boundary. A class-specific ER is selected (using a classifier) from all the ER's
    in the component tree of the image.
*/
struct CV_EXPORTS ERStat
{
public:
    //! Constructor
    explicit ERStat(int level = 256, int pixel = 0, int x = 0, int y = 0);
    //! Destructor
    ~ERStat(){};

    //! seed point and the threshold (max grey-level value)
    int pixel;
    int level;

    //! incrementally computable features
    int area;
    int perimeter;
    int euler;                 //!< euler number
    Rect rect;
    double raw_moments[2];     //!< order 1 raw moments to derive the centroid
    double central_moments[3]; //!< order 2 central moments to construct the covariance matrix
    std::deque<int> *crossings;//!< horizontal crossings
    float med_crossings;       //!< median of the crossings at three different height levels

    //! 2nd stage features
    float hole_area_ratio;
    float convex_hull_ratio;
    float num_inflexion_points;

    // TODO Other features can be added (average color, standard deviation, and such)


    // TODO shall we include the pixel list whenever available (i.e. after 2nd stage) ?
    std::vector<int> *pixels;

    //! probability that the ER belongs to the class we are looking for
    double probability;

    //! pointers preserving the tree structure of the component tree
    ERStat* parent;
    ERStat* child;
    ERStat* next;
    ERStat* prev;

    //! wenever the regions is a local maxima of the probability
    bool local_maxima;
    ERStat* max_probability_ancestor;
    ERStat* min_probability_ancestor;
};

/*!
    Base class for 1st and 2nd stages of Neumann and Matas scene text detection algorithms
    Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012

    Extracts the component tree (if needed) and filter the extremal regions (ER's) by using a given classifier.
*/
class CV_EXPORTS ERFilter : public Algorithm
{
public:

    //! callback with the classifier is made a class. By doing it we hide SVM, Boost etc.
    class CV_EXPORTS Callback
    {
    public:
        virtual ~Callback(){};
        //! The classifier must return probability measure for the region.
        virtual double eval(const ERStat& stat) = 0; //const = 0; //TODO why cannot use const = 0 here?
    };

    /*!
        the key method. Takes image on input and returns the selected regions in a vector of ERStat
        only distinctive ERs which correspond to characters are selected by a sequential classifier
        \param image   is the input image
        \param regions is output for the first stage, input/output for the second one.
    */
    virtual void run( InputArray image, std::vector<ERStat>& regions ) = 0;


    //! set/get methods to set the algorithm properties,
    virtual void setCallback(const Ptr<ERFilter::Callback>& cb) = 0;
    virtual void setThresholdDelta(int thresholdDelta) = 0;
    virtual void setMinArea(float minArea) = 0;
    virtual void setMaxArea(float maxArea) = 0;
    virtual void setMinProbability(float minProbability) = 0;
    virtual void setMinProbabilityDiff(float minProbabilityDiff) = 0;
    virtual void setNonMaxSuppression(bool nonMaxSuppression) = 0;
    virtual int  getNumRejected() = 0;
};


/*!
    Create an Extremal Region Filter for the 1st stage classifier of N&M algorithm
    Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012

    The component tree of the image is extracted by a threshold increased step by step
    from 0 to 255, incrementally computable descriptors (aspect_ratio, compactness,
    number of holes, and number of horizontal crossings) are computed for each ER
    and used as features for a classifier which estimates the class-conditional
    probability P(er|character). The value of P(er|character) is tracked using the inclusion
    relation of ER across all thresholds and only the ERs which correspond to local maximum
    of the probability P(er|character) are selected (if the local maximum of the
    probability is above a global limit pmin and the difference between local maximum and
    local minimum is greater than minProbabilityDiff).

    \param  cb                Callback with the classifier.
                              if omitted tries to load a default classifier from file trained_classifierNM1.xml
    \param  thresholdDelta    Threshold step in subsequent thresholds when extracting the component tree
    \param  minArea           The minimum area (% of image size) allowed for retreived ER's
    \param  minArea           The maximum area (% of image size) allowed for retreived ER's
    \param  minProbability    The minimum probability P(er|character) allowed for retreived ER's
    \param  nonMaxSuppression Whenever non-maximum suppression is done over the branch probabilities
    \param  minProbability    The minimum probability difference between local maxima and local minima ERs
*/
CV_EXPORTS Ptr<ERFilter> createERFilterNM1(const Ptr<ERFilter::Callback>& cb = NULL,
                                                  int thresholdDelta = 1, float minArea = 0.000025,
                                                  float maxArea = 0.13, float minProbability = 0.2,
                                                  bool nonMaxSuppression = true,
                                                  float minProbabilityDiff = 0.1);

/*!
    Create an Extremal Region Filter for the 2nd stage classifier of N&M algorithm
    Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012

    In the second stage, the ERs that passed the first stage are classified into character
    and non-character classes using more informative but also more computationally expensive
    features. The classifier uses all the features calculated in the first stage and the following
    additional features: hole area ratio, convex hull ratio, and number of outer inflexion points.

    \param  cb             Callback with the classifier
                           if omitted tries to load a default classifier from file trained_classifierNM2.xml
    \param  minProbability The minimum probability P(er|character) allowed for retreived ER's
*/
CV_EXPORTS Ptr<ERFilter> createERFilterNM2(const Ptr<ERFilter::Callback>& cb = NULL,
                                                  float minProbability = 0.85);

}
#endif // _OPENCV_ERFILTER_HPP_
