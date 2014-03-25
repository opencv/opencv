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
#include <string>

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
    ~ERStat() { }

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
        virtual ~Callback() { }
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
                              default classifier can be implicitly load with function loadClassifierNM1()
                              from file in samples/cpp/trained_classifierNM1.xml
    \param  thresholdDelta    Threshold step in subsequent thresholds when extracting the component tree
    \param  minArea           The minimum area (% of image size) allowed for retreived ER's
    \param  minArea           The maximum area (% of image size) allowed for retreived ER's
    \param  minProbability    The minimum probability P(er|character) allowed for retreived ER's
    \param  nonMaxSuppression Whenever non-maximum suppression is done over the branch probabilities
    \param  minProbability    The minimum probability difference between local maxima and local minima ERs
*/
CV_EXPORTS Ptr<ERFilter> createERFilterNM1(const Ptr<ERFilter::Callback>& cb,
                                                  int thresholdDelta = 1, float minArea = 0.00025,
                                                  float maxArea = 0.13, float minProbability = 0.4,
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
                           default classifier can be implicitly load with function loadClassifierNM2()
                           from file in samples/cpp/trained_classifierNM2.xml
    \param  minProbability The minimum probability P(er|character) allowed for retreived ER's
*/
CV_EXPORTS Ptr<ERFilter> createERFilterNM2(const Ptr<ERFilter::Callback>& cb,
                                                  float minProbability = 0.3);


/*!
    Allow to implicitly load the default classifier when creating an ERFilter object.
    The function takes as parameter the XML or YAML file with the classifier model
    (e.g. trained_classifierNM1.xml) returns a pointer to ERFilter::Callback.
*/

CV_EXPORTS Ptr<ERFilter::Callback> loadClassifierNM1(const std::string& filename);

/*!
    Allow to implicitly load the default classifier when creating an ERFilter object.
    The function takes as parameter the XML or YAML file with the classifier model
    (e.g. trained_classifierNM1.xml) returns a pointer to ERFilter::Callback.
*/

CV_EXPORTS Ptr<ERFilter::Callback> loadClassifierNM2(const std::string& filename);


// computeNMChannels operation modes
enum { ERFILTER_NM_RGBLGrad = 0,
       ERFILTER_NM_IHSGrad  = 1
     };

/*!
    Compute the different channels to be processed independently in the N&M algorithm
    Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012

    In N&M algorithm, the combination of intensity (I), hue (H), saturation (S), and gradient
    magnitude channels (Grad) are used in order to obtain high localization recall.
    This implementation also provides an alternative combination of red (R), green (G), blue (B),
    lightness (L), and gradient magnitude (Grad).

    \param  _src           Source image. Must be RGB CV_8UC3.
    \param  _channels      Output vector<Mat> where computed channels are stored.
    \param  _mode          Mode of operation. Currently the only available options are
                           ERFILTER_NM_RGBLGrad (by default) and ERFILTER_NM_IHSGrad.

*/
CV_EXPORTS void computeNMChannels(InputArray _src, OutputArrayOfArrays _channels, int _mode = ERFILTER_NM_RGBLGrad);


/*!
    Find groups of Extremal Regions that are organized as text blocks. This function implements
    the grouping algorithm described in:
    Gomez L. and Karatzas D.: Multi-script Text Extraction from Natural Scenes, ICDAR 2013.
    Notice that this implementation constrains the results to horizontally-aligned text and
    latin script (since ERFilter classifiers are trained only for latin script detection).

    The algorithm combines two different clustering techniques in a single parameter-free procedure
    to detect groups of regions organized as text. The maximally meaningful groups are fist detected
    in several feature spaces, where each feature space is a combination of proximity information
    (x,y coordinates) and a similarity measure (intensity, color, size, gradient magnitude, etc.),
    thus providing a set of hypotheses of text groups. Evidence Accumulation framework is used to
    combine all these hypotheses to get the final estimate. Each of the resulting groups are finally
    validated using a classifier in order to assest if they form a valid horizontally-aligned text block.

    \param  src            Vector of sinle channel images CV_8UC1 from wich the regions were extracted.
    \param  regions        Vector of ER's retreived from the ERFilter algorithm from each channel
    \param  filename       The XML or YAML file with the classifier model (e.g. trained_classifier_erGrouping.xml)
    \param  minProbability The minimum probability for accepting a group
    \param  groups         The output of the algorithm are stored in this parameter as list of rectangles.
*/
CV_EXPORTS void erGrouping(InputArrayOfArrays src, std::vector<std::vector<ERStat> > &regions,
                                                   const std::string& filename, float minProbablity,
                                                   std::vector<Rect > &groups);

}
#endif // _OPENCV_ERFILTER_HPP_
