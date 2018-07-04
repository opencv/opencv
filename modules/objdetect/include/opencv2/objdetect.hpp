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

#ifndef OPENCV_OBJDETECT_HPP
#define OPENCV_OBJDETECT_HPP

#include "opencv2/core.hpp"

/**
@defgroup objdetect Object Detection

Haar Feature-based Cascade Classifier for Object Detection
----------------------------------------------------------

The object detector described below has been initially proposed by Paul Viola @cite Viola01 and
improved by Rainer Lienhart @cite Lienhart02 .

First, a classifier (namely a *cascade of boosted classifiers working with haar-like features*) is
trained with a few hundred sample views of a particular object (i.e., a face or a car), called
positive examples, that are scaled to the same size (say, 20x20), and negative examples - arbitrary
images of the same size.

After a classifier is trained, it can be applied to a region of interest (of the same size as used
during the training) in an input image. The classifier outputs a "1" if the region is likely to show
the object (i.e., face/car), and "0" otherwise. To search for the object in the whole image one can
move the search window across the image and check every location using the classifier. The
classifier is designed so that it can be easily "resized" in order to be able to find the objects of
interest at different sizes, which is more efficient than resizing the image itself. So, to find an
object of an unknown size in the image the scan procedure should be done several times at different
scales.

The word "cascade" in the classifier name means that the resultant classifier consists of several
simpler classifiers (*stages*) that are applied subsequently to a region of interest until at some
stage the candidate is rejected or all the stages are passed. The word "boosted" means that the
classifiers at every stage of the cascade are complex themselves and they are built out of basic
classifiers using one of four different boosting techniques (weighted voting). Currently Discrete
Adaboost, Real Adaboost, Gentle Adaboost and Logitboost are supported. The basic classifiers are
decision-tree classifiers with at least 2 leaves. Haar-like features are the input to the basic
classifiers, and are calculated as described below. The current algorithm uses the following
Haar-like features:

![image](pics/haarfeatures.png)

The feature used in a particular classifier is specified by its shape (1a, 2b etc.), position within
the region of interest and the scale (this scale is not the same as the scale used at the detection
stage, though these two scales are multiplied). For example, in the case of the third line feature
(2c) the response is calculated as the difference between the sum of image pixels under the
rectangle covering the whole feature (including the two white stripes and the black stripe in the
middle) and the sum of the image pixels under the black stripe multiplied by 3 in order to
compensate for the differences in the size of areas. The sums of pixel values over a rectangular
regions are calculated rapidly using integral images (see below and the integral description).

To see the object detector at work, have a look at the facedetect demo:
<https://github.com/opencv/opencv/tree/3.4/samples/cpp/dbt_face_detection.cpp>

The following reference is for the detection part only. There is a separate application called
opencv_traincascade that can train a cascade of boosted classifiers from a set of samples.

@note In the new C++ interface it is also possible to use LBP (local binary pattern) features in
addition to Haar-like features. .. [Viola01] Paul Viola and Michael J. Jones. Rapid Object Detection
using a Boosted Cascade of Simple Features. IEEE CVPR, 2001. The paper is available online at
<http://research.microsoft.com/en-us/um/people/viola/Pubs/Detect/violaJones_CVPR2001.pdf>

@{
    @defgroup objdetect_c C API
@}
 */

typedef struct CvHaarClassifierCascade CvHaarClassifierCascade;

namespace cv
{

//! @addtogroup objdetect
//! @{

///////////////////////////// Object Detection ////////////////////////////

//! class for grouping object candidates, detected by Cascade Classifier, HOG etc.
//! instance of the class is to be passed to cv::partition (see cxoperations.hpp)
class CV_EXPORTS SimilarRects
{
public:
    SimilarRects(double _eps) : eps(_eps) {}
    inline bool operator()(const Rect& r1, const Rect& r2) const
    {
        double delta = eps * ((std::min)(r1.width, r2.width) + (std::min)(r1.height, r2.height)) * 0.5;
        return std::abs(r1.x - r2.x) <= delta &&
            std::abs(r1.y - r2.y) <= delta &&
            std::abs(r1.x + r1.width - r2.x - r2.width) <= delta &&
            std::abs(r1.y + r1.height - r2.y - r2.height) <= delta;
    }
    double eps;
};

/** @brief Groups the object candidate rectangles.

@param rectList Input/output vector of rectangles. Output vector includes retained and grouped
rectangles. (The Python list is not modified in place.)
@param groupThreshold Minimum possible number of rectangles minus 1. The threshold is used in a
group of rectangles to retain it.
@param eps Relative difference between sides of the rectangles to merge them into a group.

The function is a wrapper for the generic function partition . It clusters all the input rectangles
using the rectangle equivalence criteria that combines rectangles with similar sizes and similar
locations. The similarity is defined by eps. When eps=0 , no clustering is done at all. If
\f$\texttt{eps}\rightarrow +\inf\f$ , all the rectangles are put in one cluster. Then, the small
clusters containing less than or equal to groupThreshold rectangles are rejected. In each other
cluster, the average rectangle is computed and put into the output rectangle list.
 */
CV_EXPORTS   void groupRectangles(std::vector<Rect>& rectList, int groupThreshold, double eps = 0.2);
/** @overload */
CV_EXPORTS_W void groupRectangles(CV_IN_OUT std::vector<Rect>& rectList, CV_OUT std::vector<int>& weights,
                                  int groupThreshold, double eps = 0.2);
/** @overload */
CV_EXPORTS   void groupRectangles(std::vector<Rect>& rectList, int groupThreshold,
                                  double eps, std::vector<int>* weights, std::vector<double>* levelWeights );
/** @overload */
CV_EXPORTS   void groupRectangles(std::vector<Rect>& rectList, std::vector<int>& rejectLevels,
                                  std::vector<double>& levelWeights, int groupThreshold, double eps = 0.2);
/** @overload */
CV_EXPORTS   void groupRectangles_meanshift(std::vector<Rect>& rectList, std::vector<double>& foundWeights,
                                            std::vector<double>& foundScales,
                                            double detectThreshold = 0.0, Size winDetSize = Size(64, 128));

template<> CV_EXPORTS void DefaultDeleter<CvHaarClassifierCascade>::operator ()(CvHaarClassifierCascade* obj) const;

enum { CASCADE_DO_CANNY_PRUNING    = 1,
       CASCADE_SCALE_IMAGE         = 2,
       CASCADE_FIND_BIGGEST_OBJECT = 4,
       CASCADE_DO_ROUGH_SEARCH     = 8
     };

class CV_EXPORTS_W BaseCascadeClassifier : public Algorithm
{
public:
    virtual ~BaseCascadeClassifier();
    virtual bool empty() const CV_OVERRIDE = 0;
    virtual bool load( const String& filename ) = 0;
    virtual void detectMultiScale( InputArray image,
                           CV_OUT std::vector<Rect>& objects,
                           double scaleFactor,
                           int minNeighbors, int flags,
                           Size minSize, Size maxSize ) = 0;

    virtual void detectMultiScale( InputArray image,
                           CV_OUT std::vector<Rect>& objects,
                           CV_OUT std::vector<int>& numDetections,
                           double scaleFactor,
                           int minNeighbors, int flags,
                           Size minSize, Size maxSize ) = 0;

    virtual void detectMultiScale( InputArray image,
                                   CV_OUT std::vector<Rect>& objects,
                                   CV_OUT std::vector<int>& rejectLevels,
                                   CV_OUT std::vector<double>& levelWeights,
                                   double scaleFactor,
                                   int minNeighbors, int flags,
                                   Size minSize, Size maxSize,
                                   bool outputRejectLevels ) = 0;

    virtual bool isOldFormatCascade() const = 0;
    virtual Size getOriginalWindowSize() const = 0;
    virtual int getFeatureType() const = 0;
    virtual void* getOldCascade() = 0;

    class CV_EXPORTS MaskGenerator
    {
    public:
        virtual ~MaskGenerator() {}
        virtual Mat generateMask(const Mat& src)=0;
        virtual void initializeMask(const Mat& /*src*/) { }
    };
    virtual void setMaskGenerator(const Ptr<MaskGenerator>& maskGenerator) = 0;
    virtual Ptr<MaskGenerator> getMaskGenerator() = 0;
};

/** @example facedetect.cpp
This program demonstrates usage of the Cascade classifier class
\image html Cascade_Classifier_Tutorial_Result_Haar.jpg "Sample screenshot" width=321 height=254
*/
/** @brief Cascade classifier class for object detection.
 */
class CV_EXPORTS_W CascadeClassifier
{
public:
    CV_WRAP CascadeClassifier();
    /** @brief Loads a classifier from a file.

    @param filename Name of the file from which the classifier is loaded.
     */
    CV_WRAP CascadeClassifier(const String& filename);
    ~CascadeClassifier();
    /** @brief Checks whether the classifier has been loaded.
    */
    CV_WRAP bool empty() const;
    /** @brief Loads a classifier from a file.

    @param filename Name of the file from which the classifier is loaded. The file may contain an old
    HAAR classifier trained by the haartraining application or a new cascade classifier trained by the
    traincascade application.
     */
    CV_WRAP bool load( const String& filename );
    /** @brief Reads a classifier from a FileStorage node.

    @note The file may contain a new cascade classifier (trained traincascade application) only.
     */
    CV_WRAP bool read( const FileNode& node );

    /** @brief Detects objects of different sizes in the input image. The detected objects are returned as a list
    of rectangles.

    @param image Matrix of the type CV_8U containing an image where objects are detected.
    @param objects Vector of rectangles where each rectangle contains the detected object, the
    rectangles may be partially outside the original image.
    @param scaleFactor Parameter specifying how much the image size is reduced at each image scale.
    @param minNeighbors Parameter specifying how many neighbors each candidate rectangle should have
    to retain it.
    @param flags Parameter with the same meaning for an old cascade as in the function
    cvHaarDetectObjects. It is not used for a new cascade.
    @param minSize Minimum possible object size. Objects smaller than that are ignored.
    @param maxSize Maximum possible object size. Objects larger than that are ignored. If `maxSize == minSize` model is evaluated on single scale.

    The function is parallelized with the TBB library.

    @note
       -   (Python) A face detection example using cascade classifiers can be found at
            opencv_source_code/samples/python/facedetect.py
    */
    CV_WRAP void detectMultiScale( InputArray image,
                          CV_OUT std::vector<Rect>& objects,
                          double scaleFactor = 1.1,
                          int minNeighbors = 3, int flags = 0,
                          Size minSize = Size(),
                          Size maxSize = Size() );

    /** @overload
    @param image Matrix of the type CV_8U containing an image where objects are detected.
    @param objects Vector of rectangles where each rectangle contains the detected object, the
    rectangles may be partially outside the original image.
    @param numDetections Vector of detection numbers for the corresponding objects. An object's number
    of detections is the number of neighboring positively classified rectangles that were joined
    together to form the object.
    @param scaleFactor Parameter specifying how much the image size is reduced at each image scale.
    @param minNeighbors Parameter specifying how many neighbors each candidate rectangle should have
    to retain it.
    @param flags Parameter with the same meaning for an old cascade as in the function
    cvHaarDetectObjects. It is not used for a new cascade.
    @param minSize Minimum possible object size. Objects smaller than that are ignored.
    @param maxSize Maximum possible object size. Objects larger than that are ignored. If `maxSize == minSize` model is evaluated on single scale.
    */
    CV_WRAP_AS(detectMultiScale2) void detectMultiScale( InputArray image,
                          CV_OUT std::vector<Rect>& objects,
                          CV_OUT std::vector<int>& numDetections,
                          double scaleFactor=1.1,
                          int minNeighbors=3, int flags=0,
                          Size minSize=Size(),
                          Size maxSize=Size() );

    /** @overload
    This function allows you to retrieve the final stage decision certainty of classification.
    For this, one needs to set `outputRejectLevels` on true and provide the `rejectLevels` and `levelWeights` parameter.
    For each resulting detection, `levelWeights` will then contain the certainty of classification at the final stage.
    This value can then be used to separate strong from weaker classifications.

    A code sample on how to use it efficiently can be found below:
    @code
    Mat img;
    vector<double> weights;
    vector<int> levels;
    vector<Rect> detections;
    CascadeClassifier model("/path/to/your/model.xml");
    model.detectMultiScale(img, detections, levels, weights, 1.1, 3, 0, Size(), Size(), true);
    cerr << "Detection " << detections[0] << " with weight " << weights[0] << endl;
    @endcode
    */
    CV_WRAP_AS(detectMultiScale3) void detectMultiScale( InputArray image,
                                  CV_OUT std::vector<Rect>& objects,
                                  CV_OUT std::vector<int>& rejectLevels,
                                  CV_OUT std::vector<double>& levelWeights,
                                  double scaleFactor = 1.1,
                                  int minNeighbors = 3, int flags = 0,
                                  Size minSize = Size(),
                                  Size maxSize = Size(),
                                  bool outputRejectLevels = false );

    CV_WRAP bool isOldFormatCascade() const;
    CV_WRAP Size getOriginalWindowSize() const;
    CV_WRAP int getFeatureType() const;
    void* getOldCascade();

    CV_WRAP static bool convert(const String& oldcascade, const String& newcascade);

    void setMaskGenerator(const Ptr<BaseCascadeClassifier::MaskGenerator>& maskGenerator);
    Ptr<BaseCascadeClassifier::MaskGenerator> getMaskGenerator();

    Ptr<BaseCascadeClassifier> cc;
};

CV_EXPORTS Ptr<BaseCascadeClassifier::MaskGenerator> createFaceDetectionMaskGenerator();

//////////////// HOG (Histogram-of-Oriented-Gradients) Descriptor and Object Detector //////////////

//! struct for detection region of interest (ROI)
struct DetectionROI
{
   //! scale(size) of the bounding box
   double scale;
   //! set of requested locations to be evaluated
   std::vector<cv::Point> locations;
   //! vector that will contain confidence values for each location
   std::vector<double> confidences;
};

/**@brief Implementation of HOG (Histogram of Oriented Gradients) descriptor and object detector.

the HOG descriptor algorithm introduced by Navneet Dalal and Bill Triggs @cite Dalal2005 .

useful links:

https://hal.inria.fr/inria-00548512/document/

https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients

https://software.intel.com/en-us/ipp-dev-reference-histogram-of-oriented-gradients-hog-descriptor

http://www.learnopencv.com/histogram-of-oriented-gradients

http://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial

 */
struct CV_EXPORTS_W HOGDescriptor
{
public:
    enum { L2Hys = 0 //!< Default histogramNormType
         };
    enum { DEFAULT_NLEVELS = 64 //!< Default nlevels value.
         };
    /**@brief Creates the HOG descriptor and detector with default params.

    aqual to HOGDescriptor(Size(64,128), Size(16,16), Size(8,8), Size(8,8), 9, 1 )
    */
    CV_WRAP HOGDescriptor() : winSize(64,128), blockSize(16,16), blockStride(8,8),
        cellSize(8,8), nbins(9), derivAperture(1), winSigma(-1),
        histogramNormType(HOGDescriptor::L2Hys), L2HysThreshold(0.2), gammaCorrection(true),
        free_coef(-1.f), nlevels(HOGDescriptor::DEFAULT_NLEVELS), signedGradient(false)
    {}

    /** @overload
    @param _winSize sets winSize with given value.
    @param _blockSize sets blockSize with given value.
    @param _blockStride sets blockStride with given value.
    @param _cellSize sets cellSize with given value.
    @param _nbins sets nbins with given value.
    @param _derivAperture sets derivAperture with given value.
    @param _winSigma sets winSigma with given value.
    @param _histogramNormType sets histogramNormType with given value.
    @param _L2HysThreshold sets L2HysThreshold with given value.
    @param _gammaCorrection sets gammaCorrection with given value.
    @param _nlevels sets nlevels with given value.
    @param _signedGradient sets signedGradient with given value.
    */
    CV_WRAP HOGDescriptor(Size _winSize, Size _blockSize, Size _blockStride,
                  Size _cellSize, int _nbins, int _derivAperture=1, double _winSigma=-1,
                  int _histogramNormType=HOGDescriptor::L2Hys,
                  double _L2HysThreshold=0.2, bool _gammaCorrection=false,
                  int _nlevels=HOGDescriptor::DEFAULT_NLEVELS, bool _signedGradient=false)
    : winSize(_winSize), blockSize(_blockSize), blockStride(_blockStride), cellSize(_cellSize),
    nbins(_nbins), derivAperture(_derivAperture), winSigma(_winSigma),
    histogramNormType(_histogramNormType), L2HysThreshold(_L2HysThreshold),
    gammaCorrection(_gammaCorrection), free_coef(-1.f), nlevels(_nlevels), signedGradient(_signedGradient)
    {}

    /** @overload
    @param filename the file name containing  HOGDescriptor properties and coefficients of the trained classifier
    */
    CV_WRAP HOGDescriptor(const String& filename)
    {
        load(filename);
    }

    /** @overload
    @param d the HOGDescriptor which cloned to create a new one.
    */
    HOGDescriptor(const HOGDescriptor& d)
    {
        d.copyTo(*this);
    }

    /**@brief Default destructor.
    */
    virtual ~HOGDescriptor() {}

    /**@brief Returns the number of coefficients required for the classification.
    */
    CV_WRAP size_t getDescriptorSize() const;

    /** @brief Checks if detector size equal to descriptor size.
    */
    CV_WRAP bool checkDetectorSize() const;

    /** @brief Returns winSigma value
    */
    CV_WRAP double getWinSigma() const;

    /**@example peopledetect.cpp
    */
    /**@brief Sets coefficients for the linear SVM classifier.
    @param _svmdetector coefficients for the linear SVM classifier.
    */
    CV_WRAP virtual void setSVMDetector(InputArray _svmdetector);

    /** @brief Reads HOGDescriptor parameters from a file node.
    @param fn File node
    */
    virtual bool read(FileNode& fn);

    /** @brief Stores HOGDescriptor parameters in a file storage.
    @param fs File storage
    @param objname Object name
    */
    virtual void write(FileStorage& fs, const String& objname) const;

    /** @brief loads coefficients for the linear SVM classifier from a file
    @param filename Name of the file to read.
    @param objname The optional name of the node to read (if empty, the first top-level node will be used).
    */
    CV_WRAP virtual bool load(const String& filename, const String& objname = String());

    /** @brief saves coefficients for the linear SVM classifier to a file
    @param filename File name
    @param objname Object name
    */
    CV_WRAP virtual void save(const String& filename, const String& objname = String()) const;

    /** @brief clones the HOGDescriptor
    @param c cloned HOGDescriptor
    */
    virtual void copyTo(HOGDescriptor& c) const;

    /**@example train_HOG.cpp
    */
    /** @brief Computes HOG descriptors of given image.
    @param img Matrix of the type CV_8U containing an image where HOG features will be calculated.
    @param descriptors Matrix of the type CV_32F
    @param winStride Window stride. It must be a multiple of block stride.
    @param padding Padding
    @param locations Vector of Point
    */
    CV_WRAP virtual void compute(InputArray img,
                         CV_OUT std::vector<float>& descriptors,
                         Size winStride = Size(), Size padding = Size(),
                         const std::vector<Point>& locations = std::vector<Point>()) const;

    /** @brief Performs object detection without a multi-scale window.
    @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
    @param foundLocations Vector of point where each point contains left-top corner point of detected object boundaries.
    @param weights Vector that will contain confidence values for each detected object.
    @param hitThreshold Threshold for the distance between features and SVM classifying plane.
    Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient).
    But if the free coefficient is omitted (which is allowed), you can specify it manually here.
    @param winStride Window stride. It must be a multiple of block stride.
    @param padding Padding
    @param searchLocations Vector of Point includes set of requested locations to be evaluated.
    */
    CV_WRAP virtual void detect(const Mat& img, CV_OUT std::vector<Point>& foundLocations,
                        CV_OUT std::vector<double>& weights,
                        double hitThreshold = 0, Size winStride = Size(),
                        Size padding = Size(),
                        const std::vector<Point>& searchLocations = std::vector<Point>()) const;

    /** @brief Performs object detection without a multi-scale window.
    @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
    @param foundLocations Vector of point where each point contains left-top corner point of detected object boundaries.
    @param hitThreshold Threshold for the distance between features and SVM classifying plane.
    Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient).
    But if the free coefficient is omitted (which is allowed), you can specify it manually here.
    @param winStride Window stride. It must be a multiple of block stride.
    @param padding Padding
    @param searchLocations Vector of Point includes locations to search.
    */
    virtual void detect(const Mat& img, CV_OUT std::vector<Point>& foundLocations,
                        double hitThreshold = 0, Size winStride = Size(),
                        Size padding = Size(),
                        const std::vector<Point>& searchLocations=std::vector<Point>()) const;

    /** @brief Detects objects of different sizes in the input image. The detected objects are returned as a list
    of rectangles.
    @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
    @param foundLocations Vector of rectangles where each rectangle contains the detected object.
    @param foundWeights Vector that will contain confidence values for each detected object.
    @param hitThreshold Threshold for the distance between features and SVM classifying plane.
    Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient).
    But if the free coefficient is omitted (which is allowed), you can specify it manually here.
    @param winStride Window stride. It must be a multiple of block stride.
    @param padding Padding
    @param scale Coefficient of the detection window increase.
    @param finalThreshold Final threshold
    @param useMeanshiftGrouping indicates grouping algorithm
    */
    CV_WRAP virtual void detectMultiScale(InputArray img, CV_OUT std::vector<Rect>& foundLocations,
                                  CV_OUT std::vector<double>& foundWeights, double hitThreshold = 0,
                                  Size winStride = Size(), Size padding = Size(), double scale = 1.05,
                                  double finalThreshold = 2.0,bool useMeanshiftGrouping = false) const;

    /** @brief Detects objects of different sizes in the input image. The detected objects are returned as a list
    of rectangles.
    @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
    @param foundLocations Vector of rectangles where each rectangle contains the detected object.
    @param hitThreshold Threshold for the distance between features and SVM classifying plane.
    Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient).
    But if the free coefficient is omitted (which is allowed), you can specify it manually here.
    @param winStride Window stride. It must be a multiple of block stride.
    @param padding Padding
    @param scale Coefficient of the detection window increase.
    @param finalThreshold Final threshold
    @param useMeanshiftGrouping indicates grouping algorithm
    */
    virtual void detectMultiScale(InputArray img, CV_OUT std::vector<Rect>& foundLocations,
                                  double hitThreshold = 0, Size winStride = Size(),
                                  Size padding = Size(), double scale = 1.05,
                                  double finalThreshold = 2.0, bool useMeanshiftGrouping = false) const;

    /** @brief  Computes gradients and quantized gradient orientations.
    @param img Matrix contains the image to be computed
    @param grad Matrix of type CV_32FC2 contains computed gradients
    @param angleOfs Matrix of type CV_8UC2 contains quantized gradient orientations
    @param paddingTL Padding from top-left
    @param paddingBR Padding from bottom-right
    */
    CV_WRAP virtual void computeGradient(const Mat& img, CV_OUT Mat& grad, CV_OUT Mat& angleOfs,
                                 Size paddingTL = Size(), Size paddingBR = Size()) const;

    /** @brief Returns coefficients of the classifier trained for people detection (for 64x128 windows).
    */
    CV_WRAP static std::vector<float> getDefaultPeopleDetector();

    /**@example hog.cpp
    */
    /** @brief Returns coefficients of the classifier trained for people detection (for 48x96 windows).
    */
    CV_WRAP static std::vector<float> getDaimlerPeopleDetector();

    //! Detection window size. Align to block size and block stride. Default value is Size(64,128).
    CV_PROP Size winSize;

    //! Block size in pixels. Align to cell size. Default value is Size(16,16).
    CV_PROP Size blockSize;

    //! Block stride. It must be a multiple of cell size. Default value is Size(8,8).
    CV_PROP Size blockStride;

    //! Cell size. Default value is Size(8,8).
    CV_PROP Size cellSize;

    //! Number of bins used in the calculation of histogram of gradients. Default value is 9.
    CV_PROP int nbins;

    //! not documented
    CV_PROP int derivAperture;

    //! Gaussian smoothing window parameter.
    CV_PROP double winSigma;

    //! histogramNormType
    CV_PROP int histogramNormType;

    //! L2-Hys normalization method shrinkage.
    CV_PROP double L2HysThreshold;

    //! Flag to specify whether the gamma correction preprocessing is required or not.
    CV_PROP bool gammaCorrection;

    //! coefficients for the linear SVM classifier.
    CV_PROP std::vector<float> svmDetector;

    //! coefficients for the linear SVM classifier used when OpenCL is enabled
    UMat oclSvmDetector;

    //! not documented
    float free_coef;

    //! Maximum number of detection window increases. Default value is 64
    CV_PROP int nlevels;

    //! Indicates signed gradient will be used or not
    CV_PROP bool signedGradient;

    /** @brief evaluate specified ROI and return confidence value for each location
    @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
    @param locations Vector of Point
    @param foundLocations Vector of Point where each Point is detected object's top-left point.
    @param confidences confidences
    @param hitThreshold Threshold for the distance between features and SVM classifying plane. Usually
    it is 0 and should be specified in the detector coefficients (as the last free coefficient). But if
    the free coefficient is omitted (which is allowed), you can specify it manually here
    @param winStride winStride
    @param padding padding
    */
    virtual void detectROI(const cv::Mat& img, const std::vector<cv::Point> &locations,
                                   CV_OUT std::vector<cv::Point>& foundLocations, CV_OUT std::vector<double>& confidences,
                                   double hitThreshold = 0, cv::Size winStride = Size(),
                                   cv::Size padding = Size()) const;

    /** @brief evaluate specified ROI and return confidence value for each location in multiple scales
    @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
    @param foundLocations Vector of rectangles where each rectangle contains the detected object.
    @param locations Vector of DetectionROI
    @param hitThreshold Threshold for the distance between features and SVM classifying plane. Usually it is 0 and should be specified
    in the detector coefficients (as the last free coefficient). But if the free coefficient is omitted (which is allowed), you can specify it manually here.
    @param groupThreshold Minimum possible number of rectangles minus 1. The threshold is used in a group of rectangles to retain it.
    */
    virtual void detectMultiScaleROI(const cv::Mat& img,
                                     CV_OUT std::vector<cv::Rect>& foundLocations,
                                     std::vector<DetectionROI>& locations,
                                     double hitThreshold = 0,
                                     int groupThreshold = 0) const;

    /** @brief read/parse Dalal's alt model file
    @param modelfile Path of Dalal's alt model file.
    */
    void readALTModel(String modelfile);

    /** @brief Groups the object candidate rectangles.
    @param rectList  Input/output vector of rectangles. Output vector includes retained and grouped rectangles. (The Python list is not modified in place.)
    @param weights Input/output vector of weights of rectangles. Output vector includes weights of retained and grouped rectangles. (The Python list is not modified in place.)
    @param groupThreshold Minimum possible number of rectangles minus 1. The threshold is used in a group of rectangles to retain it.
    @param eps Relative difference between sides of the rectangles to merge them into a group.
    */
    void groupRectangles(std::vector<cv::Rect>& rectList, std::vector<double>& weights, int groupThreshold, double eps) const;
};

/** @brief Detect QR code in image and return minimum area of quadrangle that describes QR code.
    @param in  Matrix of the type CV_8UC1 containing an image where QR code are detected.
    @param points Output vector of vertices of a quadrangle of minimal area that describes QR code.
    @param eps_x Epsilon neighborhood, which allows you to determine the horizontal pattern of the scheme 1:1:3:1:1 according to QR code standard.
    @param eps_y Epsilon neighborhood, which allows you to determine the vertical pattern of the scheme 1:1:3:1:1 according to QR code standard.
    */
CV_EXPORTS bool detectQRCode(InputArray in, std::vector<Point> &points, double eps_x = 0.2, double eps_y = 0.1);

//! @} objdetect

}

#include "opencv2/objdetect/detection_based_tracker.hpp"

#ifndef DISABLE_OPENCV_24_COMPATIBILITY
#include "opencv2/objdetect/objdetect_c.h"
#endif

#endif
