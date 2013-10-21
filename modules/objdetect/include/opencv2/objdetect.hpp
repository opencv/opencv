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

#ifndef __OPENCV_OBJDETECT_HPP__
#define __OPENCV_OBJDETECT_HPP__

#include "opencv2/core.hpp"

typedef struct CvLatentSvmDetector CvLatentSvmDetector;
typedef struct CvHaarClassifierCascade CvHaarClassifierCascade;

namespace cv
{

///////////////////////////// Object Detection ////////////////////////////

/*
 * This is a class wrapping up the structure CvLatentSvmDetector and functions working with it.
 * The class goals are:
 * 1) provide c++ interface;
 * 2) make it possible to load and detect more than one class (model) unlike CvLatentSvmDetector.
 */
class CV_EXPORTS LatentSvmDetector
{
public:
    struct CV_EXPORTS ObjectDetection
    {
        ObjectDetection();
        ObjectDetection( const Rect& rect, float score, int classID = -1 );
        Rect rect;
        float score;
        int classID;
    };

    LatentSvmDetector();
    LatentSvmDetector( const std::vector<String>& filenames, const std::vector<String>& classNames = std::vector<String>() );
    virtual ~LatentSvmDetector();

    virtual void clear();
    virtual bool empty() const;
    bool load( const std::vector<String>& filenames, const std::vector<String>& classNames = std::vector<String>() );

    virtual void detect( const Mat& image,
                         std::vector<ObjectDetection>& objectDetections,
                         float overlapThreshold = 0.5f,
                         int numThreads = -1 );

    const std::vector<String>& getClassNames() const;
    size_t getClassCount() const;

private:
    std::vector<CvLatentSvmDetector*> detectors;
    std::vector<String> classNames;
};

// class for grouping object candidates, detected by Cascade Classifier, HOG etc.
// instance of the class is to be passed to cv::partition (see cxoperations.hpp)
class CV_EXPORTS SimilarRects
{
public:
    SimilarRects(double _eps) : eps(_eps) {}
    inline bool operator()(const Rect& r1, const Rect& r2) const
    {
        double delta = eps*(std::min(r1.width, r2.width) + std::min(r1.height, r2.height))*0.5;
        return std::abs(r1.x - r2.x) <= delta &&
            std::abs(r1.y - r2.y) <= delta &&
            std::abs(r1.x + r1.width - r2.x - r2.width) <= delta &&
            std::abs(r1.y + r1.height - r2.y - r2.height) <= delta;
    }
    double eps;
};

CV_EXPORTS   void groupRectangles(std::vector<Rect>& rectList, int groupThreshold, double eps = 0.2);
CV_EXPORTS_W void groupRectangles(CV_IN_OUT std::vector<Rect>& rectList, CV_OUT std::vector<int>& weights, int groupThreshold, double eps = 0.2);
CV_EXPORTS   void groupRectangles(std::vector<Rect>& rectList, int groupThreshold, double eps, std::vector<int>* weights, std::vector<double>* levelWeights );
CV_EXPORTS   void groupRectangles(std::vector<Rect>& rectList, std::vector<int>& rejectLevels,
                                  std::vector<double>& levelWeights, int groupThreshold, double eps = 0.2);
CV_EXPORTS   void groupRectangles_meanshift(std::vector<Rect>& rectList, std::vector<double>& foundWeights, std::vector<double>& foundScales,
                                          double detectThreshold = 0.0, Size winDetSize = Size(64, 128));

class CV_EXPORTS FeatureEvaluator
{
public:
    enum { HAAR = 0,
           LBP  = 1,
           HOG  = 2
         };

    virtual ~FeatureEvaluator();

    virtual bool read(const FileNode& node);
    virtual Ptr<FeatureEvaluator> clone() const;
    virtual int getFeatureType() const;

    virtual bool setImage(const Mat& img, Size origWinSize);
    virtual bool setWindow(Point p);

    virtual double calcOrd(int featureIdx) const;
    virtual int calcCat(int featureIdx) const;

    static Ptr<FeatureEvaluator> create(int type);
};

template<> CV_EXPORTS void DefaultDeleter<CvHaarClassifierCascade>::operator ()(CvHaarClassifierCascade* obj) const;

enum { CASCADE_DO_CANNY_PRUNING    = 1,
       CASCADE_SCALE_IMAGE         = 2,
       CASCADE_FIND_BIGGEST_OBJECT = 4,
       CASCADE_DO_ROUGH_SEARCH     = 8
     };

class CV_EXPORTS_W CascadeClassifier
{
public:
    CV_WRAP CascadeClassifier();
    CV_WRAP CascadeClassifier( const String& filename );
    virtual ~CascadeClassifier();

    CV_WRAP virtual bool empty() const;
    CV_WRAP bool load( const String& filename );
    virtual bool read( const FileNode& node );
    CV_WRAP virtual void detectMultiScale( const Mat& image,
                                   CV_OUT std::vector<Rect>& objects,
                                   double scaleFactor = 1.1,
                                   int minNeighbors = 3, int flags = 0,
                                   Size minSize = Size(),
                                   Size maxSize = Size() );

    CV_WRAP virtual void detectMultiScale( const Mat& image,
                                   CV_OUT std::vector<Rect>& objects,
                                   CV_OUT std::vector<int>& numDetections,
                                   double scaleFactor=1.1,
                                   int minNeighbors=3, int flags=0,
                                   Size minSize=Size(),
                                   Size maxSize=Size() );

    CV_WRAP virtual void detectMultiScale( const Mat& image,
                                   CV_OUT std::vector<Rect>& objects,
                                   CV_OUT std::vector<int>& rejectLevels,
                                   CV_OUT std::vector<double>& levelWeights,
                                   double scaleFactor = 1.1,
                                   int minNeighbors = 3, int flags = 0,
                                   Size minSize = Size(),
                                   Size maxSize = Size(),
                                   bool outputRejectLevels = false );


    bool isOldFormatCascade() const;
    virtual Size getOriginalWindowSize() const;
    int getFeatureType() const;
    bool setImage( const Mat& );

protected:
    virtual bool detectSingleScale( const Mat& image, int stripCount, Size processingRectSize,
                                    int stripSize, int yStep, double factor, std::vector<Rect>& candidates,
                                    std::vector<int>& rejectLevels, std::vector<double>& levelWeights, bool outputRejectLevels = false );

    virtual void detectMultiScaleNoGrouping( const Mat& image, std::vector<Rect>& candidates,
                                             std::vector<int>& rejectLevels, std::vector<double>& levelWeights,
                                             double scaleFactor, Size minObjectSize, Size maxObjectSize,
                                             bool outputRejectLevels = false );

protected:
    enum { BOOST = 0
         };
    enum { DO_CANNY_PRUNING    = CASCADE_DO_CANNY_PRUNING,
           SCALE_IMAGE         = CASCADE_SCALE_IMAGE,
           FIND_BIGGEST_OBJECT = CASCADE_FIND_BIGGEST_OBJECT,
           DO_ROUGH_SEARCH     = CASCADE_DO_ROUGH_SEARCH
         };

    friend class CascadeClassifierInvoker;

    template<class FEval>
    friend int predictOrdered( CascadeClassifier& cascade, Ptr<FeatureEvaluator> &featureEvaluator, double& weight);

    template<class FEval>
    friend int predictCategorical( CascadeClassifier& cascade, Ptr<FeatureEvaluator> &featureEvaluator, double& weight);

    template<class FEval>
    friend int predictOrderedStump( CascadeClassifier& cascade, Ptr<FeatureEvaluator> &featureEvaluator, double& weight);

    template<class FEval>
    friend int predictCategoricalStump( CascadeClassifier& cascade, Ptr<FeatureEvaluator> &featureEvaluator, double& weight);

    bool setImage( Ptr<FeatureEvaluator>& feval, const Mat& image);
    virtual int runAt( Ptr<FeatureEvaluator>& feval, Point pt, double& weight );

    class Data
    {
    public:
        struct CV_EXPORTS DTreeNode
        {
            int featureIdx;
            float threshold; // for ordered features only
            int left;
            int right;
        };

        struct CV_EXPORTS DTree
        {
            int nodeCount;
        };

        struct CV_EXPORTS Stage
        {
            int first;
            int ntrees;
            float threshold;
        };

        bool read(const FileNode &node);

        bool isStumpBased;

        int stageType;
        int featureType;
        int ncategories;
        Size origWinSize;

        std::vector<Stage> stages;
        std::vector<DTree> classifiers;
        std::vector<DTreeNode> nodes;
        std::vector<float> leaves;
        std::vector<int> subsets;
    };

    Data data;
    Ptr<FeatureEvaluator> featureEvaluator;
    Ptr<CvHaarClassifierCascade> oldCascade;

public:
    class CV_EXPORTS MaskGenerator
    {
    public:
        virtual ~MaskGenerator() {}
        virtual cv::Mat generateMask(const cv::Mat& src)=0;
        virtual void initializeMask(const cv::Mat& /*src*/) {};
    };
    void setMaskGenerator(Ptr<MaskGenerator> maskGenerator);
    Ptr<MaskGenerator> getMaskGenerator();

    void setFaceDetectionMaskGenerator();

protected:
    Ptr<MaskGenerator> maskGenerator;
};

//////////////// HOG (Histogram-of-Oriented-Gradients) Descriptor and Object Detector //////////////

// struct for detection region of interest (ROI)
struct DetectionROI
{
   // scale(size) of the bounding box
   double scale;
   // set of requrested locations to be evaluated
   std::vector<cv::Point> locations;
   // vector that will contain confidence values for each location
   std::vector<double> confidences;
};

struct CV_EXPORTS_W HOGDescriptor
{
public:
    enum { L2Hys = 0
         };
    enum { DEFAULT_NLEVELS = 64
         };

    CV_WRAP HOGDescriptor() : winSize(64,128), blockSize(16,16), blockStride(8,8),
        cellSize(8,8), nbins(9), derivAperture(1), winSigma(-1),
        histogramNormType(HOGDescriptor::L2Hys), L2HysThreshold(0.2), gammaCorrection(true),
        nlevels(HOGDescriptor::DEFAULT_NLEVELS)
    {}

    CV_WRAP HOGDescriptor(Size _winSize, Size _blockSize, Size _blockStride,
                  Size _cellSize, int _nbins, int _derivAperture=1, double _winSigma=-1,
                  int _histogramNormType=HOGDescriptor::L2Hys,
                  double _L2HysThreshold=0.2, bool _gammaCorrection=false,
                  int _nlevels=HOGDescriptor::DEFAULT_NLEVELS)
    : winSize(_winSize), blockSize(_blockSize), blockStride(_blockStride), cellSize(_cellSize),
    nbins(_nbins), derivAperture(_derivAperture), winSigma(_winSigma),
    histogramNormType(_histogramNormType), L2HysThreshold(_L2HysThreshold),
    gammaCorrection(_gammaCorrection), nlevels(_nlevels)
    {}

    CV_WRAP HOGDescriptor(const String& filename)
    {
        load(filename);
    }

    HOGDescriptor(const HOGDescriptor& d)
    {
        d.copyTo(*this);
    }

    virtual ~HOGDescriptor() {}

    CV_WRAP size_t getDescriptorSize() const;
    CV_WRAP bool checkDetectorSize() const;
    CV_WRAP double getWinSigma() const;

    CV_WRAP virtual void setSVMDetector(InputArray _svmdetector);

    virtual bool read(FileNode& fn);
    virtual void write(FileStorage& fs, const String& objname) const;

    CV_WRAP virtual bool load(const String& filename, const String& objname = String());
    CV_WRAP virtual void save(const String& filename, const String& objname = String()) const;
    virtual void copyTo(HOGDescriptor& c) const;

    CV_WRAP virtual void compute(const Mat& img,
                         CV_OUT std::vector<float>& descriptors,
                         Size winStride = Size(), Size padding = Size(),
                         const std::vector<Point>& locations = std::vector<Point>()) const;
    //with found weights output
    CV_WRAP virtual void detect(const Mat& img, CV_OUT std::vector<Point>& foundLocations,
                        CV_OUT std::vector<double>& weights,
                        double hitThreshold = 0, Size winStride = Size(),
                        Size padding = Size(),
                        const std::vector<Point>& searchLocations = std::vector<Point>()) const;
    //without found weights output
    virtual void detect(const Mat& img, CV_OUT std::vector<Point>& foundLocations,
                        double hitThreshold = 0, Size winStride = Size(),
                        Size padding = Size(),
                        const std::vector<Point>& searchLocations=std::vector<Point>()) const;
    //with result weights output
    CV_WRAP virtual void detectMultiScale(const Mat& img, CV_OUT std::vector<Rect>& foundLocations,
                                  CV_OUT std::vector<double>& foundWeights, double hitThreshold = 0,
                                  Size winStride = Size(), Size padding = Size(), double scale = 1.05,
                                  double finalThreshold = 2.0,bool useMeanshiftGrouping = false) const;
    //without found weights output
    virtual void detectMultiScale(const Mat& img, CV_OUT std::vector<Rect>& foundLocations,
                                  double hitThreshold = 0, Size winStride = Size(),
                                  Size padding = Size(), double scale = 1.05,
                                  double finalThreshold = 2.0, bool useMeanshiftGrouping = false) const;

    CV_WRAP virtual void computeGradient(const Mat& img, CV_OUT Mat& grad, CV_OUT Mat& angleOfs,
                                 Size paddingTL = Size(), Size paddingBR = Size()) const;

    CV_WRAP static std::vector<float> getDefaultPeopleDetector();
    CV_WRAP static std::vector<float> getDaimlerPeopleDetector();

    CV_PROP Size winSize;
    CV_PROP Size blockSize;
    CV_PROP Size blockStride;
    CV_PROP Size cellSize;
    CV_PROP int nbins;
    CV_PROP int derivAperture;
    CV_PROP double winSigma;
    CV_PROP int histogramNormType;
    CV_PROP double L2HysThreshold;
    CV_PROP bool gammaCorrection;
    CV_PROP std::vector<float> svmDetector;
    CV_PROP int nlevels;


   // evaluate specified ROI and return confidence value for each location
   virtual void detectROI(const cv::Mat& img, const std::vector<cv::Point> &locations,
                                   CV_OUT std::vector<cv::Point>& foundLocations, CV_OUT std::vector<double>& confidences,
                                   double hitThreshold = 0, cv::Size winStride = Size(),
                                   cv::Size padding = Size()) const;

   // evaluate specified ROI and return confidence value for each location in multiple scales
   virtual void detectMultiScaleROI(const cv::Mat& img,
                                                       CV_OUT std::vector<cv::Rect>& foundLocations,
                                                       std::vector<DetectionROI>& locations,
                                                       double hitThreshold = 0,
                                                       int groupThreshold = 0) const;

   // read/parse Dalal's alt model file
   void readALTModel(String modelfile);
   void groupRectangles(std::vector<cv::Rect>& rectList, std::vector<double>& weights, int groupThreshold, double eps) const;
};


CV_EXPORTS_W void findDataMatrix(InputArray image,
                                 CV_OUT std::vector<String>& codes,
                                 OutputArray corners = noArray(),
                                 OutputArrayOfArrays dmtx = noArray());

CV_EXPORTS_W void drawDataMatrixCodes(InputOutputArray image,
                                      const std::vector<String>& codes,
                                      InputArray corners);
}

#include "opencv2/objdetect/linemod.hpp"
#include "opencv2/objdetect/erfilter.hpp"

#endif
