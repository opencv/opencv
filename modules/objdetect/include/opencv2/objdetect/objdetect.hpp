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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#include "opencv2/core/core.hpp"

#ifdef __cplusplus
extern "C" {
#endif

/****************************************************************************************\
*                         Haar-like Object Detection functions                           *
\****************************************************************************************/

#define CV_HAAR_MAGIC_VAL    0x42500000
#define CV_TYPE_NAME_HAAR    "opencv-haar-classifier"

#define CV_IS_HAAR_CLASSIFIER( haar )                                                    \
    ((haar) != NULL &&                                                                   \
    (((const CvHaarClassifierCascade*)(haar))->flags & CV_MAGIC_MASK)==CV_HAAR_MAGIC_VAL)

#define CV_HAAR_FEATURE_MAX  3

typedef struct CvHaarFeature
{
    int  tilted;
    struct
    {
        CvRect r;
        float weight;
    } rect[CV_HAAR_FEATURE_MAX];
} CvHaarFeature;

typedef struct CvHaarClassifier
{
    int count;
    CvHaarFeature* haar_feature;
    float* threshold;
    int* left;
    int* right;
    float* alpha;
} CvHaarClassifier;

typedef struct CvHaarStageClassifier
{
    int  count;
    float threshold;
    CvHaarClassifier* classifier;

    int next;
    int child;
    int parent;
} CvHaarStageClassifier;

typedef struct CvHidHaarClassifierCascade CvHidHaarClassifierCascade;

typedef struct CvHaarClassifierCascade
{
    int  flags;
    int  count;
    CvSize orig_window_size;
    CvSize real_window_size;
    double scale;
    CvHaarStageClassifier* stage_classifier;
    CvHidHaarClassifierCascade* hid_cascade;
} CvHaarClassifierCascade;

typedef struct CvAvgComp
{
    CvRect rect;
    int neighbors;
} CvAvgComp;

/* Loads haar classifier cascade from a directory.
   It is obsolete: convert your cascade to xml and use cvLoad instead */
CVAPI(CvHaarClassifierCascade*) cvLoadHaarClassifierCascade(
                    const char* directory, CvSize orig_window_size);

CVAPI(void) cvReleaseHaarClassifierCascade( CvHaarClassifierCascade** cascade );

#define CV_HAAR_DO_CANNY_PRUNING    1
#define CV_HAAR_SCALE_IMAGE         2
#define CV_HAAR_FIND_BIGGEST_OBJECT 4
#define CV_HAAR_DO_ROUGH_SEARCH     8

CVAPI(CvSeq*) cvHaarDetectObjects( const CvArr* image,
                     CvHaarClassifierCascade* cascade,
                     CvMemStorage* storage, double scale_factor CV_DEFAULT(1.1),
                     int min_neighbors CV_DEFAULT(3), int flags CV_DEFAULT(0),
                     CvSize min_size CV_DEFAULT(cvSize(0,0)));

/* sets images for haar classifier cascade */
CVAPI(void) cvSetImagesForHaarClassifierCascade( CvHaarClassifierCascade* cascade,
                                                const CvArr* sum, const CvArr* sqsum,
                                                const CvArr* tilted_sum, double scale );

/* runs the cascade on the specified window */
CVAPI(int) cvRunHaarClassifierCascade( const CvHaarClassifierCascade* cascade,
                                       CvPoint pt, int start_stage CV_DEFAULT(0));

#ifdef __cplusplus
}

namespace cv
{
	
///////////////////////////// Object Detection ////////////////////////////

CV_EXPORTS void groupRectangles(vector<Rect>& rectList, int groupThreshold, double eps=0.2);
CV_EXPORTS void groupRectangles(vector<Rect>& rectList, vector<int>& weights, int groupThreshold, double eps=0.2);
        
class CV_EXPORTS FeatureEvaluator
{
public:    
    enum { HAAR = 0, LBP = 1 };
    virtual ~FeatureEvaluator();
    virtual bool read(const FileNode& node);
    virtual Ptr<FeatureEvaluator> clone() const;
    virtual int getFeatureType() const;
    
    virtual bool setImage(const Mat&, Size origWinSize);
    virtual bool setWindow(Point p);

    virtual double calcOrd(int featureIdx) const;
    virtual int calcCat(int featureIdx) const;

    static Ptr<FeatureEvaluator> create(int type);
};

template<> CV_EXPORTS void Ptr<CvHaarClassifierCascade>::delete_obj();
   
class CV_EXPORTS CascadeClassifier
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
    
    enum { BOOST = 0 };
    enum { DO_CANNY_PRUNING = 1, SCALE_IMAGE = 2,
           FIND_BIGGEST_OBJECT = 4, DO_ROUGH_SEARCH = 8 };

    CascadeClassifier();
    CascadeClassifier(const string& filename);
    ~CascadeClassifier();
    
    bool empty() const;
    bool load(const string& filename);
    bool read(const FileNode& node);
    void detectMultiScale( const Mat& image,
                           vector<Rect>& objects,
                           double scaleFactor=1.1,
                           int minNeighbors=3, int flags=0,
                           Size minSize=Size());
 
    bool setImage( Ptr<FeatureEvaluator>&, const Mat& );
    int runAt( Ptr<FeatureEvaluator>&, Point );

    bool is_stump_based;

    int stageType;
    int featureType;
    int ncategories;
    Size origWinSize;
    
    vector<Stage> stages;
    vector<DTree> classifiers;
    vector<DTreeNode> nodes;
    vector<float> leaves;
    vector<int> subsets;

    Ptr<FeatureEvaluator> feval;
    Ptr<CvHaarClassifierCascade> oldCascade;
};


//////////////// HOG (Histogram-of-Oriented-Gradients) Descriptor and Object Detector //////////////

struct CV_EXPORTS HOGDescriptor
{
public:
    enum { L2Hys=0 };
    
    HOGDescriptor() : winSize(64,128), blockSize(16,16), blockStride(8,8),
    	cellSize(8,8), nbins(9), derivAperture(1), winSigma(-1),
    	histogramNormType(L2Hys), L2HysThreshold(0.2), gammaCorrection(true)
    {}
    
    HOGDescriptor(Size _winSize, Size _blockSize, Size _blockStride,
                  Size _cellSize, int _nbins, int _derivAperture=1, double _winSigma=-1,
                  int _histogramNormType=L2Hys, double _L2HysThreshold=0.2, bool _gammaCorrection=false)
    : winSize(_winSize), blockSize(_blockSize), blockStride(_blockStride), cellSize(_cellSize),
    nbins(_nbins), derivAperture(_derivAperture), winSigma(_winSigma),
    histogramNormType(_histogramNormType), L2HysThreshold(_L2HysThreshold),
    gammaCorrection(_gammaCorrection)
    {}
    
    HOGDescriptor(const String& filename)
    {
        load(filename);
    }
    
    HOGDescriptor(const HOGDescriptor& d)
    {
        d.copyTo(*this);
    }
    
    virtual ~HOGDescriptor() {}
    
    size_t getDescriptorSize() const;
    bool checkDetectorSize() const;
    double getWinSigma() const;
    
    virtual void setSVMDetector(const vector<float>& _svmdetector);
    
    virtual bool read(FileNode& fn);
    virtual void write(FileStorage& fs, const String& objname) const;
    
    virtual bool load(const String& filename, const String& objname=String());
    virtual void save(const String& filename, const String& objname=String()) const;
    virtual void copyTo(HOGDescriptor& c) const;
    
    virtual void compute(const Mat& img,
                         vector<float>& descriptors,
                         Size winStride=Size(), Size padding=Size(),
                         const vector<Point>& locations=vector<Point>()) const;
    virtual void detect(const Mat& img, vector<Point>& foundLocations,
                        double hitThreshold=0, Size winStride=Size(),
                        Size padding=Size(),
                        const vector<Point>& searchLocations=vector<Point>()) const;
    virtual void detectMultiScale(const Mat& img, vector<Rect>& foundLocations,
                                  double hitThreshold=0, Size winStride=Size(),
                                  Size padding=Size(), double scale=1.05,
                                  int groupThreshold=2) const;
    virtual void computeGradient(const Mat& img, Mat& grad, Mat& angleOfs,
                                 Size paddingTL=Size(), Size paddingBR=Size()) const;
    
    static vector<float> getDefaultPeopleDetector();
    
    Size winSize;
    Size blockSize;
    Size blockStride;
    Size cellSize;
    int nbins;
    int derivAperture;
    double winSigma;
    int histogramNormType;
    double L2HysThreshold;
    bool gammaCorrection;
    vector<float> svmDetector;
};

	
}

#endif

#endif
