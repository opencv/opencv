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
#include "opencv2/features2d/features2d.hpp"

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
    int tilted;
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
                     CvSize min_size CV_DEFAULT(cvSize(0,0)), CvSize max_size CV_DEFAULT(cvSize(0,0)));

/* sets images for haar classifier cascade */
CVAPI(void) cvSetImagesForHaarClassifierCascade( CvHaarClassifierCascade* cascade,
                                                const CvArr* sum, const CvArr* sqsum,
                                                const CvArr* tilted_sum, double scale );

/* runs the cascade on the specified window */
CVAPI(int) cvRunHaarClassifierCascade( const CvHaarClassifierCascade* cascade,
                                       CvPoint pt, int start_stage CV_DEFAULT(0));


/****************************************************************************************\
*                         Latent SVM Object Detection functions                          *
\****************************************************************************************/

// DataType: STRUCT position
// Structure describes the position of the filter in the feature pyramid
// l - level in the feature pyramid
// (x, y) - coordinate in level l
typedef struct
{
    unsigned int x;
    unsigned int y;
    unsigned int l;
} CvLSVMFilterPosition;

// DataType: STRUCT filterObject
// Description of the filter, which corresponds to the part of the object
// V               - ideal (penalty = 0) position of the partial filter
//                   from the root filter position (V_i in the paper)
// penaltyFunction - vector describes penalty function (d_i in the paper)
//                   pf[0] * x + pf[1] * y + pf[2] * x^2 + pf[3] * y^2
// FILTER DESCRIPTION
//   Rectangular map (sizeX x sizeY), 
//   every cell stores feature vector (dimension = p)
// H               - matrix of feature vectors
//                   to set and get feature vectors (i,j) 
//                   used formula H[(j * sizeX + i) * p + k], where
//                   k - component of feature vector in cell (i, j)
// END OF FILTER DESCRIPTION
// xp              - auxillary parameter for internal use
//                   size of row in feature vectors 
//                   (yp = (int) (p / xp); p = xp * yp)
typedef struct{
    CvLSVMFilterPosition V;
    float fineFunction[4];
    unsigned int sizeX;
    unsigned int sizeY;
    unsigned int p;
    unsigned int xp;
    float *H;
} CvLSVMFilterObject;

// data type: STRUCT CvLatentSvmDetector
// structure contains internal representation of trained Latent SVM detector
// num_filters			- total number of filters (root plus part) in model 
// num_components		- number of components in model
// num_part_filters		- array containing number of part filters for each component
// filters				- root and part filters for all model components
// b					- biases for all model components
// score_threshold		- confidence level threshold
typedef struct CvLatentSvmDetector
{
	int num_filters;
	int num_components;
	int* num_part_filters;
	CvLSVMFilterObject** filters;
	float* b;
	float score_threshold;
}
CvLatentSvmDetector;

// data type: STRUCT CvObjectDetection
// structure contains the bounding box and confidence level for detected object 
// rect					- bounding box for a detected object
// score				- confidence level 
typedef struct CvObjectDetection
{
	CvRect rect;
	float score;
} CvObjectDetection;

//////////////// Object Detection using Latent SVM //////////////


/*
// load trained detector from a file
//
// API
// CvLatentSvmDetector* cvLoadLatentSvmDetector(const char* filename);
// INPUT
// filename				- path to the file containing the parameters of
						- trained Latent SVM detector
// OUTPUT
// trained Latent SVM detector in internal representation
*/
CVAPI(CvLatentSvmDetector*) cvLoadLatentSvmDetector(const char* filename);

/*
// release memory allocated for CvLatentSvmDetector structure
//
// API
// void cvReleaseLatentSvmDetector(CvLatentSvmDetector** detector);
// INPUT
// detector				- CvLatentSvmDetector structure to be released
// OUTPUT
*/
CVAPI(void) cvReleaseLatentSvmDetector(CvLatentSvmDetector** detector);

/*
// find rectangular regions in the given image that are likely 
// to contain objects and corresponding confidence levels
//
// API
// CvSeq* cvLatentSvmDetectObjects(const IplImage* image, 
//									CvLatentSvmDetector* detector, 
//									CvMemStorage* storage, 
//									float overlap_threshold = 0.5f);
// INPUT
// image				- image to detect objects in
// detector				- Latent SVM detector in internal representation
// storage				- memory storage to store the resultant sequence 
//							of the object candidate rectangles
// overlap_threshold	- threshold for the non-maximum suppression algorithm 
                           = 0.5f [here will be the reference to original paper]
// OUTPUT
// sequence of detected objects (bounding boxes and confidence levels stored in CvObjectDetection structures)
*/
CVAPI(CvSeq*) cvLatentSvmDetectObjects(IplImage* image, 
								CvLatentSvmDetector* detector, 
								CvMemStorage* storage, 
								float overlap_threshold CV_DEFAULT(0.5f));

#ifdef __cplusplus
}

namespace cv
{
	
///////////////////////////// Object Detection ////////////////////////////

CV_EXPORTS_W void groupRectangles(vector<Rect>& rectList, int groupThreshold, double eps=0.2);
CV_EXPORTS_W void groupRectangles(vector<Rect>& rectList, CV_OUT vector<int>& weights, int groupThreshold, double eps=0.2);
        
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
   
class CV_EXPORTS_W CascadeClassifier
{
public:
    CV_WRAP CascadeClassifier();
    CV_WRAP CascadeClassifier( const string& filename );
    virtual ~CascadeClassifier();
    
    CV_WRAP virtual bool empty() const;
    CV_WRAP bool load( const string& filename );
    bool read( const FileNode& node );
    CV_WRAP void detectMultiScale( const Mat& image,
                                   CV_OUT vector<Rect>& objects,
                                   double scaleFactor=1.1,
                                   int minNeighbors=3, int flags=0,
                                   Size minSize=Size(),
                                   Size maxSize=Size() );

    bool isOldFormatCascade() const;
    virtual Size getOriginalWindowSize() const;
    int getFeatureType() const;
    bool setImage( const Mat& );

protected:
    virtual bool detectSingleScale( const Mat& image, int stripCount, Size processingRectSize,
                                    int stripSize, int yStep, double factor, vector<Rect>& candidates );

private:
    enum { BOOST = 0 };
    enum { DO_CANNY_PRUNING = 1, SCALE_IMAGE = 2,
           FIND_BIGGEST_OBJECT = 4, DO_ROUGH_SEARCH = 8 };

    friend struct CascadeClassifierInvoker;

    template<class FEval>
    friend int predictOrdered( CascadeClassifier& cascade, Ptr<FeatureEvaluator> &featureEvaluator);

    template<class FEval>
    friend int predictCategorical( CascadeClassifier& cascade, Ptr<FeatureEvaluator> &featureEvaluator);

    template<class FEval>
    friend int predictOrderedStump( CascadeClassifier& cascade, Ptr<FeatureEvaluator> &featureEvaluator);

    template<class FEval>
    friend int predictCategoricalStump( CascadeClassifier& cascade, Ptr<FeatureEvaluator> &featureEvaluator);

    bool setImage( Ptr<FeatureEvaluator>&, const Mat& );
    int runAt( Ptr<FeatureEvaluator>&, Point );

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

        vector<Stage> stages;
        vector<DTree> classifiers;
        vector<DTreeNode> nodes;
        vector<float> leaves;
        vector<int> subsets;
    };

    Data data;
    Ptr<FeatureEvaluator> featureEvaluator;
    Ptr<CvHaarClassifierCascade> oldCascade;
};

//////////////// HOG (Histogram-of-Oriented-Gradients) Descriptor and Object Detector //////////////

struct CV_EXPORTS_W HOGDescriptor
{
public:
    enum { L2Hys=0 };
    enum { DEFAULT_NLEVELS=64 };
    
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
    
    CV_WRAP virtual void setSVMDetector(const vector<float>& _svmdetector);
    
    virtual bool read(FileNode& fn);
    virtual void write(FileStorage& fs, const String& objname) const;
    
    CV_WRAP virtual bool load(const String& filename, const String& objname=String());
    CV_WRAP virtual void save(const String& filename, const String& objname=String()) const;
    virtual void copyTo(HOGDescriptor& c) const;
    
    CV_WRAP virtual void compute(const Mat& img,
                         CV_OUT vector<float>& descriptors,
                         Size winStride=Size(), Size padding=Size(),
                         const vector<Point>& locations=vector<Point>()) const;
    CV_WRAP virtual void detect(const Mat& img, CV_OUT vector<Point>& foundLocations,
                        double hitThreshold=0, Size winStride=Size(),
                        Size padding=Size(),
                        const vector<Point>& searchLocations=vector<Point>()) const;
    CV_WRAP virtual void detectMultiScale(const Mat& img, CV_OUT vector<Rect>& foundLocations,
                                  double hitThreshold=0, Size winStride=Size(),
                                  Size padding=Size(), double scale=1.05,
                                  int groupThreshold=2) const;
    CV_WRAP virtual void computeGradient(const Mat& img, CV_OUT Mat& grad, CV_OUT Mat& angleOfs,
                                 Size paddingTL=Size(), Size paddingBR=Size()) const;
    
    static vector<float> getDefaultPeopleDetector();
    
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
    CV_PROP vector<float> svmDetector;
    CV_PROP int nlevels;
};

/****************************************************************************************\
*                                Planar Object Detection                                 *
\****************************************************************************************/

class CV_EXPORTS PlanarObjectDetector
{
public:
    PlanarObjectDetector();
    PlanarObjectDetector(const FileNode& node);
    PlanarObjectDetector(const vector<Mat>& pyr, int _npoints=300,
                         int _patchSize=FernClassifier::PATCH_SIZE,
                         int _nstructs=FernClassifier::DEFAULT_STRUCTS,
                         int _structSize=FernClassifier::DEFAULT_STRUCT_SIZE,
                         int _nviews=FernClassifier::DEFAULT_VIEWS,
                         const LDetector& detector=LDetector(),
                         const PatchGenerator& patchGenerator=PatchGenerator());
    virtual ~PlanarObjectDetector();
    virtual void train(const vector<Mat>& pyr, int _npoints=300,
                       int _patchSize=FernClassifier::PATCH_SIZE,
                       int _nstructs=FernClassifier::DEFAULT_STRUCTS,
                       int _structSize=FernClassifier::DEFAULT_STRUCT_SIZE,
                       int _nviews=FernClassifier::DEFAULT_VIEWS,
                       const LDetector& detector=LDetector(),
                       const PatchGenerator& patchGenerator=PatchGenerator());
    virtual void train(const vector<Mat>& pyr, const vector<KeyPoint>& keypoints,
                       int _patchSize=FernClassifier::PATCH_SIZE,
                       int _nstructs=FernClassifier::DEFAULT_STRUCTS,
                       int _structSize=FernClassifier::DEFAULT_STRUCT_SIZE,
                       int _nviews=FernClassifier::DEFAULT_VIEWS,
                       const LDetector& detector=LDetector(),
                       const PatchGenerator& patchGenerator=PatchGenerator());
    Rect getModelROI() const;
    vector<KeyPoint> getModelPoints() const;
    const LDetector& getDetector() const;
    const FernClassifier& getClassifier() const;
    void setVerbose(bool verbose);

    void read(const FileNode& node);
    void write(FileStorage& fs, const String& name=String()) const;
    bool operator()(const Mat& image, CV_OUT Mat& H, CV_OUT vector<Point2f>& corners) const;
    bool operator()(const vector<Mat>& pyr, const vector<KeyPoint>& keypoints,
                                       CV_OUT Mat& H, CV_OUT vector<Point2f>& corners,
                                       CV_OUT vector<int>* pairs=0) const;

protected:
    bool verbose;
    Rect modelROI;
    vector<KeyPoint> modelPoints;
    LDetector ldetector;
    FernClassifier fernClassifier;
};

}

#endif

#endif
