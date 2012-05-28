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
#include <map>
#include <deque>

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

//CVAPI(CvSeq*) cvHaarDetectObjectsForROC( const CvArr* image,
//                     CvHaarClassifierCascade* cascade, CvMemStorage* storage,
//                     CvSeq** rejectLevels, CvSeq** levelWeightds,
//                     double scale_factor CV_DEFAULT(1.1),
//                     int min_neighbors CV_DEFAULT(3), int flags CV_DEFAULT(0),
//                     CvSize min_size CV_DEFAULT(cvSize(0,0)), CvSize max_size CV_DEFAULT(cvSize(0,0)),
//                     bool outputRejectLevels = false );


CVAPI(CvSeq*) cvHaarDetectObjects( const CvArr* image,
                     CvHaarClassifierCascade* cascade, CvMemStorage* storage, 
                     double scale_factor CV_DEFAULT(1.1),
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
    int x;
    int y;
    int l;
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
typedef struct{
    CvLSVMFilterPosition V;
    float fineFunction[4];
    int sizeX;
    int sizeY;
    int numFeatures;
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
//									float overlap_threshold = 0.5f,
//                                  int numThreads = -1);
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
								float overlap_threshold CV_DEFAULT(0.5f),
                                int numThreads CV_DEFAULT(-1));

#ifdef __cplusplus
}

CV_EXPORTS CvSeq* cvHaarDetectObjectsForROC( const CvArr* image,
                     CvHaarClassifierCascade* cascade, CvMemStorage* storage,
                     std::vector<int>& rejectLevels, std::vector<double>& levelWeightds,
                     double scale_factor CV_DEFAULT(1.1),
                     int min_neighbors CV_DEFAULT(3), int flags CV_DEFAULT(0),
                     CvSize min_size CV_DEFAULT(cvSize(0,0)), CvSize max_size CV_DEFAULT(cvSize(0,0)),
                     bool outputRejectLevels = false );

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
        ObjectDetection( const Rect& rect, float score, int classID=-1 );
        Rect rect;
        float score;
        int classID;
    };

    LatentSvmDetector();
    LatentSvmDetector( const vector<string>& filenames, const vector<string>& classNames=vector<string>() );
    virtual ~LatentSvmDetector();

    virtual void clear();
    virtual bool empty() const;
    bool load( const vector<string>& filenames, const vector<string>& classNames=vector<string>() );

    virtual void detect( const Mat& image,
                         vector<ObjectDetection>& objectDetections,
                         float overlapThreshold=0.5f,
                         int numThreads=-1 );

    const vector<string>& getClassNames() const;
    size_t getClassCount() const;

private:
    vector<CvLatentSvmDetector*> detectors;
    vector<string> classNames;
};

CV_EXPORTS void groupRectangles(CV_OUT CV_IN_OUT vector<Rect>& rectList, int groupThreshold, double eps=0.2);
CV_EXPORTS_W void groupRectangles(CV_OUT CV_IN_OUT vector<Rect>& rectList, CV_OUT vector<int>& weights, int groupThreshold, double eps=0.2);
CV_EXPORTS void groupRectangles( vector<Rect>& rectList, int groupThreshold, double eps, vector<int>* weights, vector<double>* levelWeights );
CV_EXPORTS void groupRectangles(vector<Rect>& rectList, vector<int>& rejectLevels, 
                                vector<double>& levelWeights, int groupThreshold, double eps=0.2);
CV_EXPORTS void groupRectangles_meanshift(vector<Rect>& rectList, vector<double>& foundWeights, vector<double>& foundScales, 
										  double detectThreshold = 0.0, Size winDetSize = Size(64, 128));

        
class CV_EXPORTS FeatureEvaluator
{
public:    
    enum { HAAR = 0, LBP = 1, HOG = 2 };
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

template<> CV_EXPORTS void Ptr<CvHaarClassifierCascade>::delete_obj();

enum
{
	CASCADE_DO_CANNY_PRUNING=1,
	CASCADE_SCALE_IMAGE=2,
	CASCADE_FIND_BIGGEST_OBJECT=4,
	CASCADE_DO_ROUGH_SEARCH=8
};

class CV_EXPORTS_W CascadeClassifier
{
public:
    CV_WRAP CascadeClassifier();
    CV_WRAP CascadeClassifier( const string& filename );
    virtual ~CascadeClassifier();
    
    CV_WRAP virtual bool empty() const;
    CV_WRAP bool load( const string& filename );
    virtual bool read( const FileNode& node );
    CV_WRAP virtual void detectMultiScale( const Mat& image,
                                   CV_OUT vector<Rect>& objects,
                                   double scaleFactor=1.1,
                                   int minNeighbors=3, int flags=0,
                                   Size minSize=Size(),
                                   Size maxSize=Size() );

    CV_WRAP virtual void detectMultiScale( const Mat& image,
                                   CV_OUT vector<Rect>& objects,
                                   vector<int>& rejectLevels,
                                   vector<double>& levelWeights,
                                   double scaleFactor=1.1,
                                   int minNeighbors=3, int flags=0,
                                   Size minSize=Size(),
                                   Size maxSize=Size(),
                                   bool outputRejectLevels=false );


    bool isOldFormatCascade() const;
    virtual Size getOriginalWindowSize() const;
    int getFeatureType() const;
    bool setImage( const Mat& );

protected:
    //virtual bool detectSingleScale( const Mat& image, int stripCount, Size processingRectSize,
    //                                int stripSize, int yStep, double factor, vector<Rect>& candidates );

    virtual bool detectSingleScale( const Mat& image, int stripCount, Size processingRectSize,
                                    int stripSize, int yStep, double factor, vector<Rect>& candidates,
                                    vector<int>& rejectLevels, vector<double>& levelWeights, bool outputRejectLevels=false);

protected:
    enum { BOOST = 0 };
    enum { DO_CANNY_PRUNING = 1, SCALE_IMAGE = 2,
           FIND_BIGGEST_OBJECT = 4, DO_ROUGH_SEARCH = 8 };

    friend struct CascadeClassifierInvoker;

    template<class FEval>
    friend int predictOrdered( CascadeClassifier& cascade, Ptr<FeatureEvaluator> &featureEvaluator, double& weight);

    template<class FEval>
    friend int predictCategorical( CascadeClassifier& cascade, Ptr<FeatureEvaluator> &featureEvaluator, double& weight);

    template<class FEval>
    friend int predictOrderedStump( CascadeClassifier& cascade, Ptr<FeatureEvaluator> &featureEvaluator, double& weight);

    template<class FEval>
    friend int predictCategoricalStump( CascadeClassifier& cascade, Ptr<FeatureEvaluator> &featureEvaluator, double& weight);

    bool setImage( Ptr<FeatureEvaluator>&, const Mat& );
    virtual int runAt( Ptr<FeatureEvaluator>&, Point, double& weight );

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
    
    CV_WRAP virtual void setSVMDetector(InputArray _svmdetector);
    
    virtual bool read(FileNode& fn);
    virtual void write(FileStorage& fs, const String& objname) const;
    
    CV_WRAP virtual bool load(const String& filename, const String& objname=String());
    CV_WRAP virtual void save(const String& filename, const String& objname=String()) const;
    virtual void copyTo(HOGDescriptor& c) const;

    CV_WRAP virtual void compute(const Mat& img,
                         CV_OUT vector<float>& descriptors,
                         Size winStride=Size(), Size padding=Size(),
                         const vector<Point>& locations=vector<Point>()) const;
	//with found weights output
    CV_WRAP virtual void detect(const Mat& img, CV_OUT vector<Point>& foundLocations, 
						CV_OUT vector<double>& weights,
                        double hitThreshold=0, Size winStride=Size(), 
						Size padding=Size(),
                        const vector<Point>& searchLocations=vector<Point>()) const;
	//without found weights output
    virtual void detect(const Mat& img, CV_OUT vector<Point>& foundLocations,
                        double hitThreshold=0, Size winStride=Size(),
                        Size padding=Size(),
                        const vector<Point>& searchLocations=vector<Point>()) const;
	//with result weights output
    CV_WRAP virtual void detectMultiScale(const Mat& img, CV_OUT vector<Rect>& foundLocations, 
								  CV_OUT vector<double>& foundWeights, double hitThreshold=0, 
								  Size winStride=Size(), Size padding=Size(), double scale=1.05, 
								  double finalThreshold=2.0,bool useMeanshiftGrouping = false) const;
	//without found weights output
	virtual void detectMultiScale(const Mat& img, CV_OUT vector<Rect>& foundLocations, 
								  double hitThreshold=0, Size winStride=Size(),
                                  Size padding=Size(), double scale=1.05, 
								  double finalThreshold=2.0, bool useMeanshiftGrouping = false) const;

    CV_WRAP virtual void computeGradient(const Mat& img, CV_OUT Mat& grad, CV_OUT Mat& angleOfs,
                                 Size paddingTL=Size(), Size paddingBR=Size()) const;
    
    CV_WRAP static vector<float> getDefaultPeopleDetector();
	CV_WRAP static vector<float> getDaimlerPeopleDetector();
    
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


CV_EXPORTS_W void findDataMatrix(InputArray image,
                                 CV_OUT vector<string>& codes,
                                 OutputArray corners=noArray(),
                                 OutputArrayOfArrays dmtx=noArray());
CV_EXPORTS_W void drawDataMatrixCodes(InputOutputArray image,
                                      const vector<string>& codes,
                                      InputArray corners);
}

/****************************************************************************************\
*                                Datamatrix                                              *
\****************************************************************************************/

struct CV_EXPORTS CvDataMatrixCode {
  char msg[4];
  CvMat *original;
  CvMat *corners;
};

CV_EXPORTS std::deque<CvDataMatrixCode> cvFindDataMatrix(CvMat *im);

/****************************************************************************************\
*                                 LINE-MOD                                               *
\****************************************************************************************/

namespace cv {
namespace linemod {

using cv::FileNode;
using cv::FileStorage;
using cv::Mat;
using cv::noArray;
using cv::OutputArrayOfArrays;
using cv::Point;
using cv::Ptr;
using cv::Rect;
using cv::Size;

/// @todo Convert doxy comments to rst

/**
 * \brief Discriminant feature described by its location and label.
 */
struct CV_EXPORTS Feature
{
  int x; ///< x offset
  int y; ///< y offset
  int label; ///< Quantization

  Feature() : x(0), y(0), label(0) {}
  Feature(int x, int y, int label) : x(x), y(y), label(label) {}

  void read(const FileNode& fn);
  void write(FileStorage& fs) const;
};

struct CV_EXPORTS Template
{
  int width;
  int height;
  int pyramid_level;
  std::vector<Feature> features;

  void read(const FileNode& fn);
  void write(FileStorage& fs) const;
};

/**
 * \brief Represents a modality operating over an image pyramid.
 */
class QuantizedPyramid
{
public:
  // Virtual destructor
  virtual ~QuantizedPyramid() {}

  /**
   * \brief Compute quantized image at current pyramid level for online detection.
   *
   * \param[out] dst The destination 8-bit image. For each pixel at most one bit is set,
   *                 representing its classification.
   */
  virtual void quantize(Mat& dst) const =0;

  /**
   * \brief Extract most discriminant features at current pyramid level to form a new template.
   *
   * \param[out] templ The new template.
   */
  virtual bool extractTemplate(Template& templ) const =0;

  /**
   * \brief Go to the next pyramid level.
   *
   * \todo Allow pyramid scale factor other than 2
   */
  virtual void pyrDown() =0;

protected:
  /// Candidate feature with a score
  struct Candidate
  {
    Candidate(int x, int y, int label, float score)
      : f(x, y, label), score(score)
    {
    }

    /// Sort candidates with high score to the front
    bool operator<(const Candidate& rhs) const
    {
      return score > rhs.score;
    }

    Feature f;
    float score;
  };

  /**
   * \brief Choose candidate features so that they are not bunched together.
   *
   * \param[in]  candidates   Candidate features sorted by score.
   * \param[out] features     Destination vector of selected features.
   * \param[in]  num_features Number of candidates to select.
   * \param[in]  distance     Hint for desired distance between features.
   */
  static void selectScatteredFeatures(const std::vector<Candidate>& candidates,
                                      std::vector<Feature>& features,
                                      size_t num_features, float distance);
};

/**
 * \brief Interface for modalities that plug into the LINE template matching representation.
 *
 * \todo Max response, to allow optimization of summing (255/MAX) features as uint8
 */
class CV_EXPORTS Modality
{
public:
  // Virtual destructor
  virtual ~Modality() {}

  /**
   * \brief Form a quantized image pyramid from a source image.
   *
   * \param[in] src  The source image. Type depends on the modality.
   * \param[in] mask Optional mask. If not empty, unmasked pixels are set to zero
   *                 in quantized image and cannot be extracted as features.
   */
  Ptr<QuantizedPyramid> process(const Mat& src,
				    const Mat& mask = Mat()) const
  {
    return processImpl(src, mask);
  }

  virtual std::string name() const =0;

  virtual void read(const FileNode& fn) =0;
  virtual void write(FileStorage& fs) const =0;

  /**
   * \brief Create modality by name.
   *
   * The following modality types are supported:
   * - "ColorGradient"
   * - "DepthNormal"
   */
  static Ptr<Modality> create(const std::string& modality_type);

  /**
   * \brief Load a modality from file.
   */
  static Ptr<Modality> create(const FileNode& fn);

protected:
  // Indirection is because process() has a default parameter.
  virtual Ptr<QuantizedPyramid> processImpl(const Mat& src,
						const Mat& mask) const =0;
};

/**
 * \brief Modality that computes quantized gradient orientations from a color image.
 */
class CV_EXPORTS ColorGradient : public Modality
{
public:
  /**
   * \brief Default constructor. Uses reasonable default parameter values.
   */
  ColorGradient();

  /**
   * \brief Constructor.
   *
   * \param weak_threshold   When quantizing, discard gradients with magnitude less than this.
   * \param num_features     How many features a template must contain.
   * \param strong_threshold Consider as candidate features only gradients whose norms are
   *                         larger than this.
   */
  ColorGradient(float weak_threshold, size_t num_features, float strong_threshold);

  virtual std::string name() const;

  virtual void read(const FileNode& fn);
  virtual void write(FileStorage& fs) const;

  float weak_threshold;
  size_t num_features;
  float strong_threshold;

protected:
  virtual Ptr<QuantizedPyramid> processImpl(const Mat& src,
						const Mat& mask) const;
};

/**
 * \brief Modality that computes quantized surface normals from a dense depth map.
 */
class CV_EXPORTS DepthNormal : public Modality
{
public:
  /**
   * \brief Default constructor. Uses reasonable default parameter values.
   */
  DepthNormal();

  /**
   * \brief Constructor.
   *
   * \param distance_threshold   Ignore pixels beyond this distance.
   * \param difference_threshold When computing normals, ignore contributions of pixels whose
   *                             depth difference with the central pixel is above this threshold.
   * \param num_features         How many features a template must contain.
   * \param extract_threshold    Consider as candidate feature only if there are no differing
   *                             orientations within a distance of extract_threshold.
   */
  DepthNormal(int distance_threshold, int difference_threshold, size_t num_features,
              int extract_threshold);

  virtual std::string name() const;

  virtual void read(const FileNode& fn);
  virtual void write(FileStorage& fs) const;

  int distance_threshold;
  int difference_threshold;
  size_t num_features;
  int extract_threshold;

protected:
  virtual Ptr<QuantizedPyramid> processImpl(const Mat& src,
						const Mat& mask) const;
};

/**
 * \brief Debug function to colormap a quantized image for viewing.
 */
void colormap(const Mat& quantized, Mat& dst);

/**
 * \brief Represents a successful template match.
 */
struct CV_EXPORTS Match
{
  Match()
  {
  }

  Match(int x, int y, float similarity, const std::string& class_id, int template_id)
    : x(x), y(y), similarity(similarity), class_id(class_id), template_id(template_id)
  {
  }

  /// Sort matches with high similarity to the front
  bool operator<(const Match& rhs) const
  {
    // Secondarily sort on template_id for the sake of duplicate removal
    if (similarity != rhs.similarity)
      return similarity > rhs.similarity;
    else
      return template_id < rhs.template_id;
  }

  bool operator==(const Match& rhs) const
  {
    return x == rhs.x && y == rhs.y && similarity == rhs.similarity && class_id == rhs.class_id;
  }

  int x;
  int y;
  float similarity;
  std::string class_id;
  int template_id;
};

/**
 * \brief Object detector using the LINE template matching algorithm with any set of
 * modalities.
 */
class CV_EXPORTS Detector
{
public:
  /**
   * \brief Empty constructor, initialize with read().
   */
  Detector();

  /**
   * \brief Constructor.
   *
   * \param modalities       Modalities to use (color gradients, depth normals, ...).
   * \param T_pyramid        Value of the sampling step T at each pyramid level. The
   *                         number of pyramid levels is T_pyramid.size().
   */
  Detector(const std::vector< Ptr<Modality> >& modalities, const std::vector<int>& T_pyramid);

  /**
   * \brief Detect objects by template matching.
   *
   * Matches globally at the lowest pyramid level, then refines locally stepping up the pyramid.
   *
   * \param      sources   Source images, one for each modality.
   * \param      threshold Similarity threshold, a percentage between 0 and 100.
   * \param[out] matches   Template matches, sorted by similarity score.
   * \param      class_ids If non-empty, only search for the desired object classes.
   * \param[out] quantized_images Optionally return vector<Mat> of quantized images.
   * \param      masks     The masks for consideration during matching. The masks should be CV_8UC1
   *                       where 255 represents a valid pixel.  If non-empty, the vector must be
   *                       the same size as sources.  Each element must be
   *                       empty or the same size as its corresponding source.
   */
  void match(const std::vector<Mat>& sources, float threshold, std::vector<Match>& matches,
             const std::vector<std::string>& class_ids = std::vector<std::string>(),
             OutputArrayOfArrays quantized_images = noArray(),
             const std::vector<Mat>& masks = std::vector<Mat>()) const;

  /**
   * \brief Add new object template.
   *
   * \param      sources      Source images, one for each modality.
   * \param      class_id     Object class ID.
   * \param      object_mask  Mask separating object from background.
   * \param[out] bounding_box Optionally return bounding box of the extracted features.
   *
   * \return Template ID, or -1 if failed to extract a valid template.
   */
  int addTemplate(const std::vector<Mat>& sources, const std::string& class_id,
		  const Mat& object_mask, Rect* bounding_box = NULL);

  /**
   * \brief Add a new object template computed by external means.
   */
  int addSyntheticTemplate(const std::vector<Template>& templates, const std::string& class_id);

  /**
   * \brief Get the modalities used by this detector.
   *
   * You are not permitted to add/remove modalities, but you may dynamic_cast them to
   * tweak parameters.
   */
  const std::vector< Ptr<Modality> >& getModalities() const { return modalities; }

  /**
   * \brief Get sampling step T at pyramid_level.
   */
  int getT(int pyramid_level) const { return T_at_level[pyramid_level]; }

  /**
   * \brief Get number of pyramid levels used by this detector.
   */
  int pyramidLevels() const { return pyramid_levels; }

  /**
   * \brief Get the template pyramid identified by template_id.
   *
   * For example, with 2 modalities (Gradient, Normal) and two pyramid levels
   * (L0, L1), the order is (GradientL0, NormalL0, GradientL1, NormalL1).
   */
  const std::vector<Template>& getTemplates(const std::string& class_id, int template_id) const;

  int numTemplates() const;
  int numTemplates(const std::string& class_id) const;
  int numClasses() const { return static_cast<int>(class_templates.size()); }

  std::vector<std::string> classIds() const;

  void read(const FileNode& fn);
  void write(FileStorage& fs) const;

  std::string readClass(const FileNode& fn, const std::string &class_id_override = "");
  void writeClass(const std::string& class_id, FileStorage& fs) const;

  void readClasses(const std::vector<std::string>& class_ids,
                   const std::string& format = "templates_%s.yml.gz");
  void writeClasses(const std::string& format = "templates_%s.yml.gz") const;

protected:
  std::vector< Ptr<Modality> > modalities;
  int pyramid_levels;
  std::vector<int> T_at_level;

  typedef std::vector<Template> TemplatePyramid;
  typedef std::map<std::string, std::vector<TemplatePyramid> > TemplatesMap;
  TemplatesMap class_templates;

  typedef std::vector<Mat> LinearMemories;
  // Indexed as [pyramid level][modality][quantized label]
  typedef std::vector< std::vector<LinearMemories> > LinearMemoryPyramid;

  void matchClass(const LinearMemoryPyramid& lm_pyramid,
                  const std::vector<Size>& sizes,
                  float threshold, std::vector<Match>& matches,
                  const std::string& class_id,
                  const std::vector<TemplatePyramid>& template_pyramids) const;
};

/**
 * \brief Factory function for detector using LINE algorithm with color gradients.
 *
 * Default parameter settings suitable for VGA images.
 */
CV_EXPORTS Ptr<Detector> getDefaultLINE();

/**
 * \brief Factory function for detector using LINE-MOD algorithm with color gradients
 * and depth normals.
 *
 * Default parameter settings suitable for VGA images.
 */
CV_EXPORTS Ptr<Detector> getDefaultLINEMOD();

} // namespace linemod
} // namespace cv

#endif

#endif
