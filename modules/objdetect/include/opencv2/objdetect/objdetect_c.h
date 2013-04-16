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

#ifndef __OPENCV_OBJDETECT_C_H__
#define __OPENCV_OBJDETECT_C_H__

#include "opencv2/core/core_c.h"

#ifdef __cplusplus
#include <deque>
#include <vector>

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
typedef struct CvLSVMFilterPosition
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
typedef struct CvLSVMFilterObject{
    CvLSVMFilterPosition V;
    float fineFunction[4];
    int sizeX;
    int sizeY;
    int numFeatures;
    float *H;
} CvLSVMFilterObject;

// data type: STRUCT CvLatentSvmDetector
// structure contains internal representation of trained Latent SVM detector
// num_filters          - total number of filters (root plus part) in model
// num_components       - number of components in model
// num_part_filters     - array containing number of part filters for each component
// filters              - root and part filters for all model components
// b                    - biases for all model components
// score_threshold      - confidence level threshold
typedef struct CvLatentSvmDetector
{
    int num_filters;
    int num_components;
    int* num_part_filters;
    CvLSVMFilterObject** filters;
    float* b;
    float score_threshold;
} CvLatentSvmDetector;

// data type: STRUCT CvObjectDetection
// structure contains the bounding box and confidence level for detected object
// rect                 - bounding box for a detected object
// score                - confidence level
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
// filename             - path to the file containing the parameters of
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
// detector             - CvLatentSvmDetector structure to be released
// OUTPUT
*/
CVAPI(void) cvReleaseLatentSvmDetector(CvLatentSvmDetector** detector);

/*
// find rectangular regions in the given image that are likely
// to contain objects and corresponding confidence levels
//
// API
// CvSeq* cvLatentSvmDetectObjects(const IplImage* image,
//                                  CvLatentSvmDetector* detector,
//                                  CvMemStorage* storage,
//                                  float overlap_threshold = 0.5f,
//                                  int numThreads = -1);
// INPUT
// image                - image to detect objects in
// detector             - Latent SVM detector in internal representation
// storage              - memory storage to store the resultant sequence
//                          of the object candidate rectangles
// overlap_threshold    - threshold for the non-maximum suppression algorithm
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
                     double scale_factor = 1.1,
                     int min_neighbors = 3, int flags = 0,
                     CvSize min_size = cvSize(0, 0), CvSize max_size = cvSize(0, 0),
                     bool outputRejectLevels = false );

struct CvDataMatrixCode
{
  char msg[4];
  CvMat* original;
  CvMat* corners;
};

CV_EXPORTS std::deque<CvDataMatrixCode> cvFindDataMatrix(CvMat *im);

#endif


#endif /* __OPENCV_OBJDETECT_C_H__ */
