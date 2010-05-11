#ifndef CV_SHADOW_H
#define CV_SHADOW_H

#include <vector>
#include "cxtypes.h"
#include "cvtypes.h"
#include "pycvseq.hpp"

// modify cvCvtSeqToArray to take CvArr as input instead of raw data
CvArr * cvCvtSeqToArray_Shadow( const CvSeq* seq, CvArr * elements, CvSlice slice=CV_WHOLE_SEQ);

// Return a typed sequence instead of generic CvSeq
CvTypedSeq<CvRect> * cvHaarDetectObjects_Shadow( const CvArr* image, CvHaarClassifierCascade* cascade,
        CvMemStorage* storage, double scale_factor=1.1, int min_neighbors=3, int flags=0,
        CvSize min_size=cvSize(0,0) );
CvTypedSeq<CvConnectedComp> *  cvSegmentMotion_Shadow( const CvArr* mhi, CvArr* seg_mask, CvMemStorage* storage,
                                                                double timestamp, double seg_thresh );
CvTypedSeq<CvPoint> * cvApproxPoly_Shadow( const void* src_seq, int header_size, CvMemStorage* storage,
                                    int method, double parameter, int parameter2=0);

// Always return a new Mat of indices
CvMat * cvConvexHull2_Shadow( const CvArr * points, int orientation=CV_CLOCKWISE,
		                  int return_points=0);

std::vector<CvPoint> cvSnakeImage_Shadow( const CvMat * image, std::vector<CvPoint>  points,
		std::vector<float> alpha, std::vector<float> beta, std::vector<float> gamma,
		CvSize win, CvTermCriteria criteria, int calc_gradient=1 );

#endif
