#include <cxcore.h>
#include <cv.h>
#include <stdio.h>
#include "cvshadow.h"

CvArr * cvCvtSeqToArray_Shadow( const CvSeq* seq, CvArr * elements, CvSlice slice){
        CvMat stub, *mat=(CvMat *)elements;
        if(!CV_IS_MAT(mat)){
            mat = cvGetMat(elements, &stub);
        }
        cvCvtSeqToArray( seq, mat->data.ptr, slice );
        return elements;
    }

double cvArcLength_Shadow( const CvSeq * seq, CvSlice slice, int is_closed){
    return cvArcLength( seq, slice, is_closed );
}
double cvArcLength_Shadow( const CvArr * arr, CvSlice slice, int is_closed){
    return cvArcLength( arr, slice, is_closed );
}

void cvMoments_Shadow( const CvSeq * seq, CvMoments * moments, int binary ){
	cvMoments( seq, moments, binary );
}

void cvMoments_Shadow( const CvArr * seq, CvMoments * moments, int binary ){
	cvMoments( seq, moments, binary );
}


CvTypedSeq<CvRect> * cvHaarDetectObjects_Shadow( const CvArr* image, CvHaarClassifierCascade* cascade,
        CvMemStorage* storage, double scale_factor, int min_neighbors, int flags,
        CvSize min_size )
{
        return (CvTypedSeq<CvRect> *) cvHaarDetectObjects( image, cascade, storage, scale_factor,
                                                                min_neighbors, flags, min_size);
}

CvTypedSeq<CvConnectedComp> *  cvSegmentMotion_Shadow( const CvArr* mhi, CvArr* seg_mask, CvMemStorage* storage,
                                                                double timestamp, double seg_thresh ){
    return (CvTypedSeq<CvConnectedComp> *) cvSegmentMotion( mhi, seg_mask, storage, timestamp, seg_thresh );
}

CvTypedSeq<CvPoint> * cvApproxPoly_Shadow( const void* src_seq, int header_size, CvMemStorage* storage,
                                    int method, double parameter, int parameter2)
{
    return (CvTypedSeq<CvPoint> *) cvApproxPoly( src_seq, header_size, storage, method, parameter, parameter2 );
}

// Always return a new Mat of indices
CvMat * cvConvexHull2_Shadow( const CvArr * points, int orientation, int return_points){
	CvMat * hull=0;
	CvMat * points_mat=(CvMat *) points;
	CvSeq * points_seq=(CvSeq *) points;
	int npoints, type;

	if(CV_IS_MAT(points_mat)){
		npoints = MAX(points_mat->rows, points_mat->cols);
		type = return_points ? points_mat->type : CV_32S;
	}
	else if(CV_IS_SEQ(points_seq)){
		npoints = points_seq->total;
		type = return_points ? CV_SEQ_ELTYPE(points_seq) : 1;
	}
	else{
		CV_Error(CV_StsBadArg, "points must be a CvSeq or CvMat");
	}
	hull=cvCreateMat(1,npoints,type);
	cvConvexHull2(points, hull, orientation, return_points);

	return hull;
}
std::vector<CvPoint> cvSnakeImage_Shadow( const CvMat * image, std::vector<CvPoint>  points,
		std::vector<float> alpha, std::vector<float> beta,
		std::vector<float> gamma,
		CvSize win, CvTermCriteria criteria, int calc_gradient ){
	IplImage ipl_stub;
	cvSnakeImage( cvGetImage(image, &ipl_stub), &(points[0]), points.size(),
			      &((alpha)[0]), &((beta)[0]), &((gamma)[0]),
				  (alpha.size()>1 && beta.size()>1 && gamma.size()>1 ? CV_ARRAY : CV_VALUE),
				  win, criteria, calc_gradient );
	return points;
}
