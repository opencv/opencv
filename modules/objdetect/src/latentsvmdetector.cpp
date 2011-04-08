#include "precomp.hpp"
#include "_lsvmparser.h"
#include "_lsvm_matching.h" 

/*
// load trained detector from a file
//
// API
// CvLatentSvmDetector* cvLoadLatentSvmDetector(const char* filename);
// INPUT
// filename				- path to the file containing the parameters of
//						- trained Latent SVM detector
// OUTPUT
// trained Latent SVM detector in internal representation
*/
CvLatentSvmDetector* cvLoadLatentSvmDetector(const char* filename)
{
	CvLatentSvmDetector* detector = 0;
	CvLSVMFilterObject** filters = 0;
	int kFilters = 0;
	int kComponents = 0;
	int* kPartFilters = 0;
	float* b = 0;
	float scoreThreshold = 0.f;
	int err_code = 0;

	err_code = loadModel(filename, &filters, &kFilters, &kComponents, &kPartFilters, &b, &scoreThreshold);
	if (err_code != LATENT_SVM_OK) return 0;

	detector = (CvLatentSvmDetector*)malloc(sizeof(CvLatentSvmDetector));
	detector->filters = filters;
	detector->b = b;
	detector->num_components = kComponents;
	detector->num_filters = kFilters;
	detector->num_part_filters = kPartFilters;
	detector->score_threshold = scoreThreshold;

	return detector;
}

/*
// release memory allocated for CvLatentSvmDetector structure
//
// API
// void cvReleaseLatentSvmDetector(CvLatentSvmDetector** detector);
// INPUT
// detector				- CvLatentSvmDetector structure to be released
// OUTPUT
*/
void cvReleaseLatentSvmDetector(CvLatentSvmDetector** detector)
{
	free((*detector)->b);
	free((*detector)->num_part_filters);
	for (int i = 0; i < (*detector)->num_filters; i++)
	{
		free((*detector)->filters[i]->H);
		free((*detector)->filters[i]);
	}
	free((*detector)->filters);
	free((*detector));
	*detector = 0;
}

/*
// find rectangular regions in the given image that are likely 
// to contain objects and corresponding confidence levels
//
// API
// CvSeq* cvLatentSvmDetectObjects(const IplImage* image, 
//									CvLatentSvmDetector* detector, 
//									CvMemStorage* storage, 
//									float overlap_threshold = 0.5f,
                                    int numThreads = -1);
// INPUT
// image				- image to detect objects in
// detector				- Latent SVM detector in internal representation
// storage				- memory storage to store the resultant sequence 
//							of the object candidate rectangles
// overlap_threshold	- threshold for the non-maximum suppression algorithm [here will be the reference to original paper]
// OUTPUT
// sequence of detected objects (bounding boxes and confidence levels stored in CvObjectDetection structures)
*/
CvSeq* cvLatentSvmDetectObjects(IplImage* image, 
								CvLatentSvmDetector* detector, 
								CvMemStorage* storage, 
								float overlap_threshold, int numThreads)
{
	CvLSVMFeaturePyramid *H = 0;
    CvPoint *points = 0, *oppPoints = 0;
    int kPoints = 0;
    float *score = 0;    
    unsigned int maxXBorder = 0, maxYBorder = 0;
	int numBoxesOut = 0;
	CvPoint *pointsOut = 0;
	CvPoint *oppPointsOut = 0; 
    float *scoreOut = 0;
	CvSeq* result_seq = 0;
    int error = 0;

    cvConvertImage(image, image, CV_CVTIMG_SWAP_RB);
    // Getting maximum filter dimensions
	getMaxFilterDims((const CvLSVMFilterObject**)(detector->filters), detector->num_components, 
                     detector->num_part_filters, &maxXBorder, &maxYBorder);
    // Create feature pyramid with nullable border
    H = createFeaturePyramidWithBorder(image, maxXBorder, maxYBorder);
    // Search object
    error = searchObjectThresholdSomeComponents(H, (const CvLSVMFilterObject**)(detector->filters), 
        detector->num_components, detector->num_part_filters, detector->b, detector->score_threshold, 
        &points, &oppPoints, &score, &kPoints, numThreads);
    if (error != LATENT_SVM_OK)
    {
        return NULL;
    }
    // Clipping boxes
    clippingBoxes(image->width, image->height, points, kPoints);
    clippingBoxes(image->width, image->height, oppPoints, kPoints);
    // NMS procedure
    nonMaximumSuppression(kPoints, points, oppPoints, score, overlap_threshold,
                &numBoxesOut, &pointsOut, &oppPointsOut, &scoreOut);

	result_seq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvObjectDetection), storage );

	for (int i = 0; i < numBoxesOut; i++)
	{
		CvObjectDetection detection = {{0, 0, 0, 0}, 0};
		detection.score = scoreOut[i];
		CvRect bounding_box = {0, 0, 0, 0};
		bounding_box.x = pointsOut[i].x;
		bounding_box.y = pointsOut[i].y;
		bounding_box.width = oppPointsOut[i].x - pointsOut[i].x;
		bounding_box.height = oppPointsOut[i].y - pointsOut[i].y;
		detection.rect = bounding_box;
		cvSeqPush(result_seq, &detection);
	}
    cvConvertImage(image, image, CV_CVTIMG_SWAP_RB);

    freeFeaturePyramidObject(&H);
    free(points);
    free(oppPoints);
    free(score);

	return result_seq;
}
