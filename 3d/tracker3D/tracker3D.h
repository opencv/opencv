#ifndef CREATESAMPLES3D_H
#define CREATESAMPLES3D_H

// return NULL if no there was no chessboard founded

// input: the same points on two images
// output: 3D coordinates of selected points
// boxDepthCoeff is box depth (relative to max of width or height)
// maxRelativeError - max reprojection error of upper-left corner of the schessboard (in chessboard square sizes)
CvPoint3D32f* calc3DPoints(const IplImage* _img1, const IplImage* _img2,CvPoint* points1,  CvPoint* points2, CvSize innerCornersCount,
							const CvMat* intrinsic_matrix, const CvMat* _distortion_coeffs, bool undistortImage,
							float boxDepthCoeff = 1.0f, float maxRelativeError = 1.0);

CvPoint* Find3DObject(const IplImage* _img, const CvPoint3D32f* points, CvSize innerCornersCount,
					   const CvMat* intrinsic_matrix, const CvMat* _distortion_coeffs, bool undistortImage = true);

// Gets rectangular sample from 3D object
IplImage* GetSample3D(const IplImage* img, CvPoint* points);
#endif