#ifndef CREATESAMPLES_H
#define CREATESAMPLES_H
//constants

#define FRAME_WINDOW "Frame Window" // Main window name
#define FRAMES_TRACKBAR "Frames" 
#define DEPTH_TRACKBAR "Box depth" // Trackbar with box depth (in chessboard squares)
#define INITIAL_DEPTH_TRACKBAR "Init depth" // Trackbar initial box position (in chessboard squares)
#define ERROR_TRACKBAR "Max Error"
#define MAX_ERROR_VALUE 50
#define MAX_DEPTH_VALUE 50
#define MAX_INITAL_DEPTH_VALUE 100
#define MIN_INITAL_DEPTH_VALUE -100
#define IDENTITY_ERROR_VALUE 10
#define IDENTITY_DEPTH_VALUE 1
#define IDENTITY_INITIAL_DEPTH_VALUE 1

// Image deinterlacing
IplImage* Deinterlace(IplImage* src);

void InverseUndistortMap(const CvMat* mapx,const CvMat* mapy, CvMat** invmapx, CvMat** invmapy, bool interpolate=0);

//Load camera params from yaml file
int LoadCameraParams(char* filename, CvMat** intrinsic_matrix, CvMat** distortion_coeffs);

IplImage* Undistort(IplImage* src,const CvMat* intrinsic_matrix,const CvMat* distortion_coeffs);

int ShowFrame(int pos);

// if chessboardPoints = NULL using simle affine transform otherwise we must have correct oldPoints pointer
// chessBoardPoints is array 2xN
//  returns new points location
CvPoint* GetCurrentPointsPosition(IplImage* workImage, CvPoint2D32f* relCoords, CvMat* chessboardPoints = NULL, CvPoint* oldPoints=NULL, CvPoint2D32f* outCorners=0);

IplImage* GetSample(const IplImage* src,CvSize innerCornersCount, const CvPoint* points, CvPoint2D32f* chessboardCorners=0);

void createSamples2DObject(int argc, char** argv);

void createSamples3DObject(int argc, char** argv); // 3D object from two frames with automatic depth calculations (not robust)

void createSamples3DObject2(int argc, char** argv); // 3D object from one frame with manual depth settings (robust)

#endif