#ifndef CV_CALIBRATION_FORK_HPP
#define CV_CALIBRATION_FORK_HPP

#include <opencv2/core.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/calib3d/calib3d_c.h>

namespace cvfork
{
using namespace cv;

#define CV_CALIB_NINTRINSIC 18
#define CALIB_USE_QR (1 << 18)

double calibrateCamera(InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints, Size imageSize,
                                     InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
                                     OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs, OutputArray stdDeviations,
                                     OutputArray perViewErrors, int flags = 0, TermCriteria criteria = TermCriteria(
                                        TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON) );

double cvCalibrateCamera2( const CvMat* object_points,
                                const CvMat* image_points,
                                const CvMat* point_counts,
                                CvSize image_size,
                                CvMat* camera_matrix,
                                CvMat* distortion_coeffs,
                                CvMat* rotation_vectors CV_DEFAULT(NULL),
                                CvMat* translation_vectors CV_DEFAULT(NULL),
                                CvMat* stdDeviations_vector CV_DEFAULT(NULL),
                                CvMat* perViewErrors_vector CV_DEFAULT(NULL),
                                int flags CV_DEFAULT(0),
                                CvTermCriteria term_crit CV_DEFAULT(cvTermCriteria(
                                    CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,30,DBL_EPSILON)) );

double calibrateCameraCharuco(InputArrayOfArrays _charucoCorners, InputArrayOfArrays _charucoIds,
                              Ptr<aruco::CharucoBoard> &_board, Size imageSize,
                              InputOutputArray _cameraMatrix, InputOutputArray _distCoeffs,
                              OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs, OutputArray _stdDeviations, OutputArray _perViewErrors,
                              int flags = 0, TermCriteria criteria = TermCriteria(
                                    TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON) );

class CvLevMarqFork : public CvLevMarq
{
public:
    CvLevMarqFork( int nparams, int nerrs, CvTermCriteria criteria=
              cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,30,DBL_EPSILON),
              bool completeSymmFlag=false );
    bool updateAlt( const CvMat*& _param, CvMat*& _JtJ, CvMat*& _JtErr, double*& _errNorm );
    void step();
    ~CvLevMarqFork();
};
}

#endif
