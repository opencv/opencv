// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <opencv2/aruco/aruco_calib.hpp>
#include <opencv2/calib3d.hpp>

namespace cv {
namespace aruco {
using namespace std;

EstimateParameters::EstimateParameters() : pattern(ARUCO_CCW_CENTER), useExtrinsicGuess(false),
                                           solvePnPMethod(SOLVEPNP_ITERATIVE) {}

double calibrateCameraAruco(InputArrayOfArrays _corners, InputArray _ids, InputArray _counter,
                            const Ptr<Board> &board, Size imageSize, InputOutputArray _cameraMatrix,
                            InputOutputArray _distCoeffs, OutputArrayOfArrays _rvecs,
                            OutputArrayOfArrays _tvecs,
                            OutputArray _stdDeviationsIntrinsics,
                            OutputArray _stdDeviationsExtrinsics,
                            OutputArray _perViewErrors,
                            int flags, const TermCriteria& criteria) {
    // for each frame, get properly processed imagePoints and objectPoints for the calibrateCamera
    // function
    vector<Mat> processedObjectPoints, processedImagePoints;
    size_t nFrames = _counter.total();
    int markerCounter = 0;
    for(size_t frame = 0; frame < nFrames; frame++) {
        int nMarkersInThisFrame =  _counter.getMat().ptr< int >()[frame];
        vector<Mat> thisFrameCorners;
        vector<int> thisFrameIds;

        CV_Assert(nMarkersInThisFrame > 0);

        thisFrameCorners.reserve((size_t) nMarkersInThisFrame);
        thisFrameIds.reserve((size_t) nMarkersInThisFrame);
        for(int j = markerCounter; j < markerCounter + nMarkersInThisFrame; j++) {
            thisFrameCorners.push_back(_corners.getMat(j));
            thisFrameIds.push_back(_ids.getMat().ptr< int >()[j]);
        }
        markerCounter += nMarkersInThisFrame;
        Mat currentImgPoints, currentObjPoints;
        board->matchImagePoints(thisFrameCorners, thisFrameIds, currentObjPoints,
            currentImgPoints);
        if(currentImgPoints.total() > 0 && currentObjPoints.total() > 0) {
            processedImagePoints.push_back(currentImgPoints);
            processedObjectPoints.push_back(currentObjPoints);
        }
    }
    return calibrateCamera(processedObjectPoints, processedImagePoints, imageSize, _cameraMatrix, _distCoeffs, _rvecs,
                           _tvecs, _stdDeviationsIntrinsics, _stdDeviationsExtrinsics, _perViewErrors, flags, criteria);
}

double calibrateCameraAruco(InputArrayOfArrays _corners, InputArray _ids, InputArray _counter, const Ptr<Board> &board,
                            Size imageSize, InputOutputArray _cameraMatrix, InputOutputArray _distCoeffs,
                            OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs, int flags, const TermCriteria& criteria) {
    return calibrateCameraAruco(_corners, _ids, _counter, board, imageSize, _cameraMatrix, _distCoeffs,
                                _rvecs, _tvecs, noArray(), noArray(), noArray(), flags, criteria);
}

double calibrateCameraCharuco(InputArrayOfArrays _charucoCorners, InputArrayOfArrays _charucoIds,
                              const Ptr<CharucoBoard> &_board, Size imageSize,
                              InputOutputArray _cameraMatrix, InputOutputArray _distCoeffs,
                              OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs,
                              OutputArray _stdDeviationsIntrinsics,
                              OutputArray _stdDeviationsExtrinsics,
                              OutputArray _perViewErrors,
                              int flags, const TermCriteria& criteria) {
    CV_Assert(_charucoIds.total() > 0 && (_charucoIds.total() == _charucoCorners.total()));

    // Join object points of charuco corners in a single vector for calibrateCamera() function
    vector<vector<Point3f> > allObjPoints;
    allObjPoints.resize(_charucoIds.total());
    for(unsigned int i = 0; i < _charucoIds.total(); i++) {
        unsigned int nCorners = (unsigned int)_charucoIds.getMat(i).total();
        CV_Assert(nCorners > 0 && nCorners == _charucoCorners.getMat(i).total());
        allObjPoints[i].reserve(nCorners);

        for(unsigned int j = 0; j < nCorners; j++) {
            int pointId = _charucoIds.getMat(i).at< int >(j);
            CV_Assert(pointId >= 0 && pointId < (int)_board->getChessboardCorners().size());
            allObjPoints[i].push_back(_board->getChessboardCorners()[pointId]);
        }
    }
    return calibrateCamera(allObjPoints, _charucoCorners, imageSize, _cameraMatrix, _distCoeffs, _rvecs, _tvecs,
                           _stdDeviationsIntrinsics, _stdDeviationsExtrinsics, _perViewErrors, flags, criteria);
}

double calibrateCameraCharuco(InputArrayOfArrays _charucoCorners, InputArrayOfArrays _charucoIds,
                              const Ptr<CharucoBoard> &_board, Size imageSize, InputOutputArray _cameraMatrix,
                              InputOutputArray _distCoeffs, OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs,
                              int flags, const TermCriteria& criteria) {
return calibrateCameraCharuco(_charucoCorners, _charucoIds, _board, imageSize, _cameraMatrix, _distCoeffs, _rvecs,
                              _tvecs, noArray(), noArray(), noArray(), flags, criteria);
}

}
}
