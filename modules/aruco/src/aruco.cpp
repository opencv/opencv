// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "opencv2/aruco.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/utils/logger.hpp>

namespace cv {
namespace aruco {

using namespace std;

void detectMarkers(InputArray _image, const Ptr<Dictionary> &_dictionary, OutputArrayOfArrays _corners,
                   OutputArray _ids, const Ptr<DetectorParameters> &_params,
                   OutputArrayOfArrays _rejectedImgPoints) {
    ArucoDetector detector(*_dictionary, *_params);
    detector.detectMarkers(_image, _corners, _ids, _rejectedImgPoints);
}

void refineDetectedMarkers(InputArray _image, const Ptr<Board> &_board,
                           InputOutputArrayOfArrays _detectedCorners, InputOutputArray _detectedIds,
                           InputOutputArrayOfArrays _rejectedCorners, InputArray _cameraMatrix,
                           InputArray _distCoeffs, float minRepDistance, float errorCorrectionRate,
                           bool checkAllOrders, OutputArray _recoveredIdxs,
                           const Ptr<DetectorParameters> &_params) {
    RefineParameters refineParams(minRepDistance, errorCorrectionRate, checkAllOrders);
    ArucoDetector detector(_board->getDictionary(), *_params, refineParams);
    detector.refineDetectedMarkers(_image, *_board, _detectedCorners, _detectedIds, _rejectedCorners, _cameraMatrix,
                                   _distCoeffs, _recoveredIdxs);
}

void drawPlanarBoard(const Ptr<Board> &board, Size outSize, const _OutputArray &img, int marginSize, int borderBits) {
    board->generateImage(outSize, img, marginSize, borderBits);
}

void getBoardObjectAndImagePoints(const Ptr<Board> &board, InputArrayOfArrays detectedCorners, InputArray detectedIds,
                                  OutputArray objPoints, OutputArray imgPoints) {
    board->matchImagePoints(detectedCorners, detectedIds, objPoints, imgPoints);
}

int estimatePoseBoard(InputArrayOfArrays corners, InputArray ids, const Ptr<Board> &board,
                      InputArray cameraMatrix, InputArray distCoeffs, InputOutputArray rvec,
                      InputOutputArray tvec, bool useExtrinsicGuess) {
    CV_Assert(corners.total() == ids.total());

    // get object and image points for the solvePnP function
    Mat objPoints, imgPoints;
    board->matchImagePoints(corners, ids, objPoints, imgPoints);

    CV_Assert(imgPoints.total() == objPoints.total());

    if(objPoints.total() == 0) // 0 of the detected markers in board
        return 0;

    solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess);

    // divide by four since all the four corners are concatenated in the array for each marker
    return (int)objPoints.total() / 4;
}

bool estimatePoseCharucoBoard(InputArray charucoCorners, InputArray charucoIds,
                              const Ptr<CharucoBoard> &board, InputArray cameraMatrix,
                              InputArray distCoeffs, InputOutputArray rvec,
                              InputOutputArray tvec, bool useExtrinsicGuess) {
    CV_Assert((charucoCorners.getMat().total() == charucoIds.getMat().total()));
    if(charucoIds.getMat().total() < 4) return false;

    // get object and image points for the solvePnP function
    Mat objPoints, imgPoints;
    board->matchImagePoints(charucoCorners, charucoIds, objPoints, imgPoints);
    try {
        solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess);
    }
    catch (const cv::Exception& e) {
        CV_LOG_WARNING(NULL, "estimatePoseCharucoBoard: " << std::endl << e.what());
        return false;
    }

    return objPoints.total() > 0ull;
}

bool testCharucoCornersCollinear(const Ptr<CharucoBoard> &board, InputArray charucoIds) {
    return board->checkCharucoCornersCollinear(charucoIds);
}

/**
  * @brief Return object points for the system centered in a middle (by default) or in a top left corner of single
  * marker, given the marker length
  */
static Mat _getSingleMarkerObjectPoints(float markerLength, const EstimateParameters& estimateParameters) {
    CV_Assert(markerLength > 0);
    Mat objPoints(4, 1, CV_32FC3);
    // set coordinate system in the top-left corner of the marker, with Z pointing out
    if (estimateParameters.pattern == ARUCO_CW_TOP_LEFT_CORNER) {
        objPoints.ptr<Vec3f>(0)[0] = Vec3f(0.f, 0.f, 0);
        objPoints.ptr<Vec3f>(0)[1] = Vec3f(markerLength, 0.f, 0);
        objPoints.ptr<Vec3f>(0)[2] = Vec3f(markerLength, markerLength, 0);
        objPoints.ptr<Vec3f>(0)[3] = Vec3f(0.f, markerLength, 0);
    }
    else if (estimateParameters.pattern == ARUCO_CCW_CENTER) {
        objPoints.ptr<Vec3f>(0)[0] = Vec3f(-markerLength/2.f, markerLength/2.f, 0);
        objPoints.ptr<Vec3f>(0)[1] = Vec3f(markerLength/2.f, markerLength/2.f, 0);
        objPoints.ptr<Vec3f>(0)[2] = Vec3f(markerLength/2.f, -markerLength/2.f, 0);
        objPoints.ptr<Vec3f>(0)[3] = Vec3f(-markerLength/2.f, -markerLength/2.f, 0);
    }
    else
        CV_Error(Error::StsBadArg, "Unknown estimateParameters pattern");
    return objPoints;
}

void estimatePoseSingleMarkers(InputArrayOfArrays _corners, float markerLength,
                               InputArray _cameraMatrix, InputArray _distCoeffs,
                               OutputArray _rvecs, OutputArray _tvecs, OutputArray _objPoints,
                               const Ptr<EstimateParameters>& estimateParameters) {
    CV_Assert(markerLength > 0);

    Mat markerObjPoints = _getSingleMarkerObjectPoints(markerLength, *estimateParameters);
    int nMarkers = (int)_corners.total();
    _rvecs.create(nMarkers, 1, CV_64FC3);
    _tvecs.create(nMarkers, 1, CV_64FC3);

    Mat rvecs = _rvecs.getMat(), tvecs = _tvecs.getMat();

    //// for each marker, calculate its pose
    parallel_for_(Range(0, nMarkers), [&](const Range& range) {
        const int begin = range.start;
        const int end = range.end;

        for (int i = begin; i < end; i++) {
            solvePnP(markerObjPoints, _corners.getMat(i), _cameraMatrix, _distCoeffs, rvecs.at<Vec3d>(i),
                     tvecs.at<Vec3d>(i), estimateParameters->useExtrinsicGuess, estimateParameters->solvePnPMethod);
        }
    });

    if(_objPoints.needed()){
        markerObjPoints.convertTo(_objPoints, -1);
    }
}

}
}
