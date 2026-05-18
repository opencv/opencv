package org.opencv.objdetect;

import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Size;
import org.opencv.core.Scalar;
import org.opencv.objdetect.Objdetect;

/**
 * ArUco2: faster and simpler markers, boards and fractals detection.
 *
 * This class provides a more convenient Java API for the ArUco2 functionality.
 */
public class Aruco2 {

    // Enums (delegated to Objdetect)
    
    public static final int
        DICT_4X4_50 = Objdetect.Aruco2_DICT_4X4_50,
        DICT_4X4_100 = Objdetect.Aruco2_DICT_4X4_100,
        DICT_4X4_250 = Objdetect.Aruco2_DICT_4X4_250,
        DICT_4X4_1000 = Objdetect.Aruco2_DICT_4X4_1000,
        DICT_5X5_50 = Objdetect.Aruco2_DICT_5X5_50,
        DICT_5X5_100 = Objdetect.Aruco2_DICT_5X5_100,
        DICT_5X5_250 = Objdetect.Aruco2_DICT_5X5_250,
        DICT_5X5_1000 = Objdetect.Aruco2_DICT_5X5_1000,
        DICT_6X6_50 = Objdetect.Aruco2_DICT_6X6_50,
        DICT_6X6_100 = Objdetect.Aruco2_DICT_6X6_100,
        DICT_6X6_250 = Objdetect.Aruco2_DICT_6X6_250,
        DICT_6X6_1000 = Objdetect.Aruco2_DICT_6X6_1000,
        DICT_7X7_50 = Objdetect.Aruco2_DICT_7X7_50,
        DICT_7X7_100 = Objdetect.Aruco2_DICT_7X7_100,
        DICT_7X7_250 = Objdetect.Aruco2_DICT_7X7_250,
        DICT_7X7_1000 = Objdetect.Aruco2_DICT_7X7_1000,
        DICT_ARUCO_ORIGINAL = Objdetect.Aruco2_DICT_ARUCO_ORIGINAL,
        DICT_APRILTAG_16h5 = Objdetect.Aruco2_DICT_APRILTAG_16h5,
        DICT_APRILTAG_25h9 = Objdetect.Aruco2_DICT_APRILTAG_25h9,
        DICT_APRILTAG_36h10 = Objdetect.Aruco2_DICT_APRILTAG_36h10,
        DICT_APRILTAG_36h11 = Objdetect.Aruco2_DICT_APRILTAG_36h11,
        DICT_ARUCO_MIP_36h12 = Objdetect.Aruco2_DICT_ARUCO_MIP_36h12;

    public static final int
        FRACTAL_2L_6 = Objdetect.Aruco2_FRACTAL_2L_6,
        FRACTAL_3L_6 = Objdetect.Aruco2_FRACTAL_3L_6,
        FRACTAL_4L_6 = Objdetect.Aruco2_FRACTAL_4L_6,
        FRACTAL_5L_6 = Objdetect.Aruco2_FRACTAL_5L_6;

    // Functions

    public static void getFiducialMarker(Mat img, int dictionary, int id, int bitSize, boolean externalBorder) {
        Objdetect.Aruco2_getFiducialMarker(img, dictionary, id, bitSize, externalBorder);
    }

    public static void getFiducialMarker(Mat img, int dictionary, int id) {
        Objdetect.Aruco2_getFiducialMarker(img, dictionary, id);
    }

    @SuppressWarnings("unchecked")
    public static List<FiducialMarker> detectFiducialMarkers(Mat image, int dict, DetectionParameters detectorParams) {
        return (List<FiducialMarker>)(List<?>)Objdetect.Aruco2_detectFiducialMarkers_Params(image, dict, detectorParams.nativeObj);
    }

    @SuppressWarnings("unchecked")
    public static List<FiducialMarker> detectFiducialMarkers(Mat image, int dict) {
        DetectionParameters params = new DetectionParameters();
        return (List<FiducialMarker>)(List<?>)Objdetect.Aruco2_detectFiducialMarkers_Params(image, dict, params.nativeObj);
    }

    @SuppressWarnings("unchecked")
    public static List<FiducialMarker> detectFiducialMarkers(Mat image) {
        DetectionParameters params = new DetectionParameters();
        return (List<FiducialMarker>)(List<?>)Objdetect.Aruco2_detectFiducialMarkers_Params(image, DICT_ARUCO_MIP_36h12, params.nativeObj);
    }

    @SuppressWarnings("unchecked")
    public static List<FiducialMarker> detectFiducialMarkers(Mat image, MatOfInt dicts, DetectionParameters detectorParams) {
        return (List<FiducialMarker>)(List<?>)Objdetect.Aruco2_detectFiducialMarkers_Dicts(image, dicts, detectorParams.nativeObj);
    }

    @SuppressWarnings("unchecked")
    public static void drawFiducialMarkers(Mat image, List<FiducialMarker> markers, Scalar borderColor) {
        Objdetect.Aruco2_drawFiducialMarkers(image, (List<Aruco2_FiducialMarker>)(List<?>)markers, borderColor);
    }

    @SuppressWarnings("unchecked")
    public static void drawFiducialMarkers(Mat image, List<FiducialMarker> markers) {
        Objdetect.Aruco2_drawFiducialMarkers(image, (List<Aruco2_FiducialMarker>)(List<?>)markers);
    }

    public static void drawAxis(Mat image, Mat cameraMatrix, Mat distCoeffs, Mat rvec, Mat tvec, float length) {
        Objdetect.Aruco2_drawAxis(image, cameraMatrix, distCoeffs, rvec, tvec, length);
    }

    public static void getSolvePnpPoints(FiducialMarker marker, Mat objPoints, Mat imgPoints, float markerSize) {
        Objdetect.Aruco2_getSolvePnpPoints_FiducialMarker(marker.nativeObj, objPoints, imgPoints, markerSize);
    }

    public static void getSolvePnpPoints(FiducialMarker marker, Mat objPoints, Mat imgPoints) {
        Objdetect.Aruco2_getSolvePnpPoints_FiducialMarker(marker.nativeObj, objPoints, imgPoints, 1.0f);
    }

    public static void getGridBoard(Mat img, Size boardSize, int dict, int bitSize, Mat ids) {
        Objdetect.Aruco2_getGridBoard(img, boardSize, dict, bitSize, ids);
    }

    public static void getGridBoard(Mat img, Size boardSize, int dict, int bitSize) {
        Objdetect.Aruco2_getGridBoard(img, boardSize, dict, bitSize);
    }

    public static void getGridBoard(Mat img, Size boardSize, int dict) {
        Objdetect.Aruco2_getGridBoard(img, boardSize, dict);
    }

    public static boolean detectGridBoard(Mat image, Size gridSize, int dict, Aruco2_GridBoard board, Mat ids) {
        return Objdetect.Aruco2_detectGridBoard_Ids(image, gridSize, dict, board.nativeObj, ids);
    }

    public static boolean detectGridBoard(Mat image, Size gridSize, int dict, Aruco2_GridBoard board) {
        return Objdetect.Aruco2_detectGridBoard_NoIds(image, gridSize, dict, board.nativeObj);
    }

    public static void drawGridBoard(Mat image, Aruco2_GridBoard board, Scalar color, boolean drawMarkerIds) {
        Objdetect.Aruco2_drawGridBoard(image, board.nativeObj, color, drawMarkerIds);
    }

    public static void drawGridBoard(Mat image, Aruco2_GridBoard board) {
        Objdetect.Aruco2_drawGridBoard(image, board.nativeObj);
    }

    public static void getSolvePnpPoints(Aruco2_GridBoard board, Mat objPoints, Mat imgPoints, float markerSize) {
        Objdetect.Aruco2_getSolvePnpPoints_GridBoard(board.nativeObj, objPoints, imgPoints, markerSize);
    }

    public static void getSolvePnpPoints(Aruco2_GridBoard board, Mat objPoints, Mat imgPoints) {
        Objdetect.Aruco2_getSolvePnpPoints_GridBoard(board.nativeObj, objPoints, imgPoints, 1.0f);
    }

    @SuppressWarnings("unchecked")
    public static List<Diamond> detectDiamonds(Mat image, int dict) {
        return (List<Diamond>)(List<?>)Objdetect.Aruco2_detectDiamonds(image, dict);
    }

    public static void getDiamondImage(Mat img, int dictionary, int[] ids, int bitSize) {
        Objdetect.Aruco2_getDiamondImage(img, dictionary, ids, bitSize);
    }

    @SuppressWarnings("unchecked")
    public static void drawDiamonds(Mat image, List<Diamond> diamonds, Scalar color, boolean drawMarkerIds) {
        Objdetect.Aruco2_drawDiamonds(image, (List<Aruco2_Diamond>)(List<?>)diamonds, color, drawMarkerIds);
    }

    public static void getSolvePnpPoints(Diamond diamond, Mat objPoints, Mat imgPoints, float markerSize) {
        Objdetect.Aruco2_getSolvePnpPoints_Diamond(diamond.nativeObj, objPoints, imgPoints, markerSize);
    }

    public static void getFractalImage(Mat img, int ftype, int bitSize) {
        Objdetect.Aruco2_getFractalImage(img, ftype, bitSize);
    }

    @SuppressWarnings("unchecked")
    public static List<FractalMarker> detectFractals(Mat image, int ftype) {
        return (List<FractalMarker>)(List<?>)Objdetect.Aruco2_detectFractals(image, ftype);
    }

    @SuppressWarnings("unchecked")
    public static void drawFractals(Mat image, List<FractalMarker> fractals, Scalar color, boolean drawAllImagePoints) {
        Objdetect.Aruco2_drawFractals(image, (List<Aruco2_FractalMarker>)(List<?>)fractals, color, drawAllImagePoints);
    }

    public static void getSolvePnpPoints(FractalMarker fractal, Mat objPoints, Mat imgPoints, float markerSize) {
        Objdetect.Aruco2_getSolvePnpPoints_Fractal(fractal.nativeObj, objPoints, imgPoints, markerSize);
    }
}
