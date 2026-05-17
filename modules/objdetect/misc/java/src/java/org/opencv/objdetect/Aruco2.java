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

    @SuppressWarnings("unchecked")
    public static List<Marker> detectMarkers(Mat image, int dict, DetectionParameters detectorParams) {
        return (List<Marker>)(List<?>)Objdetect.detectMarkers(image, dict, detectorParams);
    }

    @SuppressWarnings("unchecked")
    public static List<Marker> detectMarkers(Mat image, int dict) {
        return (List<Marker>)(List<?>)Objdetect.detectMarkers(image, dict);
    }

    @SuppressWarnings("unchecked")
    public static List<Marker> detectMarkers(Mat image) {
        return (List<Marker>)(List<?>)Objdetect.detectMarkers(image);
    }

    @SuppressWarnings("unchecked")
    public static List<Marker> detectMarkers(Mat image, MatOfInt dicts, DetectionParameters detectorParams) {
        return (List<Marker>)(List<?>)Objdetect.detectMarkers(image, dicts, detectorParams);
    }

    public static void generateMarkerImage(Mat img, int dictionary, int id, int bitSize, boolean externalBorder) {
        Objdetect.generateMarkerImage(img, dictionary, id, bitSize, externalBorder);
    }

    public static void generateMarkerImage(Mat img, int dictionary, int id) {
        Objdetect.generateMarkerImage(img, dictionary, id);
    }

    @SuppressWarnings("unchecked")
    public static void drawDetectedMarkers(Mat image, List<Marker> markers, Scalar borderColor) {
        Objdetect.aruco2_drawDetectedMarkers(image, (List<Aruco2_Marker>)(List<?>)markers, borderColor);
    }

    @SuppressWarnings("unchecked")
    public static void drawDetectedMarkers(Mat image, List<Marker> markers) {
        Objdetect.aruco2_drawDetectedMarkers(image, (List<Aruco2_Marker>)(List<?>)markers);
    }

    public static void drawAxis(Mat image, Mat cameraMatrix, Mat distCoeffs, Mat rvec, Mat tvec, float length) {
        Objdetect.drawAxis(image, cameraMatrix, distCoeffs, rvec, tvec, length);
    }

    public static void getSolvePnpPoints(Marker marker, Mat objPoints, Mat imgPoints, float markerSize) {
        Objdetect.getSolvePnpPoints(marker, objPoints, imgPoints, markerSize);
    }

    public static void getSolvePnpPoints(Marker marker, Mat objPoints, Mat imgPoints) {
        Objdetect.getSolvePnpPoints(marker, objPoints, imgPoints);
    }

    public static void generateBoardImage(Mat img, Size boardSize, int dict, int bitSize, Mat ids) {
        Objdetect.generateBoardImage(img, boardSize, dict, bitSize, ids);
    }

    public static void generateBoardImage(Mat img, Size boardSize, int dict) {
        Objdetect.generateBoardImage(img, boardSize, dict);
    }

    public static boolean detectBoard(Mat image, Size gridSize, int dict, GridBoard board, Mat ids) {
        return Objdetect.detectBoard(image, gridSize, dict, board, ids);
    }

    public static boolean detectBoard(Mat image, Size gridSize, int dict, GridBoard board) {
        return Objdetect.detectBoard(image, gridSize, dict, board);
    }

    public static void drawDetectedBoard(Mat image, GridBoard board, Scalar color, boolean drawMarkerIds) {
        Objdetect.aruco2_drawDetectedBoard(image, board, color, drawMarkerIds);
    }

    public static void drawDetectedBoard(Mat image, GridBoard board) {
        Objdetect.aruco2_drawDetectedBoard(image, board);
    }

    public static void getSolvePnpPoints(GridBoard board, Mat objPoints, Mat imgPoints, float markerSize) {
        Objdetect.getSolvePnpPoints(board, objPoints, imgPoints, markerSize);
    }

    public static void getSolvePnpPoints(GridBoard board, Mat objPoints, Mat imgPoints) {
        Objdetect.getSolvePnpPoints(board, objPoints, imgPoints);
    }

    @SuppressWarnings("unchecked")
    public static List<Diamond> detectDiamonds(Mat image, int dict) {
        return (List<Diamond>)(List<?>)Objdetect.detectDiamonds(image, dict);
    }

    public static void generateDiamondImage(Mat img, int dictionary, int[] ids, int bitSize) {
        Objdetect.generateDiamondImage(img, dictionary, ids, bitSize);
    }

    @SuppressWarnings("unchecked")
    public static void drawDetectedDiamonds(Mat image, List<Diamond> diamonds, Scalar color, boolean drawMarkerIds) {
        Objdetect.aruco2_drawDetectedDiamonds(image, (List<Aruco2_Diamond>)(List<?>)diamonds, color, drawMarkerIds);
    }

    public static void getSolvePnpPoints(Diamond diamond, Mat objPoints, Mat imgPoints, float markerSize) {
        Objdetect.getSolvePnpPoints(diamond, objPoints, imgPoints, markerSize);
    }

    public static void generateFractalImage(Mat img, int ftype, int bitSize) {
        Objdetect.generateFractalImage(img, ftype, bitSize);
    }

    @SuppressWarnings("unchecked")
    public static List<FractalMarker> detectFractals(Mat image, int ftype) {
        return (List<FractalMarker>)(List<?>)Objdetect.detectFractals(image, ftype);
    }

    @SuppressWarnings("unchecked")
    public static void drawDetectedFractals(Mat image, List<FractalMarker> fractals, Scalar color, boolean drawAllImagePoints) {
        Objdetect.aruco2_drawDetectedFractals(image, (List<Aruco2_FractalMarker>)(List<?>)fractals, color, drawAllImagePoints);
    }

    public static void getSolvePnpPoints(FractalMarker fractal, Mat objPoints, Mat imgPoints, float markerSize) {
        Objdetect.getSolvePnpPoints(fractal, objPoints, imgPoints, markerSize);
    }
}
