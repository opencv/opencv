package org.opencv.samples.cameracalibration;

import org.opencv.core.Mat;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

public abstract class CalibrationResult {
    private static final String TAG = "OCV::CalibrationResult";

    private static final int CAMERA_MATRIX_ROWS = 3;
    private static final int CAMERA_MATRIX_COLS = 3;
    private static final int DISTORTION_COEFFICIENTS_SIZE = 5;

    public static void save(Activity activity, Mat cameraMatrix, Mat distortionCoefficients) {
        SharedPreferences sharedPref = activity.getPreferences(Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = sharedPref.edit();

        double[] cameraMatrixArray = new double[CAMERA_MATRIX_ROWS * CAMERA_MATRIX_COLS];
        cameraMatrix.get(0,  0, cameraMatrixArray);
        for (int i = 0; i < CAMERA_MATRIX_ROWS; i++) {
            for (int j = 0; j < CAMERA_MATRIX_COLS; j++) {
                int id = i * CAMERA_MATRIX_ROWS + j;
                editor.putFloat(Integer.toString(id), (float)cameraMatrixArray[id]);
            }
        }

        double[] distortionCoefficientsArray = new double[DISTORTION_COEFFICIENTS_SIZE];
        distortionCoefficients.get(0, 0, distortionCoefficientsArray);
        int shift = CAMERA_MATRIX_ROWS * CAMERA_MATRIX_COLS;
        for (int i = shift; i < DISTORTION_COEFFICIENTS_SIZE + shift; i++) {
            editor.putFloat(Integer.toString(i), (float)distortionCoefficientsArray[i-shift]);
        }

        editor.apply();
        Log.i(TAG, "Saved camera matrix: " + cameraMatrix.dump());
        Log.i(TAG, "Saved distortion coefficients: " + distortionCoefficients.dump());
    }

    public static boolean tryLoad(Activity activity, Mat cameraMatrix, Mat distortionCoefficients) {
        SharedPreferences sharedPref = activity.getPreferences(Context.MODE_PRIVATE);
        if (sharedPref.getFloat("0", -1) == -1) {
            Log.i(TAG, "No previous calibration results found");
            return false;
        }

        double[] cameraMatrixArray = new double[CAMERA_MATRIX_ROWS * CAMERA_MATRIX_COLS];
        for (int i = 0; i < CAMERA_MATRIX_ROWS; i++) {
            for (int j = 0; j < CAMERA_MATRIX_COLS; j++) {
                int id = i * CAMERA_MATRIX_ROWS + j;
                cameraMatrixArray[id] = sharedPref.getFloat(Integer.toString(id), -1);
            }
        }
        cameraMatrix.put(0, 0, cameraMatrixArray);
        Log.i(TAG, "Loaded camera matrix: " + cameraMatrix.dump());

        double[] distortionCoefficientsArray = new double[DISTORTION_COEFFICIENTS_SIZE];
        int shift = CAMERA_MATRIX_ROWS * CAMERA_MATRIX_COLS;
        for (int i = shift; i < DISTORTION_COEFFICIENTS_SIZE + shift; i++) {
            distortionCoefficientsArray[i - shift] = sharedPref.getFloat(Integer.toString(i), -1);
        }
        distortionCoefficients.put(0, 0, distortionCoefficientsArray);
        Log.i(TAG, "Loaded distortion coefficients: " + distortionCoefficients.dump());

        return true;
    }
}
