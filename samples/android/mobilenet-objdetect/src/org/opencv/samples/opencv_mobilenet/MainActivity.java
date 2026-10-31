package org.opencv.samples.opencv_mobilenet;
/*
// snippet was added for Android tutorial
//! [mobilenet_tutorial_package]
package com.example.myapplication;
//! [mobilenet_tutorial_package]
*/
//! [mobilenet_tutorial]
import android.content.Context;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.util.Log;
import android.widget.Toast;

import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Net;
import org.opencv.dnn.Dnn;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.CvType;
import org.opencv.core.CvException;

import java.io.InputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.ArrayList;
import java.util.Locale;
import java.io.ByteArrayOutputStream;

public class MainActivity extends CameraActivity implements CvCameraViewListener2 {

    @Override
    public void onResume() {
        super.onResume();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.enableView();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (OpenCVLoader.initLocal()) {
            Log.i(TAG, "OpenCV loaded successfully");
        } else {
            Log.e(TAG, "OpenCV initialization failed!");
            (Toast.makeText(this, "OpenCV initialization failed!", Toast.LENGTH_LONG)).show();
            return;
        }

        //! [init_model_from_memory]
        // Load a single-file MobileNet-SSD ONNX model from res/raw/mobilenet.onnx.
        // The resource is resolved by name at runtime so the sample still builds when the
        // (optionally downloaded) model is absent; object detection is simply disabled then.
        int modelResId = getResources().getIdentifier("mobilenet", "raw", getPackageName());
        if (modelResId != 0)
            mModelBuffer = loadFileFromResource(modelResId);

        if (mModelBuffer == null) {
            Log.e(TAG, "MobileNet ONNX model (res/raw/mobilenet.onnx) not found - object detection disabled");
        } else {
            net = Dnn.readNetFromONNX(mModelBuffer);

            outputNames = new ArrayList<>(net.getUnconnectedOutLayersNames());

            Log.i(TAG, "Network loaded successfully");
            Log.i(TAG, "OpenCV output layer names: " + outputNames);

            if (outputNames.size() != 4) {
                Log.w(TAG, "Expected 4 output layers, found " + outputNames.size());
            }
        }
        //! [init_model_from_memory]

        setContentView(R.layout.activity_main);

        // Set up camera listener.
        mOpenCvCameraView = (CameraBridgeViewBase)findViewById(R.id.CameraView);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();

        if (mModelBuffer != null)
            mModelBuffer.release();
    }

    // Load a network.
    public void onCameraViewStarted(int width, int height) {
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        final double THRESHOLD = 0.2;

        if (net == null) {
            return inputFrame.rgba();
        }

        Log.d(TAG, "handle new frame!");
        Mat frame = inputFrame.rgba();  // Keep RGBA for display
        Mat rgb = new Mat();

        // Separate RGB image for model input
        Imgproc.cvtColor(frame, rgb, Imgproc.COLOR_RGBA2RGB);
        // Forward image through network.
        //! [mobilenet_handle_frame]

        // Model input: uint8 RGB, NHWC layout [1, height, width, 3].
        int rows = frame.rows();
        int cols = frame.cols();

        byte[] inputData = new byte[
                (int)(rgb.total() * rgb.channels())
                ];
        rgb.get(0, 0, inputData);

        Mat blob = new Mat(
                new int[]{1, rows, cols, 3},
                CvType.CV_8UC1
        );

        blob.put(new int[]{0, 0, 0, 0}, inputData);

        net.setInput(blob);

        // Retrieve the four named model outputs.
        // Retrieve outputs using the names exposed by OpenCV.
        ArrayList<Mat> outputs = new ArrayList<>();

        try {
            if (outputNames.size() != 4) {
                Log.e(TAG, "Cannot run inference: expected 4 output layers, got "
                        + outputNames.size());

                blob.release();
                rgb.release();
                return frame;
            }
            net.forward(outputs, outputNames);
        } catch (CvException e) {
            Log.e(TAG, "DNN forward failed. Output names: " + outputNames, e);

            for (Mat output : outputs) {
                output.release();
            }

            blob.release();
            rgb.release();
            return frame;
        }

        if (outputs.size() != outputNames.size()) {
            Log.e(TAG, "Output count mismatch: expected "
                    + outputNames.size() + ", got " + outputs.size());

            for (Mat output : outputs) {
                output.release();
            }

            blob.release();
            rgb.release();
            return frame;
        }

// Log actual output names and shapes.
        for (int i = 0; i < outputs.size(); i++) {
            Mat out = outputs.get(i);

            Log.i(TAG, "Output[" + i + "] name=" + outputNames.get(i)
                    + ", dims=" + out.dims()
                    + ", rows=" + out.rows()
                    + ", cols=" + out.cols()
                    + ", total=" + out.total());
        }

// Identify outputs by their names, not by assumed order.
        int numIndex = findOutputIndex("num_detections");
        int boxesIndex = findOutputIndex("detection_boxes");
        int scoresIndex = findOutputIndex("detection_scores");
        int classesIndex = findOutputIndex("detection_classes");

        if (numIndex < 0 || boxesIndex < 0 ||
                scoresIndex < 0 || classesIndex < 0) {

            Log.e(TAG, "Could not map all detection outputs. "
                    + "Available names: " + outputNames);

            for (Mat output : outputs) {
                output.release();
            }

            blob.release();
            rgb.release();
            return frame;
        }

        Log.i(TAG, "Successfully mapped all four detection outputs.");


        // Reshape outputs for convenient indexing.
        Mat numDetections = outputs.get(numIndex).reshape(1, 1);

        Mat boxes = outputs.get(boxesIndex).reshape(
                1, (int)(outputs.get(boxesIndex).total() / 4)
        );

        Mat scores = outputs.get(scoresIndex).reshape(1, 1);
        Mat classes = outputs.get(classesIndex).reshape(1, 1);

        int maxDetections = Math.min(
                boxes.rows(),
                Math.min((int)scores.total(), (int)classes.total())
        );

        int count = Math.min(
                (int)Math.round(numDetections.get(0, 0)[0]),
                maxDetections
        );

        count = Math.max(0, count);

        for (int i = 0; i < count; i++) {

            double confidence = scores.get(0, i)[0];

            if (confidence > THRESHOLD) {

                int classId = (int)Math.round(classes.get(0, i)[0]);

                // Model returns normalized coordinates:
                // top, left, bottom, right.

                int top = (int)Math.round(boxes.get(i, 0)[0] * rows);
                int left = (int)Math.round(boxes.get(i, 1)[0] * cols);
                int bottom = (int)Math.round(boxes.get(i, 2)[0] * rows);
                int right = (int)Math.round(boxes.get(i, 3)[0] * cols);

                // Clamp coordinates to image boundaries.
                top = Math.max(0, Math.min(rows, top));
                left = Math.max(0, Math.min(cols, left));
                bottom = Math.max(0, Math.min(rows, bottom));
                right = Math.max(0, Math.min(cols, right));

                if (right <= left || bottom <= top)
                    continue;

                Imgproc.rectangle(
                        frame,
                        new Point(left, top),
                        new Point(right, bottom),
                        new Scalar(0, 255, 0)
                );

                // Model uses COCO classes, not PASCAL VOC.
                String label = "COCO #" + classId + ": " + confidence;

                int[] baseLine = new int[1];

                Size labelSize = Imgproc.getTextSize(
                        label,
                        Imgproc.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        1,
                        baseLine
                );

                Imgproc.rectangle(
                        frame,
                        new Point(left, Math.max(0, top - labelSize.height)),
                        new Point(left + labelSize.width, top + baseLine[0]),
                        new Scalar(255, 255, 255),
                        Imgproc.FILLED
                );

                Imgproc.putText(
                        frame,
                        label,
                        new Point(left, top),
                        Imgproc.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        new Scalar(0, 0, 0)
                );
            }
        }

        // Release native memory.
        numDetections.release();
        boxes.release();
        scores.release();
        classes.release();

        for (Mat output : outputs)
            output.release();

        blob.release();
        rgb.release();
        //! [mobilenet_handle_frame]


        return frame;
    }

    private int findOutputIndex(String expectedName) {
        for (int i = 0; i < outputNames.size(); i++) {
            String actualName = outputNames.get(i).toLowerCase(Locale.ROOT);

            if (actualName.contains(expectedName.toLowerCase(Locale.ROOT))) {
                return i;
            }
        }

        return -1;
    }

    public void onCameraViewStopped() {}

    //! [mobilenet_tutorial_resource]
    private MatOfByte loadFileFromResource(int id) {
        try (InputStream is = getResources().openRawResource(id);
             ByteArrayOutputStream buffer = new ByteArrayOutputStream()) {

            byte[] data = new byte[8192];
            int bytesRead;

            while ((bytesRead = is.read(data)) != -1) {
                buffer.write(data, 0, bytesRead);
            }

            byte[] modelData = buffer.toByteArray();

            Log.i(TAG, "Loaded ONNX model resource: "
                    + modelData.length + " bytes");

            return new MatOfByte(modelData);

        } catch (IOException e) {
            Log.e(TAG, "Failed to load ONNX model from resources", e);

            Toast.makeText(
                    this,
                    "Failed to load ONNX model!",
                    Toast.LENGTH_LONG
            ).show();

            return null;
        }
    }
    //! [mobilenet_tutorial_resource]

    private static final String TAG = "OpenCV-MobileNet";

    private MatOfByte            mModelBuffer;
    private Net                  net;
    private ArrayList<String> outputNames = new ArrayList<>();
    private CameraBridgeViewBase mOpenCvCameraView;
}
//! [mobilenet_tutorial]
