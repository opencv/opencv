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

import java.io.InputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;

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
            Log.i(TAG, "Network loaded successfully");
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
        Mat frame = inputFrame.rgba();
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);
        // Forward image through network.
        //! [mobilenet_handle_frame]

        // Model input: uint8 RGB, NHWC layout [1, height, width, 3].
        int rows = frame.rows();
        int cols = frame.cols();

        byte[] inputData = new byte[(int)(frame.total() * frame.channels())];
        frame.get(0, 0, inputData);

        Mat blob = new Mat(
            new int[]{1, rows, cols, 3},
            CvType.CV_8UC1
        );

        blob.put(new int[]{0, 0, 0, 0}, inputData);

        net.setInput(blob);

        // Retrieve the four named model outputs.
        List<Mat> outputs = new ArrayList<>();

        List<String> outputNames = Arrays.asList(
            "num_detections:0",
            "detection_boxes:0",
            "detection_scores:0",
            "detection_classes:0"
        );

        net.forward(outputs, outputNames);

        if (outputs.size() != 4) {
            Log.e(TAG, "Unexpected number of model outputs: " + outputs.size());

            for (Mat output : outputs)
                output.release();

            blob.release();
            return frame;
        }

        // Reshape outputs for convenient indexing.
        Mat numDetections = outputs.get(0).reshape(1, 1);

        Mat boxes = outputs.get(1).reshape(
            1, (int)(outputs.get(1).total() / 4)
        );

        Mat scores = outputs.get(2).reshape(1, 1);
        Mat classes = outputs.get(3).reshape(1, 1);

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

        //! [mobilenet_handle_frame]


        return frame;
    }

    public void onCameraViewStopped() {}

    //! [mobilenet_tutorial_resource]
    private MatOfByte loadFileFromResource(int id) {
       byte[] buffer;
        try {
            // load cascade file from application resources
            InputStream is = getResources().openRawResource(id);

            int size = is.available();
            buffer = new byte[size];
            int bytesRead = is.read(buffer);
            is.close();
        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Failed to ONNX model from resources! Exception thrown: " + e);
            (Toast.makeText(this, "Failed to ONNX model from resources!", Toast.LENGTH_LONG)).show();
            return null;
        }

        return new MatOfByte(buffer);
    }
    //! [mobilenet_tutorial_resource]

    private static final String TAG = "OpenCV-MobileNet";

    private MatOfByte            mModelBuffer;
    private Net                  net;
    private CameraBridgeViewBase mOpenCvCameraView;
}
//! [mobilenet_tutorial]
