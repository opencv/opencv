package org.opencv.samples.cameracalibration;

import java.util.ArrayList;
import java.util.List;

import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Range;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import android.content.res.Resources;

abstract class FrameRender {
    protected CameraCalibrator mCalibrator;

    public abstract Mat render(CvCameraViewFrame inputFrame);
}

class PreviewFrameRender extends FrameRender {
    @Override
    public Mat render(CvCameraViewFrame inputFrame) {
        return inputFrame.rgba();
    }
}

class CalibrationFrameRender extends FrameRender {
    public CalibrationFrameRender(CameraCalibrator calibrator) {
        mCalibrator = calibrator;
    }

    @Override
    public Mat render(CvCameraViewFrame inputFrame) {
        Mat rgbaFrame = inputFrame.rgba();
        Mat grayFrame = inputFrame.gray();
        mCalibrator.processFrame(grayFrame, rgbaFrame);

        return rgbaFrame;
    }
}

class UndistortionFrameRender extends FrameRender {
    public UndistortionFrameRender(CameraCalibrator calibrator) {
        mCalibrator = calibrator;
    }

    @Override
    public Mat render(CvCameraViewFrame inputFrame) {
        Mat renderedFrame = new Mat(inputFrame.rgba().size(), inputFrame.rgba().type());
        Calib3d.undistort(inputFrame.rgba(), renderedFrame,
                mCalibrator.getCameraMatrix(), mCalibrator.getDistortionCoefficients());

        return renderedFrame;
    }
}

class ComparisonFrameRender extends FrameRender {
    private int mWidth;
    private int mHeight;
    private Resources mResources;
    public ComparisonFrameRender(CameraCalibrator calibrator, int width, int height, Resources resources) {
        mCalibrator = calibrator;
        mWidth = width;
        mHeight = height;
        mResources = resources;
    }

    @Override
    public Mat render(CvCameraViewFrame inputFrame) {
        Mat undistortedFrame = new Mat(inputFrame.rgba().size(), inputFrame.rgba().type());
        Calib3d.undistort(inputFrame.rgba(), undistortedFrame,
                mCalibrator.getCameraMatrix(), mCalibrator.getDistortionCoefficients());

        Mat comparisonFrame = inputFrame.rgba();
        undistortedFrame.colRange(new Range(0, mWidth / 2)).copyTo(comparisonFrame.colRange(new Range(mWidth / 2, mWidth)));
        List<MatOfPoint> border = new ArrayList<MatOfPoint>();
        final int shift = (int)(mWidth * 0.005);
        border.add(new MatOfPoint(new Point(mWidth / 2 - shift, 0), new Point(mWidth / 2 + shift, 0),
                new Point(mWidth / 2 + shift, mHeight), new Point(mWidth / 2 - shift, mHeight)));
        Imgproc.fillPoly(comparisonFrame, border, new Scalar(255, 255, 255));

        Imgproc.putText(comparisonFrame, mResources.getString(R.string.original), new Point(mWidth * 0.1, mHeight * 0.1),
                Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 0));
        Imgproc.putText(comparisonFrame, mResources.getString(R.string.undistorted), new Point(mWidth * 0.6, mHeight * 0.1),
                Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 0));

        return comparisonFrame;
    }
}

class OnCameraFrameRender {
    private FrameRender mFrameRender;
    public OnCameraFrameRender(FrameRender frameRender) {
        mFrameRender = frameRender;
    }
    public Mat render(CvCameraViewFrame inputFrame) {
        return mFrameRender.render(inputFrame);
    }
}
