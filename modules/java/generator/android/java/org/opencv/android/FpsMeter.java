package org.opencv.android;

import java.text.DecimalFormat;

import org.opencv.core.Core;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.Log;

public class FpsMeter {
    private static final String TAG               = "FpsMeter";
    private static final int    STEP              = 20;
    private static final DecimalFormat FPS_FORMAT = new DecimalFormat("0.00");

    private int                 mFramesCounter;
    private double              mFrequency;
    private long                mprevFrameTime;
    private String              mStrfps;
    Paint                       mPaint;
    boolean                     mIsInitialized = false;
    int                         mWidth = 0;
    int                         mHeight = 0;

    public void init() {
        mFramesCounter = 0;
        mFrequency = Core.getTickFrequency();
        mprevFrameTime = Core.getTickCount();
        mStrfps = "";

        mPaint = new Paint();
        mPaint.setColor(Color.BLUE);
        mPaint.setTextSize(20);
    }

    public void measure() {
        if (!mIsInitialized) {
            init();
            mIsInitialized = true;
        } else {
            mFramesCounter++;
            if (mFramesCounter % STEP == 0) {
                long time = Core.getTickCount();
                double fps = STEP * mFrequency / (time - mprevFrameTime);
                mprevFrameTime = time;
                if (mWidth != 0 && mHeight != 0)
                    mStrfps = FPS_FORMAT.format(fps) + " FPS@" + Integer.valueOf(mWidth) + "x" + Integer.valueOf(mHeight);
                else
                    mStrfps = FPS_FORMAT.format(fps) + " FPS";
                Log.i(TAG, mStrfps);
            }
        }
    }

    public void setResolution(int width, int height) {
        Log.d(TAG, "FpsMeter.setResolution " + Integer.valueOf(mWidth) + "x" + Integer.valueOf(mHeight));
        mWidth = width;
        mHeight = height;
    }

    public void draw(Canvas canvas, float offsetx, float offsety) {
        Log.d(TAG, mStrfps);
        canvas.drawText(mStrfps, offsetx, offsety, mPaint);
    }

}
