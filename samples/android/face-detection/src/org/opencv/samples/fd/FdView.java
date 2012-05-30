package org.opencv.samples.fd;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.objdetect.CascadeClassifier;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import android.view.SurfaceHolder;

class FdView extends SampleCvViewBase {
    private static final String TAG = "Sample::FdView";
    private Mat                  mRgba;
    private Mat                  mGray;
    private File                 mCascadeFile;
    private CascadeClassifier    mCascade;
    private DetectionBaseTracker mTracker;

    public final int             CASCADE_DETECTOR = 0;
    public final int             DBT_DETECTOR     = 1;
    
    private int                  mDetectorType = CASCADE_DETECTOR;

    public static int            mFaceSize = 200;
    
    public void setMinFaceSize(float faceSize)
    {
		int height = mGray.rows();
    	if (Math.round(height * faceSize) > 0);
    	{
    		mFaceSize = Math.round(height * faceSize);
    	}
    	mTracker.setMinFaceSize(mFaceSize);
    }
    
    public void setDtetectorType(int type)
    {
    	if (mDetectorType != type)
    	{
    		mDetectorType = type;
    		
    		if (type == DBT_DETECTOR)
    		{
    			Log.i(TAG, "Detection Base Tracker enabled");
    			mTracker.start();
    		}
    		else
    		{
    			Log.i(TAG, "Cascade detectior enabled");
    			mTracker.stop();
    		}
    	}
    }

    public FdView(Context context) {
        super(context);

        try {
            InputStream is = context.getResources().openRawResource(R.raw.lbpcascade_frontalface);
            File cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE);
            mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            mCascade = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            if (mCascade.empty()) {
                Log.e(TAG, "Failed to load cascade classifier");
                mCascade = null;
            } else
                Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

            mTracker = new DetectionBaseTracker(mCascadeFile.getAbsolutePath(), 0);
            
            cascadeDir.delete();

        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
        }
    }

    @Override
	public void surfaceCreated(SurfaceHolder holder) {
        synchronized (this) {
            // initialize Mats before usage
            mGray = new Mat();
            mRgba = new Mat();
        }

        super.surfaceCreated(holder);
	}

	@Override
    protected Bitmap processFrame(VideoCapture capture) {
        capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
        capture.retrieve(mGray, Highgui.CV_CAP_ANDROID_GREY_FRAME);

        MatOfRect faces = new MatOfRect();
        
        if (mDetectorType == CASCADE_DETECTOR)
        {
        	if (mCascade != null)
                mCascade.detectMultiScale(mGray, faces, 1.1, 2, 2 // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                        , new Size(mFaceSize, mFaceSize), new Size());
        }
        else if (mDetectorType == DBT_DETECTOR)
        {
        	if (mTracker != null)
        		mTracker.detect(mGray, faces);
        }
        else
        {
        	Log.e(TAG, "Detection method is not selected!");
        }
        
        for (Rect r : faces.toArray())
            Core.rectangle(mRgba, r.tl(), r.br(), new Scalar(0, 255, 0, 255), 3);

        Bitmap bmp = Bitmap.createBitmap(mRgba.cols(), mRgba.rows(), Bitmap.Config.RGB_565/*.ARGB_8888*/);

        try {
        	Utils.matToBitmap(mRgba, bmp);
            return bmp;
        } catch(Exception e) {
        	Log.e("org.opencv.samples.puzzle15", "Utils.matToBitmap() throws an exception: " + e.getMessage());
            bmp.recycle();
            return null;
        }
    }

    @Override
    public void run() {
        super.run();

        synchronized (this) {
            // Explicitly deallocate Mats
            if (mRgba != null)
                mRgba.release();
            if (mGray != null)
                mGray.release();
            if (mCascadeFile != null)
            	mCascadeFile.delete();
            if (mTracker != null)
            	mTracker.release();

            mRgba = null;
            mGray = null;
            mCascadeFile = null;
        }
    }
}
