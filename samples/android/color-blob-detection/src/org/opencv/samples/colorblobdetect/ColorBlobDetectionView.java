package org.opencv.samples.colorblobdetect;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceHolder;
import android.view.View;
import android.view.View.OnTouchListener;

public class ColorBlobDetectionView extends SampleCvViewBase implements
		OnTouchListener {

	private Mat mRgba;
	private static final String TAG = "Example/CollorBlobDetection";
	
	public ColorBlobDetectionView(Context context)
	{
        super(context);
        setOnTouchListener(this);
	}
	
    @Override
    public void surfaceChanged(SurfaceHolder _holder, int format, int width, int height) {
        super.surfaceChanged(_holder, format, width, height);
        synchronized (this) {
            // initialize Mat before usage
            mRgba = new Mat();
        }
    }
	
	@Override
	public boolean onTouch(View v, MotionEvent event)
	{
		// TODO Auto-generated method stub
        int cols = mRgba.cols();
        int rows = mRgba.rows();
        int xoffset = (getWidth() - cols) / 2;
        int yoffset = (getHeight() - rows) / 2;

        int x = (int)event.getX() - xoffset;
        int y = (int)event.getY() - yoffset;
        
        double TouchedColor[] = mRgba.get(x,y);
        
        Log.i(TAG, "Touch coordinates: (" + x + ", " + y + ")");
        Log.i(TAG, "Touched rgba color: (" + TouchedColor[0] + ", " + TouchedColor[1] + 
        			", " + TouchedColor[2] + ", " + TouchedColor[0] + ",)");
        
        return false; // don't need subsequent touch events
	}

	@Override
	protected Bitmap processFrame(VideoCapture capture) {
		// TODO Auto-generated method stub
		capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
		
        Bitmap bmp = Bitmap.createBitmap(mRgba.cols(), mRgba.rows(), Bitmap.Config.ARGB_8888);
        try {
        	Utils.matToBitmap(mRgba, bmp);
        } catch(Exception e) {
        	Log.e(TAG, "Utils.matToBitmap() throws an exception: " + e.getMessage());
            bmp.recycle();
            bmp = null;
        }
        
        return bmp;
	}
	
    @Override
    public void run() {
        super.run();

        synchronized (this) {
            // Explicitly deallocate Mats
            if (mRgba != null)
                mRgba.release();

            mRgba = null;
        }
    }
}
