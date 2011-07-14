package org.opencv.samples.s1;

import java.util.List;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.hardware.Camera;
import android.hardware.Camera.PreviewCallback;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

public abstract class SampleViewBase extends SurfaceView implements SurfaceHolder.Callback, Runnable {

	private static final String TAG = "Sample::ViewBase";
	private Camera mCamera;
	private SurfaceHolder mHolder;
	protected int mFrameWidth;
	protected int mFrameHeight;
	private byte[] mFrame;
	private boolean mThreadRun;

	public SampleViewBase(Context context) {
		super(context);
        mHolder = getHolder();
        mHolder.addCallback(this);
	}

	public void surfaceChanged(SurfaceHolder _holder, int format, int width, int height) {
			    if ( mCamera != null) {
			        Camera.Parameters params = mCamera.getParameters();
			        List<Camera.Size> sizes = params.getSupportedPreviewSizes();
			        mFrameWidth = width;
			        mFrameHeight = height;
			        
			        //selecting optimal camera preview size
			        {
			            double minDiff = Double.MAX_VALUE;
			            for (Camera.Size size : sizes) {
			                if (Math.abs(size.height - height) < minDiff) {
			                    mFrameWidth = size.width;
			                    mFrameHeight = size.height;
			                    minDiff = Math.abs(size.height - height);
			                }
			            }
			        }
			        params.setPreviewSize(mFrameWidth, mFrameHeight);
			        mCamera.setParameters(params);
			        mCamera.startPreview();
			    }
			}

	public void surfaceCreated(SurfaceHolder holder) {
		Log.i("SAMP1", "surfaceCreated");
	    mCamera = Camera.open();
	    mCamera.setPreviewCallback(
	            new PreviewCallback() {
	                public void onPreviewFrame(byte[] data, Camera camera) {
	                    synchronized(SampleViewBase.this) {
	                        mFrame = data;
	                        //Log.i("SAMP1", "before notify");
	                        SampleViewBase.this.notify();
	                        //Log.i("SAMP1", "after notify");
	                    }
	                }
	            }
	    );
	    (new Thread(this)).start();
	}

	public void surfaceDestroyed(SurfaceHolder holder) {
		Log.i("SAMP1", "surfaceDestroyed");
	    mThreadRun = false;
	    if(mCamera != null) {
	        synchronized(this) {
	            mCamera.stopPreview();
	            mCamera.setPreviewCallback(null);
	            mCamera.release();
	            mCamera = null;
	        }
	    }
	}
	
	protected abstract Bitmap processFrame(byte[] data);

	public void run() {
	    mThreadRun = true;
	    Log.i(TAG, "Starting thread");
	    Bitmap bmp = null;
	    while(mThreadRun) {
	    	//Log.i("SAMP1", "before synchronized");
	        synchronized(this) {
	        	//Log.i("SAMP1", "in synchronized");
	            try {
	                this.wait();
	                //Log.i("SAMP1", "before processFrame");
	                bmp = processFrame(mFrame);
	                //Log.i("SAMP1", "after processFrame");
	            } catch (InterruptedException e) {
	                e.printStackTrace();
	            }
	        }
	        
	        if (bmp != null){
	        	Canvas canvas = mHolder.lockCanvas();
	        	if (canvas != null){
	        		canvas.drawBitmap(bmp, (canvas.getWidth()-mFrameWidth)/2, (canvas.getHeight()-mFrameHeight)/2, null);
	        		mHolder.unlockCanvasAndPost(canvas);
	        	}
	        }
	    }
	}
}