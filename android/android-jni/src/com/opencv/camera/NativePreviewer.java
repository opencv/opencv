package com.opencv.camera;

import java.io.IOException;
import java.lang.reflect.Method;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;

import android.content.Context;
import android.graphics.PixelFormat;
import android.hardware.Camera;
import android.hardware.Camera.PreviewCallback;
import android.hardware.Camera.Size;
import android.os.Handler;
import android.util.AttributeSet;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import com.opencv.camera.NativeProcessor.NativeProcessorCallback;
import com.opencv.camera.NativeProcessor.PoolCallback;

public class NativePreviewer extends SurfaceView implements
		SurfaceHolder.Callback, Camera.PreviewCallback, NativeProcessorCallback {

	private String whitebalance_mode = "auto";

	/** Constructor useful for defining a NativePreviewer in android layout xml
	 * 
	 * @param context
	 * @param attributes 
	 */
	public NativePreviewer(Context context, AttributeSet attributes) {
		super(context, attributes);
		listAllCameraMethods();
		// Install a SurfaceHolder.Callback so we get notified when the
		// underlying surface is created and destroyed.
		mHolder = getHolder();
		mHolder.addCallback(this);
		mHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);

		/* TODO get this working!  Can't figure out how to define these in xml 
		 */
		preview_width = attributes.getAttributeIntValue("opencv",
				"preview_width", 600);
		preview_height = attributes.getAttributeIntValue("opencv",
				"preview_height", 600);
		
		Log.d("NativePreviewer", "Trying to use preview size of " + preview_width + " " + preview_height);

		processor = new NativeProcessor();

		setZOrderMediaOverlay(false);
	}

	/**
	 * 
	 * @param context
	 * @param preview_width the desired camera preview width - will attempt to get as close to this as possible
	 * @param preview_height the desired camera preview height
	 */
	public NativePreviewer(Context context, int preview_width,
			int preview_height) {
		super(context);

		listAllCameraMethods();
		// Install a SurfaceHolder.Callback so we get notified when the
		// underlying surface is created and destroyed.
		mHolder = getHolder();
		mHolder.addCallback(this);
		mHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);

		this.preview_width = preview_width;
		this.preview_height = preview_height;

		processor = new NativeProcessor();
		setZOrderMediaOverlay(false);

	}
	/** Only call in the oncreate function of the instantiating activity
	 * 
	 * @param width desired width
	 * @param height desired height
	 */
	public void setPreviewSize(int width, int height){
		preview_width = width;
		preview_height = height;
		
		Log.d("NativePreviewer", "Trying to use preview size of " + preview_width + " " + preview_height);

	}
	
	public void setParamsFromPrefs(Context ctx){
		int size[] ={0,0};
		CameraConfig.readImageSize(ctx, size);
		int mode = CameraConfig.readCameraMode(ctx);
		setPreviewSize(size[0], size[1]);
		setGrayscale(mode == CameraConfig.CAMERA_MODE_BW ? true : false);
		whitebalance_mode = CameraConfig.readWhitebalace(ctx);
	}

	public void surfaceCreated(SurfaceHolder holder) {

	}

	public void surfaceDestroyed(SurfaceHolder holder) {
		releaseCamera();
	}

	public void surfaceChanged(SurfaceHolder holder, int format, int w, int h) {

		try {
			initCamera(mHolder);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return;
		}

		// Now that the size is known, set up the camera parameters and begin
		// the preview.

		Camera.Parameters parameters = mCamera.getParameters();
		List<Camera.Size> pvsizes = mCamera.getParameters()
				.getSupportedPreviewSizes();
		int best_width = 1000000;
		int best_height = 1000000;
		int bdist = 100000;
		for (Size x : pvsizes) {
			if (Math.abs(x.width - preview_width) < bdist) {
				bdist = Math.abs(x.width - preview_width);
				best_width = x.width;
				best_height = x.height;
			}
		}
		preview_width = best_width;
		preview_height = best_height;
		
		Log.d("NativePreviewer", "Determined compatible preview size is: (" + preview_width + "," + preview_height+")");

		Log.d("NativePreviewer","Supported params: " + mCamera.getParameters().flatten());
		
		//this is available in 8+
		//parameters.setExposureCompensation(0);
		parameters.setWhiteBalance(whitebalance_mode);
		parameters.setAntibanding(Camera.Parameters.ANTIBANDING_OFF);
		
		List<String> fmodes = mCamera.getParameters().getSupportedFocusModes();
		//for(String x: fmodes){
			
		//}

		if(parameters.get("meter-mode")!=null)
			parameters.set("meter-mode","meter-average");
		int idx = fmodes.indexOf(Camera.Parameters.FOCUS_MODE_INFINITY);
		if (idx != -1) {
			parameters.setFocusMode(Camera.Parameters.FOCUS_MODE_INFINITY);
		} else if (fmodes.indexOf(Camera.Parameters.FOCUS_MODE_FIXED) != -1) {
			parameters.setFocusMode(Camera.Parameters.FOCUS_MODE_FIXED);
		}

		if (fmodes.indexOf(Camera.Parameters.FOCUS_MODE_AUTO) != -1) {
			hasAutoFocus = true;
		}

		List<String> scenemodes = mCamera.getParameters()
				.getSupportedSceneModes();
		if (scenemodes != null)
			if (scenemodes.indexOf(Camera.Parameters.SCENE_MODE_ACTION) != -1) {
				parameters
						.setSceneMode(Camera.Parameters.SCENE_MODE_ACTION);
				Log.d("NativePreviewer","set scenemode to action");
			}

		parameters.setPreviewSize(preview_width, preview_height);

		mCamera.setParameters(parameters);

		pixelinfo = new PixelFormat();
		pixelformat = mCamera.getParameters().getPreviewFormat();
		PixelFormat.getPixelFormatInfo(pixelformat, pixelinfo);

		Size preview_size = mCamera.getParameters().getPreviewSize();
		preview_width = preview_size.width;
		preview_height = preview_size.height;
		int bufSize = preview_width * preview_height * pixelinfo.bitsPerPixel
				/ 8;

		// Must call this before calling addCallbackBuffer to get all the
		// reflection variables setup
		initForACB();
		initForPCWB();

		// Use only one buffer, so that we don't preview to many frames and bog
		// down system
		byte[] buffer = new byte[bufSize];
		addCallbackBuffer(buffer);
		setPreviewCallbackWithBuffer();

		mCamera.startPreview();

	}

	public void postautofocus(int delay) {
		if (hasAutoFocus)
			handler.postDelayed(autofocusrunner, delay);

	}

	/**
	 * Demonstration of how to use onPreviewFrame. In this case I'm not
	 * processing the data, I'm just adding the buffer back to the buffer queue
	 * for re-use
	 */
	public void onPreviewFrame(byte[] data, Camera camera) {

		if (start == null) {
			start = new Date();
		}

		processor.post(data, preview_width, preview_height, pixelformat,
				System.nanoTime(), this);

		fcount++;
		if (fcount % 100 == 0) {
			double ms = (new Date()).getTime() - start.getTime();
			Log.i("NativePreviewer", "fps:" + fcount / (ms / 1000.0));
			start = new Date();
			fcount = 0;
		}

	}

	@Override
	public void onDoneNativeProcessing(byte[] buffer) {
		addCallbackBuffer(buffer);
	}

	public void addCallbackStack(LinkedList<PoolCallback> callbackstack) {
		processor.addCallbackStack(callbackstack);
	}

	/**
	 * This must be called when the activity pauses, in Activity.onPause This
	 * has the side effect of clearing the callback stack.
	 * 
	 */
	public void onPause() {

		releaseCamera();

		addCallbackStack(null);

		processor.stop();

	}

	public void onResume() {

		processor.start();

	}

	private Method mPCWB;

	private void initForPCWB() {

		try {

			mPCWB = Class.forName("android.hardware.Camera").getMethod(
					"setPreviewCallbackWithBuffer", PreviewCallback.class);

		} catch (Exception e) {
			Log.e("NativePreviewer",
					"Problem setting up for setPreviewCallbackWithBuffer: "
							+ e.toString());
		}

	}

	/**
	 * This method allows you to add a byte buffer to the queue of buffers to be
	 * used by preview. See:
	 * http://android.git.kernel.org/?p=platform/frameworks
	 * /base.git;a=blob;f=core/java/android/hardware/Camera.java;hb=9d
	 * b3d07b9620b4269ab33f78604a36327e536ce1
	 * 
	 * @param b
	 *            The buffer to register. Size should be width * height *
	 *            bitsPerPixel / 8.
	 */
	private void addCallbackBuffer(byte[] b) {

		try {

			mAcb.invoke(mCamera, b);
		} catch (Exception e) {
			Log.e("NativePreviewer",
					"invoking addCallbackBuffer failed: " + e.toString());
		}
	}

	/**
	 * Use this method instead of setPreviewCallback if you want to use manually
	 * allocated buffers. Assumes that "this" implements Camera.PreviewCallback
	 */
	private void setPreviewCallbackWithBuffer() {
		// mCamera.setPreviewCallback(this);
		// return;
		try {

			// If we were able to find the setPreviewCallbackWithBuffer method
			// of Camera,
			// we can now invoke it on our Camera instance, setting 'this' to be
			// the
			// callback handler
			mPCWB.invoke(mCamera, this);

			// Log.d("NativePrevier","setPreviewCallbackWithBuffer: Called method");

		} catch (Exception e) {

			Log.e("NativePreviewer", e.toString());
		}
	}

	@SuppressWarnings("unused")
	private void clearPreviewCallbackWithBuffer() {
		// mCamera.setPreviewCallback(this);
		// return;
		try {

			// If we were able to find the setPreviewCallbackWithBuffer method
			// of Camera,
			// we can now invoke it on our Camera instance, setting 'this' to be
			// the
			// callback handler
			mPCWB.invoke(mCamera, (PreviewCallback) null);

			// Log.d("NativePrevier","setPreviewCallbackWithBuffer: cleared");

		} catch (Exception e) {

			Log.e("NativePreviewer", e.toString());
		}
	}

	/**
	 * These variables are re-used over and over by addCallbackBuffer
	 */
	private Method mAcb;

	private void initForACB() {
		try {

			mAcb = Class.forName("android.hardware.Camera").getMethod(
					"addCallbackBuffer", byte[].class);

		} catch (Exception e) {
			Log.e("NativePreviewer",
					"Problem setting up for addCallbackBuffer: " + e.toString());
		}
	}

	private Runnable autofocusrunner = new Runnable() {

		@Override
		public void run() {
			mCamera.autoFocus(autocallback);
		}
	};

	private Camera.AutoFocusCallback autocallback = new Camera.AutoFocusCallback() {

		@Override
		public void onAutoFocus(boolean success, Camera camera) {
			if (!success)
				postautofocus(1000);
		}
	};

	/**
	 * This method will list all methods of the android.hardware.Camera class,
	 * even the hidden ones. With the information it provides, you can use the
	 * same approach I took below to expose methods that were written but hidden
	 * in eclair
	 */
	private void listAllCameraMethods() {
		try {
			Class<?> c = Class.forName("android.hardware.Camera");
			Method[] m = c.getMethods();
			for (int i = 0; i < m.length; i++) {
				Log.d("NativePreviewer", "  method:" + m[i].toString());
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			Log.e("NativePreviewer", e.toString());
		}
	}

	private void initCamera(SurfaceHolder holder) throws InterruptedException {
		if (mCamera == null) {
			// The Surface has been created, acquire the camera and tell it
			// where
			// to draw.
			int i = 0;
			while (i++ < 5) {
				try {
					mCamera = Camera.open();
					break;
				} catch (RuntimeException e) {
					Thread.sleep(200);
				}
			}
			try {
				mCamera.setPreviewDisplay(holder);
			} catch (IOException exception) {
				mCamera.release();
				mCamera = null;

			} catch (RuntimeException e) {
				Log.e("camera", "stacktrace", e);
			}
		}
	}

	private void releaseCamera() {
		if (mCamera != null) {
			// Surface will be destroyed when we return, so stop the preview.
			// Because the CameraDevice object is not a shared resource, it's
			// very
			// important to release it when the activity is paused.
			mCamera.stopPreview();
			mCamera.release();
		}

		// processor = null;
		mCamera = null;
		mAcb = null;
		mPCWB = null;
	}

	private Handler handler = new Handler();

	private Date start;
	private int fcount = 0;
	private boolean hasAutoFocus = false;
	private SurfaceHolder mHolder;
	private Camera mCamera;

	private NativeProcessor processor;

	private int preview_width, preview_height;
	private int pixelformat;
	private PixelFormat pixelinfo;

	public void setGrayscale(boolean b) {
		processor.setGrayscale(b);
		
	}

}