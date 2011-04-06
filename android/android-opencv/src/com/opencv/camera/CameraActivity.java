package com.opencv.camera;

import java.util.LinkedList;

import android.app.Activity;
import android.content.pm.ActivityInfo;
import android.os.Bundle;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.Window;
import android.view.WindowManager;
import android.widget.LinearLayout;

import com.opencv.camera.CameraButtonsHandler.CaptureListener;
import com.opencv.opengl.GL2CameraViewer;

public abstract class CameraActivity extends Activity implements CaptureListener {

	/** Called when the activity is first created. */
	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setFullscreen();
		setOrientation();
		disableScreenTurnOff();
		setContentView(com.opencv.R.layout.camera);
		cameraButtonHandler = new CameraButtonsHandler(this,this);
		mPreview = (NativePreviewer) findViewById(com.opencv.R.id.nativepreviewer);
		LinearLayout glview_layout = (LinearLayout) findViewById(com.opencv.R.id.glview_layout);
		glview = new GL2CameraViewer(getApplication(), true, 0, 0);
		glview_layout.addView(glview);
	}

	/**
	 * Handle the capture button as follows...
	 */
	@Override
	public boolean onKeyUp(int keyCode, KeyEvent event) {

		switch (keyCode) {
		case KeyEvent.KEYCODE_CAMERA:
		case KeyEvent.KEYCODE_SPACE:
		case KeyEvent.KEYCODE_DPAD_CENTER:
			cameraButtonHandler.setIsCapture(true);
			return true;

		default:
			return super.onKeyUp(keyCode, event);
		}

	}

	/**
	 * Handle the capture button as follows... On some phones there is no
	 * capture button, only trackball
	 */
	@Override
	public boolean onTrackballEvent(MotionEvent event) {
		if (event.getAction() == MotionEvent.ACTION_UP) {
			cameraButtonHandler.setIsCapture(true);
			return true;
		}
		return super.onTrackballEvent(event);
	}

	/**
	 * Avoid that the screen get's turned off by the system.
	 */
	public void disableScreenTurnOff() {
		getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
				WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
	}

	/**
	 * Set's the orientation to landscape, as this is needed by AndAR.
	 */
	public void setOrientation() {
		setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
	}

	/**
	 * Maximize the application.
	 */
	public void setFullscreen() {
		requestWindowFeature(Window.FEATURE_NO_TITLE);
		getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
				WindowManager.LayoutParams.FLAG_FULLSCREEN);
	}

	@Override
	protected void onPause() {
		super.onPause();
		mPreview.onPause();
		glview.onPause();
	}

	@Override
	protected void onResume() {
		super.onResume();
		mPreview.setParamsFromPrefs(getApplicationContext());
		glview.onResume();
		mPreview.onResume();
		setCallbackStack();
	}

	protected void setCallbackStack() {
		LinkedList<NativeProcessor.PoolCallback> callbackstack = getCallBackStack();
		if (callbackstack == null){
			callbackstack = new LinkedList<NativeProcessor.PoolCallback>();
			callbackstack.add(glview.getDrawCallback());
		}
		mPreview.addCallbackStack(callbackstack);
	}

	/**
	 * Overide this and provide your processors to the camera
	 * 
	 * @return null for default drawing
	 */
	protected abstract LinkedList<NativeProcessor.PoolCallback> getCallBackStack();
	public void onCapture(){
		
	}

	protected NativePreviewer mPreview;
	protected GL2CameraViewer glview;
	protected CameraButtonsHandler cameraButtonHandler;
}
