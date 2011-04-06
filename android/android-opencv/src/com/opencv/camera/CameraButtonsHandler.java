package com.opencv.camera;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.ImageButton;

public class CameraButtonsHandler {
	/** Constructs a buttons handler, will register with the capture button
	 * and the camera settings button. 
	 * @param a The activity that has inflated the com.opencv.R.layout.camera
	 * as its layout.
	 */
	public CameraButtonsHandler(Activity a, CaptureListener l) {
		ImageButton capture = (ImageButton) a
				.findViewById(com.opencv.R.id.button_capture);
		ImageButton settings = (ImageButton) a
				.findViewById(com.opencv.R.id.button_camera_settings);
		capture.setOnClickListener(capture_listener);
		settings.setOnClickListener(settings_listener);
		captureListener = l;
		ctx = a;
	}
	
	public CameraButtonsHandler(Activity a) {
		ImageButton capture = (ImageButton) a
				.findViewById(com.opencv.R.id.button_capture);
		ImageButton settings = (ImageButton) a
				.findViewById(com.opencv.R.id.button_camera_settings);
		capture.setOnClickListener(capture_listener);
		settings.setOnClickListener(settings_listener);
		ctx = a;
	}
	
	
	/** Check if the capture button has been pressed
	 * @return true if the capture button has been pressed
	 */
	synchronized public boolean isCapture(){
		return capture_flag;
	}
	
	/** Reset the capture flag 
	 */
	synchronized public void resetIsCapture(){
		capture_flag = false;
	}
	
	/** Manually set the flag - call this on any event that should trigger
	 * a capture
	 * @param isCapture true if a capture should take place
	 */
	synchronized public void setIsCapture(boolean isCapture){
		capture_flag = isCapture;
		if(capture_flag && captureListener != null){
			captureListener.onCapture();
		}
	}
	
	private OnClickListener capture_listener = new View.OnClickListener() {
		@Override
		public void onClick(View v) {
			setIsCapture(true);
		}
	};
	private OnClickListener settings_listener = new View.OnClickListener() {
		@Override
		public void onClick(View v) {
			Intent configurer = new Intent(ctx,
					CameraConfig.class);
			ctx.startActivity(configurer);
		}
	};

	interface CaptureListener{
		public void onCapture();
	}
	private CaptureListener captureListener;
	private Context ctx;		
	private boolean capture_flag = false;
}
