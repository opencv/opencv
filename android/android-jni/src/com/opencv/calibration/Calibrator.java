package com.opencv.calibration;


import java.io.File;
import java.io.IOException;
import java.util.concurrent.locks.ReentrantLock;

import android.os.AsyncTask;

import com.opencv.camera.NativeProcessor;
import com.opencv.camera.NativeProcessor.PoolCallback;
import com.opencv.jni.Calibration;
import com.opencv.jni.Size;
import com.opencv.jni.image_pool;



public class Calibrator implements PoolCallback {
	private Calibration calibration;

	static public interface CalibrationCallback{
		public void onFoundChessboard(Calibrator calibrator);
		public void onDoneCalibration(Calibrator calibration, File calibfile);
		public void onFailedChessboard(Calibrator calibrator);
	}
	private CalibrationCallback callback;
	public Calibrator(CalibrationCallback callback) {
		calibration = new Calibration();
		this.callback = callback;
	}

	public void resetCalibration(){
		calibration.resetChess();
	}

	public void setPatternSize(Size size){
		Size csize = calibration.getPatternsize();
		if(size.getWidth() == csize.getWidth()&&
				   size.getHeight() == csize.getHeight())
					return;
		calibration.setPatternsize(size);	
		resetCalibration();
	}
	public void setPatternSize(int width, int height){
		Size patternsize = new Size(width,height);
		setPatternSize(patternsize);
	}
	
	private boolean capture_chess;

	ReentrantLock lock = new ReentrantLock();
	public void calibrate(File calibration_file) throws IOException{
		if(getNumberPatternsDetected() < 3){
			return;
		}
		CalibrationTask calibtask = new CalibrationTask(calibration_file);
		calibtask.execute((Object[])null);
	}

	public void queueChessCapture(){
		capture_chess = true;
	}
	
private class CalibrationTask extends AsyncTask<Object, Object, Object> {
		File calibfile;
	
		public CalibrationTask(File calib) throws IOException{
			super();
			calibfile = calib;
			calibfile.createNewFile();
		}
	
		@Override
		protected Object doInBackground(Object... params) {
			lock.lock();
			try{
				calibration.calibrate(calibfile.getAbsolutePath());
			}
			finally{
				lock.unlock();
			}
			return null;
		
		}

		@Override
		protected void onPostExecute(Object result) {			
			callback.onDoneCalibration(Calibrator.this, calibfile);
		}

	}
	

	@Override
	public void process(int idx, image_pool pool, long timestamp,
			NativeProcessor nativeProcessor) {
		if(lock.tryLock()){
			try{
				if(capture_chess){
					if(calibration.detectAndDrawChessboard(idx, pool)){
						callback.onFoundChessboard(this);
						
					}else
						callback.onFailedChessboard(this);
					capture_chess = false;
				}
			}finally{
				lock.unlock();
			}
		}
	}


	public int getNumberPatternsDetected(){
		return calibration.getNumberDetectedChessboards();
	}

	public void setCallback(CalibrationCallback callback) {
		this.callback = callback;
		
	}


}
