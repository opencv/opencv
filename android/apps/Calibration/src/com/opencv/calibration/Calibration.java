package com.opencv.calibration;

import java.io.File;
import java.io.IOException;
import java.util.LinkedList;

import android.app.Activity;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.ActivityInfo;
import android.os.Bundle;
import android.os.IBinder;
import android.view.KeyEvent;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ImageButton;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.opencv.calibration.Calibrator.CalibrationCallback;
import com.opencv.calibration.services.CalibrationService;
import com.opencv.camera.CameraConfig;
import com.opencv.camera.NativePreviewer;
import com.opencv.camera.NativeProcessor;
import com.opencv.misc.SDCardChecker;
import com.opencv.opengl.GL2CameraViewer;

public class Calibration extends Activity implements CalibrationCallback {
	private NativePreviewer mPreview;

	private GL2CameraViewer glview;
	private Calibrator calibrator;

	@Override
	public boolean onKeyUp(int keyCode, KeyEvent event) {

		switch (keyCode) {
		case KeyEvent.KEYCODE_CAMERA:
		case KeyEvent.KEYCODE_SPACE:
		case KeyEvent.KEYCODE_DPAD_CENTER:
			calibrator.queueChessCapture();
			return true;
		default:
			return super.onKeyUp(keyCode, event);
		}

	}

	@Override
	public boolean onTrackballEvent(MotionEvent event) {
		if (event.getAction() == MotionEvent.ACTION_UP) {
			calibrator.queueChessCapture();
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

	public void setNoTitle() {
		requestWindowFeature(Window.FEATURE_NO_TITLE);
	}

	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		MenuInflater inflater = getMenuInflater();
		inflater.inflate(R.menu.calibrationmenu, menu);
		return true;
	}

	@Override
	public boolean onOptionsItemSelected(MenuItem item) {
		switch (item.getItemId()) {
		case R.id.patternsize: {
			Intent sizer = new Intent(getApplicationContext(),
					ChessBoardChooser.class);
			startActivity(sizer);
		}
			break;
		case R.id.help:
			help();
			break;
		case R.id.calibrate:
			calibrate();
			break;
		case R.id.settings:
			Intent configurer = new Intent(getApplicationContext(),
					CameraConfig.class);
			startActivity(configurer);
			
		}

		return true;

	}

	private void help() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void onOptionsMenuClosed(Menu menu) {
		// TODO Auto-generated method stub
		super.onOptionsMenuClosed(menu);
	}

	private ServiceConnection mConnection = new ServiceConnection() {

		@Override
		public void onServiceDisconnected(ComponentName name) {

		}

		@Override
		public void onServiceConnected(ComponentName name, IBinder service) {

			CalibrationService calibservice = ((CalibrationService.CalibrationServiceBinder) service)
					.getService();
			if (!SDCardChecker.CheckStorage(Calibration.this))
				return;
			SDCardChecker.MakeDataDir(Calibration.this);

			File calibfile = SDCardChecker.getFile(calibservice,
					R.string.calibfile);
			try {

				Calibrator tcalib = calibrator;
				calibrator = new Calibrator(Calibration.this);
				setCallbackStack();
				calibservice.startCalibrating(Calibration.class, R.drawable.icon,tcalib, calibfile);
			} catch (IOException e) {
				e.printStackTrace();
			}

			// Tell the user about this for our demo.
			Toast.makeText(Calibration.this,
					"Starting calibration in the background.",
					Toast.LENGTH_SHORT).show();
			unbindService(this);
		}

	};

	public static File getCalibrationFile(Context ctx) {
		return SDCardChecker.getFile(ctx, R.string.calibfile);
	}

	void doBindCalibService() {
		// Establish a connection with the service. We use an explicit
		// class name because we want a specific service implementation that
		// we know will be running in our own process (and thus won't be
		// supporting component replacement by other applications).
		bindService(new Intent(Calibration.this, CalibrationService.class),
				mConnection, Context.BIND_AUTO_CREATE);
	}

	void calibrate() {
		if (calibrator.getNumberPatternsDetected() < 3) {
			Toast.makeText(this, getText(R.string.calibration_not_enough),
					Toast.LENGTH_LONG).show();
			return;
		}
		Intent calibservice = new Intent(Calibration.this,
				CalibrationService.class);
		startService(calibservice);
		doBindCalibService();

	}

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);

		setFullscreen();
		disableScreenTurnOff();
		setContentView(R.layout.camera);
		mPreview = (NativePreviewer) findViewById(R.id.nativepreviewer);
		mPreview.setPreviewSize(1000, 500);
		mPreview.setGrayscale(true);
		LinearLayout glview_layout = (LinearLayout) findViewById(R.id.glview_layout);
		glview = new GL2CameraViewer(getApplication(), false, 0, 0);
		glview_layout.addView(glview);
		calibrator = new Calibrator(this);

		ImageButton capturebutton = (ImageButton) findViewById(R.id.capture);
		capturebutton.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				calibrator.queueChessCapture();

			}
		});
		ImageButton calibbutton = (ImageButton) findViewById(R.id.calibrate);
		calibbutton.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				calibrate();
			}
		});
	}

	@Override
	protected void onDestroy() {
		super.onDestroy();
	}

	@Override
	protected void onPause() {
		super.onPause();

		mPreview.onPause();

		glview.onPause();

	}

	protected void setCallbackStack() {
		calibrator.setPatternSize(ChessBoardChooser.getPatternSize(this));

		LinkedList<NativeProcessor.PoolCallback> callbackstack = new LinkedList<NativeProcessor.PoolCallback>();
		callbackstack.add(calibrator);
		callbackstack.add(glview.getDrawCallback());
		mPreview.addCallbackStack(callbackstack);
		updateNumber(calibrator);
	}

	@Override
	protected void onResume() {
		super.onResume();
		int size[] ={0,0};
		CameraConfig.readImageSize(getApplicationContext(), size);
		int mode = CameraConfig.readCameraMode(getApplicationContext());
		mPreview.setPreviewSize(size[0], size[1]);
		mPreview.setGrayscale(mode == CameraConfig.CAMERA_MODE_BW ? true : false);
		
		glview.onResume();
		mPreview.onResume();
		setCallbackStack();

	}

	void updateNumber(Calibrator calibrator) {
		TextView numbertext = (TextView) findViewById(R.id.numberpatterns);
		int numdetectd = calibrator.getNumberPatternsDetected();
		if (numdetectd > 2) {
			numbertext
					.setTextColor(getResources().getColor(R.color.good_color));

		} else
			numbertext.setTextColor(getResources().getColor(R.color.bad_color));

		numbertext.setText(String.valueOf(numdetectd));

	}

	@Override
	public void onFoundChessboard(final Calibrator calibrator) {
		runOnUiThread(new Runnable() {

			@Override
			public void run() {
				Toast.makeText(Calibration.this,
						"Captured a calibration pattern!", Toast.LENGTH_SHORT)
						.show();
				updateNumber(calibrator);

			}
		});

	}

	@Override
	public void onDoneCalibration(Calibrator calibration, File calibfile) {

	}

	@Override
	public void onFailedChessboard(final Calibrator calibrator) {
		runOnUiThread(new Runnable() {

			@Override
			public void run() {
				Toast.makeText(
						Calibration.this,
						"No pattern found.  Make sure its the right dimensions, and close enough...",
						Toast.LENGTH_LONG).show();
				updateNumber(calibrator);

			}
		});

	}

}