package com.theveganrobot.cvcamera;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.LinkedList;
import java.util.Scanner;

import android.app.Activity;
import android.app.AlertDialog;
import android.app.Dialog;
import android.app.ProgressDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.Gravity;
import android.view.KeyEvent;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup.LayoutParams;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageButton;
import android.widget.LinearLayout;
import android.widget.Toast;

import com.opencv.camera.CameraConfig;
import com.opencv.camera.NativePreviewer;
import com.opencv.camera.NativeProcessor;
import com.opencv.camera.NativeProcessor.PoolCallback;
import com.opencv.jni.image_pool;
import com.opencv.opengl.GL2CameraViewer;
import com.theveganrobot.cvcamera.jni.Processor;
import com.theveganrobot.cvcamera.jni.cvcamera;

public class CVCamera extends Activity {

	static final int DIALOG_CALIBRATING = 0;
	static final int DIALOG_CALIBRATION_FILE = 1;
	private static final int DIALOG_OPENING_TUTORIAL = 2;
	private static final int DIALOG_TUTORIAL_FAST = 3;
	private static final int DIALOG_TUTORIAL_SURF = 4;
	private static final int DIALOG_TUTORIAL_STAR = 5;
	private static final int DIALOG_TUTORIAL_CHESS = 6;
	private boolean captureChess;

	ProgressDialog makeCalibDialog() {
		ProgressDialog progressDialog;
		progressDialog = new ProgressDialog(this);
		progressDialog.setMessage("Callibrating. Please wait...");
		progressDialog.setCancelable(false);

		return progressDialog;
	}

	void toasts(int id) {
		switch (id) {
		case DIALOG_OPENING_TUTORIAL:
			Toast.makeText(this, "Try clicking the menu for CV options.",
					Toast.LENGTH_LONG).show();
			break;
		case DIALOG_TUTORIAL_FAST:
			Toast.makeText(this, "Detecting and Displaying FAST features",
					Toast.LENGTH_LONG).show();
			break;
		case DIALOG_TUTORIAL_SURF:
			Toast.makeText(this, "Detecting and Displaying SURF features",
					Toast.LENGTH_LONG).show();
			break;
		case DIALOG_TUTORIAL_STAR:
			Toast.makeText(this, "Detecting and Displaying STAR features",
					Toast.LENGTH_LONG).show();
			break;
		case DIALOG_TUTORIAL_CHESS:
			Toast.makeText(
					this,
					"Calibration Mode, Point at a chessboard pattern and press the camera button, space,"
							+ "or the DPAD to capture.", Toast.LENGTH_LONG)
					.show();
			break;

		default:
			break;
		}

	}

	@Override
	protected Dialog onCreateDialog(int id) {
		Dialog dialog;
		switch (id) {
		case DIALOG_CALIBRATING:
			dialog = makeCalibDialog();
			break;

		case DIALOG_CALIBRATION_FILE:
			dialog = makeCalibFileAlert();
			break;
		default:
			dialog = null;
		}
		return dialog;
	}

	private Dialog makeCalibFileAlert() {
		AlertDialog.Builder builder = new AlertDialog.Builder(this);
		builder.setMessage(calib_text)
				.setTitle("camera.yml at " + calib_file_loc)
				.setCancelable(false)
				.setPositiveButton("Ok", new DialogInterface.OnClickListener() {
					public void onClick(DialogInterface dialog, int id) {

					}
				});
		AlertDialog alert = builder.create();
		return alert;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see android.app.Activity#onKeyUp(int, android.view.KeyEvent)
	 */
	@Override
	public boolean onKeyUp(int keyCode, KeyEvent event) {

		switch (keyCode) {
		case KeyEvent.KEYCODE_CAMERA:
		case KeyEvent.KEYCODE_SPACE:
		case KeyEvent.KEYCODE_DPAD_CENTER:
			captureChess = true;
			return true;

		default:
			return super.onKeyUp(keyCode, event);
		}

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see android.app.Activity#onKeyLongPress(int, android.view.KeyEvent)
	 */
	@Override
	public boolean onKeyLongPress(int keyCode, KeyEvent event) {

		return super.onKeyLongPress(keyCode, event);
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
		menu.add("FAST");
		menu.add("STAR");
		menu.add("SURF");
		menu.add("Chess");
		menu.add("Settings");
		return true;
	}

	private NativePreviewer mPreview;
	private GL2CameraViewer glview;

	@Override
	public boolean onOptionsItemSelected(MenuItem item) {
		LinkedList<PoolCallback> defaultcallbackstack = new LinkedList<PoolCallback>();
		defaultcallbackstack.addFirst(glview.getDrawCallback());
		if (item.getTitle().equals("FAST")) {

			defaultcallbackstack.addFirst(new FastProcessor());
			toasts(DIALOG_TUTORIAL_FAST);
		}

		else if (item.getTitle().equals("Chess")) {

			defaultcallbackstack.addFirst(new CalibrationProcessor());
			toasts(DIALOG_TUTORIAL_CHESS);

		}

		else if (item.getTitle().equals("STAR")) {

			defaultcallbackstack.addFirst(new STARProcessor());
			toasts(DIALOG_TUTORIAL_STAR);

		}

		else if (item.getTitle().equals("SURF")) {

			defaultcallbackstack.addFirst(new SURFProcessor());
			toasts(DIALOG_TUTORIAL_SURF);

		}

		else if (item.getTitle().equals("Settings")) {

			Intent intent = new Intent(this,CameraConfig.class);
			startActivity(intent);
		}

		mPreview.addCallbackStack(defaultcallbackstack);
		
		return true;
	}

	@Override
	public void onOptionsMenuClosed(Menu menu) {
		// TODO Auto-generated method stub
		super.onOptionsMenuClosed(menu);
	}

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);

		setFullscreen();
		disableScreenTurnOff();

		FrameLayout frame = new FrameLayout(this);

		// Create our Preview view and set it as the content of our activity.
		mPreview = new NativePreviewer(getApplication(), 640, 480);

		LayoutParams params = new LayoutParams(LayoutParams.WRAP_CONTENT,
				LayoutParams.WRAP_CONTENT);
		params.height = getWindowManager().getDefaultDisplay().getHeight();
		params.width = (int) (params.height * 4.0 / 2.88);

		LinearLayout vidlay = new LinearLayout(getApplication());

		vidlay.setGravity(Gravity.CENTER);
		vidlay.addView(mPreview, params);
		frame.addView(vidlay);

		// make the glview overlay ontop of video preview
		mPreview.setZOrderMediaOverlay(false);

		glview = new GL2CameraViewer(getApplication(), false, 0, 0);
		glview.setZOrderMediaOverlay(true);

		LinearLayout gllay = new LinearLayout(getApplication());

		gllay.setGravity(Gravity.CENTER);
		gllay.addView(glview, params);
		frame.addView(gllay);

		ImageButton capture_button = new ImageButton(getApplicationContext());
		capture_button.setImageDrawable(getResources().getDrawable(
				android.R.drawable.ic_menu_camera));
		capture_button.setLayoutParams(new LayoutParams(
				LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT));
		capture_button.setOnClickListener(new View.OnClickListener() {

			@Override
			public void onClick(View v) {
				captureChess = true;

			}
		});

		LinearLayout buttons = new LinearLayout(getApplicationContext());
		buttons.setLayoutParams(new LayoutParams(LayoutParams.WRAP_CONTENT,
				LayoutParams.WRAP_CONTENT));

		buttons.addView(capture_button);

		Button focus_button = new Button(getApplicationContext());
		focus_button.setLayoutParams(new LayoutParams(
				LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT));
		focus_button.setText("Focus");
		focus_button.setOnClickListener(new View.OnClickListener() {

			@Override
			public void onClick(View v) {
				mPreview.postautofocus(100);
			}
		});
		buttons.addView(focus_button);

		frame.addView(buttons);
		setContentView(frame);
		toasts(DIALOG_OPENING_TUTORIAL);
	}

	@Override
	public boolean onTrackballEvent(MotionEvent event) {
		if (event.getAction() == MotionEvent.ACTION_UP) {
			captureChess = true;
			return true;
		}
		return super.onTrackballEvent(event);
	}

	@Override
	protected void onPause() {
		super.onPause();

		// clears the callback stack
		mPreview.onPause();

		glview.onPause();

	}

	@Override
	protected void onResume() {
		super.onResume();
		glview.onResume();
		mPreview.setParamsFromPrefs(getApplicationContext());
		// add an initiall callback stack to the preview on resume...
		// this one will just draw the frames to opengl
		LinkedList<NativeProcessor.PoolCallback> cbstack = new LinkedList<PoolCallback>();
		cbstack.add(glview.getDrawCallback());
		mPreview.addCallbackStack(cbstack);
		mPreview.onResume();

	}

	// final processor so taht these processor callbacks can access it
	final Processor processor = new Processor();

	class FastProcessor implements NativeProcessor.PoolCallback {

		@Override
		public void process(int idx, image_pool pool, long timestamp,
				NativeProcessor nativeProcessor) {
			processor.detectAndDrawFeatures(idx, pool, cvcamera.DETECT_FAST);

		}

	}

	class STARProcessor implements NativeProcessor.PoolCallback {

		@Override
		public void process(int idx, image_pool pool, long timestamp,
				NativeProcessor nativeProcessor) {
			processor.detectAndDrawFeatures(idx, pool, cvcamera.DETECT_STAR);

		}

	}

	class SURFProcessor implements NativeProcessor.PoolCallback {

		@Override
		public void process(int idx, image_pool pool, long timestamp,
				NativeProcessor nativeProcessor) {
			processor.detectAndDrawFeatures(idx, pool, cvcamera.DETECT_SURF);

		}

	}

	String calib_text = null;
	String calib_file_loc = null;

	class CalibrationProcessor implements NativeProcessor.PoolCallback {

		boolean calibrated = false;

		@Override
		public void process(int idx, image_pool pool, long timestamp,
				NativeProcessor nativeProcessor) {

			if (calibrated) {
				processor.drawText(idx, pool, "Calibrated successfully");
				return;
			}
			if (processor.getNumberDetectedChessboards() == 10) {

				File opencvdir = new File(
						Environment.getExternalStorageDirectory(), "opencv");
				if (!opencvdir.exists()) {
					opencvdir.mkdir();
				}
				File calibfile = new File(opencvdir, "camera.yml");

				calib_file_loc = calibfile.getAbsolutePath();
				processor.calibrate(calibfile.getAbsolutePath());
				Log.i("chessboard", "calibrated");
				calibrated = true;
				processor.resetChess();
				runOnUiThread(new Runnable() {

					@Override
					public void run() {
						removeDialog(DIALOG_CALIBRATING);

					}
				});

				try {

					StringBuilder text = new StringBuilder();
					String NL = System.getProperty("line.separator");
					Scanner scanner = new Scanner(calibfile);

					try {
						while (scanner.hasNextLine()) {
							text.append(scanner.nextLine() + NL);
						}
					} finally {
						scanner.close();
					}

					calib_text = text.toString();

					runOnUiThread(new Runnable() {

						@Override
						public void run() {
							showDialog(DIALOG_CALIBRATION_FILE);

						}
					});

				} catch (FileNotFoundException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}

			} else if (captureChess
					&& processor.detectAndDrawChessboard(idx, pool)) {

				runOnUiThread(new Runnable() {

					String numchess = String.valueOf(processor
							.getNumberDetectedChessboards());

					@Override
					public void run() {
						Toast.makeText(CVCamera.this,
								"Detected " + numchess + " of 10 chessboards",
								Toast.LENGTH_SHORT).show();

					}
				});
				Log.i("cvcamera",
						"detected a chessboard, n chess boards found: "
								+ String.valueOf(processor
										.getNumberDetectedChessboards()));

			}

			captureChess = false;

			if (processor.getNumberDetectedChessboards() == 10) {
				runOnUiThread(new Runnable() {

					@Override
					public void run() {
						showDialog(DIALOG_CALIBRATING);

					}
				});

				processor.drawText(idx, pool, "Calibrating, please wait.");
			}
			if (processor.getNumberDetectedChessboards() < 10) {

				processor.drawText(idx, pool,
						"found " + processor.getNumberDetectedChessboards()
								+ "/10 chessboards");
			}

		}

	}

}
