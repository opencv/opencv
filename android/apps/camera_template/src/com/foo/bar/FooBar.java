package com.foo.bar;

import java.util.LinkedList;

import android.app.Activity;
import android.app.AlertDialog;
import android.app.Dialog;
import android.content.DialogInterface;
import android.os.Bundle;
import android.util.Log;
import android.view.Gravity;
import android.view.KeyEvent;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.Window;
import android.view.WindowManager;
import android.view.ViewGroup.LayoutParams;
import android.widget.FrameLayout;
import android.widget.LinearLayout;

//make sure you have the OpenCV project open, so that the
//android-sdk can find it!
import com.foo.bar.jni.BarBar;
import com.foo.bar.jni.FooBarStruct;
import com.opencv.camera.NativePreviewer;
import com.opencv.camera.NativeProcessor;
import com.opencv.camera.NativeProcessor.PoolCallback;
import com.opencv.jni.image_pool;
import com.opencv.opengl.GL2CameraViewer;

public class FooBar extends Activity {

	private final int FOOBARABOUT = 0;

	@Override
	protected Dialog onCreateDialog(int id) {
		Dialog dialog;
		switch (id) {
		case FOOBARABOUT:
			AlertDialog.Builder builder = new AlertDialog.Builder(this);
			builder.setTitle(R.string.about_title);
			builder.setMessage(R.string.about_str);
			builder.setPositiveButton(R.string.ok,
					new DialogInterface.OnClickListener() {

						@Override
						public void onClick(DialogInterface dialog, int which) {
							dismissDialog(FOOBARABOUT);

						}
					});
			dialog = builder.create();
		default:
			dialog = null;
		}
		return dialog;
	}

	/*
	 * Handle the capture button as follows...
	 */
	@Override
	public boolean onKeyUp(int keyCode, KeyEvent event) {

		switch (keyCode) {
		case KeyEvent.KEYCODE_CAMERA:
		case KeyEvent.KEYCODE_SPACE:
		case KeyEvent.KEYCODE_DPAD_CENTER:
			// capture button pressed here
			return true;

		default:
			return super.onKeyUp(keyCode, event);
		}

	}

	/*
	 * Handle the capture button as follows... On some phones there is no
	 * capture button, only trackball
	 */
	@Override
	public boolean onTrackballEvent(MotionEvent event) {
		if (event.getAction() == MotionEvent.ACTION_UP) {
			// capture button pressed
			return true;
		}
		return super.onTrackballEvent(event);
	}

	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		menu.add(R.string.about_menu);
		return true;
	}

	private NativePreviewer mPreview;
	private GL2CameraViewer glview;

	@Override
	public boolean onOptionsItemSelected(MenuItem item) {
		// example menu
		if (item.getTitle().equals(
				getResources().getString(R.string.about_menu))) {
			showDialog(FOOBARABOUT);
		}
		return true;
	}

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);

		requestWindowFeature(Window.FEATURE_NO_TITLE);
		getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
				WindowManager.LayoutParams.FLAG_FULLSCREEN);

		FrameLayout frame = new FrameLayout(this);

		// Create our Preview view and set it as the content of our activity.
		mPreview = new NativePreviewer(getApplication(), 300, 300);

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
		glview.setLayoutParams(new LayoutParams(LayoutParams.FILL_PARENT,
				LayoutParams.FILL_PARENT));
		frame.addView(glview);

		setContentView(frame);

	}

	@Override
	protected void onPause() {
		super.onPause();

		// IMPORTANT
		// must tell the NativePreviewer of a pause
		// and the glview - so that they can release resources and start back up
		// properly
		// failing to do this will cause the application to crash with no
		// warning
		// on restart
		// clears the callback stack
		mPreview.onPause();

		glview.onPause();

	}

	@Override
	protected void onResume() {
		super.onResume();

		// resume the opengl viewer first
		glview.onResume();

		// add an initial callback stack to the preview on resume...
		// this one will just draw the frames to opengl
		LinkedList<NativeProcessor.PoolCallback> cbstack = new LinkedList<PoolCallback>();

		// SpamProcessor will be called first
		cbstack.add(new SpamProcessor());

		// then the same idx and pool will be passed to
		// the glview callback -
		// so operate on the image at idx, and modify, and then
		// it will be drawn in the glview
		// or remove this, and call glview manually in SpamProcessor
		// cbstack.add(glview.getDrawCallback());

		mPreview.addCallbackStack(cbstack);
		mPreview.onResume();

	}

	class SpamProcessor implements NativeProcessor.PoolCallback {

		FooBarStruct foo = new FooBarStruct();
		BarBar barbar = new BarBar();

		@Override
		public void process(int idx, image_pool pool, long timestamp,
				NativeProcessor nativeProcessor) {

			// example of using the jni generated FoobarStruct;
			int nImages = foo.pool_image_count(pool);
			Log.i("foobar", "Number of images in pool: " + nImages);

			// call a function - this function does absolutely nothing!
			barbar.crazy();

			// sample processor
			// this gets called every frame in the order of the list
			// first add to the callback stack linked list will be the
			// first called
			// the idx and pool may be used to get the cv::Mat
			// that is the latest frame being passed.
			// pool.getClass(idx)

			// these are what the glview.getDrawCallback() calls
			glview.drawMatToGL(idx, pool);
			glview.requestRender();

		}

	}

}
