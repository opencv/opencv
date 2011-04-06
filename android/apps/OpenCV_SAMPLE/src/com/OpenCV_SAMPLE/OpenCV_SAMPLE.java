package com.OpenCV_SAMPLE;

import java.util.LinkedList;

import android.os.Bundle;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;

import com.OpenCV_SAMPLE.jni.CVSample;
import com.opencv.camera.CameraActivity;
import com.opencv.camera.NativeProcessor;
import com.opencv.camera.NativeProcessor.PoolCallback;
import com.opencv.jni.Mat;
import com.opencv.jni.image_pool;

public class OpenCV_SAMPLE extends CameraActivity {

	private int do_what = R.id.cv_menu_nothing;

	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		MenuInflater menu_flater = new MenuInflater(this);
		menu_flater.inflate(R.menu.sample_menu, menu);
		return true;
	}

	@Override
	public boolean onMenuItemSelected(int featureId, MenuItem item) {
		switch (item.getItemId()) {
		case R.id.cv_menu_blur:
		case R.id.cv_menu_canny:
		case R.id.cv_menu_invert:
		case R.id.cv_menu_nothing:
			do_what = item.getItemId();
			break;
		default:
			return false;

		}
		return true;
	}

	/** Called when the activity is first created. */
	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
	}

	@Override
	protected LinkedList<PoolCallback> getCallBackStack() {
		LinkedList<PoolCallback> list = new LinkedList<NativeProcessor.PoolCallback>();
		list.add(samplePoolCallback);
		return list;
	}

	CVSample cvsample = new CVSample();
	Mat canny = new Mat();
	PoolCallback samplePoolCallback = new PoolCallback() {

		@Override
		public void process(int idx, image_pool pool, long timestamp,
				NativeProcessor nativeProcessor) {
			Mat grey = pool.getGrey(idx);
			Mat color = pool.getImage(idx);
			Mat draw_img = color;
			switch (do_what) {
			case R.id.cv_menu_blur:
				cvsample.blur(draw_img, 5);
				break;
			case R.id.cv_menu_canny:
				cvsample.canny(grey, canny, 15);
				draw_img = canny;
				break;
			case R.id.cv_menu_invert:
				cvsample.invert(draw_img);
				break;
			case R.id.cv_menu_nothing:
				break;
			}
			pool.addImage(idx + 1, draw_img);
			glview.getDrawCallback().process(idx + 1, pool, timestamp,
					nativeProcessor);
		}
	};

}