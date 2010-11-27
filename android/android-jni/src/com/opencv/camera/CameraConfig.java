package com.opencv.camera;

import com.opencv.R;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.SharedPreferences.Editor;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.AdapterView.OnItemSelectedListener;
import android.widget.Spinner;

public class CameraConfig extends Activity {
	public static final String CAMERA_SETTINGS = "CAMERA_SETTINGS";
	public static final String CAMERA_MODE = "camera_mode";
	public static final String IMAGE_WIDTH = "IMAGE_WIDTH";
	public static final String IMAGE_HEIGHT = "IMAGE_HEIGHT";
	public static final int CAMERA_MODE_BW = 0;
	public static final int CAMERA_MODE_COLOR = 1;

	public static int readCameraMode(Context ctx) {
		// Restore preferences
		SharedPreferences settings = ctx.getSharedPreferences(CAMERA_SETTINGS,
				0);
		int mode = settings.getInt(CAMERA_MODE, CAMERA_MODE_BW);
		return mode;
	}

	static public void setCameraMode(Context context, String mode) {
		int m = 0;
		if (mode.equals("BW")) {
			m = CAMERA_MODE_BW;
		} else if (mode.equals("color"))
			m = CAMERA_MODE_COLOR;
		setCameraMode(context, m);
	}

	private static String sizeToString(int[] size) {
		return size[0] + "x" + size[1];
	}

	private static void parseStrToSize(String ssize, int[] size) {
		String sz[] = ssize.split("x");
		size[0] = Integer.valueOf(sz[0]);
		size[1] = Integer.valueOf(sz[1]);
	}

	public static void readImageSize(Context ctx, int[] size) {
		// Restore preferences
		SharedPreferences settings = ctx.getSharedPreferences(CAMERA_SETTINGS,
				0);
		size[0] = settings.getInt(IMAGE_WIDTH, 600);
		size[1] = settings.getInt(IMAGE_HEIGHT, 600);

	}

	public static void setCameraMode(Context ctx, int mode) {
		// Restore preferences
		SharedPreferences settings = ctx.getSharedPreferences(CAMERA_SETTINGS,
				0);
		Editor editor = settings.edit();
		editor.putInt(CAMERA_MODE, mode);
		editor.commit();
	}

	public static void setImageSize(Context ctx, String strsize) {
		int size[] = { 0, 0 };
		parseStrToSize(strsize, size);
		setImageSize(ctx, size[0], size[1]);
	}

	public static void setImageSize(Context ctx, int width, int height) {
		// Restore preferences
		SharedPreferences settings = ctx.getSharedPreferences(CAMERA_SETTINGS,
				0);
		Editor editor = settings.edit();
		editor.putInt(IMAGE_WIDTH, width);
		editor.putInt(IMAGE_HEIGHT, height);
		editor.commit();
	}

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		// TODO Auto-generated method stub
		super.onCreate(savedInstanceState);
		setContentView(R.layout.camerasettings);
		int mode = readCameraMode(this);
		int size[] = { 0, 0 };
		readImageSize(this, size);

		final Spinner size_spinner;
		final Spinner mode_spinner;
		size_spinner = (Spinner) findViewById(R.id.image_size);
		mode_spinner = (Spinner) findViewById(R.id.camera_mode);

		String strsize = sizeToString(size);
		String strmode = modeToString(mode);

		String sizes[] = getResources().getStringArray(R.array.image_sizes);

		int i = 1;
		for (String x : sizes) {
			if (x.equals(strsize))
				break;
			i++;
		}
		if(i <= sizes.length)
			size_spinner.setSelection(i-1);

		i = 1;
		String modes[] =  getResources().getStringArray(R.array.camera_mode);
		for (String x :modes) {
			if (x.equals(strmode))
				break;
			i++;
		}
		if(i <= modes.length)
			mode_spinner.setSelection(i-1);

		size_spinner.setOnItemSelectedListener(new OnItemSelectedListener() {

			@Override
			public void onItemSelected(AdapterView<?> arg0, View spinner,
					int position, long arg3) {
				Object o = size_spinner.getItemAtPosition(position);
				if (o != null)
					setImageSize(spinner.getContext(), (String) o);
			}

			@Override
			public void onNothingSelected(AdapterView<?> arg0) {

			}
		});
		mode_spinner.setOnItemSelectedListener(new OnItemSelectedListener() {

			@Override
			public void onItemSelected(AdapterView<?> arg0, View spinner,
					int position, long arg3) {
				Object o = mode_spinner.getItemAtPosition(position);
				if (o != null)
					setCameraMode(spinner.getContext(), (String) o);

			}

			@Override
			public void onNothingSelected(AdapterView<?> arg0) {

			}
		});

	}

	private String modeToString(int mode) {
		switch (mode) {
		case CAMERA_MODE_BW:
			return "BW";
		case CAMERA_MODE_COLOR:
			return "color";
		default:
			return "";
		}
	}
}
