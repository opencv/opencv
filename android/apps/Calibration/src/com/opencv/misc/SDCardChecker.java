package com.opencv.misc;

import java.io.File;

import android.content.Context;
import android.os.Environment;
import android.widget.Toast;

import com.opencv.calibration.R;

public class SDCardChecker {

	public static File createThumb(Context ctx, File workingDir) {
		return new File(workingDir, "thumb.jpg");
	}

	public static File getDir(Context ctx, String relativename) {
		return new File(Environment.getExternalStorageDirectory()
				+ relativename);
	}

	public static File getDir(Context ctx, int id) {
		return new File(Environment.getExternalStorageDirectory()
				+ ctx.getResources().getString(id));
	}

	public static File getFile(Context ctx, int id) {
		return new File(Environment.getExternalStorageDirectory()
				+ ctx.getResources().getString(id));
	}

	public static void MakeDataDir(Context ctx) {
		File dir = getDir(ctx, R.string.sdcarddir);
		dir.mkdirs();
	}

	public static boolean CheckStorage(Context ctx) {
		boolean mExternalStorageAvailable = false;
		boolean mExternalStorageWriteable = false;
		String state = Environment.getExternalStorageState();

		if (Environment.MEDIA_MOUNTED.equals(state)) {
			// We can read and write the media
			mExternalStorageAvailable = mExternalStorageWriteable = true;
		} else if (Environment.MEDIA_MOUNTED_READ_ONLY.equals(state)) {
			// We can only read the media
			mExternalStorageAvailable = true;
			mExternalStorageWriteable = false;
		} else {
			// Something else is wrong. It may be one of many other states, but
			// all we need
			// to know is we can neither read nor write
			mExternalStorageAvailable = mExternalStorageWriteable = false;
		}
		boolean goodmount = mExternalStorageAvailable
				&& mExternalStorageWriteable;
		if (!goodmount) {
			Toast.makeText(ctx, ctx.getString(R.string.sdcard_error_msg),
					Toast.LENGTH_LONG).show();
		}
		return goodmount;
	}

}
