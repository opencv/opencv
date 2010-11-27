package com.opencv.calibration;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import android.app.Activity;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.widget.TextView;

import com.opencv.R;

public class CalibrationViewer extends Activity {

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		// TODO Auto-generated method stub
		super.onCreate(savedInstanceState);
		setContentView(R.layout.calibrationviewer);

		Bundle extras = getIntent().getExtras();
		String filename = extras.getString("calibfile");
		if (filename != null) {
			TextView text = (TextView) findViewById(R.id.calibtext);
			text.setMovementMethod(new ScrollingMovementMethod());
			try {
				BufferedReader reader = new BufferedReader(new FileReader(
						filename));
				while (reader.ready()) {
					text.append(reader.readLine() +"\n");
				}

			} catch (FileNotFoundException e) {
				Log.e("opencv", "could not open calibration file at:"
						+ filename);
			} catch (IOException e) {
				Log.e("opencv", "error reading file: "
						+ filename);
			}
		}

	}

}
