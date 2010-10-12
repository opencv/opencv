package com.opencv.calibration;

import com.opencv.jni.Size;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.SharedPreferences.Editor;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.AdapterView.OnItemSelectedListener;
import android.widget.Spinner;

public class ChessBoardChooser extends Activity {
	public static final String CHESS_SIZE = "chess_size";
	public static final int DEFAULT_WIDTH = 6;
	public static final int DEFAULT_HEIGHT = 8;
	public static final int LOWEST = 3;

	class DimChooser implements OnItemSelectedListener {
		private String dim;

		public DimChooser(String dim) {
			this.dim = dim;
		}

		@Override
		public void onItemSelected(AdapterView<?> arg0, View arg1, int pos,
				long arg3) {
			SharedPreferences settings = getSharedPreferences(CHESS_SIZE, 0);
			Editor editor = settings.edit();
			editor.putInt(dim, pos + LOWEST);
			editor.commit();
		}

		@Override
		public void onNothingSelected(AdapterView<?> arg0) {
		}
	}

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		// TODO Auto-generated method stub
		super.onCreate(savedInstanceState);
		setContentView(R.layout.chesssizer);
		// Restore preferences
		SharedPreferences settings = getSharedPreferences(CHESS_SIZE, 0);
		int width = settings.getInt("width", 6);

		int height = settings.getInt("height", 8);

		Spinner wspin, hspin;
		wspin = (Spinner) findViewById(R.id.rows);
		hspin = (Spinner) findViewById(R.id.cols);

		wspin.setSelection(width - LOWEST);
		hspin.setSelection(height - LOWEST);

		wspin.setOnItemSelectedListener(new DimChooser("width"));
		hspin.setOnItemSelectedListener(new DimChooser("height"));

	}

	public static Size getPatternSize(Context ctx) {
		SharedPreferences settings = ctx.getSharedPreferences(CHESS_SIZE, 0);
		int width = settings.getInt("width", 6);

		int height = settings.getInt("height", 8);

		return new Size(width, height);
	}

}
