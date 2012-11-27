package org.opencv.samples.tutorial5;

import org.opencv.android.JavaCameraView;

import android.content.Context;
import android.util.AttributeSet;

public class CustomJavaCameraView extends JavaCameraView {

	public CustomJavaCameraView(Context context, AttributeSet attrs) {
		super(context, attrs);
	}
	
    @Override
    protected boolean connectCamera(int width, int height) {
    	boolean result = super.connectCamera(width, height);
    	return result;
    }
}
