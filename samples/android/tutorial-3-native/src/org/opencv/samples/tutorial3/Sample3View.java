package org.opencv.samples.tutorial3;

import android.content.Context;
import android.graphics.Bitmap;

class Sample3View extends SampleViewBase {
	
	private int mFrameSize;
	private Bitmap mBitmap;
	private int[] mRGBA;

    public Sample3View(Context context) {
        super(context);
    }

	@Override
	protected void onPreviewStarted(int previewWidtd, int previewHeight) {
		mFrameSize = previewWidtd * previewHeight;
		mRGBA = new int[mFrameSize];
		mBitmap = Bitmap.createBitmap(previewWidtd, previewHeight, Bitmap.Config.ARGB_8888);
	}

	@Override
	protected void onPreviewStopped() {
		if(mBitmap != null) {
			mBitmap.recycle();
			mBitmap = null;
		}
		mRGBA = null;
		
		
	}

    @Override
    protected Bitmap processFrame(byte[] data) {
        int[] rgba = mRGBA;

        FindFeatures(getFrameWidth(), getFrameHeight(), data, rgba);

        Bitmap bmp = mBitmap; 
        bmp.setPixels(rgba, 0/* offset */, getFrameWidth() /* stride */, 0, 0, getFrameWidth(), getFrameHeight());
        return bmp;
    }

    public native void FindFeatures(int width, int height, byte yuv[], int[] rgba);

    static {
        System.loadLibrary("native_sample");
    }
}
