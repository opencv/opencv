package com.opencv.utils;

import java.nio.ByteBuffer;

import com.opencv.jni.Mat;
import com.opencv.jni.Size;
import com.opencv.jni.opencv;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;

public class BitmapBridge {
	static void copyBitmap(Bitmap bmap, Mat mat) throws Exception {
		if ((bmap.getConfig() == null) || bmap.getConfig() == Config.ARGB_8888)
			throw new Exception("bad config");
		Size sz = new Size(bmap.getWidth(), bmap.getHeight());
		mat.create(sz, opencv.CV_8UC4);
		ByteBuffer buffer = ByteBuffer.allocate(4 * bmap.getWidth()
				* bmap.getHeight());
		bmap.copyPixelsToBuffer(buffer);
		opencv.copyBufferToMat(mat, buffer);

	}

	static Bitmap matToBitmap(Mat mat) {
		Bitmap bmap = Bitmap.createBitmap(mat.getCols(), mat.getRows(),
				Config.ARGB_8888);
		ByteBuffer buffer = ByteBuffer.allocate(4 * bmap.getWidth()
				* bmap.getHeight());
		opencv.copyMatToBuffer(buffer, mat);
		bmap.copyPixelsFromBuffer(buffer);
		return bmap;
	}
}
