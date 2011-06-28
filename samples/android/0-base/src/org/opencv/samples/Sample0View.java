package org.opencv.samples;

import java.util.List;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.hardware.Camera;
import android.hardware.Camera.PreviewCallback;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

class Sample0View extends SurfaceView implements SurfaceHolder.Callback, Runnable{
    private static final String TAG = "Sample0Base::View";
    
    private Camera camera;
    private SurfaceHolder holder;
    private int frame_width;
    private int frame_height;
    private byte[] frame;
    
    private boolean mThreadRun;
   

	public Sample0View(Context context) {
		super(context);
		holder = getHolder();
		holder.addCallback(this);
	}
	

	public void surfaceChanged(SurfaceHolder _holder, int format, int width, int height) {
        if ( camera != null) {
            Camera.Parameters params = camera.getParameters();
            List<Camera.Size> sizes = params.getSupportedPreviewSizes();
            frame_width = width;
            frame_height = height;
            
        	//selecting optimal camera preview size
            {
                double minDiff = Double.MAX_VALUE;
                for (Camera.Size size : sizes) {
                    if (Math.abs(size.height - height) < minDiff) {
                    	frame_width = size.width;
                    	frame_height = size.height;
                        minDiff = Math.abs(size.height - height);
                    }
                }
            }
            params.setPreviewSize(frame_width, frame_height);
            camera.setParameters(params);
        	camera.startPreview();
        }
    }

	public void surfaceCreated(SurfaceHolder holder) {
		camera = Camera.open();
		camera.setPreviewCallback(
			new PreviewCallback() {
				public void onPreviewFrame(byte[] data, Camera camera) {
					synchronized(Sample0View.this)
					{
						frame = data;
						Sample0View.this.notify();
					}
				}
			}
		);
    	(new Thread(this)).start();
	}

	public void surfaceDestroyed(SurfaceHolder holder) {
		mThreadRun = false;
		if(camera != null) {
			camera.stopPreview();
			camera.setPreviewCallback(null);
			camera.release();
			camera = null;
		}
	}

	public void run() {
		mThreadRun = true;
		Log.i(TAG, "Starting thread");
		while(mThreadRun) {
			byte[] data = null;
			synchronized(this)
			{
				try {
					this.wait();
					data = frame;
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			
			Canvas canvas = holder.lockCanvas();
			
			int frameSize = frame_width*frame_height;
			int[] rgba = new int[frameSize];
			
			Sample0Base a = (Sample0Base)getContext();
			int view_mode = a.view_mode;
			if(view_mode == Sample0Base.view_mode_gray) {
				for(int i = 0; i < frameSize; i++) {
					int y = (0xff & ((int)data[i]));
					rgba[i] = 0xff000000 + (y << 16) + (y << 8) + y;
				}
			}
			else if (view_mode == Sample0Base.view_mode_rgba) {
				for(int i = 0; i < frame_height; i++)
					for(int j = 0; j < frame_width; j++) {
						int y = (0xff & ((int)data[i*frame_width+j]));
						int u = (0xff & ((int)data[frameSize + (i >> 1) * frame_width + (j & ~1) + 0]));
						int v = (0xff & ((int)data[frameSize + (i >> 1) * frame_width + (j & ~1) + 1]));
						if (y < 16) y = 16;
						
						int r = Math.round(1.164f * (y - 16) + 1.596f * (v - 128)                     );
						int g = Math.round(1.164f * (y - 16) - 0.813f * (v - 128) - 0.391f * (u - 128));
						int b = Math.round(1.164f * (y - 16)                      + 2.018f * (u - 128));
						
						if (r < 0) r = 0; if (r > 255) r = 255;
						if (g < 0) g = 0; if (g > 255) g = 255;
						if (b < 0) b = 0; if (b > 255) b = 255;
						
						rgba[i*frame_width+j] = 0xff000000 + (b << 16) + (g << 8) + r;
					}
			}
			
			Bitmap bmp = Bitmap.createBitmap(frame_width, frame_height, Bitmap.Config.ARGB_8888);
			bmp.setPixels(rgba, 0/*offset*/, frame_width /*stride*/, 0, 0, frame_width, frame_height);
			
			canvas.drawBitmap(bmp, (canvas.getWidth()-frame_width)/2, (canvas.getHeight()-frame_height)/2, null);
			holder.unlockCanvasAndPost(canvas);
			}
	}
}