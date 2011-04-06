package com.opencv.camera;

import java.util.LinkedList;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import android.graphics.PixelFormat;
import android.util.Log;

import com.opencv.jni.image_pool;
import com.opencv.jni.opencv;

/** The NativeProcessor is a native processing stack engine.
 * 
 * What this means is that the NativeProcessor handles loading
 * live camera frames into native memory space, i.e. the image_pool
 * and then calling a stack of PoolCallback's and passing them the
 * image_pool.
 * 
 * The image_pool index 0 is populated with the live video image
 * 
 * And any modifications to this the pool are in place, so you may
 * pass on changes to the pool to the next PoolCallback in the stack.
 *
 */
public class NativeProcessor {
	/** Users that would like to be able to have access to live video frames
	 * should implement a PoolCallback
	 * the idx and pool contain the images, specifically at idx == 0 is the
	 * live video frame.
	 */
	static public interface PoolCallback {
		void process(int idx, image_pool pool, long timestamp,
				NativeProcessor nativeProcessor);
	}

	

	/**At every frame, each PoolCallback is called in order and is passed the
	 * the same pool and index
	 * 
	 * @param stack  A list of PoolCallback objects, that will be called in order
	 */
	public void addCallbackStack(LinkedList<PoolCallback> stack) {

		try {
			while (!stacklock.tryLock(10, TimeUnit.MILLISECONDS)) {

			}
			try {
				nextStack = stack;
			} finally {
				stacklock.unlock();
			}
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();

		}

	}

	/**
	 * Create a NativeProcessor. The processor will not start running until
	 * start is called, at which point it will operate in its own thread and
	 * sleep until a post is called. The processor should not be started until
	 * an onSurfaceChange event, and should be shut down when the surface is
	 * destroyed by calling interupt.
	 * 
	 */
	public NativeProcessor() {
		gray_scale_only = false;
	}

	
	
	/** Grayscale only is much faster because the yuv does not get decoded, and grayscale is only one
	 * byter per pixel - giving fast opengl texture loading.
	 * 
	 * You still have access to the whole yuv image, but grayscale is only immediately available to
	 * use without further effort.
	 * 
	 * Suggestion - use grayscale only and save your yuv images to disk if you would like color images
	 * 
	 * Also, in grayscale mode, the images in the pool are only single channel, so please keep this in mind
	 * when accessing the color images - check the cv::Mat::channels() or cv::Mat::type() if your messing
	 * with color channels
	 * 
	 * @param grayscale true if you want to only process grayscale images
	 */
	public void setGrayscale(boolean grayscale){
		gray_scale_only = grayscale;
	}
	

	/**
	 * A callback that allows the NativeProcessor to pass back the buffer when
	 * it has completed processing a frame.
	 */
	static protected interface NativeProcessorCallback {
		/**
		 * Called after processing, meant to be recieved by the NativePreviewer
		 * wich reuses the byte buffer for the camera preview...
		 * 
		 * @param buffer
		 *            the buffer passed to the NativeProcessor with post.
		 */
		void onDoneNativeProcessing(byte[] buffer);
	}

	
	protected void stop() {
		mthread.interrupt();
		try {
			mthread.join();
		} catch (InterruptedException e) {
			Log.w("NativeProcessor",
					"interupted while stoping " + e.getMessage());
		}
		mthread = null;
	}

	protected void start() {
		mthread = new ProcessorThread();
		mthread.start();
	}
	/**
	 * post is used to notify the processor that a preview frame is ready, this
	 * will return almost immediately. if the processor is busy, returns false
	 * and is essentially a nop.
	 * 
	 * @param buffer
	 *            a preview frame from the Android Camera onPreviewFrame
	 *            callback
	 * @param width
	 *            of preview frame
	 * @param height
	 *            of preview frame
	 * @param format
	 *            of preview frame
	 * @return true if the processor wasn't busy and accepted the post, false if
	 *         the processor is still processing.
	 */

	protected boolean post(byte[] buffer, int width, int height, int format,
			long timestamp, NativeProcessorCallback callback) {

		lock.lock();
		try {
			NPPostObject pobj = new NPPostObject(buffer, width, height, format,
					timestamp, callback);
			postobjects.addFirst(pobj);
		} finally {
			lock.unlock();
		}
		return true;

	}
	
	private class ProcessorThread extends Thread {

		private void process(NPPostObject pobj) throws Exception {

			if (pobj.format == PixelFormat.YCbCr_420_SP) {
				// add as color image, because we know how to decode this
				opencv.addYUVtoPool(pool, pobj.buffer, 0, pobj.width,
						pobj.height, gray_scale_only);

			} else if (pobj.format == PixelFormat.YCbCr_422_SP) {
				// add as gray image, because this format is not coded
				// for...//TODO figure out how to decode this
				// format
				opencv.addYUVtoPool(pool, pobj.buffer, 0, pobj.width,
						pobj.height, true);
			} else
				throw new Exception("bad pixel format!");

			for (PoolCallback x : stack) {
				if (interrupted()) {
					throw new InterruptedException(
							"Native Processor interupted while processing");
				}
				x.process(0, pool, pobj.timestamp, NativeProcessor.this);
			}

			pobj.done(); // tell the postobject that we're done doing
							// all the processing.

		}

		@Override
		public void run() {

			try {
				while (true) {
					yield();

					while (!stacklock.tryLock(5, TimeUnit.MILLISECONDS)) {
					}
					try {
						if (nextStack != null) {
							stack = nextStack;
							nextStack = null;
						}
					} finally {
						stacklock.unlock();
					}

					NPPostObject pobj = null;

					while (!lock.tryLock(5, TimeUnit.MILLISECONDS)) {
					}
					try {
						if (postobjects.isEmpty())
							continue;
						pobj = postobjects.removeLast();

					} finally {
						lock.unlock();

					}

					if (interrupted())
						throw new InterruptedException();

					if (stack != null && pobj != null)
						process(pobj);

				}
			} catch (InterruptedException e) {

				Log.i("NativeProcessor",
						"native processor interupted, ending now");

			} catch (Exception e) {

				e.printStackTrace();
			} finally {

			}
		}

	}
	
	static private class NPPostObject {
		public NPPostObject(byte[] buffer, int width, int height, int format,
				long timestamp, NativeProcessorCallback callback) {
			this.buffer = buffer;
			this.width = width;
			this.height = height;
			this.format = format;
			this.timestamp = timestamp;
			this.callback = callback;
		}

		public void done() {
			callback.onDoneNativeProcessing(buffer);

		}

		int width, height;
		byte[] buffer;
		int format;
		long timestamp;
		NativeProcessorCallback callback;
	}


	private LinkedList<NPPostObject> postobjects = new LinkedList<NPPostObject>();

	private image_pool pool = new image_pool();

	private final Lock lock = new ReentrantLock();

	private LinkedList<PoolCallback> stack = new LinkedList<PoolCallback>();
	private boolean gray_scale_only;
	
	private Lock stacklock = new ReentrantLock();

	private LinkedList<PoolCallback> nextStack;
	
	private ProcessorThread mthread;

}