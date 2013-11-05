package org.opencv.highgui;

import java.util.List;
import java.util.LinkedList;

import org.opencv.core.Mat;
import org.opencv.core.Size;

// C++: class VideoCapture
//javadoc: VideoCapture
public class VideoCapture {

    protected final long nativeObj;

    protected VideoCapture(long addr) {
        nativeObj = addr;
    }

    //
    // C++: VideoCapture::VideoCapture()
    //

    // javadoc: VideoCapture::VideoCapture()
    public VideoCapture()
    {

        nativeObj = n_VideoCapture();

        return;
    }

    //
    // C++: VideoCapture::VideoCapture(int device)
    //

    // javadoc: VideoCapture::VideoCapture(device)
    public VideoCapture(int device)
    {

        nativeObj = n_VideoCapture(device);

        return;
    }

    //
    // C++: double VideoCapture::get(int propId)
    //

/**
 * Returns the specified "VideoCapture" property.
 *
 * Note: When querying a property that is not supported by the backend used by
 * the "VideoCapture" class, value 0 is returned.
 *
 * @param propId property identifier; it can be one of the following:
 *   * CV_CAP_PROP_FRAME_WIDTH width of the frames in the video stream.
 *   * CV_CAP_PROP_FRAME_HEIGHT height of the frames in the video stream.
 *
 * @see <a href="http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get">org.opencv.highgui.VideoCapture.get</a>
 */
    public double get(int propId)
    {

        double retVal = n_get(nativeObj, propId);

        return retVal;
    }

    public List<Size> getSupportedPreviewSizes()
    {
        String[] sizes_str = n_getSupportedPreviewSizes(nativeObj).split(",");
        List<Size> sizes = new LinkedList<Size>();

        for (String str : sizes_str) {
            String[] wh = str.split("x");
            sizes.add(new Size(Double.parseDouble(wh[0]), Double.parseDouble(wh[1])));
        }

        return sizes;
    }

    //
    // C++: bool VideoCapture::grab()
    //

    // javadoc: VideoCapture::grab()
    public boolean grab()
    {

        boolean retVal = n_grab(nativeObj);

        return retVal;
    }

    //
    // C++: bool VideoCapture::isOpened()
    //

    // javadoc: VideoCapture::isOpened()
    public boolean isOpened()
    {

        boolean retVal = n_isOpened(nativeObj);

        return retVal;
    }

    //
    // C++: bool VideoCapture::open(int device)
    //

    // javadoc: VideoCapture::open(device)
    public boolean open(int device)
    {

        boolean retVal = n_open(nativeObj, device);

        return retVal;
    }

    //
    // C++: bool VideoCapture::read(Mat image)
    //

    // javadoc: VideoCapture::read(image)
    public boolean read(Mat image)
    {

        boolean retVal = n_read(nativeObj, image.nativeObj);

        return retVal;
    }

    //
    // C++: void VideoCapture::release()
    //

    // javadoc: VideoCapture::release()
    public void release()
    {

        n_release(nativeObj);

        return;
    }

    //
    // C++: bool VideoCapture::retrieve(Mat image, int channel = 0)
    //

    // javadoc: VideoCapture::retrieve(image, channel)
    public boolean retrieve(Mat image, int channel)
    {

        boolean retVal = n_retrieve(nativeObj, image.nativeObj, channel);

        return retVal;
    }

    // javadoc: VideoCapture::retrieve(image)
    public boolean retrieve(Mat image)
    {

        boolean retVal = n_retrieve(nativeObj, image.nativeObj);

        return retVal;
    }

    //
    // C++: bool VideoCapture::set(int propId, double value)
    //

/**
 * Sets a property in the "VideoCapture".
 *
 * @param propId property identifier; it can be one of the following:
 *   * CV_CAP_PROP_FRAME_WIDTH width of the frames in the video stream.
 *   * CV_CAP_PROP_FRAME_HEIGHT height of the frames in the video stream.
 * @param value value of the property.
 *
 * @see <a href="http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-set">org.opencv.highgui.VideoCapture.set</a>
 */
    public boolean set(int propId, double value)
    {

        boolean retVal = n_set(nativeObj, propId, value);

        return retVal;
    }

    @Override
    protected void finalize() throws Throwable {
        n_delete(nativeObj);
        super.finalize();
    }

    // C++: VideoCapture::VideoCapture()
    private static native long n_VideoCapture();

    // C++: VideoCapture::VideoCapture(string filename)
    private static native long n_VideoCapture(java.lang.String filename);

    // C++: VideoCapture::VideoCapture(int device)
    private static native long n_VideoCapture(int device);

    // C++: double VideoCapture::get(int propId)
    private static native double n_get(long nativeObj, int propId);

    // C++: bool VideoCapture::grab()
    private static native boolean n_grab(long nativeObj);

    // C++: bool VideoCapture::isOpened()
    private static native boolean n_isOpened(long nativeObj);

    // C++: bool VideoCapture::open(string filename)
    private static native boolean n_open(long nativeObj, java.lang.String filename);

    // C++: bool VideoCapture::open(int device)
    private static native boolean n_open(long nativeObj, int device);

    // C++: bool VideoCapture::read(Mat image)
    private static native boolean n_read(long nativeObj, long image_nativeObj);

    // C++: void VideoCapture::release()
    private static native void n_release(long nativeObj);

    // C++: bool VideoCapture::retrieve(Mat image, int channel = 0)
    private static native boolean n_retrieve(long nativeObj, long image_nativeObj, int channel);

    private static native boolean n_retrieve(long nativeObj, long image_nativeObj);

    // C++: bool VideoCapture::set(int propId, double value)
    private static native boolean n_set(long nativeObj, int propId, double value);

    private static native String n_getSupportedPreviewSizes(long nativeObj);

    // native support for java finalize()
    private static native void n_delete(long nativeObj);

}
