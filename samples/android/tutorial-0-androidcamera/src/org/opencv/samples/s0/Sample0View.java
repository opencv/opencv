package org.opencv.samples.tutorial0;

import android.content.Context;
import android.graphics.Bitmap;

class Sample0View extends SampleViewBase {
    public Sample0View(Context context) {
        super(context);
    }

    @Override
    protected Bitmap processFrame(byte[] data) {
        int frameSize = getFrameWidth() * getFrameHeight();
        int[] rgba = new int[frameSize];

        int view_mode = Sample0Base.viewMode;
        if (view_mode == Sample0Base.VIEW_MODE_GRAY) {
            for (int i = 0; i < frameSize; i++) {
                int y = (0xff & ((int) data[i]));
                rgba[i] = 0xff000000 + (y << 16) + (y << 8) + y;
            }
        } else if (view_mode == Sample0Base.VIEW_MODE_RGBA) {
            for (int i = 0; i < getFrameHeight(); i++)
                for (int j = 0; j < getFrameWidth(); j++) {
                    int y = (0xff & ((int) data[i * getFrameWidth() + j]));
                    int u = (0xff & ((int) data[frameSize + (i >> 1) * getFrameWidth() + (j & ~1) + 0]));
                    int v = (0xff & ((int) data[frameSize + (i >> 1) * getFrameWidth() + (j & ~1) + 1]));
                    y = y < 16 ? 16 : y;

                    int r = Math.round(1.164f * (y - 16) + 1.596f * (v - 128));
                    int g = Math.round(1.164f * (y - 16) - 0.813f * (v - 128) - 0.391f * (u - 128));
                    int b = Math.round(1.164f * (y - 16) + 2.018f * (u - 128));

                    r = r < 0 ? 0 : (r > 255 ? 255 : r);
                    g = g < 0 ? 0 : (g > 255 ? 255 : g);
                    b = b < 0 ? 0 : (b > 255 ? 255 : b);

                    rgba[i * getFrameWidth() + j] = 0xff000000 + (b << 16) + (g << 8) + r;
                }
        }

        Bitmap bmp = Bitmap.createBitmap(getFrameWidth(), getFrameHeight(), Bitmap.Config.ARGB_8888);
        bmp.setPixels(rgba, 0/* offset */, getFrameWidth() /* stride */, 0, 0, getFrameWidth(), getFrameHeight());
        return bmp;
    }
}