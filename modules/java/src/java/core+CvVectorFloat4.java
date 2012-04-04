package org.opencv.core;

public class CvVectorFloat4 extends CvVectorFloat {
    private static final int _ch = 4; //xxxC4

    public CvVectorFloat4() {
        super(_ch);
    }

    public CvVectorFloat4(long addr) {
        super(_ch, addr);
    }

    public CvVectorFloat4(Mat m) {
        super(_ch, m);
    }

    public CvVectorFloat4(float[] a) {
        super(_ch, a);
    }

}
