package org.opencv.core;

public class CvVectorFloat6 extends CvVectorFloat {
    private static final int _ch = 6; //xxxC6

    public CvVectorFloat6() {
        super(_ch);
    }

    public CvVectorFloat6(long addr) {
        super(_ch, addr);
    }

    public CvVectorFloat6(Mat m) {
        super(_ch, m);
    }

    public CvVectorFloat6(float[] a) {
        super(_ch, a);
    }
}
