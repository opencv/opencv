package org.opencv.core;

public class CvVectorDouble extends CvVector {
    private static final int _d = CvType.CV_64F;

    public CvVectorDouble(int ch) {
        super(_d, ch);
    }

    public CvVectorDouble() {
        this(1);
    }

    public CvVectorDouble(int ch, long addr) {
        super(_d, ch, addr);
    }

    public CvVectorDouble(int ch, Mat m) {
        super(_d, ch, m);
    }

    public CvVectorDouble(int ch, double[] a) {
        super(_d, ch);
        if(a!=null) {
            int cnt = a.length / ch;
            create(cnt);
            put(0, 0, a);
        }
    }

    public double[] toPrimitiveArray(double[] a) {
        int cnt = (int) total() * channels;
        if(cnt == 0)
            return new double[0];//null;
        double[] res = a;
        if(res==null || res.length<cnt)
            res = new double[cnt];
        get(0, 0, res); //TODO: check ret val!
        return res;
    }
}
