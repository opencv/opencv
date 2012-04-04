package org.opencv.core;

public class CvVectorFloat extends CvVector {
    private static final int _d = CvType.CV_32F;

    public CvVectorFloat(int ch) {
        super(_d, ch);
    }

    public CvVectorFloat(int ch, long addr) {
        super(_d, ch, addr);
    }

    public CvVectorFloat(long addr) {
        super(_d, 1, addr);
    }

    public CvVectorFloat(int ch, Mat m) {
        super(_d, ch, m);
    }

    public CvVectorFloat(int ch, float[] a) {
        super(_d, ch);
        if(a!=null) {
            int cnt = a.length / ch;
            create(cnt);
            put(0, 0, a);
        }
    }

    public float[] toArray(float[] a) {
        int cnt = (int) total() * channels;
        if(cnt == 0)
            return new float[0];//null;
        float[] res = a;
        if(res==null || res.length<cnt)
            res = new float[cnt];
        get(0, 0, res); //TODO: check ret val!
        return res;
    }
}
