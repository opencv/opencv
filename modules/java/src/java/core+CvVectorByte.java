package org.opencv.core;

public class CvVectorByte extends CvVector {
    private static final int _d = CvType.CV_8U;

    public CvVectorByte(int ch) {
        super(_d, ch);
    }

    public CvVectorByte() {
        super(_d, 1);
    }

    public CvVectorByte(int ch, long addr) {
        super(_d, ch, addr);
    }

    public CvVectorByte(long addr) {
        super(_d, 1, addr);
    }

    public CvVectorByte(int ch, Mat m) {
        super(_d, ch, m);
    }

    public CvVectorByte(int ch, byte[] a) {
        super(_d, ch);
        if(a!=null) {
            int cnt = a.length / ch;
            create(cnt);
            put(0, 0, a);
        }
    }

    public byte[] toArray(byte[] a) {
        int cnt = (int) total() * channels;
        if(cnt == 0)
            return new byte[0];//null;
        byte[] res = a;
        if(res==null || res.length<cnt)
            res = new byte[cnt];
        get(0, 0, res); //TODO: check ret val!
        return res;
    }

}
