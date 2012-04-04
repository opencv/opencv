package org.opencv.core;

public class CvVectorPoint2f extends CvVectorFloat {
    private static final int _ch = 2; //xxxC2

    public CvVectorPoint2f() {
        super(_ch);
    }

    public CvVectorPoint2f(long addr) {
        super(_ch, addr);
    }

    public CvVectorPoint2f(Mat m) {
        super(_ch, m);
    }

    public CvVectorPoint2f(Point...a) {
        super(_ch);
        if(a==null || a.length==0)
            return;
        int cnt = a.length;
        create(cnt);
        float buff[] = new float[_ch * cnt];
        for(int i=0; i<cnt; i++) {
        	Point p = a[i];
            buff[_ch*i+0] = (float) p.x;
            buff[_ch*i+1] = (float) p.y;
        }
        put(0, 0, buff); //TODO: check ret val!
    }

    public Point[] toArray(Point[] a) {
        float buff[] = super.toArray(null);
        if(buff.length == 0)
            return new Point[0]; //null;
        int cnt = buff.length / _ch;
        Point[] res = a;
        if(a==null || a.length<cnt)
            res = new Point[cnt];
        for(int i=0; i<cnt; i++)
            res[i] = new Point(buff[i*_ch], buff[i*_ch+1]);
        return res;
    }
}
