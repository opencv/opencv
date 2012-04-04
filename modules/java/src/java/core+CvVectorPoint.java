package org.opencv.core;

public class CvVectorPoint extends CvVectorInt {
    private static final int _ch = 2; //xxxC2

    public CvVectorPoint() {
        super(_ch);
    }

    public CvVectorPoint(long addr) {
        super(_ch, addr);
    }

    public CvVectorPoint(Mat m) {
        super(_ch, m);
    }

    public CvVectorPoint(Point[] a) {
        super(_ch);
        if(a==null)
            return;
        int cnt = a.length;
        create(cnt);
        int buff[] = new int[_ch * cnt];
        for(int i=0; i<cnt; i++) {
        	Point p = a[i];
            buff[_ch*i+0] = (int) p.x;
            buff[_ch*i+1] = (int) p.y;
        }
        put(0, 0, buff); //TODO: check ret val!
    }

    public Point[] toArray(Point[] a) {
        int buff[] = super.toArray(null);
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
