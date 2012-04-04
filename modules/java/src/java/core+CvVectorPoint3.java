package org.opencv.core;

public class CvVectorPoint3 extends CvVectorInt {
    private static final int _ch = 3; //xxxC2

    public CvVectorPoint3() {
        super(_ch);
    }

    public CvVectorPoint3(long addr) {
        super(_ch, addr);
    }

    public CvVectorPoint3(Mat m) {
        super(_ch, m);
    }

    public CvVectorPoint3(Point3[] a) {
        super(_ch);
        if(a==null)
            return;
        int cnt = a.length;
        create(cnt);
        int buff[] = new int[_ch * cnt];
        for(int i=0; i<cnt; i++) {
        	Point3 p = a[i];
            buff[_ch*i]   = (int) p.x;
            buff[_ch*i+1] = (int) p.y;
            buff[_ch*i+2] = (int) p.z;
        }
        put(0, 0, buff); //TODO: check ret val!
    }

    public Point3[] toArray(Point3[] a) {
        int buff[] = super.toArray(null);
        if(buff.length == 0)
            return new Point3[0]; //null;
        int cnt = buff.length / _ch;
        Point3[] res = a;
        if(a==null || a.length<cnt)
            res = new Point3[cnt];
        for(int i=0; i<cnt; i++)
            res[i] = new Point3(buff[i*_ch], buff[i*_ch+1], buff[i*_ch+2]);
        return res;
    }
}
